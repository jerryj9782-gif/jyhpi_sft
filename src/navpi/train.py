"""Single GPU / torchrun DDP full-parameter SFT. Importing this module starts nothing."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from safetensors.torch import save_file
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import yaml

from .checkpoints import load_model, sha256
from .config import ModelConfig
from .data import CachedDataset, ManifestDataset, to_device
from .schedule import apply_phase, group_lr_factor
from .validation import validate


def read_config(filename, overrides=()):
    cfg = yaml.safe_load(Path(filename).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"Unknown configuration override {key}")
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def dataset_from_config(cfg, split, model_config):
    data = cfg["data"]
    manifest = data[f"{split}_manifest"]
    if data.get(f"{split}_cache"):
        expected = {"manifest_sha256": sha256(manifest), "stats_sha256": sha256(data["stats"]),
                    "tokenizer_sha256": sha256(data["tokenizer"]),
                    "image_size": model_config.image_size, "max_text_len": model_config.max_text_len,
                    "cameras": ["cam_front_center", "cam_front_left", "cam_front_right"]}
        return CachedDataset(data[f"{split}_cache"], expected=expected)
    return ManifestDataset(manifest, data["root"], data["stats"], data["tokenizer"],
                           image_size=model_config.image_size, max_text_len=model_config.max_text_len)


def make_optimizer(model, cfg, *, cuda):
    groups = {name: [] for name in ("base", "vision", "projector", "expert", "head")}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise RuntimeError(f"Full SFT unexpectedly froze {name}")
        group = "base"
        if "vision_tower" in name:
            group = "vision"
        elif "multi_modal_projector" in name:
            group = "projector"
        elif "gemma_expert" in name or name.startswith("time_mlp"):
            group = "expert"
        elif name.startswith(("action_in_proj.", "action_out_proj.")):
            group = "head"
        groups[group].append(parameter)
    rates = {"base": cfg["lr"], "vision": cfg.get("vision_lr", cfg["lr"]),
             "projector": cfg.get("projector_lr", cfg["lr"]),
             "expert": cfg.get("expert_lr", cfg["lr"]), "head": cfg["head_lr"]}
    return torch.optim.AdamW([{"params": params, "lr": rates[name], "name": name}
                             for name, params in groups.items()],
        betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg["weight_decay"], fused=cuda)


def lr_factor(step, *, total, warmup, minimum):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1, (step - warmup) / max(1, total - warmup))
    return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * progress))


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def save_checkpoint(model, optimizer, scheduler, cfg, output, step, epoch, next_batch, device, rank, world,
                    validation_state=None):
    local_rng = rng_state(device)
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, local_rng)
    else:
        states[0] = local_rng
    if rank == 0:
        target = output / f"step_{step:08d}"
        temporary = output / f".step_{step:08d}.partial"
        if target.exists() or temporary.exists():
            raise FileExistsError(f"Checkpoint already exists: {target}")
        temporary.mkdir(parents=True)
        save_file({key: tensor.detach().cpu().contiguous() for key, tensor in model.state_dict().items()},
                  str(temporary / "model.safetensors"))
        (temporary / "model_config.json").write_text(json.dumps(model.config.to_dict(), indent=2))
        (temporary / "run_config.json").write_text(json.dumps(cfg, indent=2))
        (temporary / "data_hashes.json").write_text(json.dumps({key: sha256(cfg["data"][key])
            for key in ("train_manifest", "val_manifest", "stats", "tokenizer") if cfg["data"].get(key)}, indent=2))
        shutil.copy2(cfg["data"]["stats"], temporary / "norm_stats.json")
        shutil.copy2(cfg["data"]["tokenizer"], temporary / "tokenizer.model")
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "step": step, "epoch": epoch, "next_batch": next_batch,
                    "rng": states, "world_size": world, "validation": validation_state,
                    "format_version": 2}, temporary / "training.pt")
        (temporary / "COMPLETE").write_text("ok\n")
        temporary.rename(target)
    if world > 1:
        dist.barrier()


def run(cfg, *, resume=None, warm_start=None):
    world, rank, local_rank = (int(os.getenv(name, default)) for name, default in
                               (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA. For CPU checks use pytest, not the training command.")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training configuration requires BF16-capable GPUs")
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60), device_id=device)
    t = cfg["train"]
    if t["epochs"] < 1 or t["micro_batch"] < 1 or t["workers"] < 0:
        raise ValueError("epochs, micro_batch must be positive; workers must be nonnegative")
    torch.set_num_threads(t["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed = t["seed"] + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    output = Path(t["output"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "run_config.json").exists() and not resume:
            raise FileExistsError("Use a new output directory or --resume")
    if world > 1:
        dist.barrier()
    if resume and not (Path(resume) / "COMPLETE").is_file():
        raise ValueError("Resume requires a completed training checkpoint")
    if warm_start and sha256(Path(warm_start) / "norm_stats.json") != sha256(cfg["data"]["stats"]):
        raise ValueError("Warm start requires the checkpoint's normalization statistics")
    model, loading = load_model(resume or warm_start or cfg["init_checkpoint"], base=not bool(resume or warm_start),
                               config=ModelConfig(**cfg.get("model", {})), checkpointing=t["checkpointing"])
    model.to(device).train()
    dataset = dataset_from_config(cfg, "train", model.config)
    val_dataset = dataset_from_config(cfg, "val", model.config) if t.get("validation", False) else None
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                 seed=t["seed"], drop_last=True)
    options = {"num_workers": t["workers"], "pin_memory": True,
               "persistent_workers": t["workers"] > 0}
    if t["workers"]:
        options["prefetch_factor"] = t["prefetch_factor"]
    # Dedicated generator prevents loader creation/skipping from consuming flow-matching RNG.
    loader = DataLoader(dataset, batch_size=t["micro_batch"], sampler=sampler, drop_last=True,
                        generator=torch.Generator().manual_seed(t["seed"]), **options)
    if not len(loader):
        raise ValueError("Dataset too small for world size and micro batch")
    accum = t["grad_accum"]
    if accum < 1:
        raise ValueError("grad_accum must be >= 1")
    optimizer = make_optimizer(model, t, cuda=True)
    steps_per_epoch = math.ceil(len(loader) / accum)
    total_steps = steps_per_epoch * t["epochs"]
    from functools import partial
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [partial(group_lr_factor,
        name=group["name"], cfg=t, total=total_steps, steps_per_epoch=steps_per_epoch)
        for group in optimizer.param_groups])
    step, start_epoch, skip = 0, 0, 0
    validation_state = {"best_ade": float("inf"), "bad_epochs": 0, "best_step": None}
    if resume:
        resume = Path(resume)
        if not (resume / "COMPLETE").is_file():
            raise ValueError("Resume requires a completed training checkpoint")
        old = json.loads((resume / "run_config.json").read_text())
        # Exact resumption requires the same schedule, data ordering and accumulation.
        for key in ("epochs", "micro_batch", "grad_accum", "seed", "lr", "head_lr", "warmup_ratio", "min_lr_ratio",
                    "vision_mode", "vision_freeze_epochs", "vision_lr", "vision_warmup_steps", "projector_lr",
                    "expert_lr", "weight_decay", "grad_clip", "validation", "val_max_samples", "val_sample_steps"):
            if old["train"].get(key) != t.get(key):
                raise ValueError(f"Resume changed train.{key}; start a new run from model weights instead")
        hashes = json.loads((resume / "data_hashes.json").read_text())
        for split_key in hashes:
            if hashes[split_key] != sha256(cfg["data"][split_key]):
                raise ValueError(f"Resume data changed: {split_key}")
        state = torch.load(resume / "training.pt", map_location="cpu", weights_only=False)
        if state.get("format_version") != 2:
            raise ValueError("Older optimizer group format: use --warm-start to begin a new schedule")
        if state["world_size"] != world:
            raise ValueError("Exact resume requires the original world size")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, start_epoch, skip = state["step"], state["epoch"], state["next_batch"]
        restore_rng(state["rng"][rank], device)
        validation_state = state.get("validation") or validation_state
        del state
    execution_model = model
    if t["compile"]:
        # Inductor requires its own warmup benchmark; optional, never silently enabled.
        execution_model = torch.compile(model, mode="default", dynamic=False)
    if world > 1:
        # pi0.5's last prefix FFN and final prefix norm do not affect action loss.
        # find_unused_parameters handles these inherited dead ends without freezing any block.
        execution_model = DDP(execution_model, device_ids=[local_rank], broadcast_buffers=False,
                              gradient_as_bucket_view=True, bucket_cap_mb=t["bucket_cap_mb"],
                              find_unused_parameters=True)
    if rank == 0:
        (output / "run_config.json").write_text(json.dumps(cfg, indent=2))
        print(json.dumps({"loading": loading, "world_size": world,
                          "effective_batch": world * t["micro_batch"] * accum,
                          "steps": total_steps, "parameters_dtype": "float32",
                          "compute_dtype": "bfloat16"}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    loss_sum = torch.zeros((), device=device)
    samples_window = 0
    times = []
    max_steps = t.get("max_steps")
    if max_steps and step >= max_steps:
        raise ValueError("Checkpoint has already reached max_steps; increase that limit to continue")
    last_saved = -1
    early_stop = False
    previous_phase = None
    for epoch in range(start_epoch, t["epochs"]):
        sampler.set_epoch(epoch)
        for index, batch in enumerate(loader):
            if epoch == start_epoch and index < skip:
                continue
            phase = apply_phase(model, t, step, steps_per_epoch)
            if phase != previous_phase:
                if rank == 0:
                    print(json.dumps({"phase": phase, "step": step}), flush=True)
                previous_phase = phase
            group_start = index // accum * accum
            group_size = min(accum, len(loader) - group_start)
            boundary = (index + 1) % accum == 0 or index + 1 == len(loader)
            context = execution_model.no_sync() if world > 1 and not boundary else nullcontext()
            batch = to_device(batch, device)
            with context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = execution_model(**batch)
                    scaled_loss = loss / group_size
                scaled_loss.backward()
            loss_sum += loss.detach() / group_size
            samples_window += len(batch["actions"])
            if not boundary:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"], error_if_nonfinite=True)
            step_lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
            optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
            step += 1
            # Synchronization is deliberate: reported throughput includes actual GPU completion.
            torch.cuda.synchronize(device)
            duration = time.perf_counter() - started
            reduced = torch.stack((loss_sum, grad_norm.float(), torch.tensor(duration, device=device)))
            if world > 1:
                dist.all_reduce(reduced[:2], op=dist.ReduceOp.SUM)
                dist.all_reduce(reduced[2:], op=dist.ReduceOp.MAX)
                reduced[:2] /= world
            duration = reduced[2].item()
            peaks = torch.tensor([torch.cuda.max_memory_allocated(device),
                                  torch.cuda.max_memory_reserved(device)], device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
            metrics = {"step": step, "epoch": epoch + (index + 1) / len(loader),
                       "loss": reduced[0].item(), "grad_norm": reduced[1].item(),
                       "seconds": duration, "samples_per_second": samples_window * world / duration,
                       "lr": step_lrs["base"], "group_lrs": step_lrs, "phase": phase,
                       "peak_allocated_gib": peaks[0].item() / 2**30,
                       "peak_reserved_gib": peaks[1].item() / 2**30}
            if rank == 0:
                with (output / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)
            times.append(metrics)
            at_end = index + 1 == len(loader)
            stopping = bool(max_steps and step >= max_steps)
            if at_end and val_dataset is not None:
                validation_metrics = validate(model, val_dataset, cfg["data"], t, device, rank, world)
                validation_metrics.update(epoch=epoch + 1, step=step)
                if validation_metrics["ade_m"] < validation_state["best_ade"] - t.get("early_stop_min_delta", 0):
                    validation_state.update(best_ade=validation_metrics["ade_m"], bad_epochs=0, best_step=step)
                else:
                    validation_state["bad_epochs"] += 1
                patience = t.get("early_stop_patience", 0)
                early_stop = bool(patience and epoch + 1 >= t.get("min_epochs", 1)
                                  and validation_state["bad_epochs"] >= patience)
                if rank == 0:
                    with (output / "validation.jsonl").open("a") as f:
                        f.write(json.dumps(validation_metrics) + "\n")
                    print(json.dumps({"validation": validation_metrics, "early_stop": early_stop}), flush=True)
            save = t["save_checkpoints"] and (at_end or stopping or
                    (t["save_every"] and step % t["save_every"] == 0))
            if save:
                save_checkpoint(model, optimizer, scheduler, cfg, output, step,
                                epoch + int(at_end), 0 if at_end else index + 1, device, rank, world,
                                validation_state=validation_state)
                last_saved = step
                if rank == 0 and validation_state["best_step"] == step:
                    (output / "best.json").write_text(json.dumps({**validation_state,
                        "checkpoint": str(output / f"step_{step:08d}"), "selection": "validation ADE"}, indent=2))
            loss_sum.zero_(); samples_window = 0
            started = time.perf_counter()
            if stopping or early_stop:
                break
        skip = 0
        if (max_steps and step >= max_steps) or early_stop:
            break
    if rank == 0:
        (output / "summary.json").write_text(json.dumps({"completed_steps": step,
            "last_saved_step": last_saved, "early_stopped": early_stop, "validation": validation_state,
            "warmup_excluded_metrics": times[2:]}, indent=2))
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_80gb.yaml")
    parser.add_argument("--set", action="append", default=[])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume")
    group.add_argument("--warm-start", help="Preserve trained NAVSIM head, reset optimizer and schedule")
    args = parser.parse_args()
    run(read_config(args.config, args.set), resume=args.resume, warm_start=args.warm_start)


if __name__ == "__main__":
    main()
