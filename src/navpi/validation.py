"""Physical trajectory validation with fixed per-token noise and exact DDP sharding."""
from contextlib import nullcontext
import hashlib
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from .data import Normalizer, read_rows, to_device


def token_noise(tokens, device, seed):
    values = []
    for token in tokens:
        derived = (seed + int(hashlib.sha256(token.encode()).hexdigest()[:15], 16)) % 2**63
        values.append(torch.randn((8, 3), device=device,
            generator=torch.Generator(device=device).manual_seed(derived)))
    return torch.stack(values)


@torch.inference_mode()
def validate(model, dataset, data_cfg, train_cfg, device, rank=0, world=1):
    rows = read_rows(data_cfg["val_manifest"])
    limit = train_cfg.get("val_max_samples", 0) or len(rows)
    limit = min(limit, len(rows))
    if limit < 1 or len(dataset) != len(rows):
        raise ValueError("Validation requires a nonempty matching manifest/dataset")
    indices = list(range(rank, limit, world))  # no DistributedSampler duplicate padding
    loader = DataLoader(Subset(dataset, indices), batch_size=train_cfg["val_batch_size"],
        shuffle=False, num_workers=train_cfg["workers"], pin_memory=device.type == "cuda")
    normalizer = Normalizer(data_cfg["stats"])
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    was_training, was_frozen = model.training, model.vision_frozen
    model.eval()
    offset = 0
    for batch in loader:
        count = len(batch["actions"])
        selected = [rows[i] for i in indices[offset:offset + count]]
        batch = to_device(batch, device)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with amp:
            output = model.sample(batch["images"], batch["tokens"], batch["token_mask"],
                steps=train_cfg["val_sample_steps"],
                noise=token_noise([row["token"] for row in selected], device, train_cfg["seed"]))
        output = normalizer.decode_actions(output)
        if not torch.isfinite(output).all():
            raise FloatingPointError("Nonfinite validation trajectory")
        truth = torch.as_tensor(np.asarray([row["actions"] for row in selected]), device=device)
        displacement = torch.linalg.vector_norm(output[..., :2] - truth[..., :2], dim=-1)
        heading = ((output[..., 2] - truth[..., 2] + np.pi) % (2 * np.pi) - np.pi).abs()
        totals += torch.stack((displacement.mean(1).sum(), displacement[:, -1].sum(),
                              heading.mean(1).sum(), totals.new_tensor(count)))
        offset += count
    if world > 1:
        dist.all_reduce(totals)
    model.train(was_training)
    model.set_vision_frozen(was_frozen)
    return {"ade_m": (totals[0] / totals[3]).item(), "fde_m": (totals[1] / totals[3]).item(),
            "heading_mae_rad": (totals[2] / totals[3]).item(), "samples": int(totals[3].item()),
            "metric": "trajectory displacement, not PDM"}
