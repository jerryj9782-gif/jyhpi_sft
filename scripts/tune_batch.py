"""Explicitly requested future GPU benchmark. Each candidate uses fresh torchrun processes.

Includes forward, backward, DDP, gradient clipping, fused AdamW states and warmup.
No model checkpoints are written. This DOES perform temporary optimizer updates.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import math
import yaml
from navpi.train import read_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sft_80gb.yaml")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    p.add_argument("--checkpointing", nargs="+", choices=["none", "vision", "full"], default=["full", "none"])
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--memory-budget-gib", type=float, default=74)
    p.add_argument("--output", type=Path, default=Path("runs/batch_tuning"))
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--vision-mode", choices=["full", "frozen"], default="full")
    p.add_argument("--target-batch", type=int, default=64)
    a = p.parse_args()
    if a.steps < 4 or a.gpus < 1 or a.target_batch < 1 or min(a.batches) < 1:
        p.error("Need >= 4 steps and positive gpus, batches, target-batch")
    a.output.mkdir(parents=True, exist_ok=False)
    results = []
    for mode in a.checkpointing:
        for batch in a.batches:
            target = a.output / f"{mode}_b{batch}"
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       f"--nproc_per_node={a.gpus}", "-m", "navpi.train", "--config", a.config]
            overrides = [*a.set, f"train.output={target}", f"train.micro_batch={batch}",
                         "train.grad_accum=1", f"train.checkpointing={mode}",
                         f"train.max_steps={a.steps}", "train.save_checkpoints=false", "train.validation=false",
                         f"train.vision_mode={a.vision_mode}"]
            for item in overrides:
                command.extend(["--set", item])
            logfile = a.output / f"{mode}_b{batch}.log"
            print(f"Benchmarking {mode}, micro batch {batch}; {logfile}", flush=True)
            env = {**os.environ, "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
            with logfile.open("w") as log:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            result = {"checkpointing": mode, "micro_batch": batch, "exit_code": completed.returncode}
            if completed.returncode == 0:
                metrics = json.loads((target / "summary.json").read_text())["warmup_excluded_metrics"]
                if not metrics:
                    raise RuntimeError("No measured steps after warmup")
                peak = max(row["peak_reserved_gib"] for row in metrics)
                result.update(peak_reserved_gib=peak,
                    samples_per_second=sum(row["samples_per_second"] for row in metrics) / len(metrics),
                    within_budget=peak <= a.memory_budget_gib)
            else:
                logtext = logfile.read_text()
                result["oom"] = "out of memory" in logtext.lower() or "OutOfMemoryError" in logtext
                if not result["oom"]:
                    results.append(result)
                    (a.output / "results.json").write_text(json.dumps(results, indent=2))
                    raise RuntimeError(f"Non-OOM failure; inspect {logfile}")
            results.append(result)
            (a.output / "results.json").write_text(json.dumps(results, indent=2))
            if result.get("oom"):
                break
    feasible = [row for row in results if row.get("within_budget")]
    best = max(feasible, key=lambda row: row["samples_per_second"]) if feasible else None
    (a.output / "recommendation.json").write_text(json.dumps(best, indent=2))
    if best:
        config = read_config(a.config, a.set)
        config["train"].update(micro_batch=best["micro_batch"], checkpointing=best["checkpointing"],
            grad_accum=max(1, math.ceil(a.target_batch / (a.gpus * best["micro_batch"]))),
            output="runs/navsim_tuned")
        (a.output / "recommended.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        print("Effective batch:", a.gpus * best["micro_batch"] * config["train"]["grad_accum"])
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
