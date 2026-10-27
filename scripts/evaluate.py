"""Validation ADE/FDE/heading; these metrics are NOT NAVSIM PDM/EPDMS scores."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from navpi.checkpoints import load_model
from navpi.data import ManifestDataset, Normalizer, to_device
from navpi.train import read_config
from navpi.checkpoints import sha256
from navpi.validation import token_noise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sft_80gb.yaml")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest", help="Defaults to validation; test is always an explicit choice")
    a = p.parse_args()
    if a.batch_size < 1 or a.steps < 1:
        p.error("batch-size and steps must be positive")
    cfg = read_config(a.config); data = cfg["data"]
    a.output.mkdir(parents=True, exist_ok=False)
    model, _ = load_model(a.checkpoint)
    device = torch.device(a.device); model.to(device).eval()
    dataset = ManifestDataset(a.manifest or data["val_manifest"], data["root"],
        a.checkpoint / "norm_stats.json", a.checkpoint / "tokenizer.model",
        image_size=model.config.image_size, max_text_len=model.config.max_text_len)
    normalizer = Normalizer(a.checkpoint / "norm_stats.json")
    totals = np.zeros(5, np.float64)
    offset = 0
    with (a.output / "predictions.jsonl").open("w") as f, torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=a.batch_size, shuffle=False, num_workers=2):
            rows = dataset.rows[offset:offset + len(batch["actions"])]
            noises = token_noise([row["token"] for row in rows], device, a.seed)
            batch = to_device(batch, device)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with autocast:
                prediction = model.sample(batch["images"], batch["tokens"], batch["token_mask"],
                                          steps=a.steps, noise=noises)
            prediction = normalizer.decode_actions(prediction).cpu().numpy()
            if not np.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite predicted trajectory")
            truth = np.asarray([row["actions"] for row in rows])
            errors = np.linalg.norm(prediction[..., :2] - truth[..., :2], axis=-1)
            heading = np.abs((prediction[..., 2] - truth[..., 2] + np.pi) % (2 * np.pi) - np.pi)
            velocity = np.asarray([row["state"][1:3] for row in rows])
            cv = velocity[:, None] * np.arange(.5, 4.01, .5)[None, :, None]
            cv_error = np.linalg.norm(cv - truth[..., :2], axis=-1)
            totals += [errors.mean(1).sum(), errors[:, -1].sum(), heading.mean(1).sum(),
                       cv_error.mean(1).sum(), cv_error[:, -1].sum()]
            for row, trajectory in zip(rows, prediction, strict=True):
                f.write(json.dumps({"token": row["token"], "trajectory": trajectory.tolist()}) + "\n")
            offset += len(rows)
    if not offset:
        raise ValueError("Empty evaluation manifest")
    metrics = dict(zip(("ade_m", "fde_m", "heading_mae_rad", "cv_ade_m", "cv_fde_m"), totals / offset))
    metrics.update(samples=offset, checkpoint=str(a.checkpoint), steps=a.steps, seed=a.seed,
                   manifest_sha256=sha256(a.manifest or data["val_manifest"]),
                   checkpoint_sha256=sha256(a.checkpoint / "model.safetensors"),
                   note="Displacement metrics only; not PDM/EPDMS")
    (a.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
