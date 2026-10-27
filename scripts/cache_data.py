"""Optional CPU preprocessing; cache RGB uint8, tokens and targets on a local SSD."""
import argparse
import json
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from navpi.checkpoints import sha256
from navpi.config import ModelConfig
from navpi.data import CAMERAS
from navpi.train import read_config, dataset_from_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sft_80gb.yaml")
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    cfg = read_config(a.config)
    cfg["data"][f"{a.split}_cache"] = None
    model = ModelConfig(**cfg.get("model", {}))
    dataset = dataset_from_config(cfg, a.split, model)
    if a.output.exists():
        raise FileExistsError(a.output)
    temporary = a.output.with_name(a.output.name + ".partial")
    temporary.mkdir(parents=True, exist_ok=False)
    arrays = None
    offset = 0
    for batch in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=a.workers):
        values = {key: value.numpy() for key, value in batch.items()}
        if arrays is None:
            arrays = {key: np.lib.format.open_memmap(temporary / f"{key}.npy", mode="w+",
                        dtype=value.dtype, shape=(len(dataset), *value.shape[1:])) for key, value in values.items()}
        count = len(batch["images"])
        for key, value in values.items():
            arrays[key][offset:offset + count] = value
        offset += count
        if offset % 1024 < 64:
            print(f"Cached {offset}/{len(dataset)}", flush=True)
    for array in arrays.values():
        array.flush()
    data = cfg["data"]
    metadata = {"samples": len(dataset), "image_size": model.image_size, "max_text_len": model.max_text_len,
        "cameras": list(CAMERAS), "manifest_sha256": sha256(data[f"{a.split}_manifest"]),
        "stats_sha256": sha256(data["stats"]), "tokenizer_sha256": sha256(data["tokenizer"])}
    (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2))
    temporary.rename(a.output)
    print(f"Ready: {a.output}")


if __name__ == "__main__":
    main()
