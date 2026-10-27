"""Portable JSONL manifests and optional pre-resized uint8 mmap caches."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import sentencepiece as spm
import torch
from torch.utils.data import Dataset

CAMERAS = ("cam_front_center", "cam_front_left", "cam_front_right")
SENSOR_CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0")


def read_rows(path):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


class Normalizer:
    def __init__(self, filename):
        self.stats = json.loads(Path(filename).read_text())["norm_stats"]
        for key, dim in (("state", 5), ("actions", 3)):
            for field in ("q01", "q99"):
                if np.asarray(self.stats[key][field]).shape != (dim,):
                    raise ValueError(f"{key}.{field} must have {dim} values")
            low, high = self.limits(key)
            if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high < low):
                raise ValueError(f"Invalid statistics: {key}")

    def limits(self, key):
        return (np.asarray(self.stats[key]["q01"], np.float32),
                np.asarray(self.stats[key]["q99"], np.float32))

    def encode(self, value, key):
        lo, hi = self.limits(key)
        return (np.asarray(value, np.float32) - lo) / (hi - lo + 1e-6) * 2 - 1

    def decode_actions(self, value):
        lo, hi = self.limits("actions")
        if isinstance(value, torch.Tensor):
            lo = torch.as_tensor(lo, device=value.device)
            hi = torch.as_tensor(hi, device=value.device)
        result = (value + 1) / 2 * (hi - lo + 1e-6) + lo
        result[..., 2] = (result[..., 2] + np.pi) % (2 * np.pi) - np.pi
        return result


class Tokenizer:
    def __init__(self, filename, max_len=128):
        self.processor = spm.SentencePieceProcessor(model_file=str(filename))
        self.max_len = max_len

    def encode(self, prompt, state):
        # Same pi0.5 state-token format; only the five real ego features, no robot padding.
        state = np.clip(state, -1, 1)
        bins = np.linspace(-1, 1, 257)[:-1]
        codes = np.clip(np.digitize(state, bins) - 1, 0, 255)
        prompt = prompt.strip().replace("_", " ").replace("\n", " ")
        text = f"Task: {prompt}, State: {' '.join(map(str, codes))};\nAction: "
        ids = self.processor.encode(text, add_bos=True)
        if len(ids) > self.max_len:
            raise ValueError(f"Prompt needs {len(ids)} tokens, configured max={self.max_len}; refusing truncation")
        mask = np.arange(self.max_len) < len(ids)
        return np.pad(np.array(ids, np.int64), (0, self.max_len - len(ids))), mask


def resize_image(image, size=224):
    image = image.convert("RGB")
    # Deterministic letterbox; no independent crop/rotation that would corrupt geometry.
    image.thumbnail((size, size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return np.asarray(canvas, np.uint8).transpose(2, 0, 1).copy()


def image_path(root, name):
    path = Path(name)
    if path.is_absolute():
        if path.is_file():
            return path
        # Old manifests stored an obsolete dataset prefix. Use explicit supplied root.
        parts = path.parts
        if "sensor_blobs" in parts:
            path = Path(*parts[parts.index("sensor_blobs"):])
        else:
            raise FileNotFoundError(name)
    return Path(root) / path


class ManifestDataset(Dataset):
    def __init__(self, manifest, root, stats, tokenizer, *, image_size=224, max_text_len=128):
        self.rows = read_rows(manifest)
        self.root = root
        self.image_size = image_size
        self.normalizer = Normalizer(stats)
        encoder = Tokenizer(tokenizer, max_text_len)
        self.tokens, self.masks, self.actions = [], [], []
        for row in self.rows:
            state, actions = np.asarray(row["state"]), np.asarray(row["actions"])
            if state.shape != (5,) or actions.shape != (8, 3):
                raise ValueError(f"Bad NAVSIM shape for {row.get('token')}")
            if not np.isfinite(state).all() or not np.isfinite(actions).all():
                raise ValueError("Nonfinite manifest entry")
            ids, mask = encoder.encode(row["prompt"], self.normalizer.encode(state, "state"))
            self.tokens.append(ids); self.masks.append(mask)
            self.actions.append(self.normalizer.encode(actions, "actions"))
        self.tokens, self.masks, self.actions = map(np.asarray, (self.tokens, self.masks, self.actions))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        images = []
        for camera in CAMERAS:
            with Image.open(image_path(self.root, self.rows[index]["images"][camera])) as im:
                images.append(resize_image(im, self.image_size))
        return {"images": torch.from_numpy(np.stack(images)),
                "tokens": torch.from_numpy(self.tokens[index].copy()),
                "token_mask": torch.from_numpy(self.masks[index].copy()),
                "actions": torch.from_numpy(self.actions[index].copy())}


class CachedDataset(Dataset):
    def __init__(self, path, *, expected=None):
        self.path = Path(path)
        self.meta = json.loads((self.path / "metadata.json").read_text())
        if expected:
            for key, value in expected.items():
                if self.meta.get(key) != value:
                    raise ValueError(f"Stale/incompatible cache: {key}")
        self.arrays = None

    def __len__(self):
        return self.meta["samples"]

    def __getitem__(self, index):
        if self.arrays is None:
            self.arrays = {key: np.load(self.path / f"{key}.npy", mmap_mode="r")
                           for key in ("images", "tokens", "token_mask", "actions")}
        return {key: torch.from_numpy(value[index].copy()) for key, value in self.arrays.items()}


def to_device(batch, device):
    result = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    result["images"] = result["images"].float().div_(127.5).sub_(1)
    return result
