"""Inference API: physical ego-relative x/y in meters and heading in radians."""
from contextlib import nullcontext
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from .checkpoints import load_model
from .data import CAMERAS, Normalizer, Tokenizer, resize_image


class NavPiPolicy:
    def __init__(self, checkpoint, device="cuda", steps=10):
        checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        self.model, _ = load_model(checkpoint, base=False)
        self.model.to(self.device).eval()
        self.normalizer = Normalizer(checkpoint / "norm_stats.json")
        self.tokenizer = Tokenizer(checkpoint / "tokenizer.model", self.model.config.max_text_len)
        self.steps = steps

    @torch.inference_mode()
    def predict(self, images, state, prompt, *, seed=42):
        pixels = np.stack([resize_image(Image.fromarray(np.asarray(images[c], np.uint8)),
                                       self.model.config.image_size) for c in CAMERAS])
        state = np.asarray(state, np.float32)
        if state.shape != (5,) or not np.isfinite(state).all():
            raise ValueError("state = [speed, vx, vy, ax, ay]")
        ids, mask = self.tokenizer.encode(prompt, self.normalizer.encode(state, "state"))
        pixels = torch.from_numpy(pixels).unsqueeze(0).to(self.device).float() / 127.5 - 1
        ids = torch.from_numpy(ids).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn((1, 8, 3), generator=generator, device=self.device)
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
        with autocast:
            actions = self.model.sample(pixels, ids, mask, steps=self.steps, noise=noise)
        return self.normalizer.decode_actions(actions)[0].cpu().numpy().astype(np.float32)
