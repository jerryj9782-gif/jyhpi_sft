"""Explicit, checkpointed architecture. Never infer NAVSIM shapes from robot weights."""
from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    action_dim: int = 3
    horizon: int = 8
    dt: float = 0.5
    image_size: int = 224
    patch_size: int = 14
    max_text_len: int = 64
    state_dim: int = 5
    vocab_size: int = 257152
    vlm_width: int = 2048
    vlm_mlp: int = 16384
    expert_width: int = 1024
    expert_mlp: int = 4096
    depth: int = 18
    heads: int = 8
    kv_heads: int = 1
    head_dim: int = 256
    vision_width: int = 1152
    vision_mlp: int = 4304
    vision_depth: int = 27
    vision_heads: int = 16

    def __post_init__(self):
        if (self.action_dim, self.horizon, self.dt) != (3, 8, 0.5):
            raise ValueError("This model outputs exactly 8 x (x, y, heading), at 0.5 s intervals")
        if self.heads % self.kv_heads:
            raise ValueError("heads must be divisible by kv_heads")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def tiny(cls):
        """Same execution path with small dimensions for CPU correctness tests only."""
        return cls(image_size=28, patch_size=14, max_text_len=8, vocab_size=128,
                   vlm_width=32, vlm_mlp=64, expert_width=16, expert_mlp=32,
                   depth=2, heads=2, kv_heads=1, head_dim=16,
                   vision_width=32, vision_mlp=64, vision_depth=2, vision_heads=4)
