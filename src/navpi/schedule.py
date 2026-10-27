"""Stable optimizer groups and a resumable ViT warmup -> full SFT schedule."""
import math


def vision_unfreeze_step(cfg, steps_per_epoch):
    mode = cfg.get("vision_mode", "full")
    if mode not in ("staged", "full", "frozen"):
        raise ValueError("vision_mode must be staged, full or frozen")
    if mode == "full":
        return 0
    if mode == "frozen":
        return math.inf
    epochs = cfg.get("vision_freeze_epochs", 1)
    if not 0 <= epochs < cfg["epochs"]:
        raise ValueError("vision_freeze_epochs must be >= 0 and < total epochs")
    return math.ceil(steps_per_epoch * epochs)


def apply_phase(model, cfg, step, steps_per_epoch):
    frozen = step < vision_unfreeze_step(cfg, steps_per_epoch)
    model.set_vision_frozen(frozen)
    return "vision_frozen" if frozen else "full_sft"


def cosine_factor(step, total, warmup, minimum):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1, (step - warmup) / max(1, total - warmup))
    return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * progress))


def group_lr_factor(step, *, name, cfg, total, steps_per_epoch):
    if name == "vision":
        start = vision_unfreeze_step(cfg, steps_per_epoch)
        if step < start:
            return 0.0
        return cosine_factor(step - start, max(1, total - start),
            cfg.get("vision_warmup_steps", 50), cfg["min_lr_ratio"])
    return cosine_factor(step, total, max(1, int(total * cfg["warmup_ratio"])), cfg["min_lr_ratio"])
