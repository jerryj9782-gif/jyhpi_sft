"""Strict streaming checkpoint loading, including safetensors tied-weight aliases."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .config import ModelConfig
from .model import NavPi

HEAD_KEYS = {f"action_{side}_proj.{kind}" for side in ("in", "out") for kind in ("weight", "bias")}
EMBED = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
LM_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"
EXPERT_HEAD = "paligemma_with_expert.gemma_expert.lm_head.weight"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weight_plan(model, filename, *, base):
    state = model.state_dict()
    with safe_open(str(filename), framework="pt", device="cpu") as reader:
        keys = set(reader.keys())
        aliases = reader.metadata() or {}
        mapping, missing, mismatches = {}, [], []
        for name, tensor in state.items():
            if base and name in HEAD_KEYS:
                continue
            source = name if name in keys else aliases.get(name)
            if source is None and name == EMBED and LM_HEAD in keys:
                source = LM_HEAD
            if source not in keys:
                missing.append(name)
            elif list(reader.get_slice(source).get_shape()) != list(tensor.shape):
                mismatches.append(name)
            else:
                mapping[name] = source
        ignored = keys - set(mapping.values())
        allowed = (HEAD_KEYS | {EXPERT_HEAD, LM_HEAD}) if base else set()
        unexpected = ignored - allowed
        if missing or mismatches or unexpected:
            raise ValueError(f"Incompatible checkpoint: missing={missing}, shape={mismatches}, unexpected={sorted(unexpected)}")
    return {"mapping": mapping, "reinitialized": sorted(HEAD_KEYS) if base else [],
            "ignored": sorted(ignored), "loaded_tensors": len(mapping),
            "parameters": sum(p.numel() for p in model.parameters())}


def load_weights(model, filename, *, base):
    plan = weight_plan(model, filename, base=base)
    state = model.state_dict()
    with safe_open(str(filename), framework="pt", device="cpu") as reader, torch.no_grad():
        for name, source in plan["mapping"].items():
            state[name].copy_(reader.get_tensor(source))
    return {k: v for k, v in plan.items() if k != "mapping"}


def load_model(path, *, base=False, config=None, checkpointing="none"):
    path = Path(path)
    if path.is_dir():
        if not base:
            config = ModelConfig(**json.loads((path / "model_config.json").read_text()))
        path = path / "model.safetensors"
    config = config or ModelConfig()
    # Construct once on CPU. Avoids uninitialized rotary/position buffers from to_empty.
    model = NavPi(config, checkpointing=checkpointing)
    report = load_weights(model, path, base=base)
    return model, report
