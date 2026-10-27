"""CPU-only official pi0.5 Orbax -> OpenPI-format BF16 safetensors conversion."""
import os
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from safetensors.torch import save_file
from .config import ModelConfig
from .vendor.jax_conversion import slice_paligemma_state_dict, slice_gemma_state_dict


def flatten(tree, prefix=""):
    result = {}
    for key, value in tree.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def convert_parameters(params, config=None):
    c = config or ModelConfig()
    paligemma = NS(vision_config=NS(hidden_size=c.vision_width, num_hidden_layers=c.vision_depth),
        text_config=NS(hidden_size=c.vlm_width, num_hidden_layers=c.depth,
                       num_attention_heads=c.heads, head_dim=c.head_dim))
    expert = NS(hidden_size=c.expert_width, num_hidden_layers=c.depth,
                num_attention_heads=c.heads, head_dim=c.head_dim, vocab_size=c.vocab_size)
    base, suffix = slice_paligemma_state_dict(flatten(params["PaliGemma"]), paligemma)
    base.update(slice_gemma_state_dict(suffix, expert, num_expert=1, checkpoint_dir="", pi05=True))
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        for key, target in (("kernel", "weight"), ("bias", "bias")):
            value = params[name][key]
            if isinstance(value, dict):
                value = value["value"]
            base[f"{name}.{target}"] = torch.from_numpy(value.T if key == "kernel" else value)
    return base


def convert(params_directory, output):
    # Set before importing JAX; never reserve GPU memory during checkpoint preparation.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    torch.set_num_threads(2)
    import jax
    import orbax.checkpoint as ocp
    with ocp.PyTreeCheckpointer() as checkpointer:
        path = Path(params_directory).resolve()
        metadata = checkpointer.metadata(path)
        item = {"params": metadata["params"]}
        params = checkpointer.restore(path, ocp.args.PyTreeRestore(item=item,
            restore_args=jax.tree.map(lambda _: ocp.ArrayRestoreArgs(
                restore_type=np.ndarray, dtype=np.float32), item)))["params"]
    tensors = convert_parameters(params)
    # Strict shape/key validation before serialization, without allocating a second model.
    from .model import NavPi
    from .checkpoints import HEAD_KEYS
    with torch.device("meta"):
        expected = NavPi(ModelConfig()).state_dict()
    if set(tensors) != set(expected):
        raise ValueError(f"Conversion key mismatch: {set(tensors) ^ set(expected)}")
    for name in expected.keys() - HEAD_KEYS:
        if tensors[name].shape != expected[name].shape:
            raise ValueError(f"Conversion shape mismatch: {name}")
    del expected, params
    for name in tensors:
        tensors[name] = tensors[name].to(torch.bfloat16).contiguous()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    save_file(tensors, str(partial), metadata={"source": "gs://openpi-assets/checkpoints/pi05_base",
              "format": "OpenPI PyTorch", "storage_dtype": "bfloat16"})
    partial.replace(target)
