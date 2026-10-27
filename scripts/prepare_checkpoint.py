"""Fetch official pi0.5 and convert on CPU, or reuse an OpenPI-format checkpoint.

Never treat a LeRobot-format checkpoint as interchangeable. Structural validation
must pass. Existing files are reused; --expected-sha256 can verify transferred bytes.
"""
import argparse
import json
from pathlib import Path
import shutil
import urllib.request
import torch
from navpi.checkpoints import sha256, weight_plan
from navpi.config import ModelConfig
from navpi.model import NavPi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, help="Local OpenPI PyTorch safetensors")
    p.add_argument("--hf-repo", help="Explicit OpenPI-format mirror; no default third-party source")
    p.add_argument("--revision", help="Immutable commit hash required with --hf-repo")
    p.add_argument("--tokenizer-source", type=Path)
    p.add_argument("--output", type=Path, default=Path("checkpoints/pi05_base"))
    p.add_argument("--tokenizer-output", type=Path, default=Path("assets/tokenizer/tokenizer.model"))
    p.add_argument("--expected-sha256")
    p.add_argument("--jax-source", type=Path, help="Existing official Orbax checkpoint root")
    p.add_argument("--download-dir", type=Path, default=Path(".checkpoint_download/pi05_base"))
    a = p.parse_args()
    if sum(bool(x) for x in (a.source, a.hf_repo, a.jax_source)) > 1:
        p.error("Choose only one checkpoint source")
    if a.hf_repo and (not a.revision or len(a.revision) != 40):
        p.error("--hf-repo requires a full immutable 40-character commit revision")
    a.output.mkdir(parents=True, exist_ok=True)
    target = a.output / "model.safetensors"
    existed = target.exists()
    if not target.exists():
        if a.source:
            source = a.source / "model.safetensors" if a.source.is_dir() else a.source
        elif a.hf_repo and a.revision:
            from huggingface_hub import hf_hub_download
            source = Path(hf_hub_download(a.hf_repo, "model.safetensors", revision=a.revision))
        else:
            import importlib.util
            if any(importlib.util.find_spec(name) is None for name in ("jax", "orbax", "google_crc32c")):
                p.error("Official CPU conversion needs: python -m pip install -e '.[checkpoint]'")
            from navpi.download import download_official
            from navpi.convert_checkpoint import convert
            params = a.jax_source / "params" if a.jax_source else download_official(a.download_dir)
            convert(params, target)
            source = None
        if source:
            temporary = target.with_suffix(".partial")
            shutil.copyfile(source, temporary)
            temporary.rename(target)
    digest = sha256(target)
    if a.expected_sha256 and digest != a.expected_sha256:
        raise ValueError("Checkpoint SHA256 mismatch")
    a.tokenizer_output.parent.mkdir(parents=True, exist_ok=True)
    if not a.tokenizer_output.exists():
        if a.tokenizer_source:
            shutil.copyfile(a.tokenizer_source, a.tokenizer_output)
        else:
            temporary = a.tokenizer_output.with_suffix(".partial")
            urllib.request.urlretrieve("https://storage.googleapis.com/big_vision/paligemma_tokenizer.model", temporary)
            temporary.rename(a.tokenizer_output)
    tokenizer_sha = sha256(a.tokenizer_output)
    if tokenizer_sha != "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6":
        raise ValueError("Tokenizer differs from the official PaliGemma tokenizer")
    # Full-size parameter shapes without allocating the model or touching CUDA.
    with torch.device("meta"):
        model = NavPi(ModelConfig())
    plan = weight_plan(model, target, base=True)
    del plan["mapping"]
    record = {"checkpoint_sha256": digest, "checkpoint_bytes": target.stat().st_size,
              "tokenizer_sha256": tokenizer_sha, "validation": plan,
              "source": str(a.source or a.jax_source) if a.source or a.jax_source else a.hf_repo or
                  ("existing local file" if existed else "gs://openpi-assets/checkpoints/pi05_base"),
              "revision": a.revision,
              "original_base": "gs://openpi-assets/checkpoints/pi05_base",
              "format": "OpenPI PyTorch conversion; new 3D head initialized at training startup"}
    provenance = a.output / "provenance.json"
    if provenance.exists():
        previous = json.loads(provenance.read_text())
        if previous.get("checkpoint_sha256") == digest:
            record["source"] = previous.get("source", record["source"])
    provenance.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
