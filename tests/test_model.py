from copy import deepcopy
from pathlib import Path
import pytest
from safetensors.torch import save_file
import torch
from navpi.config import ModelConfig
from navpi.model import NavPi
from navpi.checkpoints import HEAD_KEYS, EMBED, LM_HEAD, EXPERT_HEAD, load_weights, weight_plan


def batch(n=2):
    return dict(images=torch.randn(n, 3, 3, 28, 28), tokens=torch.randint(0, 128, (n, 8)),
                token_mask=torch.tensor([[True] * 5 + [False] * 3] * n),
                actions=torch.randn(n, 8, 3), noise=torch.randn(n, 8, 3), time=torch.full((n,), .4))


def test_native_shape_and_component_gradients():
    torch.manual_seed(1)
    model = NavPi(ModelConfig.tiny())
    inputs = batch()
    loss = model(**inputs); loss.backward()
    assert torch.isfinite(loss)
    for pattern in ("vision_tower", "multi_modal_projector", "language_model", "gemma_expert", "time_mlp", "action_in", "action_out"):
        grads = [p.grad for n, p in model.named_parameters() if pattern in n and p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads), pattern
        assert all(torch.isfinite(g).all() for g in grads), pattern
    assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
    model.eval()
    assert model.sample(inputs["images"], inputs["tokens"], inputs["token_mask"], steps=2).shape == (2, 8, 3)
    with pytest.raises(ValueError, match="without padding"):
        model(**{**inputs, "actions": torch.zeros(2, 8, 32)})


@pytest.mark.parametrize("checkpointing", ["none", "vision", "full"])
def test_sdpa_matches_eager_and_cached_inference(checkpointing):
    torch.manual_seed(2)
    model = NavPi(ModelConfig.tiny(), checkpointing=checkpointing)
    inputs = batch()
    reference = deepcopy(model)
    reference.attention_backend = "eager"
    loss = model(**inputs); other = reference(**inputs)
    torch.testing.assert_close(loss, other, atol=2e-6, rtol=2e-6)
    loss.backward(); other.backward()
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=1e-4, msg=name)
    model.eval()
    with torch.no_grad():
        cache, valid = model._cache_prefix(inputs["images"], inputs["tokens"], inputs["token_mask"])
        direct = model.velocity(inputs["images"], inputs["tokens"], inputs["token_mask"], inputs["actions"], inputs["time"])
        cached = model._cached_velocity(cache, valid, inputs["actions"], inputs["time"])
    torch.testing.assert_close(direct, cached, atol=3e-6, rtol=2e-5)


def test_checkpointing_gradient_equivalence():
    torch.manual_seed(3)
    model = NavPi(ModelConfig.tiny(), checkpointing="none")
    other = deepcopy(model); other.set_checkpointing("full")
    inputs = batch()
    model(**inputs).backward(); other(**inputs).backward()
    for p, q in zip(model.parameters(), other.parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad)


def test_base_loader_handles_tied_embedding_and_only_resets_head(tmp_path):
    torch.manual_seed(5)
    source = NavPi(ModelConfig.tiny())
    weights = {k: v.clone() for k, v in source.state_dict().items()}
    weights[LM_HEAD] = weights.pop(EMBED)
    weights[EXPERT_HEAD] = torch.randn(128, 16)
    weights["action_in_proj.weight"] = torch.randn(16, 32)
    weights["action_out_proj.weight"] = torch.randn(32, 16)
    weights["action_out_proj.bias"] = torch.randn(32)
    filename = tmp_path / "base.safetensors"
    save_file(weights, filename, metadata={EMBED: LM_HEAD})
    target = NavPi(ModelConfig.tiny())
    head_before = {k: target.state_dict()[k].clone() for k in HEAD_KEYS}
    report = load_weights(target, filename, base=True)
    assert set(report["reinitialized"]) == HEAD_KEYS
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, head_before[key] if key in HEAD_KEYS else source.state_dict()[key])
    del weights["time_mlp_in.weight"]
    save_file(weights, tmp_path / "broken.safetensors", metadata={EMBED: LM_HEAD})
    with pytest.raises(ValueError, match="missing"):
        load_weights(target, tmp_path / "broken.safetensors", base=True)


def test_real_checkpoint_shapes_without_allocating_full_model():
    filename = Path("checkpoints/pi05_base/model.safetensors")
    if not filename.is_file():
        pytest.skip("Local pretrained checkpoint not installed")
    with torch.device("meta"):
        model = NavPi(ModelConfig())
    plan = weight_plan(model, filename, base=True)
    assert plan["loaded_tensors"] == 807
    assert plan["parameters"] == 3353374451


def test_padding_tokens_do_not_affect_predictions():
    model = NavPi(ModelConfig.tiny()).eval()
    inputs = batch()
    altered = inputs["tokens"].clone()
    altered[~inputs["token_mask"]] = 126
    with torch.no_grad():
        original = model(**inputs)
        new = model(**{**inputs, "tokens": altered})
    torch.testing.assert_close(original, new, atol=1e-6, rtol=1e-6)


def test_bfloat16_autocast_retains_float32_parameters():
    model = NavPi(ModelConfig.tiny())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(**batch())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
