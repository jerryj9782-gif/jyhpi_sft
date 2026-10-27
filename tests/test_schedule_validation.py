from functools import partial
import json
import numpy as np
import pytest
import torch
from navpi.config import ModelConfig
from navpi.model import NavPi
from navpi.schedule import apply_phase, group_lr_factor, vision_unfreeze_step
from navpi.train import make_optimizer
from navpi.validation import validate, token_noise
from test_model import batch


def test_staged_vision_no_decay_then_unfreezes_and_resume_lr():
    torch.manual_seed(8)
    model = NavPi(ModelConfig.tiny(), checkpointing="full").train()
    cfg = dict(epochs=3, vision_mode="staged", vision_freeze_epochs=1, vision_warmup_steps=2,
               lr=1e-3, vision_lr=1e-4, head_lr=1e-2, weight_decay=.1, min_lr_ratio=.1, warmup_ratio=.1)
    optimizer = make_optimizer(model, cfg, cuda=False)
    factors = [partial(group_lr_factor, name=g["name"], cfg=cfg, total=6, steps_per_epoch=2)
               for g in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factors)
    before = next(model.vision.parameters()).detach().clone()
    for step in range(3):
        apply_phase(model, cfg, step, 2)
        model(**batch()).backward()
        assert all(p.requires_grad for p in model.parameters())
        vision_grads = [p.grad for p in model.vision.parameters()]
        projector = model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear
        assert projector.weight.grad.abs().sum() > 0
        if step < 2:
            assert all(g is None for g in vision_grads)
        else:
            assert any(g is not None and g.abs().sum() > 0 for g in vision_grads)
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        if step < 2:
            torch.testing.assert_close(before, next(model.vision.parameters()), rtol=0, atol=0)
    assert not torch.equal(before, next(model.vision.parameters()))
    other = make_optimizer(model, cfg, cuda=False)
    resumed = torch.optim.lr_scheduler.LambdaLR(other, factors)
    other.load_state_dict(optimizer.state_dict()); resumed.load_state_dict(scheduler.state_dict())
    optimizer.step(); scheduler.step(); other.step(); resumed.step()
    assert scheduler.get_last_lr() == resumed.get_last_lr()
    assert vision_unfreeze_step({**cfg, "vision_mode": "full"}, 2) == 0
    assert vision_unfreeze_step({**cfg, "vision_mode": "frozen"}, 2) == float("inf")
    with pytest.raises(ValueError):
        vision_unfreeze_step({**cfg, "vision_freeze_epochs": 3}, 2)


def test_validation_noise_and_batch_independence(tmp_path):
    class FakeModel(torch.nn.Module):
        vision_frozen = True
        def set_vision_frozen(self, value): self.vision_frozen = value
        def sample(self, images, tokens, token_mask, steps, noise): return noise
    rows = [{"token": str(i), "actions": [[0., 0., 0.]] * 8} for i in range(5)]
    manifest = tmp_path / "val.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"norm_stats": {"actions": {"q01": [-1.] * 3, "q99": [1.] * 3},
                                               "state": {"q01": [-1.] * 5, "q99": [1.] * 5}}}))
    data = [{"images": torch.zeros(3, 3, 28, 28, dtype=torch.uint8), "tokens": torch.zeros(8, dtype=torch.long),
             "token_mask": torch.ones(8, dtype=torch.bool), "actions": torch.zeros(8, 3)} for _ in rows]
    options = dict(val_batch_size=2, workers=0, val_sample_steps=1, seed=42)
    model = FakeModel().train()
    cfg = {"val_manifest": manifest, "stats": stats}
    first = validate(model, data, cfg, options, torch.device("cpu"))
    second = validate(model, data, cfg, {**options, "val_batch_size": 3}, torch.device("cpu"))
    assert first["samples"] == second["samples"] == 5
    assert first["ade_m"] == pytest.approx(second["ade_m"], abs=1e-7)
    assert model.training and model.vision_frozen
    assert torch.equal(token_noise(["0", "1"], "cpu", 42)[1], token_noise(["1"], "cpu", 42)[0])
