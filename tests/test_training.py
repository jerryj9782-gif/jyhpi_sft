import json
from copy import deepcopy
import numpy as np
import pytest
import torch
from navpi.config import ModelConfig
from navpi.model import NavPi
from navpi.train import make_optimizer, lr_factor, rng_state, restore_rng, save_checkpoint
from navpi.checkpoints import load_model
from test_model import batch


def test_accumulation_matches_full_batch_including_short_final_window():
    torch.manual_seed(10)
    first = NavPi(ModelConfig.tiny(), checkpointing="full")
    other = deepcopy(first)
    inputs = batch(3)
    first(**inputs).backward()
    for i in range(3):
        (other(**{k: v[i:i + 1] for k, v in inputs.items()}) / 3).backward()
    for (name, p), (_, q) in zip(first.named_parameters(), other.named_parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)


def test_fp32_adamw_updates_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(11)
    model = NavPi(ModelConfig.tiny())
    options = dict(lr=1e-5, head_lr=1e-4, weight_decay=.01)
    opt = make_optimizer(model, options, cuda=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.)
    inputs = batch()
    before = model.action_out_proj.weight.detach().clone()
    model(**inputs).backward(); opt.step(); scheduler.step(); opt.zero_grad(set_to_none=True)
    assert not torch.equal(before, model.action_out_proj.weight)
    for state in opt.state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
    for name in ("stats.json", "tokenizer.model", "train.jsonl"):
        (tmp_path / name).write_text("dummy")
    cfg = {"data": {"stats": str(tmp_path / "stats.json"), "tokenizer": str(tmp_path / "tokenizer.model"),
                    "train_manifest": str(tmp_path / "train.jsonl")}, "train": options}
    save_checkpoint(model, opt, scheduler, cfg, tmp_path, 1, 0, 2, torch.device("cpu"), 0, 1)
    restored, _ = load_model(tmp_path / "step_00000001")
    for p, q in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    saved = torch.load(tmp_path / "step_00000001/training.pt", weights_only=False)
    fresh_opt = make_optimizer(restored, options, cuda=False)
    fresh_opt.load_state_dict(saved["optimizer"])
    assert saved["next_batch"] == 2
    for m, o in ((model, opt), (restored, fresh_opt)):
        m(**inputs).backward(); o.step()
    for p, q in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_rng_resume():
    state = rng_state(torch.device("cpu"))
    first = torch.rand(10); nfirst = np.random.rand(10)
    restore_rng(state, torch.device("cpu"))
    torch.testing.assert_close(torch.rand(10), first)
    np.testing.assert_array_equal(np.random.rand(10), nfirst)


def test_warmup_and_cosine_bounds():
    values = [lr_factor(i, total=100, warmup=10, minimum=.1) for i in range(101)]
    assert values[0] == .1 and values[9] == 1 and values[-1] == .1
    assert all(.1 <= x <= 1 for x in values)
