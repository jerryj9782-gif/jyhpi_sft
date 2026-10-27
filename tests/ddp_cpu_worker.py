"""Two tiny CPU/Gloo ranks: DDP no_sync gradients vs an equivalent global batch."""
from contextlib import nullcontext
from copy import deepcopy
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from navpi.config import ModelConfig
from navpi.model import NavPi

torch.set_num_threads(1)
torch.manual_seed(120)
dist.init_process_group("gloo")
rank = dist.get_rank()
assert dist.get_world_size() == 2
model = NavPi(ModelConfig.tiny(), checkpointing="full")
reference = deepcopy(model)
wrapped = DistributedDataParallel(model, find_unused_parameters=True, gradient_as_bucket_view=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4)
for step in range(2):
    model.set_vision_frozen(step == 0)
    reference.set_vision_frozen(step == 0)
    inputs = dict(images=torch.randn(4, 3, 3, 28, 28), tokens=torch.randint(0, 128, (4, 8)),
                  token_mask=torch.ones(4, 8, dtype=torch.bool), actions=torch.randn(4, 8, 3),
                  noise=torch.randn(4, 8, 3), time=torch.full((4,), .4))
    reference(**inputs).backward()
    for micro in range(2):
        index = rank * 2 + micro
        context = wrapped.no_sync() if micro == 0 else nullcontext()
        with context:
            (wrapped(**{k: v[index:index + 1] for k, v in inputs.items()}) / 2).backward()
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=2e-4, msg=name)
    optimizer.step(); ref_optimizer.step()
    # Tiny roundoff in near-zero gradients is amplified by Adam epsilon; compare grads above.
    optimizer.zero_grad(set_to_none=True); ref_optimizer.zero_grad(set_to_none=True)
    # Re-align reference to avoid compounding harmless CPU reduction roundoff next iteration.
    reference.load_state_dict(model.state_dict())
    ref_optimizer.load_state_dict(optimizer.state_dict())
if rank == 0:
    print("PASS: 2 CPU Gloo ranks; frozen -> unfrozen ViT, 2 accumulated microbatches; gradients match global batch")
dist.destroy_process_group()
