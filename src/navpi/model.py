"""pi0.5 dual-stream flow model, adapted from openpi_old.

Checkpoint names are preserved for the vision tower, projector, VLM, action
expert and time MLP. Only action_in/out_proj are new, three-dimensional layers.
No global transformers patching, JAX, LeRobot or robot action padding is needed.
"""
import math
from contextlib import nullcontext
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

from .config import ModelConfig
from .vendor.configuration_gemma import GemmaConfig
from .vendor.modeling_gemma import GemmaModel, apply_rotary_pos_emb, _gated_residual
from .vendor.modeling_siglip import SiglipVisionModel


def container(**modules):
    result = nn.Module()
    for name, module in modules.items():
        result.add_module(name, module)
    return result


def attention(q, k, v, mask, scale, backend="sdpa"):
    # Explicit expansion also works on the math/CPU SDPA backend in torch 2.8.
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    if backend == "eager":  # correctness reference, never the training default
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        return scores.softmax(-1).to(v.dtype) @ v
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                         dropout_p=0.0, is_causal=False, scale=scale)


class NavPi(nn.Module):
    def __init__(self, config: ModelConfig, checkpointing="full"):
        super().__init__()
        self.config = config
        c = config
        def text_config(width, mlp, adaptive):
            return GemmaConfig(vocab_size=c.vocab_size, hidden_size=width,
                               intermediate_size=mlp, num_hidden_layers=c.depth,
                               num_attention_heads=c.heads, num_key_value_heads=c.kv_heads,
                               head_dim=c.head_dim, hidden_activation="gelu_pytorch_tanh",
                               use_adarms=adaptive, adarms_cond_dim=width if adaptive else None,
                               attention_dropout=0.0, use_cache=False)
        vision_config = SiglipVisionConfig(hidden_size=c.vision_width,
            intermediate_size=c.vision_mlp, num_hidden_layers=c.vision_depth,
            num_attention_heads=c.vision_heads, image_size=c.image_size,
            patch_size=c.patch_size, vision_use_head=False, hidden_act="gelu_pytorch_tanh")
        vision_config._attn_implementation = "sdpa"
        expert = GemmaModel(text_config(c.expert_width, c.expert_mlp, True))
        expert.embed_tokens = None
        self.paligemma_with_expert = container(
            paligemma=container(model=container(
                language_model=GemmaModel(text_config(c.vlm_width, c.vlm_mlp, False)),
                vision_tower=SiglipVisionModel(vision_config),
                multi_modal_projector=container(linear=nn.Linear(c.vision_width, c.vlm_width)))),
            gemma_expert=container(model=expert))
        self.action_in_proj = nn.Linear(3, c.expert_width)
        self.action_out_proj = nn.Linear(c.expert_width, 3)
        self.time_mlp_in = nn.Linear(c.expert_width, c.expert_width)
        self.time_mlp_out = nn.Linear(c.expert_width, c.expert_width)
        self.attention_backend = "sdpa"
        self.vision_frozen = False
        # FP32 master parameters and Adam moments; BF16 autocast is applied by the runner.
        self.float()
        self.set_checkpointing(checkpointing)

    @property
    def vlm(self):
        return self.paligemma_with_expert.paligemma.model.language_model

    @property
    def expert(self):
        return self.paligemma_with_expert.gemma_expert.model

    @property
    def vision(self):
        return self.paligemma_with_expert.paligemma.model.vision_tower

    def set_checkpointing(self, mode):
        if mode not in ("none", "vision", "full"):
            raise ValueError(mode)
        self.checkpointing = mode
        if mode == "none":
            self.vision.gradient_checkpointing_disable()
        else:
            self.vision.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

    def set_vision_frozen(self, frozen):
        # Keep DDP's registered parameter set fixed across the stage transition.
        # no_grad yields grad=None: AdamW skips both updates and weight decay.
        self.vision_frozen = bool(frozen)
        self.vision.train(self.training and not self.vision_frozen)

    def embed_prefix(self, images, tokens):
        b, cameras, channels, height, width = images.shape
        if cameras != 3 or channels != 3 or (height, width) != (self.config.image_size,) * 2:
            raise ValueError("Expected [B, 3 cameras, RGB, image_size, image_size]")
        # One ViT invocation for all three cameras, maintaining F0,L0,R0 ordering.
        with torch.no_grad() if self.vision_frozen else nullcontext():
            image_features = self.vision(images.flatten(0, 1)).last_hidden_state
        image_features = self.paligemma_with_expert.paligemma.model.multi_modal_projector.linear(image_features)
        image_features = image_features.reshape(b, -1, self.config.vlm_width)
        text_features = self.vlm.embed_tokens(tokens) * math.sqrt(self.config.vlm_width)
        return torch.cat((image_features, text_features.to(image_features.dtype)), dim=1)

    def embed_suffix(self, actions, time):
        fraction = torch.linspace(0, 1, self.config.expert_width // 2, device=time.device)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        angle = time[:, None].float() * (2 * math.pi / period[None])
        time_emb = torch.cat((angle.sin(), angle.cos()), dim=-1)
        cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))
        return self.action_in_proj(actions), cond

    def _qkv(self, layer, hidden, cond):
        normed, gate = layer.input_layernorm(hidden, cond=cond)
        shape = (*normed.shape[:2], -1, self.config.head_dim)
        return [getattr(layer.self_attn, name)(normed).view(shape).transpose(1, 2)
                for name in ("q_proj", "k_proj", "v_proj")], gate

    def _finish(self, layer, hidden, out, gate, cond):
        out = out.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1)
        hidden = _gated_residual(hidden, layer.self_attn.o_proj(out), gate)
        normed, gate = layer.post_attention_layernorm(hidden, cond=cond)
        return _gated_residual(hidden, layer.mlp(normed), gate)

    def _layer(self, index, prefix, suffix, cond, mask, positions):
        layers = (self.vlm.layers[index], self.expert.layers[index])
        p, pg = self._qkv(layers[0], prefix, None)
        s, sg = self._qkv(layers[1], suffix, cond)
        q, k, v = [torch.cat((p[i], s[i]), dim=2) for i in range(3)]
        cos, sin = self.vlm.rotary_emb(q, positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        out = attention(q, k, v, mask, layers[0].self_attn.scaling, self.attention_backend)
        n = prefix.shape[1]
        # The final prefix output is never consumed by the action expert; skip its FFN.
        next_prefix = (prefix if index == self.config.depth - 1 else
                       self._finish(layers[0], prefix, out[:, :, :n], pg, None))
        return (next_prefix,
                self._finish(layers[1], suffix, out[:, :, n:], sg, cond))

    def _prefix_mask(self, tokens_mask):
        count = 3 * (self.config.image_size // self.config.patch_size) ** 2
        valid = F.pad(tokens_mask, (count, 0), value=True)
        return valid, valid.long().cumsum(1) - 1

    def velocity(self, images, tokens, token_mask, noisy_actions, time):
        prefix = self.embed_prefix(images, tokens)
        suffix, cond = self.embed_suffix(noisy_actions, time)
        valid, _ = self._prefix_mask(token_mask)
        p = prefix.shape[1]
        valid = F.pad(valid, (0, self.config.horizon), value=True)
        positions = valid.long().cumsum(1) - 1
        # Padding query results are ignored; only key padding needs masking.
        block = torch.ones((valid.shape[1], valid.shape[1]), dtype=torch.bool, device=valid.device)
        block[:p, p:] = False
        mask = block[None, None] & valid[:, None, None, :]
        for index in range(self.config.depth):
            function = partial(self._layer, index)
            args = (prefix, suffix, cond, mask, positions)
            if self.training and self.checkpointing == "full":
                prefix, suffix = checkpoint(function, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                prefix, suffix = function(*args)
        suffix, _ = self.expert.norm(suffix, cond=cond)
        return self.action_out_proj(suffix).float()

    def forward(self, images, tokens, token_mask, actions, noise=None, time=None):
        if actions.shape[1:] != (8, 3):
            raise ValueError("NAVSIM targets must be [B, 8, 3], without padding")
        noise = torch.randn_like(actions) if noise is None else noise
        if time is None:
            # Beta(1.5, 1) inverse CDF, same distribution as openpi without CPU sampling.
            time = torch.rand(actions.shape[0], device=actions.device).pow(1 / 1.5) * .999 + .001
        xt = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        velocity = self.velocity(images, tokens, token_mask, xt, time)
        return F.mse_loss(velocity, noise.float() - actions.float())

    def _cache_prefix(self, images, tokens, token_mask):
        prefix = self.embed_prefix(images, tokens)
        valid, positions = self._prefix_mask(token_mask)
        cache = []
        for index, layer in enumerate(self.vlm.layers):
            (q, k, v), gate = self._qkv(layer, prefix, None)
            cos, sin = self.vlm.rotary_emb(q, positions)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            cache.append((k, v))
            if index == self.config.depth - 1:
                break
            out = attention(q, k, v, valid[:, None, None, :], layer.self_attn.scaling, self.attention_backend)
            prefix = self._finish(layer, prefix, out, gate, None)
        return cache, valid

    def _cached_velocity(self, cache, valid, actions, time):
        suffix, cond = self.embed_suffix(actions, time)
        positions = valid.sum(1, keepdim=True) + torch.arange(8, device=actions.device)[None]
        mask = F.pad(valid, (0, 8), value=True)[:, None, None, :]
        for layer, (pk, pv) in zip(self.expert.layers, cache, strict=True):
            (q, k, v), gate = self._qkv(layer, suffix, cond)
            cos, sin = self.vlm.rotary_emb(q, positions)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k, v = torch.cat((pk, k), dim=2), torch.cat((pv, v), dim=2)
            out = attention(q, k, v, mask, layer.self_attn.scaling, self.attention_backend)
            suffix = self._finish(layer, suffix, out, gate, cond)
        suffix, _ = self.expert.norm(suffix, cond=cond)
        return self.action_out_proj(suffix).float()

    @torch.inference_mode()
    def sample(self, images, tokens, token_mask, *, steps=10, noise=None):
        if steps < 1:
            raise ValueError("steps must be positive")
        if self.training:
            raise RuntimeError("Call model.eval() before sampling")
        cache, valid = self._cache_prefix(images, tokens, token_mask)
        actions = (torch.randn((images.shape[0], 8, 3), device=images.device)
                   if noise is None else noise.float())
        for index in range(steps):
            time = torch.full((images.shape[0],), 1 - index / steps, device=images.device)
            actions = actions - self._cached_velocity(cache, valid, actions, time) / steps
        return actions
