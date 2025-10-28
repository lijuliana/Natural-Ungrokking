"""Nanochat-style autoregressive transformer, matched to the v1 paper.

v1 Table 1: 4L / 256d / 2H, vocab 8192, 11.5M total params, "Muon + AdamW
for embeddings, scalars, and gains". The Dec-2025 nanochat baseline
(rotary, qk-norm, relu^2 MLP, untied head, embedding norm, logit softcap,
zero-init projections) gives only 7.3M at this shape; the missing 4.2M is
exactly two vocab x d tables, i.e. modded-nanogpt-style value embeddings
on alternating layers (last layer always included) with learnable mixing
scalars (v = l1*v + l2*ve). Exact v1 source unavailable; fidelity is
validated against val_bpb ~ 1.149-1.152 and the behavioral gate
(see DECISIONS.md).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    n_layer: int = 4
    d_model: int = 256
    n_head: int = 2
    ctx_len: int = 2048

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


def _rmsnorm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def _rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, Dh) with Dh even
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Value embedding on alternating layers, last layer always included."""
    return layer_idx % 2 == (n_layer - 1) % 2


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        # ve mixing scalars (modded-nanogpt convention): v = l[0]*v + l[1]*ve
        self.ve_lambdas = (nn.Parameter(torch.tensor([0.5, 0.5]))
                           if has_ve(layer_idx, cfg.n_layer) else None)

    def forward(self, x, ve, cos, sin):
        B, T, C = x.shape
        H, Dh = self.cfg.n_head, self.cfg.head_dim
        q = self.wq(x).view(B, T, H, Dh).transpose(1, 2)
        k = self.wk(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.wv(x).view(B, T, H, Dh).transpose(1, 2)
        if self.ve_lambdas is not None:
            ve = ve.view(B, T, H, Dh).transpose(1, 2)
            v = self.ve_lambdas[0] * v + self.ve_lambdas[1] * ve
        q, k = _rotary(q, cos, sin), _rotary(k, cos, sin)
        q, k = _rmsnorm(q), _rmsnorm(k)  # qk-norm for stability
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.down = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.relu(self.up(x)).square())


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn = Attention(cfg, layer_idx)
        self.mlp = MLP(cfg)

    def forward(self, x, ve, cos, sin):
        x = x + self.attn(_rmsnorm(x), ve, cos, sin)
        x = x + self.mlp(_rmsnorm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.value_embeds = nn.ModuleDict(
            {str(i): nn.Embedding(cfg.vocab_size, cfg.d_model)
             for i in range(cfg.n_layer) if has_ve(i, cfg.n_layer)})
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        half = cfg.head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(half) / half))
        t = torch.arange(cfg.ctx_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("rope_cos", freqs.cos()[None, None], persistent=False)
        self.register_buffer("rope_sin", freqs.sin()[None, None], persistent=False)

        self.init_weights()

    def init_weights(self):
        # nanochat scheme: zero-init residual projections and lm_head so the
        # model starts as (approximately) the identity over token embeddings
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_out, fan_in = m.weight.shape
                std = (1.0 / math.sqrt(fan_in)) * min(1.0, math.sqrt(fan_out / fan_in))
                nn.init.normal_(m.weight, mean=0.0, std=std)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.lm_head.weight)
        for b in self.blocks:
            nn.init.zeros_(b.attn.wo.weight)
            nn.init.zeros_(b.mlp.down.weight)
            if b.attn.ve_lambdas is not None:
                with torch.no_grad():
                    b.attn.ve_lambdas.copy_(torch.tensor([0.5, 0.5]))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.size(1)
        cos, sin = self.rope_cos[:, :, :T], self.rope_sin[:, :, :T]
        x = _rmsnorm(self.wte(idx))
        for i, b in enumerate(self.blocks):
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = b(x, ve, cos, sin)
        logits = self.lm_head(_rmsnorm(x)).float()
        return 15.0 * torch.tanh(logits / 15.0)  # nanochat logit softcap

    def loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self(idx)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def matrix_params(self):
        """2-D block weights for Muon (not embeddings/head/ve)."""
        return [p for n, p in self.named_parameters()
                if p.ndim == 2 and n.startswith("blocks")]

    def embed_params(self, exclude_head: bool = False):
        return [p for n, p in self.named_parameters()
                if ("wte" in n or "value_embeds" in n
                    or (not exclude_head and "lm_head" in n))]

    def head_params(self):
        return [p for n, p in self.named_parameters() if "lm_head" in n]

    def scalar_params(self):
        return [p for n, p in self.named_parameters() if "ve_lambdas" in n]
