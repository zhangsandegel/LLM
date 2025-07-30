import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelArgs:
    def __init__(self, dim: int, max_seq_len: int):
        self.dim = dim
        self.max_seq_len = max_seq_len


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # [seq_len, dim // 2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis  # shape: [seq_len, dim // 2], complex dtype


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape to [batch_size, seq_len, dim//2, 2] then to complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis[:xq_.size(1)]  # [seq_len, dim//2]
    freqs_cis = freqs_cis.unsqueeze(0)  # [1, seq_len, dim//2]

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.max_seq_len = args.max_seq_len

        self.wq = nn.Linear(self.dim, self.dim, bias=False)
        self.wk = nn.Linear(self.dim, self.dim, bias=False)
        self.wv = nn.Linear(self.dim, self.dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

        self.freqs_cis = precompute_freqs_cis(self.dim, self.max_seq_len * 2)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x).view(batch_size, seq_len, self.dim)
        xk = self.wk(x).view(batch_size, seq_len, self.dim)
        xv = self.wv(x).view(batch_size, seq_len, self.dim)

        # rotary embedding
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=self.freqs_cis)

        # scaled dot-product attention
        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(self.dim)
        scores = F.softmax(scores, dim=-1)
        out = torch.matmul(scores, xv)

        return self.wo(out)
