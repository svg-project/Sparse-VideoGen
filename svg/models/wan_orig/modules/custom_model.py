# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import sparse_attention

from ....timer import time_logging_decorator
from .model import WanRMSNorm, rope_apply, WanSelfAttention

__all__ = ['WanModel']


class WanSelfAttention_Sparse(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 layer_idx=0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.layer_idx = layer_idx

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, t):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        with time_logging_decorator("selfattn - linear"):
            # query, key, value function
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            
        with time_logging_decorator("selfattn - qk norm"):
            # QK Norm
            q = self.norm_q(q)
            k = self.norm_k(k)
            
            q = q.view(b, s, n, d)
            k = k.view(b, s, n, d)
            v = v.view(b, s, n, d)

        with time_logging_decorator("selfattn - rope"):
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)

        with time_logging_decorator("selfattn - flash attention"):
            x = sparse_attention(
                q=q,
                k=k,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size,
                layer_idx=self.layer_idx,
                timestep=t)

        with time_logging_decorator("selfattn - linear"):
            # output
            x = x.flatten(2)
            x = self.o(x)
        return x


def replace_sparse_forward():
    WanSelfAttention.forward = WanSelfAttention_Sparse.forward