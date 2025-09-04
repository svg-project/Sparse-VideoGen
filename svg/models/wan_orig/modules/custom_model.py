# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import sys

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import sparse_attention
from .model import WanModel, WanRMSNorm, WanSelfAttention, sinusoidal_embedding_1d

__all__ = ["WanModel"]

try:
    sys.path.append("svg/kernels/build/")
    import _kernels

    from .model import rope_apply

    @amp.autocast(enabled=False)
    def _get_freqs(freqs, grid_sizes, c):
        f, h, w = grid_sizes[0].tolist()
        seq_len = f * h * w

        # split freqs
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        freqs = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, -1)

        return freqs

    @amp.autocast(enabled=False)
    def apply_rotary_emb(query: torch.Tensor, key: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor):
        n, c = query.size(2), query.size(3) // 2

        freqs_real = freqs.real.squeeze(0).squeeze(0).contiguous().to(torch.float32)
        freqs_imag = freqs.imag.squeeze(0).squeeze(0).contiguous().to(torch.float32)

        # Input query and key are torch.float32 due to qk norm
        _kernels.apply_qk_rope_inplace_cossin_complex(query, key, freqs_real, freqs_imag, 0)  # len_text_prompt = 0
        return query, key

    ENABLE_FAST_KERNEL = True
except ImportError:
    import warnings

    warnings.warn("Could not import RoPE / Norm kernels! Falling back to PyTorch implementation.")

    # This function is equivalent to rope_apply in the model.py file. We rewrite it to use _get_freqs.
    @amp.autocast(enabled=False)
    def rope_apply(x, grid_sizes, freqs):
        n, c = x.size(2), x.size(3) // 2
        freqs = _get_freqs(freqs, grid_sizes, c)

        # precompute multipliers
        x_rotated = torch.view_as_complex(x.to(torch.float64).unflatten(3, (-1, 2)))
        # apply rotary embedding
        x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)

        return x_out.float()

    @amp.autocast(enabled=False)
    def apply_rotary_emb(query: torch.Tensor, key: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor):
        query = rope_apply(query, grid_sizes, freqs)
        key = rope_apply(key, grid_sizes, freqs)
        return query, key

    ENABLE_FAST_KERNEL = False


class WanSelfAttention_Sparse(nn.Module):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, layer_idx=0):
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

        # query, key, value function
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # QK Norm. The output is torch.float32
        q = self.norm_q(q)
        k = self.norm_k(k)

        q = q.view(b, s, n, d)
        k = k.view(b, s, n, d)
        v = v.view(b, s, n, d)

        # [B, S, N, D] -> [B, N, S, D]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        q, k = apply_rotary_emb(q, k, grid_sizes, freqs)
        # q = rope_apply(q, grid_sizes, freqs)
        # k = rope_apply(k, grid_sizes, freqs)

        x = sparse_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
            layer_idx=self.layer_idx,
            timestep=t,
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanModel_Sparse(WanModel):
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Print memory allocated in MB
        if torch.cuda.is_available():
            allocated_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"CUDA Memory Allocated: {allocated_memory:.2f} MB")

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if ENABLE_FAST_KERNEL:
            head_dim = self.dim // self.num_heads
            freqs = _get_freqs(self.freqs, grid_sizes, head_dim // 2)

        # arguments
        kwargs = dict(e=e0, t=t, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs, context=context, context_lens=context_lens)

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]


def replace_sparse_forward():
    WanSelfAttention.forward = WanSelfAttention_Sparse.forward
    WanModel.forward = WanModel_Sparse.forward
