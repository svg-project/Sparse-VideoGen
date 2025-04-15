# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import json

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from .utils import create_block_mask_cached, generate_temporal_head_mask_mod, generate_dense_mask_mod
from .placement import wan_sparse_head_placement, wan_hidden_states_placement, ref_wan_sparse_head_placement, ref_wan_hidden_states_placement 
from ....timer import time_logging_decorator

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    "sparse_attention",
    "flash_attention",
    "attention",
]


flex_attention = torch.compile(flex_attention, dynamic=False)
torch._dynamo.config.cache_size_limit = 192 * 3
torch._dynamo.config.accumulated_cache_size_limit = 192 * 3

class Wan_SparseAttn:
    num_sampled_rows = 32
    attention_masks = None

    context_length = 0
    num_frame = 21
    frame_size = 3600

    first_layers_fp = 0
    first_times_fp = 0

    sample_mse_max_row = 10000
    block_mask = None
    
    record_attention = False

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Hunyuan_SparseAttn requires PyTorch 2.0, please upgrade PyTorch.")


    @classmethod
    @time_logging_decorator("SVG - sample mse")
    def sample_mse(self, query, key, value):
        assert len(self.attention_masks) == 2

        cfg, num_heads, seq_len, dim = query.size()
        num_sampled_rows = min(self.num_sampled_rows, seq_len)
        sampled_rows = torch.randint(low=0, high=self.sample_mse_max_row, size=(num_sampled_rows,))
        sampled_q = query[:, :, sampled_rows, :]
        sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (dim**0.5)
    
        sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1)
        sampled_golden_hidden_states = torch.matmul(sampled_attn_weights, value)  # (1, seq_len, dim)

        sampled_mses = torch.zeros(len(self.attention_masks), cfg, num_heads, device=query.device, dtype=query.dtype)

        # Only have Tri-diagonal and Striped
        for mask_idx, attn_mask in enumerate(self.attention_masks):
            sampled_attention_mask = attn_mask[sampled_rows, :]
            sampled_attention_scores = sampled_qk_scores.masked_fill(sampled_attention_mask == 0, float('-inf'))
            sampled_attn_weights = F.softmax(sampled_attention_scores, dim=-1)
            sampled_hidden_states = torch.matmul(sampled_attn_weights, value)
            mse = torch.mean((sampled_hidden_states - sampled_golden_hidden_states) ** 2, dim=(2, 3))
            sampled_mses[mask_idx] = mse

        return sampled_mses

    @classmethod
    @time_logging_decorator("SVG - sparse flex attention")
    def sparse_flex_attention(self, query, key, value, block_mask):
        return flex_attention(query, key, value, block_mask=block_mask)
    
    @classmethod
    @time_logging_decorator("SVG - dense flex attention")
    def dense_flex_attention(self, query, key, value):
        return flex_attention(query, key, value, block_mask=self.dense_block_mask)
    
    @classmethod
    @time_logging_decorator("SVG - sparse head placement")
    def sparse_head_placement(self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size):
        
        query_out, key_out, value_out = ref_wan_sparse_head_placement(query, key, value, best_mask_idx, context_length, num_frame, frame_size)

        return query_out, key_out, value_out

    @classmethod
    @time_logging_decorator("SVG - fast sparse head placement")
    def fast_sparse_head_placement(self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size):

        wan_sparse_head_placement(query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size)

        return query_out, key_out, value_out

    @classmethod
    @time_logging_decorator("SVG - hidden states placement")
    def hidden_states_placement(self, \
        hidden_states, output_hidden_states, \
        best_mask_idx, context_length, num_frame, frame_size
    ):
        ref_wan_hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size)

    @classmethod
    @time_logging_decorator("SVG - fast hidden states placement")
    def fast_hidden_states_placement(self, \
        hidden_states, output_hidden_states, \
        best_mask_idx, context_length, num_frame, frame_size
    ):
        wan_hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size)

    @classmethod
    @time_logging_decorator("SVG - attention core logic")
    def attention_core_logic(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        timestep,
        layer_idx,
    ):
        cfg, num_heads, seq_len, dim = query.size()
        
        context_length, num_frame, frame_size = self.context_length, self.num_frame, self.frame_size

        assert seq_len == context_length + num_frame * frame_size, \
            f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        sampled_mses = self.sample_mse(query, key, value)
        best_mask_idx = torch.argmin(sampled_mses, dim=0)

        if self.record_attention:
            with open(self.record_path, "a") as file:
                file.write(json.dumps({
                    "Time": timestep.item(),
                    "Layer": layer_idx,
                    "Choice": best_mask_idx.tolist(),
                    "SampledMSEs": sampled_mses.tolist(),
                }) + "\n")
                
        output_hidden_states = torch.zeros_like(query)
        query_out, key_out, value_out = torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)

        query_out, key_out, value_out = self.fast_sparse_head_placement(query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size)

        hidden_states = self.sparse_flex_attention(query_out, key_out, value_out, block_mask=self.block_mask)

        self.fast_hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size)

        return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)
    

def sparse_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
    layer_idx=0,
    timestep=None
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """

    with time_logging_decorator("sparse attn - prepare"):
        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        assert q.device.type == 'cuda' and q.size(-1) <= 256

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # params
        b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        # preprocess query
        q = half(q)
        k = half(k)
        v = half(v)

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if q_scale is not None:
            q = q * q_scale

    # Determine if we use Full Attention to calculate  # TODO  
    full_attention_flag = False
    if layer_idx < 42 * Wan_SparseAttn.first_layers_fp:
        full_attention_flag = True
    if timestep > 1000 * (1 - Wan_SparseAttn.first_times_fp):
        full_attention_flag = True

    if full_attention_flag:    
        with time_logging_decorator("sparse attn - dense attention"):
            x = Wan_SparseAttn.dense_flex_attention(q, k, v)
                    
            # x = Wan_SparseAttn.attention_core_logic(
            #     q, k, v, timestep, layer_idx,
            #     dense=True
            # )
    else:
        with time_logging_decorator("sparse attn - core attention"):
                
            x = Wan_SparseAttn.attention_core_logic(
                q, k, v, timestep, layer_idx,
            )

    with time_logging_decorator("sparse attn - prepare"):
        x = x.transpose(1, 2).contiguous()

    return x.type(out_dtype)
    
    
def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
    layer_idx=0,
    timestep=None
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(
            device=q.device, non_blocking=True
        )
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(
            device=k.device, non_blocking=True
        )
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            "Flash attention 3 is not available, use flash attention 2 instead."
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                "Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance."
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p
        )

        out = out.transpose(1, 2).contiguous()
        return out

def prepare_dense_attention(cfg_size, num_head, head_dim, dtype, device, context_length, prompt_length, num_frame, frame_size
):

    seq_len = context_length + num_frame * frame_size
    query, key, value = [torch.zeros((1, cfg_size * num_head, seq_len, head_dim), dtype=dtype, device=device) for _ in range(3)]

    mask_mod = generate_dense_mask_mod()
    block_mask = create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)

    hidden_states = flex_attention(query, key, value, block_mask=block_mask)

    return block_mask


def prepare_flexattention(cfg_size, num_head, head_dim, dtype, device, context_length, prompt_length, num_frame, frame_size, \
    diag_width=1, multiplier=2
):
    assert diag_width == multiplier
    seq_len = context_length + num_frame * frame_size
    query, key, value = [torch.zeros((1, cfg_size * num_head, seq_len, head_dim), dtype=dtype, device=device) for _ in range(3)]

    mask_mod = generate_temporal_head_mask_mod(context_length, prompt_length, num_frame, frame_size, mul=multiplier)
    block_mask = create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)

    hidden_states = flex_attention(query, key, value, block_mask=block_mask)

    return block_mask