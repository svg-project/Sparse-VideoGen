import json
from typing import Optional, Tuple

import flashinfer
import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from diffusers.models.embeddings import apply_rotary_emb
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.nn.attention.flex_attention import flex_attention

from ...kernels.triton.permute import apply_inverse_permutation_triton, permute_tensor_by_labels_triton
from ...kmeans_utils import (
    batch_kmeans_Euclid,
    density_calculation,
    dynamic_block_sparse_fwd_flashinfer,
    identify_dynamic_map,
)
from ...logger import logger
from ...timer import time_logging_decorator
from ...utils.misc import Color
from .placement import (
    hunyuan_hidden_states_placement,
    hunyuan_sparse_head_placement,
    ref_hunyuan_hidden_states_placement,
    ref_hunyuan_sparse_head_placement,
)
from .utils import create_block_mask_cached, generate_temporal_head_mask_mod

flex_attention = torch.compile(flex_attention, dynamic=False)
torch._dynamo.config.cache_size_limit = 192 * 3
torch._dynamo.config.accumulated_cache_size_limit = 192 * 3


class HunyuanVideoAttnProcessor2_0_FlashAttention:
    """
    This is a custom attention processor that replaces the original attention implementation with flash attention.
    The original implementation is based on the FSDP + mask implementation, which is SLOW. We switch to flash attention + varlen for efficiency.
    """

    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        # 5. Attention - We switch to flash attention for efficiency. That's the only difference from its original implementation.
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        # query: (B * H, L, D)
        _query = query.permute(2, 0, 1, 3).flatten(1, 2)
        _key = key.permute(2, 0, 1, 3).flatten(1, 2)
        _value = value.permute(2, 0, 1, 3).flatten(1, 2)
        cu_seqlens_q = torch.tensor(
            [0, attention_mask.sum(), attention_mask.numel()], dtype=torch.int32, device=query.device
        )
        cu_seqlens_kv = torch.tensor(
            [0, attention_mask.sum(), attention_mask.numel()], dtype=torch.int32, device=query.device
        )
        max_seqlen_q = attention_mask.numel()
        max_seqlen_kv = attention_mask.numel()
        # _query: (L, B * H, D)
        hidden_states = flash_attn_varlen_func(
            _query, _key, _value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
        )
        hidden_states = hidden_states.permute(1, 0, 2).reshape(query.shape)

        # out: (B, L, H * D)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


# Define the fast kernels
try:
    import sys

    sys.path.append("svg/kernels/build/")
    import _kernels

    def apply_qk_norm(attn_norm_q, attn_norm_k, query, key):
        """Fast RMSNorm.Input is (B, H, L, D)"""
        assert query.is_contiguous() and key.is_contiguous(), "Query and Key must be contiguous"

        _kernels.rms_norm_forward(query.view(-1, query.shape[-1]), attn_norm_q.weight, attn_norm_q.eps)
        _kernels.rms_norm_forward(key.view(-1, key.shape[-1]), attn_norm_k.weight, attn_norm_k.eps)

        return query, key

    def apply_qk_rope_single(query, key, freqs_cis, encoder_hidden_states):
        assert query.is_contiguous() and key.is_contiguous(), "Query and Key must be contiguous"

        txt_len = encoder_hidden_states.shape[1]
        cos, sin = freqs_cis[0], freqs_cis[1]
        _kernels.apply_qk_rope_inplace_cossin_txtlast(query, key, cos, sin, txt_len)

        return query, key

    def apply_qk_rope_double(query, key, freqs_cis):
        assert query.is_contiguous() and key.is_contiguous(), "Query and Key must be contiguous"

        # Apply RoPE if needed.
        if freqs_cis is not None:
            cos, sin = freqs_cis[0], freqs_cis[1]
            _kernels.apply_qk_rope_inplace_cossin_txtlast(query, key, cos, sin, 0)

        return query, key

    ENABLE_FAST_KERNEL = True

    logger.info(f"{Color.green}Using Fast CUDA and Triton Kernels{Color.reset}")


except ImportError:
    ENABLE_FAST_KERNEL = False

    def apply_qk_norm(attn_norm_q, attn_norm_k, query, key):
        if attn_norm_q is not None:
            query = attn_norm_q(query)
        if attn_norm_k is not None:
            key = attn_norm_k(key)
        return query, key

    def apply_qk_rope_single(query, key, image_rotary_emb, encoder_hidden_states):

        txt_len = encoder_hidden_states.shape[1]
        img_q, txt_q = query[:, :, :-txt_len], query[:, :, -txt_len:]
        img_k, txt_k = key[:, :, :-txt_len], key[:, :, -txt_len:]

        img_q = apply_rotary_emb(img_q, image_rotary_emb)
        img_k = apply_rotary_emb(img_k, image_rotary_emb)

        query = torch.cat([img_q, txt_q], dim=2)
        key = torch.cat([img_k, txt_k], dim=2)

        return query, key

    def apply_qk_rope_double(query, key, image_rotary_emb):

        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)
        return query, key

    logger.info(f"{Color.red}Disable Fast CUDA and Triton Kernels{Color.reset}")


class Hunyuan_SVGAttn_Processor2_0:
    """
    Supports Sparse VideoGen.
    """

    num_sampled_rows = 32
    attention_masks = None

    prompt_length = 0
    context_length = 256
    num_frame = 33
    frame_size = 3600

    first_layers_fp = 0
    first_times_fp = 0

    sample_mse_max_row = 10000
    block_mask = None

    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Hunyuan_SparseAttn requires PyTorch 2.0, please upgrade PyTorch.")

    @time_logging_decorator("Level 2 - get_qkv")
    def get_qkv(self, attn, hidden_states):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        return query, key, value

    @time_logging_decorator("Level 2 - get_transpose_qkv")
    def get_transpose_qkv(self, attn, query, key, value):
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()

        return query, key, value

    @time_logging_decorator("Level 2 - get_qk_norm")
    def get_qk_norm(self, attn, query, key):
        query, key = apply_qk_norm(attn.norm_q, attn.norm_k, query, key)
        return query, key

    @time_logging_decorator("Level 2 - get_rotary_emb")
    def get_rotary_emb(self, attn, query, key, image_rotary_emb, encoder_hidden_states):
        if image_rotary_emb is not None:
            if attn.add_q_proj is None and encoder_hidden_states is not None:
                # Triggered in Single Transformer Block
                query, key = apply_qk_rope_single(query, key, image_rotary_emb, encoder_hidden_states)
            else:
                # Triggered in Double Transformer Block
                query, key = apply_qk_rope_double(query, key, image_rotary_emb)

        return query, key

    @time_logging_decorator("Level 2 - get_encoder_condition_and_concat")
    def get_encoder_condition_and_concat(self, attn, query, key, value, encoder_hidden_states):
        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        return query, key, value

    @time_logging_decorator("Level 2 - get_cu_max_seqlen")
    def get_cu_max_seqlen(self, attention_mask, device):
        cu_seqlens_q = torch.tensor([0, attention_mask.sum(), attention_mask.numel()], dtype=torch.int32, device=device)
        cu_seqlens_kv = torch.tensor(
            [0, attention_mask.sum(), attention_mask.numel()], dtype=torch.int32, device=device
        )
        max_seqlen_q = attention_mask.numel()
        max_seqlen_kv = attention_mask.numel()
        return cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv

    @time_logging_decorator("Level 2 - get_o")
    def get_o(self, attn, hidden_states, encoder_hidden_states):
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query, key, value = self.get_qkv(attn, hidden_states)

        query, key, value = self.get_transpose_qkv(attn, query, key, value)

        # 2. QK normalization
        query, key = self.get_qk_norm(attn, query, key)

        # 3. Rotational positional embeddings applied to latent stream
        query, key = self.get_rotary_emb(attn, query, key, image_rotary_emb, encoder_hidden_states)

        # 4. Encoder condition QKV projection and normalization
        query, key, value = self.get_encoder_condition_and_concat(attn, query, key, value, encoder_hidden_states)

        # 5. Calculate the attention
        # ========================================================================
        cu_max_seqlens = self.get_cu_max_seqlen(attention_mask, query.device)
        hidden_states = self.attention_core_logic(query, key, value, timestep, self.layer_idx, cu_max_seqlens)
        # ========================================================================

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        hidden_states, encoder_hidden_states = self.get_o(attn, hidden_states, encoder_hidden_states)

        return hidden_states, encoder_hidden_states

    @time_logging_decorator("Level 3 - sample_mse")
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
            sampled_attention_scores = sampled_qk_scores.masked_fill(sampled_attention_mask == 0, float("-inf"))
            sampled_attn_weights = F.softmax(sampled_attention_scores, dim=-1)
            sampled_hidden_states = torch.matmul(sampled_attn_weights, value)
            mse = torch.mean((sampled_hidden_states - sampled_golden_hidden_states) ** 2, dim=(2, 3))
            sampled_mses[mask_idx] = mse

        return sampled_mses

    @time_logging_decorator("Level 3 - sparse_flex_attention")
    def sparse_flex_attention(self, query, key, value, block_mask):
        return flex_attention(query, key, value, block_mask=block_mask)

    @time_logging_decorator("Level 3 - sparse_head_placement")
    def sparse_head_placement(
        self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
    ):

        query_out, key_out, value_out = ref_hunyuan_sparse_head_placement(
            query, key, value, best_mask_idx, context_length, num_frame, frame_size
        )

        return query_out, key_out, value_out

    @time_logging_decorator("Level 3 - fast_sparse_head_placement")
    def fast_sparse_head_placement(
        self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
    ):

        hunyuan_sparse_head_placement(
            query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
        )

        return query_out, key_out, value_out

    @time_logging_decorator("Level 3 - hidden_states_placement")
    def hidden_states_placement(
        self, hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
    ):
        ref_hunyuan_hidden_states_placement(
            hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
        )

    @time_logging_decorator("Level 3 - fast_hidden_states_placement")
    def fast_hidden_states_placement(
        self, hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
    ):
        hunyuan_hidden_states_placement(
            hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
        )

    @time_logging_decorator("Level 3 - Dense FlashInfer Attention")
    def flashinfer_attention(self, query, key, value, cu_max_seqlens):
        """
        VarlenFlashInfer Attention. Input is (B, H, L, D).
        """
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = cu_max_seqlens
        out = flashinfer_varlen_func(query, key, value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        return out

    @time_logging_decorator("Level 3 - Dense Flash Attention")
    def flash_attention(self, query, key, value, cu_max_seqlens):
        """
        Varlen FlashAttention. Input is (B, H, L, D). Need to be converted to (L, B * H, D) for FlashAttention.
        """
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = cu_max_seqlens

        # query: (B * H, L, D)
        _query = query.permute(2, 0, 1, 3).flatten(1, 2)
        _key = key.permute(2, 0, 1, 3).flatten(1, 2)
        _value = value.permute(2, 0, 1, 3).flatten(1, 2)
        # _query: (L, B * H, D)
        hidden_states = flash_attn_varlen_func(
            _query, _key, _value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
        )
        hidden_states = hidden_states.permute(1, 0, 2).reshape(query.shape)

        # hidden_states: (B, H, L, D)
        return hidden_states

    @time_logging_decorator("Level 2 - attention core logic")
    def attention_core_logic(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        timestep: torch.Tensor,
        layer_idx: int,
        cu_max_seqlens: Tuple[torch.Tensor, torch.Tensor, int, int],
    ):
        cfg, num_heads, seq_len, dim = query.size()

        context_length, num_frame, frame_size = self.context_length, self.num_frame, self.frame_size

        assert (
            seq_len == context_length + num_frame * frame_size
        ), f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        # Determine if we use Full Attention to calculate
        full_attention_flag = False

        if self.layer_idx < self.first_layers_fp:
            full_attention_flag = True
        if timestep[0] > self.first_times_fp:
            full_attention_flag = True

        # print(f"Full Attention Flag: {full_attention_flag}")
        # print(
        #     f"Layer Index: {self.layer_idx}, First Layers FP: {self.first_layers_fp}, First Times FP: {self.first_times_fp}, Timestep: {timestep[0]}"
        # )

        if full_attention_flag:
            output_hidden_states = self.flash_attention(query, key, value, cu_max_seqlens)
            return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)
        else:
            sampled_mses = self.sample_mse(query, key, value)
            best_mask_idx = torch.argmin(sampled_mses, dim=0)

            output_hidden_states = torch.zeros_like(query)

            query_out, key_out, value_out = torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)

            query_out, key_out, value_out = self.fast_sparse_head_placement(
                query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size
            )

            hidden_states = self.sparse_flex_attention(query_out, key_out, value_out, block_mask=self.block_mask)

            self.fast_hidden_states_placement(
                hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size
            )

            return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)


def prepare_flexattention(
    cfg_size,
    num_head,
    head_dim,
    dtype,
    device,
    context_length,
    prompt_length,
    num_frame,
    frame_size,
    diag_width=1,
    multiplier=2,
):
    assert diag_width == multiplier
    seq_len = context_length + num_frame * frame_size
    query, key, value = [
        torch.zeros((1, cfg_size * num_head, seq_len, head_dim), dtype=dtype, device=device) for _ in range(3)
    ]

    mask_mod = generate_temporal_head_mask_mod(context_length, prompt_length, num_frame, frame_size, mul=multiplier)
    block_mask = create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)

    _ = flex_attention(query, key, value, block_mask=block_mask)

    return block_mask


# ---- Semantic Aware Permutation Processor ----
class Hunyuan_SAPAttn_Processor2_0(Hunyuan_SVGAttn_Processor2_0):
    """
    Semantic Aware Permutation Attention
    """

    # centroids
    num_q_centroids = 0
    num_k_centroids = 0
    top_p_kmeans = 0
    min_kc_ratio = 0

    centroids_init = {}
    q_centroids = {}
    k_centroids = {}

    kmeans_iter_init = 0
    kmeans_iter_step = 0
    zero_step_kmeans_init = False

    logging_file = None

    @time_logging_decorator("Level 3.7 - kmeans init")
    def kmeans_init(self, query, key, layer_idx):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
            query.view(cfg * num_heads, seq_len, dim), n_clusters=self.num_q_centroids, max_iters=self.kmeans_iter_init
        )
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
            key.view(cfg * num_heads, seq_len, dim), n_clusters=self.num_k_centroids, max_iters=self.kmeans_iter_init
        )

        self.q_centroids[layer_idx] = qcentroids
        self.k_centroids[layer_idx] = kcentroids

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    @time_logging_decorator("Level 3.7 - kmeans step")
    def kmeans_step(self, query, key, layer_idx):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
            query.view(cfg * num_heads, seq_len, dim),
            n_clusters=self.num_q_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.q_centroids[layer_idx],
        )
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
            key.view(cfg * num_heads, seq_len, dim),
            n_clusters=self.num_k_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.k_centroids[layer_idx],
        )

        self.q_centroids[layer_idx] = qcentroids
        self.k_centroids[layer_idx] = kcentroids

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    @time_logging_decorator("Level 3.5 - kmeans clustering")
    def kmeans_clustering(self, query, key, layer_idx):
        if layer_idx not in self.centroids_init.keys():
            qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_init(
                query, key, layer_idx
            )
            self.centroids_init[layer_idx] = True
            print(f"Centroids initialized at layer {layer_idx}. Init step: {self.kmeans_iter_init}")
        else:
            assert self.centroids_init[layer_idx], f"Centroids not initialized at layer {layer_idx}"
            qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_step(
                query, key, layer_idx
            )

        return qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter

    @time_logging_decorator("Level 3 - semantic aware permutation")
    def semantic_aware_permutation(self, query, key, value, timestep, layer_idx):
        cfg, num_heads, seq_len, dim = query.size()

        # 1. Kmeans clustering
        qlabels, qcentroids, qcluster_sizes, qiter, klabels, kcentroids, kcluster_sizes, kiter = self.kmeans_clustering(
            query, key, layer_idx
        )

        # 2. Identify dynamic map
        q_cluster_sizes = qcluster_sizes.view(cfg, num_heads, self.num_q_centroids)
        k_cluster_sizes = kcluster_sizes.view(cfg, num_heads, self.num_k_centroids)

        dynamic_map = identify_dynamic_map(
            qcentroids.view(cfg, num_heads, self.num_q_centroids, dim),
            kcentroids.view(cfg, num_heads, self.num_k_centroids, dim),
            q_cluster_sizes,
            k_cluster_sizes,
            self.top_p_kmeans,
            self.min_kc_ratio,
        )

        # 3. Permute the query, key, value
        q_permuted, q_sorted_indices = permute_tensor_by_labels_triton(query, qlabels, dim=2)
        k_permuted, k_sorted_indices = permute_tensor_by_labels_triton(key, klabels, dim=2)
        v_permuted, _ = permute_tensor_by_labels_triton(value, klabels, dim=2, sorted_indices=k_sorted_indices)

        return q_permuted, k_permuted, v_permuted, dynamic_map, q_cluster_sizes, k_cluster_sizes, q_sorted_indices

    @time_logging_decorator("Level 3 - dynamic map post processing")
    def dynamic_map_post_processing(
        self,
        q_perm,
        k_perm,
        v_perm,
        query,
        key,
        value,
        dyn_map,
        qc_sz_s,
        kc_sz_s,
        q_sorted_indices,
        video_length,
        context_length,
        prompt_length,
        unprompt_length,
    ):
        # Update the query, key, value
        query[:, :, :-context_length, :] = q_perm
        key[:, :, :-context_length, :] = k_perm
        value[:, :, :-context_length, :] = v_perm

        # Update the dynamic map
        dyn_map = F.pad(dyn_map, (0, 2, 0, 2), value=0)
        dyn_map[:, :, -2, :-1] = True
        dyn_map[:, :, :-1, -2] = True
        dyn_map[:, :, -1, -1] = True

        # Update the cluster sizes
        qc_sz_s = F.pad(qc_sz_s, (0, 2), value=0)
        qc_sz_s[:, :, -2] = prompt_length
        qc_sz_s[:, :, -1] = unprompt_length
        kc_sz_s = F.pad(kc_sz_s, (0, 2), value=0)
        kc_sz_s[:, :, -2] = prompt_length
        kc_sz_s[:, :, -1] = unprompt_length

        # Update the sorted indices
        q_sorted_indices = F.pad(q_sorted_indices, (0, context_length), value=0)
        q_sorted_indices[:, video_length:] = torch.arange(
            video_length, video_length + context_length, device=q_sorted_indices.device
        )

        q_sorted_indices = q_sorted_indices.unsqueeze(0)

        return query, key, value, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices

    @time_logging_decorator("Level 3 - prepare video part")
    def prepare_video_part(self, query, key, value):
        video_length = self.num_frame * self.frame_size
        query_video = query[:, :, :video_length, :].contiguous()
        key_video = key[:, :, :video_length, :].contiguous()
        value_video = value[:, :, :video_length, :].contiguous()

        attn_output = torch.zeros_like(query)
        return query_video, key_video, value_video, attn_output

    @time_logging_decorator("Level 2 - attention core logic")
    def attention_core_logic(self, query, key, value, timestep, layer_idx, cu_max_seqlens):
        cfg, num_heads, seq_len, dim = query.size()
        assert cfg == 1, "Batch size must be 1 for kmeans block sparse attention"

        prompt_length, context_length, num_frame, frame_size = (
            self.prompt_length,
            self.context_length,
            self.num_frame,
            self.frame_size,
        )

        assert (
            seq_len == context_length + num_frame * frame_size
        ), f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        # Determine if we use Full Attention to calculate
        full_attention_flag = False

        if self.layer_idx < self.first_layers_fp:
            full_attention_flag = True
        if timestep[0] > self.first_times_fp:
            full_attention_flag = True

        if full_attention_flag:
            if self.zero_step_kmeans_init:
                video_length = self.num_frame * self.frame_size
                query_video = query[:, :, :video_length, :].contiguous()
                key_video = key[:, :, :video_length, :].contiguous()
                self.kmeans_clustering(query_video, key_video, layer_idx)

            output_hidden_states = self.flashinfer_attention(query, key, value, cu_max_seqlens)
            return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)
        else:
            # Due to Hunyuan's design, we need to split the query, key, value into 1. video part, 2. prompt part, 3. Unused prompt part
            video_length = num_frame * frame_size
            unprompt_length = context_length - prompt_length

            # 1. Video part
            query_video, key_video, value_video, attn_output = self.prepare_video_part(query, key, value)

            # Core logic
            q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices = self.semantic_aware_permutation(
                query_video, key_video, value_video, timestep, layer_idx
            )

            # Post-processing to make sure Part 1 and Part 2 can attend each other, while Part 3 is only attended by itself
            q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices = self.dynamic_map_post_processing(
                q_perm,
                k_perm,
                v_perm,
                query,
                key,
                value,
                dyn_map,
                qc_sz_s,
                kc_sz_s,
                q_sorted_indices,
                video_length,
                context_length,
                prompt_length,
                unprompt_length,
            )

            output_permuted = dynamic_block_sparse_fwd_flashinfer(
                q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, is_cpu=False
            )

            # attn_output = apply_inverse_permutation(output_permuted, q_sorted_indices, dim=2)
            attn_output = apply_inverse_permutation_triton(output_permuted, q_sorted_indices, dim=2)

            # Save time, layer, density information to logging file
            if self.logging_file is not None:
                # Create log entry
                densities = density_calculation(dyn_map, qc_sz_s, kc_sz_s)

                avg_density = densities.mean().item()
                log_entry = {
                    "timestep": timestep[0].item(),
                    "layer": layer_idx,
                    "avg_density": avg_density,
                    "density": densities.tolist(),
                }

                # print(f"Time Step: {timestep[0].item()} Layer: {layer_idx} Density: {avg_density}")

                # Append to log file
                with open(self.logging_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            return attn_output.reshape(cfg, num_heads, seq_len, dim)


def flashinfer_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
    """
    Launcher for the Flashinfer varlen kernel for HunyuanVideo.

    Args:
        q (torch.Tensor): Query tensor, shape [B, H, S, D].
        k (torch.Tensor): Key tensor, shape [B, H, S, D].
        v (torch.Tensor): Value tensor, shape [B, H, S, D].
        block_mask_map (torch.Tensor): Boolean mask, shape [B, H, qc_num, kc_num]. Currently must on CPU.
        block_row_sz (torch.Tensor): Query block sizes, shape [B, H, qc_num]. Currently must on CPU.
        block_col_sz (torch.Tensor): Key block sizes, shape [B, H, kc_num]. Currently must on CPU.
    """

    # Create block mask map
    B, H, S, D = q.shape
    block_row_sz = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    block_col_sz = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
    block_mask_map = torch.tensor(
        [
            [
                True,
                False,
            ],
            [False, True],
        ],
        device=q.device,
        dtype=torch.bool,
    )

    # Expand dims
    block_row_sz = block_row_sz.repeat(B, H, 1)
    block_col_sz = block_col_sz.repeat(B, H, 1)
    block_mask_map = block_mask_map.repeat(B, H, 1, 1)

    # Input shape check
    qc_num = block_row_sz.shape[-1]
    kc_num = block_col_sz.shape[-1]
    assert block_mask_map.shape == (B, H, qc_num, kc_num)

    # Check if block_col_sz and block_row_sz are the same for each head
    assert torch.all(block_col_sz.sum(dim=2) == block_col_sz.sum(dim=2)[0, 0])
    assert torch.all(block_row_sz.sum(dim=2) == block_row_sz.sum(dim=2)[0, 0])

    # Prepare flashinfer wrapper
    float_workspace_buffer = torch.empty(128 * 1024 * 1024, device=q.device)
    wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(float_workspace_buffer, backend="auto")

    # Reshape inputs to (B * H, ...)
    q = q.reshape(B * H, S, D)
    k = k.reshape(B * H, S, D)
    v = v.reshape(B * H, S, D)
    block_mask_map = block_mask_map.reshape(B * H, qc_num, kc_num)
    block_row_sz = block_row_sz.reshape(B * H, qc_num)
    block_col_sz = block_col_sz.reshape(B * H, kc_num)

    wrapper.plan(
        block_mask_map=block_mask_map,
        block_row_sz=block_row_sz,
        block_col_sz=block_col_sz,
        num_qo_heads=B * H,
        num_kv_heads=B * H,
        head_dim=D,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )

    o = wrapper.run(q, k, v)  # [num_qo_heads, qo_len, head_dim]
    o = o.reshape(B, H, S, D)
    return o
