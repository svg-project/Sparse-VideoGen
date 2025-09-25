import sys
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from torch.nn.attention.flex_attention import (
    flex_attention,
)

from .placement import (
    hidden_states_placement,
    ref_hidden_states_placement,
    ref_sparse_head_placement,
    sparse_head_placement,
)
from .utils import create_block_mask_cached, generate_temporal_head_mask_mod

try:
    sys.path.append("svg/kernels/build/")
    import _kernels

    def qk_norm(attn, query, key):
        if attn.norm_q is not None:
            _kernels.layer_norm_forward(query.view(-1, query.shape[-1]), attn.norm_q.weight, attn.norm_q.bias)
        if attn.norm_k is not None:
            _kernels.layer_norm_forward(key.view(-1, key.shape[-1]), attn.norm_k.weight, attn.norm_k.bias)
        return query, key

    def rotary_emb(image_rotary_emb, query, key, text_seq_length):
        cos, sin = image_rotary_emb
        _kernels.apply_qk_rope_inplace_cossin(query, key, cos, sin, text_seq_length)
        return query, key

except ImportError:
    import warnings

    warnings.warn("Could not import RoPE / Norm kernels! Falling back to PyTorch implementation.")

    def qk_norm(attn, query, key):
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        return query, key

    def rotary_emb(image_rotary_emb, query, key, text_seq_length):
        query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
        key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
        return query, key


flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
torch._dynamo.config.cache_size_limit = 192 * 3
torch._dynamo.config.accumulated_cache_size_limit = 192 * 3


# Use this class to save attention qkv
class CogVideoX_SparseAttn_Processor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    version = None
    context_length = 0
    num_frame = 0
    frame_size = 0

    first_layers_fp = 0
    first_times_fp = 0

    num_sampled_rows = 32
    attention_masks = None
    block_mask = None

    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def get_qkv(self, attn, hidden_states):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        return query, key, value

    def process_before_linear(self, attn, hidden_states, encoder_hidden_states):
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        return hidden_states, batch_size, sequence_length

    def transpose_qkv(self, attn, query, key, value, batch_size):
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        return query, key, value, head_dim

    def get_o(self, attn, hidden_states, batch_size, head_dim):
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def split_hidden_states(self, hidden_states, text_seq_length):
        encoder_hidden_states, hidden_states = hidden_states.split([text_seq_length, hidden_states.size(1) - text_seq_length], dim=1)
        return encoder_hidden_states, hidden_states

    def flash_attention(self, query, key, value):
        output_hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        return output_hidden_states

    def sample_mse(self, query, key, value):

        assert len(self.attention_masks) == 2

        cfg, num_heads, seq_len, dim = query.size()
        num_sampled_rows = min(self.num_sampled_rows, seq_len)
        sampled_rows = torch.randint(low=0, high=seq_len, size=(num_sampled_rows,))
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

    def sparse_flex_attention(self, query, key, value, block_mask):
        return flex_attention(query, key, value, block_mask=block_mask)

    def sparse_head_placement(self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size):
        query_out, key_out, value_out = ref_sparse_head_placement(query, key, value, best_mask_idx, context_length, num_frame, frame_size)
        return query_out, key_out, value_out

    def fast_sparse_head_placement(self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size):
        sparse_head_placement(query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size)
        return query_out, key_out, value_out

    def hidden_states_placement(self, hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size):
        ref_hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size)

    def fast_hidden_states_placement(self, hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size):
        hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size)

    def attention_core_logic(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, timestep):
        cfg, num_heads, seq_len, dim = query.size()

        context_length, num_frame, frame_size = self.context_length, self.num_frame, self.frame_size

        assert seq_len == context_length + num_frame * frame_size, f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        # Determine if we use Full Attention to calculate
        full_attention_flag = False
        if self.layer_idx < 42 * self.first_layers_fp:
            full_attention_flag = True
        if timestep[0] > 1000 * (1 - self.first_times_fp):
            full_attention_flag = True

        if full_attention_flag:
            output_hidden_states = self.flash_attention(query, key, value)
            return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)
        else:

            sampled_mses = self.sample_mse(query, key, value)
            best_mask_idx = torch.argmin(sampled_mses, dim=0)
            output_hidden_states = torch.zeros_like(query)
            query_out, key_out, value_out = torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)

            query_out, key_out, value_out = self.fast_sparse_head_placement(query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size)
            # query_out, key_out, value_out = self.sparse_head_placement(query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size)

            hidden_states = self.sparse_flex_attention(query_out, key_out, value_out, block_mask=self.block_mask)

            self.fast_hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size)
            # self.hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size)

            return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states, batch_size, sequence_length = self.process_before_linear(attn, hidden_states, encoder_hidden_states)
        query, key, value = self.get_qkv(attn, hidden_states)
        query, key, value, head_dim = self.transpose_qkv(attn, query, key, value, batch_size)

        query, key = qk_norm(attn, query, key)

        query, key = rotary_emb(image_rotary_emb, query, key, text_seq_length)

        # ========================================================================
        hidden_states = self.attention_core_logic(query, key, value, timestep)
        # ========================================================================

        hidden_states = self.get_o(attn, hidden_states, batch_size, head_dim)
        encoder_hidden_states, hidden_states = self.split_hidden_states(hidden_states, text_seq_length)

        return hidden_states, encoder_hidden_states


def prepare_flexattention(cfg_size, num_head, head_dim, dtype, device, context_length, num_frame, frame_size, diag_width=1, multiplier=2):
    seq_len = context_length + num_frame * frame_size
    query, key, value = [torch.zeros((1, cfg_size * num_head, seq_len, head_dim), dtype=dtype, device=device) for _ in range(3)]

    # NOTE: multiplier == diag_width
    assert diag_width == multiplier
    mask_mod = generate_temporal_head_mask_mod(context_length, num_frame, frame_size, mul=multiplier, attn_sink=False)
    block_mask = create_block_mask_cached(mask_mod, 1, cfg_size * num_head, seq_len, seq_len, device=device, _compile=True)
    hidden_states = flex_attention(query, key, value, block_mask=block_mask)
    hidden_states = flex_attention(
        query.view(cfg_size, num_head, seq_len, head_dim), key.view(cfg_size, num_head, seq_len, head_dim), value.view(cfg_size, num_head, seq_len, head_dim), block_mask=block_mask
    )
    return block_mask
