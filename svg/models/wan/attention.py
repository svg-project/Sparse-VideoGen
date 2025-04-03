import sys
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from torch.nn.attention.flex_attention import (
    flex_attention,
)

from .placement import wan_sparse_head_placement, wan_hidden_states_placement, ref_wan_sparse_head_placement, ref_wan_hidden_states_placement
from .utils import generate_temporal_head_mask_mod, create_block_mask_cached

flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
torch._dynamo.config.cache_size_limit = 192 * 3
torch._dynamo.config.accumulated_cache_size_limit = 192 * 3


class WanAttn_SparseAttn_Processor2_0:
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
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def get_qkv(self, attn, hidden_states, encoder_hidden_states):
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        return query, key, value
    
    def get_qk_norm(self, attn, query, key):
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        return query, key
    
    def get_transpose_qkv(self, attn, query, key, value):
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        return query, key, value
    
    def get_rotary_emb(self, query, key, rotary_emb):

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        
        return query, key
            
    def get_o(self, attn, query, hidden_states, hidden_states_img):
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)    
    
        return hidden_states
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query, key, value = self.get_qkv(attn, hidden_states, encoder_hidden_states)

        query, key = self.get_qk_norm(attn, query, key)

        query, key, value = self.get_transpose_qkv(attn, query, key, value)
        
        query, key = self.get_rotary_emb(query, key, rotary_emb)


        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # ========================================================================
        if timestep is None: # Cross Attention in Wan
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else: # The main attention
            hidden_states = self.attention_core_logic(query, key, value, timestep)
        # ========================================================================
        
        hidden_states = self.get_o(attn, query, hidden_states, hidden_states_img)

        return hidden_states
    
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

    def sparse_flex_attention(self, query, key, value, block_mask):
        return flex_attention(query, key, value, block_mask=block_mask)

    def sparse_head_placement(self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size):
        
        query_out, key_out, value_out = ref_wan_sparse_head_placement(query, key, value, best_mask_idx, context_length, num_frame, frame_size)

        return query_out, key_out, value_out

    def fast_sparse_head_placement(self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size):

        wan_sparse_head_placement(query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size)

        return query_out, key_out, value_out

    def hidden_states_placement(self, \
        hidden_states, output_hidden_states, \
        best_mask_idx, context_length, num_frame, frame_size
    ):
        ref_wan_hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size)

    def fast_hidden_states_placement(self, \
        hidden_states, output_hidden_states, \
        best_mask_idx, context_length, num_frame, frame_size
    ):
        wan_hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size)

    def flash_attention(self, query, key, value):
        output_hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
        return output_hidden_states
    
    def attention_core_logic(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        timestep,
    ):
        cfg, num_heads, seq_len, dim = query.size()
        
        context_length, num_frame, frame_size = self.context_length, self.num_frame, self.frame_size

        assert seq_len == context_length + num_frame * frame_size, \
            f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        # Determine if we use Full Attention to calculate
        full_attention_flag = False

        if self.layer_idx < self.num_layers * self.first_layers_fp:
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

            hidden_states = self.sparse_flex_attention(query_out, key_out, value_out, block_mask=self.block_mask)

            self.fast_hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, context_length, num_frame, frame_size)

            return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)


def prepare_flexattention(cfg_size, num_head, head_dim, dtype, device, context_length, prompt_length, num_frame, frame_size, \
    diag_width=1, multiplier=2
):
    assert diag_width == multiplier, f"{diag_width} is not equivalent to {multiplier}"
    
    seq_len = context_length + num_frame * frame_size
    query, key, value = [torch.zeros((cfg_size, num_head, seq_len, head_dim), dtype=dtype, device=device) for _ in range(3)]

    mask_mod = generate_temporal_head_mask_mod(context_length, prompt_length, num_frame, frame_size, mul=multiplier)
    block_mask = create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)

    hidden_states = flex_attention(query, key, value, block_mask=block_mask)

    return block_mask