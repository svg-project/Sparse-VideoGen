"""Mask Mod for Image2Video"""

import math
from math import floor, ceil
import torch
from torch import Tensor


from functools import lru_cache
from typing import Optional, List

import torch
from torch.nn.attention.flex_attention import (
    create_block_mask,
)


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda", _compile=False):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device, _compile=_compile)
    return block_mask

def generate_temporal_head_mask_mod(context_length: int = 226, prompt_length: int = 226, num_frames: int = 13, token_per_frame: int = 1350, mul: int = 2):
    
    def round_to_multiple(idx):
        return ceil(idx / 128) * 128
        
    def temporal_mask_mod(b, h, q_idx, kv_idx):
        two_frame = round_to_multiple(mul * token_per_frame)
        temporal_head_mask = (torch.abs(q_idx - kv_idx) <= two_frame)

        # return temporal_head_mask
        first_frame_mask = (kv_idx < token_per_frame)
        video_mask = first_frame_mask | temporal_head_mask
        return video_mask
    
    return temporal_mask_mod

def generate_dense_mask_mod():
    def dense_mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= 0) # True
    return dense_mask_mod

def sparsity_to_width(sparsity, context_length, num_frame, frame_size):
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len ** 2
    
    sparsity = (sparsity * total_elements - 2 * seq_len * context_length) / total_elements
      
    width = seq_len * (1 - math.sqrt(1 - sparsity))
    width_frame = width / frame_size
    
    return width_frame

def get_attention_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size):
    
    from termcolor import colored

    allocated = torch.cuda.memory_allocated() / 1e9
    print(colored(f"Allocated Memory: {allocated:.2f} GB", "yellow"))

    attention_mask = torch.zeros((context_length + num_frame * frame_size, context_length + num_frame * frame_size), device="cpu")

    # TODO: fix hard coded mask
    if mask_name == "spatial":
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")
        
        pixel_attn_mask[:, :frame_size] = 1 # First Frame Sink
        
        block_size, block_thres = 128, frame_size * 2
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1
        attention_mask = pixel_attn_mask
    else:
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")

        pixel_attn_mask[:, :frame_size] = 1 # First Frame Sink
        
        block_size, block_thres = 128, frame_size * 2
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1

        pixel_attn_mask = pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame).permute(1, 0, 3, 2).reshape(frame_size * num_frame, frame_size * num_frame)
        attention_mask = pixel_attn_mask

    attention_mask = attention_mask[:sample_mse_max_row].cuda()
    return attention_mask