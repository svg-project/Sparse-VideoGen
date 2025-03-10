import math
from math import floor
import os
import random
from functools import lru_cache

import numpy as np
import torch
from torch.nn.attention.flex_attention import (
    create_block_mask,
)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda", _compile=False):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device, _compile=_compile)
    return block_mask

def generate_temporal_head_mask_mod(prompt_length: int = 226, num_frames: int = 13, token_per_frame: int = 1350, mul: int = 2, attn_sink: bool = False):
    
    def round_to_multiple(idx):
        return floor(idx / 128) * 128
        
    def temporal_mask_mod(b, h, q_idx, kv_idx):
        first_row_mask = q_idx < prompt_length
        if attn_sink:
            first_column_mask = kv_idx < (prompt_length + token_per_frame)
        else:
            first_column_mask = kv_idx < prompt_length

        two_frame = round_to_multiple(mul * token_per_frame)
        temporal_head_mask = (torch.abs(q_idx - kv_idx) < two_frame)
        return first_column_mask | first_row_mask | temporal_head_mask
    
    return temporal_mask_mod

def sparsity_to_width(sparsity, context_length, num_frame, frame_size):
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len ** 2
    
    sparsity = (sparsity * total_elements - 2 * seq_len * context_length) / total_elements
      
    width = seq_len * (1 - math.sqrt(1 - sparsity))
    width_frame = width / frame_size
    
    return width_frame

def get_attention_mask(mask_name, context_length, num_frame, frame_size):
    # TODO: Replace with real implementation

    attention_mask = torch.zeros((context_length + num_frame * frame_size, context_length + num_frame * frame_size)).cuda()
    if mask_name == "spatial":
        attention_mask[:context_length, :] = 1
        attention_mask[:, :context_length] = 1
        block_size, block_thres = 128, frame_size * 1.5
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    attention_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1
        # attention_mask = torch.load("/data/home/xihaocheng/andy_develop/I2VSparse/sparseattn/v1.5/mask_tensor/mask_spatial.pt", map_location="cuda")
    elif mask_name == "temporal":
        pixel_attn_mask = torch.zeros_like(attention_mask[context_length:, context_length:])

        block_size, block_thres = 128, frame_size * 1.5
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1

        pixel_attn_mask = pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame)\
            .permute(1, 0, 3, 2).reshape(frame_size * num_frame, frame_size * num_frame)
        attention_mask[context_length:, context_length:] = pixel_attn_mask
        # attention_mask = torch.load("/data/home/xihaocheng/andy_develop/I2VSparse/sparseattn/v1.5/mask_tensor/mask_temporal.pt", map_location="cuda")
    return attention_mask
