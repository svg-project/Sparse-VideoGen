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
    if mask_name == "spatial":
        attention_mask = torch.load("/home/andy/I2VSparse/sparseattn/v1.5/mask_tensor/mask_spatial.pt", map_location="cuda")
    elif mask_name == "temporal":
        attention_mask = torch.load("/home/andy/I2VSparse/sparseattn/v1.5/mask_tensor/mask_temporal.pt", map_location="cuda")
    return attention_mask
