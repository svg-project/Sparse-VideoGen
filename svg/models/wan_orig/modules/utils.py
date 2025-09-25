"""Mask Mod for Image2Video"""

from functools import lru_cache
from math import ceil, floor
from typing import List, Optional

import torch
from torch import Tensor
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
        temporal_head_mask = torch.abs(q_idx - kv_idx) <= two_frame

        first_frame_mask = kv_idx < token_per_frame

        video_mask = first_frame_mask | temporal_head_mask

        return video_mask

    return temporal_mask_mod


def generate_dense_mask_mod():
    def dense_mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= 0  # True

    return dense_mask_mod
