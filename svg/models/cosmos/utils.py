"""Mask Mod for Image2Video"""

import math
from functools import lru_cache
from math import ceil
from typing import Tuple

import flashinfer
import numpy as np
import sympy as sp
import torch
from torch.nn.attention.flex_attention import (
    create_block_mask,
)

from ...logger import logger


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda", _compile=False):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device, _compile=_compile)
    return block_mask


def generate_temporal_head_mask_mod(
    context_length: int = 226, prompt_length: int = 226, num_frames: int = 13, token_per_frame: int = 1350, mul: int = 2
):

    def round_to_multiple(idx):
        return ceil(idx / 128) * 128

    def temporal_mask_mod(b, h, q_idx, kv_idx):
        two_frame = round_to_multiple(mul * token_per_frame)
        temporal_head_mask = torch.abs(q_idx - kv_idx) <= two_frame

        # return temporal_head_mask
        first_frame_mask = kv_idx < token_per_frame
        video_mask = first_frame_mask | temporal_head_mask
        return video_mask

    return temporal_mask_mod


def generate_dense_mask_mod():
    def dense_mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= 0  # True

    return dense_mask_mod


def sparsity_to_width(sparsity, context_length, num_frame, frame_size):
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len**2

    sparsity = (sparsity * total_elements - 2 * seq_len * context_length) / total_elements

    width = seq_len * (1 - math.sqrt(1 - sparsity))
    width_frame = width / frame_size

    return width_frame


def get_attention_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size):
    """
    Generate the emulated attention mask. Used for online profiling.
    """

    from termcolor import colored

    allocated = torch.cuda.memory_allocated() / 1e9
    print(colored(f"Allocated Memory: {allocated:.2f} GB", "yellow"))

    attention_mask = torch.zeros(
        (context_length + num_frame * frame_size, context_length + num_frame * frame_size), device="cpu"
    )

    # TODO: fix hard coded mask
    if mask_name == "spatial":
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")

        pixel_attn_mask[:, :frame_size] = 1  # First Frame Sink

        block_size, block_thres = 128, frame_size * 2
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1
        attention_mask = pixel_attn_mask
    else:
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")

        pixel_attn_mask[:, :frame_size] = 1  # First Frame Sink

        block_size, block_thres = 128, frame_size * 2
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1

        pixel_attn_mask = (
            pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame)
            .permute(1, 0, 3, 2)
            .reshape(frame_size * num_frame, frame_size * num_frame)
        )
        attention_mask = pixel_attn_mask

    attention_mask = attention_mask[:sample_mse_max_row].cuda()
    return attention_mask


def get_factor(num_frames: int, num_tokens_per_frame: int) -> int:
    """
    Get the factor of num_frames * num_tokens_per_frame.
    Find the maximum factor that is less than 128.
    """
    # factors = sp.divisors(num_frames * num_tokens_per_frame)
    factors = sp.divisors(num_tokens_per_frame)
    # sort it, from large to small
    factors.sort(reverse=True)

    for factor in factors:
        if factor < 256:
            return factor

    raise ValueError(f"No factor found for {num_frames} * {num_tokens_per_frame}")


def gen_temporal_mask(
    num_frames: int,
    num_tokens_per_frame: int,
    multiplier: float,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Generate temporal mask for temporal attention head (reordered sliding window).
    Will be used for flashinfer sparse attention.

    abs(q_idx - kv_idx) < multiplier * num_tokens_per_frame

    Args:
        multiplier (int): width of sliding window

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]: Attention mask (row, column) in BSR format and block size
    """
    # TODO: Autosearch row_block_size and column_block_size
    row_block_size = column_block_size = get_factor(num_frames, num_tokens_per_frame)

    assert (num_tokens_per_frame * num_frames) % row_block_size == 0 and (
        num_tokens_per_frame * num_frames
    ) % column_block_size == 0
    num_row_blocks = num_col_blocks = num_frames * num_tokens_per_frame // row_block_size

    attn_mask = np.full((num_row_blocks, num_col_blocks), -1)
    host_row_indices = np.zeros(num_row_blocks + 1, dtype=np.int32)

    for i in range(num_row_blocks):
        for j in range(num_col_blocks):
            row_token_idx = i * row_block_size + row_block_size // 2
            col_token_idx = j * column_block_size + column_block_size // 2

            # Diagonal region
            if abs(row_token_idx - col_token_idx) < multiplier * num_tokens_per_frame:
                attn_mask[i, j] = j
                host_row_indices[i + 1] += 1
            # First frame region
            elif col_token_idx <= num_tokens_per_frame:
                attn_mask[i, j] = j
                host_row_indices[i + 1] += 1

    host_row_indices = np.cumsum(host_row_indices)
    host_column_indices = attn_mask[attn_mask != -1]

    sparsity = len(host_column_indices) / (num_row_blocks * num_col_blocks)
    logger.info(
        f"Flashinfer temporal sparsity: {sparsity * 100:.2f}% | Block size: {row_block_size}x{column_block_size}"
    )

    row_indices = torch.from_numpy(host_row_indices).to(torch.int32).cuda()
    # This padding is to avoid irregular memory access in flashinfer kernel
    host_column_indices = np.concatenate((host_column_indices, [0] * 256))
    column_indices = torch.from_numpy(host_column_indices).to(torch.int32).cuda()

    return row_indices, column_indices, (row_block_size, column_block_size)


def flashinfer_sparse_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    temporal_mask_metadata: Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]],
) -> torch.Tensor:
    """
    Forward pass for spatial attention head.
    Separate calculation of text attention sink and video sparse attention

    Args:
        q,k,v: [seq_len, num_heads, head_dim]
        temporal_mask_metadata: metadata for temporal attention head

    Returns:
        torch.Tensor: output tensor
    """
    row_indices, column_indices, block_size = temporal_mask_metadata

    cfg, num_heads, seq_len, head_dim = q.shape[0], q.shape[1], q.shape[2], q.shape[3]

    q = q.permute(2, 0, 1, 3).reshape(seq_len, num_heads, head_dim)
    k = k.permute(2, 0, 1, 3).reshape(seq_len, num_heads, head_dim)
    v = v.permute(2, 0, 1, 3).reshape(seq_len, num_heads, head_dim)

    # Assert the dimension of video modality
    assert q.shape[0] % block_size[0] == 0, f"Query length {q.shape[0]} % block_size {block_size[0]} != 0"
    assert k.shape[0] % block_size[1] == 0, f"Key length {k.shape[0]} % block_size {block_size[1]} != 0"
    assert k.shape[0] == v.shape[0], f"Key length {k.shape[0]} != Value length {v.shape[0]}"

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    bsr_wrapper = flashinfer.BlockSparseAttentionWrapper(workspace)

    bsr_wrapper.plan(
        row_indices,
        column_indices,
        q.shape[0],  # video length
        k.shape[0],  # video length
        block_size[0],
        block_size[1],
        q.shape[1],  # num_qo_heads
        k.shape[1],  # num_kv_heads
        q.shape[2],  # head_dim
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )
    o_image = bsr_wrapper.run(q, k, v, return_lse=False)

    o_image = o_image.reshape(seq_len, cfg, num_heads, head_dim).permute(1, 2, 0, 3).contiguous()

    return o_image
