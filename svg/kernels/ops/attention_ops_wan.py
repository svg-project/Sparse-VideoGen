import os
from dataclasses import dataclass
from typing import Tuple

import flashinfer
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch


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
    return 1


def visualize_attention_mask(attn_mask: np.ndarray, sparsity: float, block_size: Tuple[int, int]):
    os.makedirs("figures", exist_ok=True)

    # Create a colormap for visualization
    # -1 values will be light gray, other values will range from blue to red
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.get_cmap("coolwarm").copy()
    cmap.set_bad(color="lightgray")
    masked_data = np.ma.masked_where(attn_mask == -1, attn_mask)
    plt.imshow(masked_data, cmap=cmap, interpolation="nearest")

    plt.colorbar(label="Column Index")
    plt.title(f"Attention Mask Visualization | Temporal sparsity: {sparsity:.2f} | Block size: {block_size[0]}x{block_size[1]}")
    plt.xlabel("Column Block Index")
    plt.ylabel("Row Block Index")

    plt.savefig("figures/attention_mask.png")
    plt.close()


def gen_temporal_mask(
    num_frames: int,
    num_tokens_per_frame: int,
    multiplier: float,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Generate temporal mask for temporal attention head (reordered sliding window).
    abs(q_idx - kv_idx) < multiplier * num_tokens_per_frame

    Args:
        multiplier (int): width of sliding window

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]: Attention mask (row, column) in BSR format and block size
    """
    # TODO: Autosearch row_block_size and column_block_size
    row_block_size = column_block_size = get_factor(num_frames, num_tokens_per_frame)

    assert (num_tokens_per_frame * num_frames) % row_block_size == 0 and (num_tokens_per_frame * num_frames) % column_block_size == 0
    num_row_blocks = num_col_blocks = num_frames * num_tokens_per_frame // row_block_size

    attn_mask = np.full((num_row_blocks, num_col_blocks), -1)
    host_row_indices = np.zeros(num_row_blocks + 1, dtype=np.int32)

    for i in range(num_row_blocks):
        for j in range(num_col_blocks):
            row_token_idx = i * row_block_size + row_block_size // 2
            col_token_idx = j * column_block_size + column_block_size // 2
            if abs(row_token_idx - col_token_idx) < multiplier * num_tokens_per_frame:
                attn_mask[i, j] = j
                host_row_indices[i + 1] += 1

    host_row_indices = np.cumsum(host_row_indices)
    host_column_indices = attn_mask[attn_mask != -1]

    sparsity = len(host_column_indices) / (num_row_blocks * num_col_blocks)
    print(f"Temporal sparsity: {sparsity:.2f} | Block size: {row_block_size}x{column_block_size}")
    print("Visualize temporal attention mask:")
    visualize_attention_mask(attn_mask, sparsity, (row_block_size, column_block_size))

    row_indices = torch.from_numpy(host_row_indices).to(torch.int32).cuda()
    # This padding is to avoid irregular memory access in flashinfer kernel
    host_column_indices = np.concatenate((host_column_indices, [0] * 256))
    column_indices = torch.from_numpy(host_column_indices).to(torch.int32).cuda()

    return row_indices, column_indices, (row_block_size, column_block_size)


def ref_gen_temporal_mask(
    num_frames: int,
    num_tokens_per_frame: int,
    multiplier: float,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Generate temporal mask for temporal attention head (reordered sliding window).
    abs(q_idx - kv_idx) < multiplier * num_tokens_per_frame

    Args:
        num_frames (int): number of frames
        num_tokens_per_frame (int): number of tokens per frame
        multiplier (float): multiplier for the temporal mask

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]: Attention mask (row, column) in BSR format and block size
    """
    # TODO: Autosearch row_block_size and column_block_size
    row_block_size = column_block_size = get_factor(num_frames, num_tokens_per_frame)

    assert (num_tokens_per_frame * num_frames) % row_block_size == 0 and (num_tokens_per_frame * num_frames) % column_block_size == 0
    num_row_blocks = num_col_blocks = num_frames * num_tokens_per_frame // row_block_size

    attn_mask = np.full((num_row_blocks, num_col_blocks), -1)
    host_row_indices = np.zeros(num_row_blocks + 1, dtype=np.int32)

    for i in range(num_row_blocks):
        for j in range(num_col_blocks):
            row_token_idx = i * row_block_size + row_block_size // 2
            col_token_idx = j * column_block_size + column_block_size // 2
            if abs(row_token_idx - col_token_idx) < multiplier * num_tokens_per_frame:
                attn_mask[i, j] = j
                host_row_indices[i + 1] += 1
    return attn_mask


@dataclass
class WanFAMetadata:
    num_frames: int
    num_tokens_per_frame: int

    temporal_mask_metadata: Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]
    workspace: torch.Tensor


def wan_sparse_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: WanFAMetadata,
) -> torch.Tensor:
    """
    Forward pass for spatial attention head.
    Separate calculation of text attention sink and video sparse attention

    Args:
        q,k,v: [seq_len, num_heads, head_dim]
        metadata (FAMetadata): metadata for spatial attention head

    Returns:
        torch.Tensor: output tensor
    """
    row_indices, column_indices, block_size = metadata.temporal_mask_metadata

    # Assert the dimension of video modality
    assert q.shape[0] % block_size[0] == 0, f"Query length {q.shape[0]} % block_size {block_size[0]} != 0"
    assert k.shape[0] % block_size[1] == 0, f"Key length {k.shape[0]} % block_size {block_size[1]} != 0"
    assert k.shape[0] == v.shape[0], f"Key length {k.shape[0]} != Value length {v.shape[0]}"

    bsr_wrapper = flashinfer.BlockSparseAttentionWrapper(metadata.workspace)
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

    return o_image
