from dataclasses import dataclass
from typing import Tuple

import flashinfer
import numpy as np
import torch


def _gen_temporal_mask(
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
    assert num_tokens_per_frame % 10 == 0
    row_block_size = column_block_size = num_tokens_per_frame // 10

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

    print("Generated temporal attention mask:")
    print(attn_mask)

    host_row_indices = np.cumsum(host_row_indices)
    host_column_indices = attn_mask[attn_mask != -1]

    print("Temporal sparsity: ", len(host_column_indices) / (num_row_blocks * num_col_blocks))

    row_indices = torch.from_numpy(host_row_indices).to(torch.int32).cuda()
    # This padding is to avoid irregular memory access in flashinfer kernel
    host_column_indices = np.concatenate((host_column_indices, [0] * 256))
    column_indices = torch.from_numpy(host_column_indices).to(torch.int32).cuda()

    return row_indices, column_indices, (row_block_size, column_block_size)


def _gen_spatial_mask(
    num_frames: int,
    num_tokens_per_frame: int,
    multiplier: int,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Generate spatial mask for spatial attention head (sliding window + attention sink).
    q_idx < 1*num_tokens_per_frame | abs(frame_idx(q_idx) - frame_idx(kv_idx)) < multiplier

    Args:
        num_frames (int): number of total frames for the video
        multiplier (int): width of sliding window

    Returns:
        Attention mask (row, column) in BSR format and block size, refs:
        1. https://ieeexplore.ieee.org/document/8742660
        2. https://docs.flashinfer.ai/api/sparse.html
        row tensor: (num_frames+1,)
        column tensor: (multiplier * num_frames,)
    """
    assert multiplier >= 0, "width of sliding window must be at least one frame"
    assert num_frames > 0

    # init numpy array 2D (num_frames, num_frames) with -1
    attn_mask = np.full((num_frames, num_frames), -1)
    host_row_indices = np.zeros(num_frames + 1, dtype=np.int32)

    for i in range(num_frames):
        for j in range(num_frames):
            if abs(i - j) <= multiplier or j == 0:
                attn_mask[i, j] = j
                host_row_indices[i + 1] += 1

    print("Generated spatial attention mask:")
    print(attn_mask)

    host_row_indices = np.cumsum(host_row_indices)
    host_column_indices = attn_mask[attn_mask != -1]

    print("Spatial sparsity: ", len(host_column_indices) / (num_frames * num_frames))

    row_indices = torch.from_numpy(host_row_indices).to(torch.int32).cuda()
    # This padding is to avoid irregular memory access in flashinfer kernel
    host_column_indices = np.concatenate((host_column_indices, [0] * 256))
    column_indices = torch.from_numpy(host_column_indices).to(torch.int32).cuda()

    return row_indices, column_indices, (num_tokens_per_frame, num_tokens_per_frame)


@dataclass
class FAMetadata:
    len_text_promt: int
    num_frames: int
    num_tokens_per_frame: int

    temporal_mask_metadata: Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]
    spatial_mask_metadata: Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]
    workspace: torch.Tensor


def init_sparse_attn(
    len_text_prompt: int,
    num_frames: int,
    num_tokens_per_frame: int,
    temporal_multiplier: float,
    spatial_multiplier: int,
) -> FAMetadata:
    temporal_mask_metadata = _gen_temporal_mask(num_frames, num_tokens_per_frame, temporal_multiplier)
    spatial_mask_metadata = _gen_spatial_mask(num_frames, num_tokens_per_frame, spatial_multiplier)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8).cuda()

    return FAMetadata(
        len_text_prompt,
        num_frames,
        num_tokens_per_frame,
        temporal_mask_metadata,
        spatial_mask_metadata,
        workspace_buffer,
    )


def sparse_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: FAMetadata,
    sparse_pattern: str = "temporal",
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
    assert sparse_pattern in ["temporal", "spatial"]

    if sparse_pattern == "temporal":
        row_indices, column_indices, block_size = metadata.temporal_mask_metadata
    else:
        row_indices, column_indices, block_size = metadata.spatial_mask_metadata

    bsr_wrapper = flashinfer.BlockSparseAttentionWrapper(metadata.workspace)
    len_text_prompt = metadata.len_text_promt
    bsr_wrapper.plan(
        row_indices,
        column_indices,
        q.shape[0] - len_text_prompt,  # separate text and video
        k.shape[0] - len_text_prompt,  # separate text and video
        block_size[0],
        block_size[1],
        q.shape[1],  # num_qo_heads
        k.shape[1],  # num_kv_heads
        q.shape[2],  # head_dim
    )
    o_image_l, o_image_l_lse = bsr_wrapper.run(q[len_text_prompt:, :, :], k[len_text_prompt:, :, :], v[len_text_prompt:, :, :], return_lse=True)

    o_image_r, o_image_r_lse = flashinfer.single_prefill_with_kv_cache(
        q[len_text_prompt:, :, :],
        k[:len_text_prompt, :, :],
        v[:len_text_prompt, :, :],
        causal=False,
        return_lse=True,
    )

    o_image, _ = flashinfer.merge_state(o_image_l, o_image_l_lse, o_image_r, o_image_r_lse)
    o_text = flashinfer.single_prefill_with_kv_cache(
        q[:len_text_prompt, :, :],
        k,
        v,
        causal=False,
        return_lse=False,
    )

    return torch.cat([o_text, o_image], dim=0)
