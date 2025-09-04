from itertools import product
from typing import Tuple

import flashinfer
import numpy as np
import pytest
import torch
from ops.attention_ops import (
    FAMetadata,
    _gen_spatial_mask,
    _gen_temporal_mask,
    sparse_attn_forward,
)

NUM_TOKEN_PER_FRAME = 3600

torch.manual_seed(0)


def ref_gen_temporal_mask(
    num_frames: int,
    num_tokens_per_frame: int,
    multiplier: float,
) -> np.array:
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

    return attn_mask


def ref_gen_spatial_mask(
    num_frames: int,
    num_tokens_per_frame: int,
    multiplier: int,
) -> np.array:
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

    return attn_mask


def gen_mask_block2element(
    block_mask: np.ndarray,
    block_size: Tuple[int, int],
    len_text_prompt: int,
) -> torch.Tensor:
    # block_mask <0 set 0; >=0 set 1
    block_mask = (block_mask >= 0).astype(np.int32)

    block_mask = np.repeat(block_mask, block_size[1], axis=1)
    block_mask = np.repeat(block_mask, block_size[0], axis=0)

    tmp = np.ones((block_mask.shape[0], len_text_prompt), dtype=np.int32)
    block_mask = np.concatenate([tmp, block_mask], axis=1)

    tmp = np.ones((len_text_prompt, block_mask.shape[1]), dtype=np.int32)
    block_mask = np.concatenate([tmp, block_mask], axis=0)

    # cast into bool
    block_mask = block_mask.astype(np.bool_)
    out = torch.from_numpy(block_mask).to(torch.bool).cuda()

    return out


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def ref_attn_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    o_ref = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)

    return o_ref


def ref_torch_attn_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
):
    # Shape is (seq_len, num_heads, head_dim)
    M = q.shape[0]
    N = k.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]

    o_shape = q.shape
    q = q.squeeze()
    k = k.squeeze()
    v = v.squeeze()

    # Transpose for easier matmul: (num_heads, seq_len, head_dim)
    q = q.permute(1, 0, 2)
    k = k.permute(1, 0, 2)
    v = v.permute(1, 0, 2)

    attn_output = torch.empty_like(q)

    scale = head_dim**0.5

    for h in range(num_heads):
        q_h = q[h, :, :]  # (seq_len, head_dim)
        k_h = k[h, :, :].T  # (head_dim, seq_len)
        v_h = v[h, :, :]  # (seq_len, head_dim)

        # (seq_len, seq_len)
        attn_scores = torch.matmul(q_h, k_h) / scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # attn_scores = attn_scores.to(torch.float32)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        # attn_weights = attn_weights.to(q_h.dtype)

        # (seq_len, head_dim)
        output_h = torch.matmul(attn_weights, v_h)

        attn_output[h, :, :] = output_h

    attn_output = attn_output.permute(1, 0, 2)

    return attn_output


# parameters = list(product([5,13], [16,32,77], [24,32], [64,128], [0,1,2]))
parameters = list(product([21], [16], [24, 32], [64, 128], [2]))


@pytest.mark.parametrize("num_frames, len_text_prompt, num_heads, head_dim, mul", parameters)
@torch.inference_mode()
def test_spatial_sparse_attn(num_frames, len_text_prompt, num_heads, head_dim, mul):
    seq_len = num_frames * NUM_TOKEN_PER_FRAME + len_text_prompt

    q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16).cuda()
    k = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16).cuda()
    v = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16).cuda()

    mask_metadata = _gen_spatial_mask(num_frames, NUM_TOKEN_PER_FRAME, mul)
    _, _, block_size = mask_metadata

    ref_block_mask = ref_gen_spatial_mask(num_frames, NUM_TOKEN_PER_FRAME, mul)
    ref_mask = gen_mask_block2element(ref_block_mask, block_size, len_text_prompt)

    # o_ref = ref_attn_impl(q, k, v, ref_mask) # This will fail due to flashinfer's bug
    o_ref = ref_torch_attn_impl(q, k, v, ref_mask)

    fa_metadata = FAMetadata(
        len_text_prompt,
        num_frames,
        NUM_TOKEN_PER_FRAME,
        None,
        mask_metadata,
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8).cuda(),
    )
    o = sparse_attn_forward(q, k, v, fa_metadata, "spatial")

    try:
        assert_close(o, o_ref)
    except Exception as e:
        import IPython

        IPython.embed()


# parameters = list(product([5,13], [16,32,77], [24,32], [64,128], [0.5,1,1.4,1.8]))
parameters = list(product([21], [16], [24, 32], [64, 128], [1.8]))


@pytest.mark.parametrize("num_frames, len_text_prompt, num_heads, head_dim, mul", parameters)
@torch.inference_mode()
def test_temporal_sparse_attn(num_frames, len_text_prompt, num_heads, head_dim, mul):
    seq_len = num_frames * NUM_TOKEN_PER_FRAME + len_text_prompt

    q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16).cuda()
    k = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16).cuda()
    v = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16).cuda()

    mask_metadata = _gen_temporal_mask(num_frames, NUM_TOKEN_PER_FRAME, mul)
    _, _, block_size = mask_metadata

    ref_block_mask = ref_gen_temporal_mask(num_frames, NUM_TOKEN_PER_FRAME, mul)
    ref_mask = gen_mask_block2element(ref_block_mask, block_size, len_text_prompt)

    # o_ref = ref_attn_impl(q, k, v, ref_mask) # This will fail due to flashinfer's bug
    o_ref = ref_torch_attn_impl(q, k, v, ref_mask)

    fa_metadata = FAMetadata(
        len_text_prompt,
        num_frames,
        NUM_TOKEN_PER_FRAME,
        mask_metadata,
        None,
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8).cuda(),
    )
    o = sparse_attn_forward(q, k, v, fa_metadata, "temporal")

    assert_close(o, o_ref)
