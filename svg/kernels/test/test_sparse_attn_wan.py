from itertools import product
from test.test_sparse_attn import assert_close, gen_mask_block2element
from typing import Tuple

import flashinfer
import numpy as np
import pytest
import torch
from ops.attention_ops_wan import (
    WanFAMetadata,
    gen_temporal_mask,
    ref_gen_temporal_mask,
    wan_sparse_attn_forward,
)

torch.manual_seed(0)


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


parameters = list(product([(21, 3600)], [77, 226], [24, 40], [64, 128], [3, 6.2]))


# parameters = list(product([(21, 3600)], [77], [24], [128], [3]))
@pytest.mark.parametrize("video_config, len_text_prompt, num_heads, head_dim, mul", parameters)
@torch.inference_mode()
def test_temporal_sparse_attn(video_config, len_text_prompt, num_heads, head_dim, mul):
    num_frames, num_tokens_per_frame = video_config
    seq_len = num_frames * num_tokens_per_frame

    q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16).cuda()
    k = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16).cuda()
    v = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16).cuda()

    mask_metadata = gen_temporal_mask(num_frames, num_tokens_per_frame, mul)
    _, _, block_size = mask_metadata

    ref_block_mask = ref_gen_temporal_mask(num_frames, num_tokens_per_frame, mul)
    ref_mask = gen_mask_block2element(ref_block_mask, block_size, 0)

    o_ref = ref_torch_attn_impl(q, k, v, ref_mask)

    fa_metadata = WanFAMetadata(
        num_frames,
        num_tokens_per_frame,
        mask_metadata,
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8).cuda(),
    )
    o = wan_sparse_attn_forward(q, k, v, fa_metadata)

    assert_close(o, o_ref)
