import numpy as np
import pytest
import scipy as sp
import torch
import flashinfer


def _test_variable_block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_mask_map: torch.Tensor,
    block_row_sz: torch.Tensor,
    block_col_sz: torch.Tensor,
):
    # qkv: HND
    assert torch.all(block_col_sz.sum(dim=1) == block_col_sz.sum(dim=1)[0])
    assert torch.all(block_row_sz.sum(dim=1) == block_row_sz.sum(dim=1)[0])

    float_workspace_buffer = torch.empty(128 * 1024 * 1024, device=q.device)
    wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
        float_workspace_buffer, backend="auto"
    )

    wrapper.plan(
        block_mask_map=block_mask_map,
        block_row_sz=block_row_sz,
        block_col_sz=block_col_sz,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )

    o = wrapper.run(q, k, v)  # [num_qo_heads, qo_len, head_dim]
    o = o.reshape(num_kv_heads, -1, *o.shape[-2:])
    return o