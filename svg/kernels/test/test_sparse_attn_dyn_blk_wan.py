import flashinfer
import torch
import pytest
from ops.attention_ops_wan_dyn_blk import _test_variable_block_sparse_attention



def random_partition_batch(
    seq_len: int,
    num_blocks: int,
    bsz: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    assert seq_len >= num_blocks
    sizes = torch.empty((bsz, num_blocks), dtype=dtype, device=device)
    for i in range(bsz):
        cut_pts = torch.randperm(seq_len - 1, device=device)[: num_blocks - 1] + 1
        cut_pts, _ = torch.sort(cut_pts)
        row_sizes = torch.diff(
            torch.cat(
                (
                    torch.tensor([0], device=device),
                    cut_pts,
                    torch.tensor([seq_len], device=device),
                )
            )
        )
        sizes[i] = row_sizes

    assert sizes.min() >= 1
    assert sizes.max() <= seq_len
    assert torch.all(sizes.sum(dim=-1) == seq_len)

    return sizes


def _ref_attention(
    q: torch.Tensor,  # [gqa_group_size, qo_len, head_dim]
    k: torch.Tensor,  # [1, kv_len, head_dim]
    v: torch.Tensor,  # [1, kv_len, head_dim]
    block_mask_map: torch.Tensor,  # [MB, NB]
    block_row_sz: torch.Tensor,  # [MB]
    block_col_sz: torch.Tensor,  # [NB]
) -> torch.Tensor:
    # convert block mask map to element mask
    def _block_mask_to_element_mask(
        block_mask_map: torch.Tensor,  # [MB, NB] – bool
        block_row_sz: torch.Tensor,  # [MB]     – int (rows per block-row)
        block_col_sz: torch.Tensor,  # [NB]     – int (cols per block-col)
    ) -> torch.Tensor:
        block_row_sz = block_row_sz.to(block_mask_map.device, dtype=torch.long)
        block_col_sz = block_col_sz.to(block_mask_map.device, dtype=torch.long)
        expanded_rows = torch.repeat_interleave(block_mask_map, block_row_sz, dim=0)
        element_mask = torch.repeat_interleave(expanded_rows, block_col_sz, dim=1)

        return element_mask

    dense_mask = _block_mask_to_element_mask(
        block_mask_map, block_row_sz, block_col_sz
    ).to(dtype=torch.bool, device=q.device)

    q = q.transpose(0, 1).contiguous()
    k = k.transpose(0, 1).contiguous()
    v = v.transpose(0, 1).contiguous()
    o = flashinfer.prefill.single_prefill_with_kv_cache(
        q, k, v, custom_mask=dense_mask
    )  # [qo_len, gqa_group_size, head_dim]
    o = o.transpose(0, 1).contiguous()

    return o


@pytest.mark.parametrize("num_qo_heads", [1, 4, 16])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seq_len", [256, 4096, 8192])
@pytest.mark.parametrize("num_blocks_row", [10, 20])
@pytest.mark.parametrize("num_blocks_col", [50, 100])
@pytest.mark.parametrize("block_density", [0.2, 0.7, 0.9])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_variable_block_sparse_attention_wrapper(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    num_blocks_row: int,
    num_blocks_col: int,
    block_density: float,
    dtype: torch.dtype,
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")
    if seq_len // num_blocks_row < 1:
        pytest.skip("seq_len must be greater than num_blocks_row")
    if seq_len // num_blocks_col < 1:
        pytest.skip("seq_len must be greater than num_blocks_col")

    block_row_sz = random_partition_batch(seq_len, num_blocks_row, num_kv_heads)
    block_col_sz = random_partition_batch(seq_len, num_blocks_col, num_kv_heads)
    block_mask_map = (
        torch.rand(num_kv_heads, num_blocks_row, num_blocks_col) > block_density
    )
    block_mask_map = block_mask_map.to(dtype=torch.bool, device="cpu")

    q = torch.randn(num_qo_heads, seq_len, head_dim, device="cuda:0", dtype=dtype)
    k = torch.randn(num_kv_heads, seq_len, head_dim, device="cuda:0", dtype=dtype)
    v = torch.randn(num_kv_heads, seq_len, head_dim, device="cuda:0", dtype=dtype)

    o = _test_variable_block_sparse_attention(
        q,
        k,
        v,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        block_mask_map,
        block_row_sz,
        block_col_sz,
    )

    # Make compatible with GQA
    q = q.reshape(num_kv_heads, -1, *q.shape[-2:])
    for kv_head_idx in range(num_kv_heads):
        o_ref = _ref_attention(
            q[kv_head_idx],
            k[kv_head_idx : kv_head_idx + 1, :, :],
            v[kv_head_idx : kv_head_idx + 1, :, :],
            block_mask_map[kv_head_idx],
            block_row_sz[kv_head_idx],
            block_col_sz[kv_head_idx],
        )
        torch.testing.assert_close(o[kv_head_idx], o_ref, atol=1e-2, rtol=1e-2)

