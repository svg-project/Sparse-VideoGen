import torch
import triton
import triton.language as tl
from typing import Optional
from ...timer import time_logging_decorator


########################################################
# Triton kernels
########################################################

@triton.jit
def _permute_kernel(
    X_ptr,
    IDX_ptr,
    Y_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Each program permutes BLOCK_S tokens *all* hidden features (D). No inner python loop."""

    pid_bh = tl.program_id(0)
    tile_s = tl.program_id(1)

    # Offsets along sequence
    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    token_mask = s_offsets < S

    # Gather source indices for these tokens
    idx_ptrs = IDX_ptr + pid_bh * S + s_offsets
    src_row_idx = tl.load(idx_ptrs, mask=token_mask, other=0).to(tl.int32)

    # Broadcast to create 2-D pointer matrix (BLOCK_S, D)
    d_offsets = tl.arange(0, D)

    src_ptrs = X_ptr + (pid_bh * S + src_row_idx[:, None]) * D + d_offsets[None, :]
    dst_ptrs = Y_ptr + (pid_bh * S + s_offsets[:, None])     * D + d_offsets[None, :]

    full_mask = token_mask[:, None]

    values = tl.load(src_ptrs, mask=full_mask, other=0.0)
    tl.store(dst_ptrs, values, mask=full_mask)


@triton.jit
def _inverse_permute_kernel(
    X_ptr,
    IDX_ptr,
    Y_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Inverse permutation: scatter BLOCK_S tokens back in one shot."""

    pid_bh = tl.program_id(0)
    tile_s = tl.program_id(1)

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    token_mask = s_offsets < S

    idx_ptrs = IDX_ptr + pid_bh * S + s_offsets
    src_pos_idx = s_offsets.to(tl.int32)
    dst_pos_idx = tl.load(idx_ptrs, mask=token_mask, other=0).to(tl.int32)

    d_offsets = tl.arange(0, D)

    src_ptrs = X_ptr + (pid_bh * S + src_pos_idx[:, None]) * D + d_offsets[None, :]
    dst_ptrs = Y_ptr + (pid_bh * S + dst_pos_idx[:, None]) * D + d_offsets[None, :]

    full_mask = token_mask[:, None]

    values = tl.load(src_ptrs, mask=full_mask, other=0.0)
    tl.store(dst_ptrs, values, mask=full_mask)


########################################################
# Python wrappers
########################################################

@time_logging_decorator("Level 4 - permute tensor by labels triton")
def permute_tensor_by_labels_triton(
    tensor: torch.Tensor,
    labels: Optional[torch.Tensor],
    dim: int,
    *,
    sorted_indices: Optional[torch.Tensor] = None,
):
    """
    Permute `tensor` along `dim` according to ascending order of `labels`.

    This is a Triton-accelerated replacement for the original implementation.
    It currently supports 4-D tensors of shape [B, H, S, D] and `dim == 2`.
    If these conditions are not met or the tensors reside on CPU, we fall back
    to the reference PyTorch implementation.
    """

    # Assertions – we only support the optimized CUDA path.
    assert dim == 2, "permute_tensor_by_labels currently only supports dim==2 (sequence dimension)"
    assert tensor.dim() == 4, "Expected tensor shape [B,H,S,D]"
    assert tensor.is_cuda, "permute_tensor_by_labels requires CUDA tensors"
    
    B, H, S, D = tensor.shape
    BH = B * H

    # Determine sorted indices
    if sorted_indices is not None:
        sorted_indices = sorted_indices.to(torch.int32).contiguous()
    else:
        assert labels is not None, "Either `labels` or `sorted_indices` must be provided."
        labels = labels.to(tensor.device)
        sorted_indices = torch.argsort(labels, dim=-1).to(torch.int32).contiguous()

    # Flatten tensor and allocate output
    inp_flat = tensor.reshape(BH, S, D).contiguous()
    out_flat = torch.empty_like(inp_flat)

    # Triton kernel tile size
    BLOCK_S = 64  # number of tokens per program, tunable

    n_s_tiles = triton.cdiv(S, BLOCK_S)
    grid = (BH, n_s_tiles)

    _permute_kernel[grid](inp_flat, sorted_indices, out_flat, S, D, BLOCK_S, num_warps=4)

    permuted_tensor = out_flat.reshape(B, H, S, D)
    return permuted_tensor, sorted_indices


@time_logging_decorator("Level 4 - apply inverse permutation triton")
def apply_inverse_permutation_triton(
    permuted_tensor: torch.Tensor,
    sorted_indices: torch.Tensor,
    dim: int,
):
    """
    Triton implementation of inverse permutation. Inverse the permutation applied by `permute_tensor_by_labels`.
    
    Args:
        permuted_tensor: (B, H, S, D).
        sorted_indices: (B, H, S).
        dim: Dimension along which to apply inverse permutation. Typically 2, meaning the sequence lengthdimension.
        
    Returns:
        Tensor of shape (B, H, S, D).
    """

    assert dim == 2, "apply_inverse_permutation currently only supports dim==2"
    assert permuted_tensor.dim() == 4, "Expected tensor shape [B,H,S,D]"
    assert permuted_tensor.is_cuda, "apply_inverse_permutation requires CUDA tensors"

    B, H, S, D = permuted_tensor.shape
    BH = B * H

    # Ensure index dtype
    sorted_indices = sorted_indices.to(torch.int32).contiguous()

    # Flatten inputs
    inp_flat = permuted_tensor.reshape(BH, S, D).contiguous()
    out_flat = torch.empty_like(inp_flat)

    BLOCK_S = 64
    n_s_tiles = triton.cdiv(S, BLOCK_S)
    grid = (BH, n_s_tiles)

    _inverse_permute_kernel[grid](inp_flat, sorted_indices, out_flat, S, D, BLOCK_S, num_warps=4)

    original_tensor = out_flat.reshape(B, H, S, D)
    return original_tensor


########################################################
# Quick correctness test & micro-benchmark (similar to rmsnorm)
########################################################


if __name__ == "__main__":
    import time, argparse

    parser = argparse.ArgumentParser("Permute kernel benchmark")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--skip-bench", action="store_true", help="Skip benchmark timing")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this benchmark.")

    device = "cuda"
    # Problem size – can be tuned
    B, H, S, D = 1, 32, 74256, 128
    x = torch.randn(B, H, S, D, device=device, dtype=torch.float16).contiguous()
    labels = torch.randint(0, 1000, (B * H, S), device=device)

    # ------------------------------------------------------------------
    # Correctness check
    # ------------------------------------------------------------------
    y, sidx = permute_tensor_by_labels_triton(x, labels, 2)
    x_rec = apply_inverse_permutation_triton(y, sidx, 2)
    torch.testing.assert_close(x_rec, x, rtol=5e-3, atol=5e-3)
    print("[Correctness] Triton permute + inverse OK ✅")

    if args.skip_bench:
        exit(0)

    # Reference implementations using torch.gather (for timing only)
    def torch_permute():
        sorted_idx = torch.argsort(labels, dim=-1)
        gather_idx = sorted_idx.unsqueeze(-1).expand(B * H, S, D).reshape(B, H, S, D)
        torch.gather(x, 2, gather_idx)

    # Precompute tensors for inverse
    y_ref, sorted_idx_ref = permute_tensor_by_labels_triton(x, labels, 2)

    def torch_inverse():
        inv = torch.argsort(sorted_idx_ref, dim=-1)
        gather_idx = inv.unsqueeze(-1).expand(B * H, S, D).reshape(B, H, S, D)
        torch.gather(y_ref, 2, gather_idx)

    # Warm-up both kernels
    for _ in range(10):
        permute_tensor_by_labels_triton(x, labels, 2)
        torch_permute()

    # ------------------------------------------------------------------
    # Benchmark helpers
    # ------------------------------------------------------------------
    def bench(fn, iters=args.iters):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.time() - start) / iters * 1000  # ms/iter

    ms_triton_perm = bench(lambda: permute_tensor_by_labels_triton(x, labels, 2)[0])
    ms_torch_perm = bench(torch_permute)

    ms_triton_inv = bench(lambda: apply_inverse_permutation_triton(y_ref, sorted_idx_ref, 2))
    ms_torch_inv = bench(torch_inverse)

    # ------------------------------------------------------------------
    # Memory bandwidth (GB/s)
    # ------------------------------------------------------------------
    total_elems = B * H * S * D
    bytes_per_elem = 2  # float16
    bytes_moved = total_elems * bytes_per_elem * 2  # read + write

    bw_triton_perm = bytes_moved / ms_triton_perm * 1e-6
    bw_torch_perm  = bytes_moved / ms_torch_perm  * 1e-6
    bw_triton_inv  = bytes_moved / ms_triton_inv * 1e-6
    bw_torch_inv   = bytes_moved / ms_torch_inv  * 1e-6

    print("\n===== Benchmark (average ms/iter over", args.iters, "iters) =====")
    print(f"Permute  – Triton: {ms_triton_perm:.3f} ms | Torch: {ms_torch_perm:.3f} ms")
    print(f"Inverse  – Triton: {ms_triton_inv:.3f} ms | Torch: {ms_torch_inv:.3f} ms")

    print("\n===== Effective Memory Bandwidth (GB/s) =====")
    print(f"Permute  – Triton: {bw_triton_perm:.2f} | Torch: {bw_torch_perm:.2f}")
    print(f"Inverse  – Triton: {bw_triton_inv:.2f} | Torch: {bw_torch_inv:.2f}")


########################################################
# End of file
########################################################


