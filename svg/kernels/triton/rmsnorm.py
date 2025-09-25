import torch
import triton
import triton.language as tl

from .utils import flatten_if_batched


@triton.jit
def _rms_norm_fwd_fused(
    X,
    Y,
    W,
    Rstd,
    x_stride,
    y_stride,
    N: tl.constexpr,  # number of columns in X,
    N2: tl.constexpr,
    eps,  # epsilon to avoid division by zero
    BLOCK_M: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    mask = cols < N

    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]

    x = tl.load(x_ptr, mask=mask[None, :], other=0.0).to(tl.float32)

    # Compute variance
    _var = x * x
    var = tl.sum(_var, axis=1) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Write mean / rstd
    tl.store(Rstd + rows, rstd)
    rstd = tl.reshape(rstd, (BLOCK_M, 1))

    # Normalize and apply linear transformation
    w = tl.load(W + cols)
    x_hat = x * rstd
    y = x_hat * w

    # Write output
    y = y.to(Y.type.element_ty)
    tl.store(y_ptr, y, mask=mask[None, :])


def triton_rmsnorm_forward(x, w, eps):
    """
    Forward pass of the RMSNorm.

    Args:
        x (torch.Tensor): Input tensor, High precision.
        w (torch.Tensor): RMSNorm weight tensor.
        eps (float): RMSNorm epsilon value.

    Returns:
        y (torch.Tensor): Output tensor, High precision.
        (w, rstd, num_warps) (tuple): RMSNorm weight tensor, rstd tensor, and number of warps.
    """
    assert x.is_contiguous(), "Input must be contiguous"
    # Change batched 3D input to 2D
    [x], batched, BS = flatten_if_batched(x)

    # allocate output
    M, N = x.shape
    y = torch.empty_like(x, dtype=x.dtype)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

    # heuristics for number of warps
    num_warps = 8
    
    # Avoid illegal memory access
    N2 = triton.next_power_of_2(N)
    
    if N <= 512:
        BLOCK_M = 32
    else:
        BLOCK_M = 1

    # Call the triton kernel
    _rms_norm_fwd_fused[(triton.cdiv(M, BLOCK_M),)](  #
        x,
        y,
        w,
        rstd,  #
        x.stride(0),
        y.stride(0),
        N,
        N2,
        eps,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
    )

    # Recover 2D to 3D
    if batched:
        y = y.reshape(BS, -1, y.shape[-1])

    return y

if __name__ == "__main__":
    """Quick correctness test & micro-benchmark vs. torch.nn.functional.rms_norm."""
    import time
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, D = 40, 74256, 128
    eps = 1e-6

    x = torch.randn(B, N, D, device=device, dtype=torch.float32).contiguous()
    w = torch.randn(D, device=device, dtype=torch.float32).contiguous()

    # ------------------------------------------------------------------
    # Correctness check
    # ------------------------------------------------------------------
    y_triton = triton_rmsnorm_forward(x, w, eps)
    y_torch = F.rms_norm(x, (D,), w, eps)
    max_diff = (y_triton - y_torch).abs().max().item()
    print(f"Max |triton - torch|: {max_diff:e}")

    # Warm-up both kernels
    for _ in range(10):
        triton_rmsnorm_forward(x, w, eps)
        F.rms_norm(x, (D,), w, eps)

    def bench(fn, iters: int = 50):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.time() - start) / iters * 1000  # ms / iter

    triton_ms = bench(lambda: triton_rmsnorm_forward(x, w, eps))
    torch_ms = bench(lambda: F.rms_norm(x, (D,), w, eps))

    print(f"Triton RMSNorm : {triton_ms:.3f} ms/iter")
    print(f"PyTorch RMSNorm: {torch_ms:.3f} ms/iter")