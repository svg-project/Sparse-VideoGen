import torch
import triton
import triton.language as tl

from .utils import flatten_if_batched


########################################################
# Elementwise_affine=True
########################################################

@triton.jit
def _layer_norm_param_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    x_stride,  # how much to increase the pointer when moving by 1 row
    y_stride,  # how much to increase the pointer when moving by 1 row
    N: tl.constexpr,  # number of columns in X,
    N2: tl.constexpr,  # number of columns in X,
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

    # Compute mean and Variance
    mean = tl.sum(x, axis=1, keep_dims=True) / N
    # Compute variance
    _var = (x - mean) * (x - mean)
    var = tl.sum(_var, axis=1, keep_dims=True) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Write mean / rstd
    _mean = tl.reshape(mean, (BLOCK_M))
    _rstd = tl.reshape(rstd, (BLOCK_M))
    tl.store(Mean + rows, _mean)
    tl.store(Rstd + rows, _rstd)

    # Normalize and apply linear transformation
    x_hat = (x - mean) * rstd

    w = tl.load(W + cols)
    b = tl.load(B + cols)
    
    x_hat = x_hat * w + b

    # Write output
    x_hat = x_hat.to(Y.type.element_ty)
    tl.store(y_ptr, x_hat, mask=mask[None, :])


def triton_layernorm_param_forward(x, w, b, eps):
    # Change batched 3D input to 2D
    [x], batched, BS = flatten_if_batched(x)

    # allocate output
    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float32)
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # heuristics for number of warps
    num_warps = 8

    N2 = triton.next_power_of_2(N)
    
    if N <= 512:
        BLOCK_M = 32
    else:
        BLOCK_M = 1

    # enqueue kernel
    _layer_norm_param_fwd_fused[(triton.cdiv(M, BLOCK_M),)](  #
        x,
        y,
        w,
        b,
        mean,
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


########################################################
# Elementwise_affine=False
########################################################


@triton.jit
def _layer_norm_noparam_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    x_stride,  # how much to increase the pointer when moving by 1 row
    y_stride,  # how much to increase the pointer when moving by 1 row
    N: tl.constexpr,  # number of columns in X,
    N2: tl.constexpr,  # number of columns in X,
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

    # Compute mean and Variance
    mean = tl.sum(x, axis=1, keep_dims=True) / N
    # Compute variance
    _var = (x - mean) * (x - mean)
    var = tl.sum(_var, axis=1, keep_dims=True) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Write mean / rstd
    _mean = tl.reshape(mean, (BLOCK_M))
    _rstd = tl.reshape(rstd, (BLOCK_M))
    tl.store(Mean + rows, _mean)
    tl.store(Rstd + rows, _rstd)

    # Normalize and apply linear transformation
    x_hat = (x - mean) * rstd

    # Write output
    x_hat = x_hat.to(Y.type.element_ty)
    tl.store(y_ptr, x_hat, mask=mask[None, :])


def triton_layernorm_noparam_forward(x, eps):
    assert x.is_contiguous(), "Input must be contiguous"

    # Change batched 3D input to 2D
    [x], batched, BS = flatten_if_batched(x)

    # allocate output
    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float32)
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # heuristics for number of warps
    num_warps = 8

    N2 = triton.next_power_of_2(N)
    
    if N <= 512:
        BLOCK_M = 32
    else:
        BLOCK_M = 1

    # enqueue kernel
    _layer_norm_noparam_fwd_fused[(triton.cdiv(M, BLOCK_M),)](  #
        x,
        y,
        mean,
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


########################################################
# Router
########################################################

def triton_layernorm_forward(x, w, b, eps, elementwise_affine=True):
    if elementwise_affine:
        assert w is not None and b is not None
        return triton_layernorm_param_forward(x, w, b, eps)
    else:
        assert w is None and b is None
        return triton_layernorm_noparam_forward(x, eps)


if __name__ == "__main__":
    """Quick correctness check and micro-benchmark vs. torch.nn.functional.layer_norm.

    Run this file directly to get a rough idea of speed-ups.  Designed to finish in a
    few seconds on a single GPU.  Adjust B, N, D as needed.
    """
    import time
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Problem sizes (batch, seq_len, hidden_dim)
    B, N, D = 1, 74256, 40 * 128
    eps = 1e-5

    x = torch.randn(B, N, D, device=device, dtype=torch.float32).contiguous()
    w = torch.randn(D, device=device, dtype=torch.float32).contiguous()
    b = torch.randn(D, device=device, dtype=torch.float32).contiguous()

    # ------------------------------------------------------------------
    # Correctness check
    # ------------------------------------------------------------------
    y_triton = triton_layernorm_forward(x, w, b, eps)
    y_torch = F.layer_norm(x, (D,), w, b, eps)
    max_diff = (y_triton - y_torch).abs().max().item()
    print(f"Max |triton - torch|: {max_diff:e}")

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------
    for _ in range(10):
        triton_layernorm_forward(x, w, b, eps)
        F.layer_norm(x, (D,), w, b, eps)

    # ------------------------------------------------------------------
    # Benchmark helpers
    # ------------------------------------------------------------------
    def bench(fn, iters: int = 50):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                fn()
        torch.cuda.synchronize()
        return (time.time() - start) / iters * 1000  # ms / iter

    triton_ms = bench(lambda: triton_layernorm_forward(x, w, b, eps))
    torch_ms = bench(lambda: F.layer_norm(x, (D,), w, b, eps))

    print(f"Triton LayerNorm: {triton_ms:.3f} ms/iter")
    print(f"PyTorch LayerNorm: {torch_ms:.3f} ms/iter")