import torch
import triton
import triton.language as tl

from .utils import flatten_if_batched


################################################################################
# Modulate scale and shift
################################################################################

@triton.jit
def _modulate_shift_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    SCALE,  # pointer to the weights
    SHIFT,  # pointer to the biases
    x_stride,  # how much to increase the pointer when moving by 1 row
    y_stride,  # how much to increase the pointer when moving by 1 row
    N: tl.constexpr,  # number of columns in X,
    N2: tl.constexpr,  # number of columns in X,
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

    scale = tl.load(SCALE + cols)
    shift = tl.load(SHIFT + cols)
    
    y = x * (1 + scale) + shift

    # Write output
    y = y.to(Y.type.element_ty)
    tl.store(y_ptr, y, mask=mask[None, :])


def triton_modulate_shift_forward(x, scale, shift, output_dtype=torch.float32):
    """
    Modulate scale and shift. y = x * (1 + scale) + shift
    """
    # Change batched 3D input to 2D
    [x], batched, BS = flatten_if_batched(x)

    # allocate output
    M, N = x.shape
    y = torch.empty_like(x, dtype=output_dtype)
    # heuristics for number of warps
    num_warps = 8

    N2 = triton.next_power_of_2(N)
    
    if N <= 512:
        BLOCK_M = 32
    else:
        BLOCK_M = 1

    # enqueue kernel
    _modulate_shift_fwd_fused[(triton.cdiv(M, BLOCK_M),)](  #
        x,
        y,
        scale,
        shift,
        x.stride(0),
        y.stride(0),
        N,
        N2,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
    )

    # Recover 2D to 3D
    if batched:
        y = y.reshape(BS, -1, y.shape[-1])

    return y


################################################################################
# Modulate gate residual
################################################################################

@triton.jit
def _modulate_gate_residual_fwd_fused(
    R,  # pointer to the residual
    X,  # pointer to the input
    Y,  # pointer to the output
    GATE,  # pointer to the gate
    r_stride,  # how much to increase the pointer when moving by 1 row
    x_stride,  # how much to increase the pointer when moving by 1 row
    y_stride,  # how much to increase the pointer when moving by 1 row
    N: tl.constexpr,  # number of columns in X,
    N2: tl.constexpr,  # number of columns in X,
    BLOCK_M: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    mask = cols < N

    r_ptr = R + rows[:, None] * r_stride + cols[None, :]
    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]

    r = tl.load(r_ptr, mask=mask[None, :], other=0.0).to(tl.float32)
    x = tl.load(x_ptr, mask=mask[None, :], other=0.0).to(tl.float32)

    gate = tl.load(GATE + cols)
    
    y = r + x * gate

    # Write output
    y = y.to(Y.type.element_ty)
    tl.store(y_ptr, y, mask=mask[None, :])


def triton_modulate_gate_residual_forward(residual, x, gate, output_dtype=torch.float32):
    """
    Modulate gate residual. y = residual + x * gate
    """
    # Change batched 3D input to 2D
    [residual, x], batched, BS = flatten_if_batched(residual, x)

    # allocate output
    M, N = x.shape
    y = torch.empty_like(x, dtype=output_dtype)
    # heuristics for number of warps
    num_warps = 8

    N2 = triton.next_power_of_2(N)
    
    if N <= 512:
        BLOCK_M = 32
    else:
        BLOCK_M = 1

    # enqueue kernel
    _modulate_gate_residual_fwd_fused[(triton.cdiv(M, BLOCK_M),)](  #
        residual,
        x,
        y,
        gate,
        residual.stride(0),
        x.stride(0),
        y.stride(0),
        N,
        N2,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
    )

    # Recover 2D to 3D
    if batched:
        y = y.reshape(BS, -1, y.shape[-1])

    return y
