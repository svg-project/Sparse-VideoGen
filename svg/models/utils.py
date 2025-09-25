import torch

def pseudo_quantize_absmax_perhead(
    x: torch.Tensor,
    num_bits: int = 8,
    hadamard: torch.Tensor = None
) -> torch.Tensor:
    """
    1) Subtract mean along S dimension to make data smoother.
    2) Reshape D dimension into groups of size `group_size`, and use absolute max (symmetric range) to compute scale.
    (We do NOT store group_min; only store group_max for each group.)
    3) Quantize and dequantize each group.
    4) Add back the mean.

    Symmetric quantization formula:
        group_max = max( abs(x_reshaped) ) per group
        scale = group_max / (levels/2)
        x_quant = round( clamp( x_reshaped / scale, -levels/2, +levels/2 ) )
        x_dequant = x_quant * scale

    Args:
        x (torch.Tensor): Input tensor of shape [B, H, S, D].
        group_size (int): Group size for the D dimension.
        num_bits (int): Number of quantization bits (default=8).

    Returns:
        torch.Tensor: The dequantized tensor of the same shape as input.
    """
    B, H, S, D = x.shape

    x_smooth = x                 # [B, H, S, D]
    
    # HXI: Hadamard
    if hadamard is not None:
        x_smooth = torch.matmul(x_smooth, hadamard)

    # 2) Reshape to [B, H, S, D//group_size, group_size]
    x_reshaped = x_smooth.view(B, H, 1, S * D)

    # 2a) Compute absolute max within each group
    group_absmax = x_reshaped.abs().max(dim=-1, keepdim=True).values  # [B, H, S, D//group_size, 1]

    # 2b) Compute scale (symmetric range)
    levels = (1 << num_bits) - 1  # e.g. 255 for 8 bits
    # Usually for signed values in symmetric quantization:
    #   scale = group_absmax / (levels/2)
    # so that we can represent from -group_absmax ~ +group_absmax
    half_levels = levels / 2.0  # e.g. 255 / 2 = 127.5 (approx)
    epsilon = 1e-8
    group_absmax = torch.clamp(group_absmax, min=epsilon)
    scale = group_absmax / half_levels  # [B, H, S, D//group_size, 1]

    # 3) Quantize: x_reshaped / scale -> round -> clamp
    # Range is approx [-127.5, +127.5] for 8-bit
    x_quant = torch.round(x_reshaped / scale)
    x_quant = torch.clamp(x_quant, min=-half_levels, max=+half_levels)

    # 3a) Dequantize
    x_dequant = x_quant * scale  # [B, H, S, D//group_size, group_size]

    # 3b) Reshape back
    x_dequant = x_dequant.view(B, H, S, D)
    x_rec = x_dequant  # [B, H, S, D]
    return x_rec


def visualize_sparse_bsr(row_ptr, col_idx, block_size, grid_size=(20, 20)):
    """
    可视化 Block Sparse Row (BSR) 稀疏矩阵结构（仅结构，不含数值）。
    
    参数:
        row_ptr (Tensor): 长度为 (num_block_rows + 1), int32/int64
        col_idx (Tensor): 长度为 (num_blocks,), 每个是 block 的列索引
        block_size (tuple): (row_block_size, col_block_size)
        grid_size (tuple): (max_rows, max_cols) 压缩显示的字符图尺寸
    返回:
        str: 可打印的字符图，表示稀疏结构
    """
    if not isinstance(row_ptr, torch.Tensor):
        row_ptr = torch.tensor(row_ptr)
    if not isinstance(col_idx, torch.Tensor):
        col_idx = torch.tensor(col_idx)

    row_block_size, col_block_size = block_size
    num_block_rows = row_ptr.shape[0] - 1
    num_blocks = col_idx.shape[0]

    # Estimate the total size of the dense matrix
    num_rows = num_block_rows * row_block_size
    max_block_col = col_idx.max().item() if col_idx.numel() > 0 else 0
    num_cols = (max_block_col + 1) * col_block_size

    # Construct dense structure mask
    dense = torch.zeros((num_rows, num_cols), dtype=torch.float32)
    for block_row in range(num_block_rows):
        start = row_ptr[block_row].item()
        end = row_ptr[block_row + 1].item()
        for i in range(start, end):
            block_col = col_idx[i].item()
            r_start = block_row * row_block_size
            r_end = r_start + row_block_size
            c_start = block_col * col_block_size
            c_end = c_start + col_block_size
            dense[r_start:r_end, c_start:c_end] = 1.0

    # Compress to character map
    dense = dense.numpy()
    total_rows, total_cols = dense.shape
    max_rows, max_cols = grid_size

    def cdiv(a, b):
        return (a + (b - 1)) // b

    row_step = max(1, cdiv(total_rows, max_rows))
    col_step = max(1, cdiv(total_cols, max_cols))

    def summarize_block(patch):
        mean = patch.mean()
        if mean == 1:
            return "█" * 2
        elif mean == 0:
            return "  "
        else:
            return "░" * 2

    vis = ""
    for r in range(0, total_rows, row_step):
        for c in range(0, total_cols, col_step):
            patch = dense[r:r+row_step, c:c+col_step]
            vis += summarize_block(patch)
        vis += "\n"

    return vis