import torch

def flatten_if_batched(*tensors):
    """
    Flattens all input tensors from (B, N, D_i) to (B * N, D_i) if they are batched (3D).

    Args:
        *tensors: Any number of input tensors, each must have shape (B, N, D_i) or (N, D_i)

    Returns:
        flat_tensors: List of flattened tensors
        batched: Boolean flag indicating whether inputs were batched
        batch_size: Batch size if batched, else None
    """
    if not tensors:
        raise ValueError("At least one tensor must be provided.")

    first = tensors[0]
    assert len(first.shape) in [
        2,
        3,
    ], "Input tensors must be batched (3D) or not batched (2D)"

    if len(first.shape) == 3:  # batched
        batched = True
        batch_size = first.shape[0]
        assert all(t.shape[0] == batch_size for t in tensors), "All input tensors must have the same batch size"
        assert all(
            t.shape[1] == first.shape[1] for t in tensors
        ), "All input tensors must have the same sequence length"
        flat_tensors = [t.reshape(-1, t.shape[-1]) for t in tensors]
    else:
        batched = False
        batch_size = None
        flat_tensors = list(tensors)

    return flat_tensors, batched, batch_size