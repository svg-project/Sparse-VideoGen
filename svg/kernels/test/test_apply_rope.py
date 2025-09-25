import sys

sys.path.append("build/")

from itertools import product

import _kernels
import pytest
import torch


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def different_proportion(a, b):
    return torch.sum(a != b).item() / a.numel()


def ref_host_apply_rope(
    x,
    cos,
    sin,
) -> torch.Tensor:
    # Ref: /diffusers/models/embeddings.py#L1171
    assert x.dim() == 4
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    return out


parameters = list(product([1, 3, 5], [16, 32], [151, 1037, 6778], [64, 128, 256], [15, 35, 77]))


@pytest.mark.parametrize("bsz, num_heads, total_seq_len, head_dim, len_text_prompt", parameters)
@torch.inference_mode()
def test_apply_rope(bsz, num_heads, total_seq_len, head_dim, len_text_prompt):
    if len_text_prompt >= total_seq_len:
        pytest.skip("len_text_prompt >= total_seq_len")
    valid_seq_len = total_seq_len - len_text_prompt
    q = torch.randn(bsz, num_heads, total_seq_len, head_dim, dtype=torch.bfloat16).cuda()
    k = torch.randn(bsz, num_heads, total_seq_len, head_dim, dtype=torch.bfloat16).cuda()
    cos = torch.randn(valid_seq_len, head_dim, dtype=torch.float32).cuda()
    sin = torch.randn(valid_seq_len, head_dim, dtype=torch.float32).cuda()

    q_image_host = ref_host_apply_rope(q[:, :, len_text_prompt:, :], cos, sin)
    k_image_host = ref_host_apply_rope(k[:, :, len_text_prompt:, :], cos, sin)
    q_text_host = q[:, :, :len_text_prompt, :].clone()
    k_text_host = k[:, :, :len_text_prompt, :].clone()

    _kernels.apply_qk_rope_inplace_cossin(q, k, cos, sin, len_text_prompt)

    assert_close(q[:, :, len_text_prompt:, :], q_image_host)
    assert_close(k[:, :, len_text_prompt:, :], k_image_host)
    assert_close(q[:, :, :len_text_prompt, :], q_text_host)
    assert_close(k[:, :, :len_text_prompt, :], k_text_host)

    print(f"{different_proportion(q[:,:,len_text_prompt:,:], q_image_host) * 100:.4f}% elements are different for q_image_host")
    print(f"{different_proportion(k[:,:,len_text_prompt:,:], k_image_host) * 100:.4f}% elements are different for k_image_host")
    print(f"{different_proportion(q[:,:,:len_text_prompt,:], q_text_host) * 100:.4f}% elements are different for q_text_host")
    print(f"{different_proportion(k[:,:,:len_text_prompt,:], k_text_host) * 100:.4f}% elements are different for k_text_host")
