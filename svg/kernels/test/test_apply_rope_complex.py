import sys
sys.path.append('build/')

import _kernels

import torch
import pytest
from itertools import product
from typing import List, Tuple

def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
        torch.float32: (1e-3, 1e-3),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    
def different_proportion(a, b):
    return torch.sum(a != b).item() / a.numel()

def ref_host_apply_rope_complex(
    x,
    freqs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 4
    # Apply rotary embedding to image part
    x_rotated = torch.view_as_complex(x.to(torch.float64).unflatten(3, (-1, 2)))
    x_out = torch.view_as_real(x_rotated * freqs.unsqueeze(0).unsqueeze(0)).flatten(3, 4)
    
    out = x_out.type_as(x)
    return out

torch.manual_seed(0)

parameters = list(product([1,3,5], [16,32], [151,1037,6778], [64,128,256], [15,35,77]))
@pytest.mark.parametrize("bsz, num_heads, total_seq_len, head_dim, len_text_prompt", parameters)
@torch.inference_mode()
def test_apply_rope_complex(bsz, num_heads, total_seq_len, head_dim, len_text_prompt):
    if len_text_prompt >= total_seq_len:
        pytest.skip("len_text_prompt >= total_seq_len")
    valid_seq_len = total_seq_len - len_text_prompt
    
    q = torch.randn(bsz, num_heads, total_seq_len, head_dim, dtype=torch.float16).cuda()
    k = torch.randn(bsz, num_heads, total_seq_len, head_dim, dtype=torch.float16).cuda()
    freqs = torch.complex(torch.randn(valid_seq_len, head_dim // 2, dtype=torch.float32).cuda(),
                           torch.randn(valid_seq_len, head_dim // 2, dtype=torch.float32).cuda())
    
    q_image_host = ref_host_apply_rope_complex(q[:,:,len_text_prompt:,:], freqs)
    k_image_host = ref_host_apply_rope_complex(k[:,:,len_text_prompt:,:], freqs)
    q_text_host = q[:,:,:len_text_prompt,:].clone()
    k_text_host = k[:,:,:len_text_prompt,:].clone()

    freqs_real = freqs.real.contiguous()
    freqs_imag = freqs.imag.contiguous()
    _kernels.apply_qk_rope_inplace_cossin_complex(q, k, freqs_real, freqs_imag, len_text_prompt)

    assert_close(q[:,:,len_text_prompt:,:], q_image_host)
    assert_close(k[:,:,len_text_prompt:,:], k_image_host)
    assert_close(q[:,:,:len_text_prompt,:], q_text_host)
    assert_close(k[:,:,:len_text_prompt,:], k_text_host)

    # print(f"{different_proportion(q[:,:,len_text_prompt:,:], q_image_host) * 100:.4f}% elements are different for q_image_host")
    # print(f"{different_proportion(k[:,:,len_text_prompt:,:], k_image_host) * 100:.4f}% elements are different for k_image_host")
    # if len_text_prompt > 0:
    #     print(f"{different_proportion(q[:,:,:len_text_prompt,:], q_text_host) * 100:.4f}% elements are different for q_text_host")
    #     print(f"{different_proportion(k[:,:,:len_text_prompt,:], k_text_host) * 100:.4f}% elements are different for k_text_host")