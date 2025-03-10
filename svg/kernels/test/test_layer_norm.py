import sys
sys.path.append('/ssd/data/xihaocheng/Hunyuan/I2VSparse/kernels/build/')

import _kernels

import torch
import pytest
from itertools import product

def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
        
def different_proportion(a, b):
    return torch.sum(a != b).item() / a.numel()
    
def ref_host_layer_norm(
    input,
    gemma,
    beta,
) -> torch.Tensor:
    return torch.nn.functional.layer_norm(input, [input.size(-1)], gemma, beta, 1e-5)

parameters = list(product([1,7,31,55,95,128,512], [32, 64,128,256]))
@pytest.mark.parametrize("batch_size, head_dim", parameters)
@torch.inference_mode()
def test_layer_norm(batch_size, head_dim):
    input = torch.randn(batch_size, head_dim, dtype=torch.bfloat16).cuda()
    gemma = torch.randn(head_dim, dtype=torch.bfloat16).cuda()
    beta = torch.randn(head_dim, dtype=torch.bfloat16).cuda()
    
    output_host = ref_host_layer_norm(input, gemma, beta)
    _kernels.layer_norm_forward(input, gemma, beta)

    assert_close(input, output_host)
    print(f"{different_proportion(input, output_host) * 100:.4f}% elements are different")