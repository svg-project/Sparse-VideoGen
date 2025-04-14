import sys
sys.path.append('build/')

import _kernels

import torch
import pytest
from itertools import product

torch.manual_seed(0)

def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    
def different_proportion(a, b):
    return torch.sum(a != b).item() / a.numel()
    
def ref_host_rms_norm(
    input,
    gemma
) -> torch.Tensor:
    return torch.nn.functional.rms_norm(input, [input.size(-1)], gemma, 1e-5)

def replica_host_rms_norm(
    input,
    gemma
) -> torch.Tensor:
    input_dtype = input.dtype
    input = input.to(torch.float32)
    variance = input.pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + 1e-5)
    return gemma * input.to(input_dtype)

parameters = list(product([1,7,31,55,95,128,512], [32, 64,128,256]))
@pytest.mark.parametrize("batch_size, head_dim", parameters)
@torch.inference_mode()
def test_rms_norm(batch_size, head_dim):    
    input = torch.randn(batch_size, head_dim, dtype=torch.bfloat16).cuda()
    gemma = torch.randn(head_dim, dtype=torch.bfloat16).cuda()

    output_host = ref_host_rms_norm(input, gemma)
    output_replica = replica_host_rms_norm(input, gemma)
    _kernels.rms_norm_forward(input, gemma, 1e-5)
    
    output_host = output_host.to(input)
    output_replica = output_replica.to(input)

    assert_close(input, output_host)
    assert_close(output_replica, output_host)
    print(f"{different_proportion(input, output_host) * 100:.4f}% elements are different")
    print(f"{different_proportion(output_replica, output_host) * 100:.4f}% elements are different")
