import sys
sys.path.append('/ssd/data/xihaocheng/Hunyuan/I2VSparse/kernels/build/')

import _kernels

import torch
from itertools import product
from typing import List, Tuple


def bench_apply_rope(
    param,
    ln_func,
    iter_warmup: int = 10,
    iter_total: int = 100,
) -> List:
    time_list = []
    for bsz, num_heads, stride_seq_len, head_dim, len_text_prompt in param:
        valid_seq_len = stride_seq_len - len_text_prompt
        
        q = torch.randn(bsz, num_heads, stride_seq_len, head_dim, dtype=torch.float16).cuda()
        k = torch.randn(bsz, num_heads, stride_seq_len, head_dim, dtype=torch.float16).cuda()
        cos = torch.randn(valid_seq_len, head_dim, dtype=torch.float32).cuda()
        sin = torch.randn(valid_seq_len, head_dim, dtype=torch.float32).cuda()
        
        for _ in range(iter_warmup):
            ln_func(q, k, cos, sin, len_text_prompt)
            
        local_time_list = []
        for _ in range(iter_total):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            ln_func(q, k, cos, sin, len_text_prompt)
            end.record()
            torch.cuda.synchronize()
            local_time_list.append(start.elapsed_time(end))
        # avg time
        time_list.append(sum(local_time_list) / len(local_time_list))
    return time_list

def ref_torch_impl(
    q,
    k,
    cos,
    sin,
    len_text_prompt
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    def _ref_diffuser_impl(
        x,
        cos,
        sin
    ) -> torch.Tensor:
        assert x.dim() == 4
        x = x[:,:,len_text_prompt:,:]
        y = x[:,:,:len_text_prompt,:]
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        out = torch.cat([y, out], dim=2)
        return out
    
    return _ref_diffuser_impl(q, cos, sin), _ref_diffuser_impl(k, cos, sin)

def ref_customized_impl(
    q,
    k,
    cos,
    sin,
    len_text_prompt
) -> Tuple[torch.Tensor, torch.Tensor]:
    _kernels.apply_qk_rope_inplace_cossin(q, k, cos, sin, len_text_prompt)
    return q, k

parameters = list(product([1,4], [32], [1024,4096,16384], [64], [117]))
torch_time = bench_apply_rope(parameters, ref_torch_impl)
narrow_time = bench_apply_rope(parameters, ref_customized_impl)

assert len(torch_time) == len(parameters)
assert len(narrow_time) == len(parameters)

torch_bandwidth = []
narrow_bandwidth = []
for i in range(len(parameters)):
    bsz = parameters[i][0]
    num_heads = parameters[i][1]
    seq_len = parameters[i][2]
    head_dim = parameters[i][3]
    len_text_prompt = parameters[i][4]
    
    latency_torch = torch_time[i]
    latency_narrow = narrow_time[i]
    
    torch_bandwidth.append(num_heads * bsz * head_dim * (seq_len-len_text_prompt) * 2 / latency_torch * 1e-6)
    narrow_bandwidth.append(num_heads * bsz * head_dim * (seq_len-len_text_prompt) * 2 / latency_narrow * 1e-6)

print(f"Test args in [bsz,num_heads,seq_len,head_dim]: {parameters}")
print(f"Diffusers's Bandwidth: {torch_bandwidth}")
print(f"Customized's Bandwidth: {narrow_bandwidth}")