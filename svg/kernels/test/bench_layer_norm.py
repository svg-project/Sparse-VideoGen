import sys

sys.path.append("build/")

from itertools import product
from typing import List

import _kernels
import torch


def bench_layer_norm(
    param,
    ln_func,
    iter_warmup: int = 10,
    iter_total: int = 100,
) -> List:
    time_list = []
    for batch_size, head_dim in param:
        input = torch.randn(batch_size, head_dim, dtype=torch.float16).cuda()
        gemma = torch.randn(head_dim, dtype=torch.float16).cuda()
        beta = torch.randn(head_dim, dtype=torch.float16).cuda()

        for _ in range(iter_warmup):
            ln_func(input, gemma, beta)

        local_time_list = []
        for _ in range(iter_total):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            ln_func(input, gemma, beta)
            end.record()
            torch.cuda.synchronize()
            local_time_list.append(start.elapsed_time(end))
        # avg time
        time_list.append(sum(local_time_list) / len(local_time_list))
    return time_list


def ref_torch_impl(
    input,
    gemma,
    beta,
) -> torch.Tensor:
    return torch.nn.functional.layer_norm(input, [input.size(-1)], gemma, beta, 1e-5)


def ref_narrow_impl(input, gemma, beta) -> torch.Tensor:
    _kernels.layer_norm_forward(input, gemma, beta)
    return input


num_total_loading = 4096 * 16 * 1024
# parameters = list(product([65536], [64,128,256,512]))
parameters = [(num_total_loading // x, x) for x in [32, 64, 128, 256]]
torch_time = bench_layer_norm(parameters, ref_torch_impl)
narrow_time = bench_layer_norm(parameters, ref_narrow_impl)

assert len(torch_time) == len(parameters)
assert len(narrow_time) == len(parameters)

torch_bandwidth = []
narrow_bandwidth = []
for i in range(len(parameters)):
    bsz = parameters[i][0]
    head_dim = parameters[i][1]
    latency_torch = torch_time[i]
    latency_narrow = narrow_time[i]

    torch_bandwidth.append(bsz * head_dim * 2 / latency_torch * 1e-6)
    narrow_bandwidth.append(bsz * head_dim * 2 / latency_narrow * 1e-6)

print(f"Test args in [bsz,head_dim]: {parameters}")
print(f"Diffusers's Bandwidth: {torch_bandwidth}")
print(f"Customized's Bandwidth: {narrow_bandwidth}")
