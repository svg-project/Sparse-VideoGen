import sys

sys.path.append("build/")

from itertools import product
from typing import List, Tuple

import _kernels
import torch


def bench_apply_rope_complex(
    param,
    rope_func,
    iter_warmup: int = 10,
    iter_total: int = 100,
) -> List:
    time_list = []
    for bsz, num_heads, stride_seq_len, head_dim, len_text_prompt in param:
        valid_seq_len = stride_seq_len - len_text_prompt

        q = torch.randn(bsz, num_heads, stride_seq_len, head_dim, dtype=torch.float16).cuda()
        k = torch.randn(bsz, num_heads, stride_seq_len, head_dim, dtype=torch.float16).cuda()
        freqs = torch.complex(torch.randn(valid_seq_len, head_dim // 2, dtype=torch.float32).cuda(), torch.randn(valid_seq_len, head_dim // 2, dtype=torch.float32).cuda())

        for _ in range(iter_warmup):
            rope_func(q, k, freqs, len_text_prompt)

        local_time_list = []
        for _ in range(iter_total):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            rope_func(q, k, freqs, len_text_prompt)
            end.record()
            torch.cuda.synchronize()
            local_time_list.append(start.elapsed_time(end))
        # avg time
        time_list.append(sum(local_time_list) / len(local_time_list))
    return time_list


def ref_torch_impl(q, k, freqs, len_text_prompt) -> torch.Tensor:

    def _ref_diffuser_impl(
        hidden_states,
        freqs,
    ) -> torch.Tensor:
        # Skip text prompt part
        text_part = hidden_states[:, :, :len_text_prompt, :]
        image_part = hidden_states[:, :, len_text_prompt:, :]

        # Apply rotary embedding to image part
        x_rotated = torch.view_as_complex(image_part.to(torch.float64).unflatten(3, (-1, 2)))
        x_out = torch.view_as_real(x_rotated * freqs.unsqueeze(0).unsqueeze(0)).flatten(3, 4)
        out = x_out.type_as(hidden_states)

        return torch.cat([text_part, out], dim=2)

    # Concatenate text and image parts
    return _ref_diffuser_impl(q, freqs), _ref_diffuser_impl(k, freqs)


def ref_customized_impl(q, k, freqs, len_text_prompt) -> torch.Tensor:
    freqs_real = freqs.real.contiguous()
    freqs_imag = freqs.imag.contiguous()

    # import IPython; IPython.embed()
    _kernels.apply_qk_rope_inplace_cossin_complex(q, k, freqs_real, freqs_imag, len_text_prompt)
    return q, k


parameters = list(product([1, 4], [32], [1024, 4096, 16384], [64], [117]))
torch_time = bench_apply_rope_complex(parameters, ref_torch_impl)
narrow_time = bench_apply_rope_complex(parameters, ref_customized_impl)

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

    torch_bandwidth.append(num_heads * bsz * head_dim * (seq_len - len_text_prompt) * 2 / latency_torch * 1e-6)
    narrow_bandwidth.append(num_heads * bsz * head_dim * (seq_len - len_text_prompt) * 2 / latency_narrow * 1e-6)

print(f"Test args in [bsz,num_heads,seq_len,head_dim,len_text_prompt]: {parameters}")
print(f"Complex RoPE Bandwidth: {torch_bandwidth}")
print(f"Narrow RoPE Bandwidth: {narrow_bandwidth}")
