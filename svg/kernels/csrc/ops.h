#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/extension.h>

#include "norm/narrow_layer_norm.cuh"
#include "norm/narrow_rms_norm.cuh"
#include "pytorch_extension_utils.h"
#include "rope/rope_enc.cuh"
#include "rope/rope_enc_txtlast.cuh"
#include "rope/rope_enc_complex.cuh"
/*
	input: [m, n] Row-major; assume n is reduce dim
	output: [m, n] Row-major
	gemma: [n]
	beta: [n]
*/
void layer_norm_forward(torch::Tensor input,
						torch::Tensor gemma,
						torch::Tensor beta)
{
	CHECK_INPUT(input);
	CHECK_INPUT(gemma);
	CHECK_INPUT(beta);

	CHECK_SHAPE(beta, gemma);
	CHECK_EQ(input.dim(), 2);
	CHECK_EQ(beta.dim(), 1);
	CHECK_EQ(input.size(1), beta.size(0));

	const int head_dim = input.size(1);

	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(input.scalar_type(), c_type, [&]
												   {
		DEBUG_CUDA_CALL(
			narrow_layernorm_inplace<c_type>(cutlass::MatrixCoord(input.size(0), input.size(1)),
										static_cast<c_type*>(input.data_ptr()),
										static_cast<c_type*>(gemma.data_ptr()),
										static_cast<c_type*>(beta.data_ptr()),
										at::cuda::getCurrentCUDAStream()));
		return true; });
	TORCH_CHECK(success, "Customized call failed");
}

/*
	input: [m, n] Row-major; assume n is reduce dim
	output: [m, n] Row-major
	gemma: [n]
	beta: [n]
*/
void rms_norm_forward(torch::Tensor input,
					  torch::Tensor gemma,
					  float epsilon = 1e-5)
{
	CHECK_INPUT(input);
	CHECK_INPUT(gemma);

	CHECK_EQ(input.dim(), 2);
	CHECK_EQ(gemma.dim(), 1);
	CHECK_EQ(input.size(1), gemma.size(0));

	const int head_dim = input.size(1);

	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(input.scalar_type(), c_type, [&]
												   {
		DEBUG_CUDA_CALL(
			narrow_rmsnorm_inplace<c_type>(cutlass::MatrixCoord(input.size(0), input.size(1)),
								static_cast<c_type*>(input.data_ptr()),
								static_cast<c_type*>(gemma.data_ptr()),
								epsilon,
								at::cuda::getCurrentCUDAStream()));
		return true; });
	TORCH_CHECK(success, "Customized call failed");
}

/*
	q: [bsz, num_qo_heads, total_seq_len, head_dim]
	k: [bsz, num_ko_heads, total_seq_len, head_dim]
	cos_cache: [seq_len, head_dim]
	sin_cache: [seq_len, head_dim]
	len_text_prompt: int
	NOTE (Yilong): first len_text_prompt will be skipped during RoPE
*/
void apply_qk_rope_inplace_cossin(torch::Tensor q,
								  torch::Tensor k,
								  torch::Tensor cos_cache,
								  torch::Tensor sin_cache,
								  uint32_t len_text_prompt)
{
	CHECK_INPUT(q);
	CHECK_INPUT(k);
	CHECK_INPUT(cos_cache);
	CHECK_INPUT(sin_cache);

	CHECK_EQ(cos_cache.dtype(), torch::kFloat);
	CHECK_EQ(sin_cache.dtype(), torch::kFloat);

	CHECK_EQ(q.dim(), 4);
	CHECK_EQ(k.dim(), 4);
	CHECK_EQ(cos_cache.dim(), 2);
	CHECK_EQ(sin_cache.dim(), 2);

	const uint32_t bsz = q.size(0);
	const uint32_t num_qo_heads = q.size(1);
	const uint32_t stride_seq_len = q.size(2);
	const uint32_t head_dim = q.size(3);
	const uint32_t num_kv_heads = k.size(1);
	const uint32_t valid_seq_len = stride_seq_len - len_text_prompt;

	assert(valid_seq_len > 0);

	CHECK_EQ(q.size(0), k.size(0));
	CHECK_EQ(q.size(2), k.size(2));
	CHECK_EQ(q.size(3), k.size(3));
	CHECK_EQ(cos_cache.size(0), valid_seq_len);
	CHECK_EQ(cos_cache.size(1), head_dim);
	CHECK_EQ(sin_cache.size(0), valid_seq_len);
	CHECK_EQ(sin_cache.size(1), head_dim);

	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&]
												   {
		DEBUG_CUDA_CALL(ApplyQKRotaryCosSinCacheInPlace<c_type>(static_cast<c_type*>(q.data_ptr()),
																static_cast<c_type*>(k.data_ptr()),
																cos_cache.data_ptr<float>(),
																sin_cache.data_ptr<float>(),
																bsz,
																num_qo_heads,
																num_kv_heads,
																stride_seq_len,
																len_text_prompt,
																head_dim,
																at::cuda::getCurrentCUDAStream()));
		return true; });
	TORCH_CHECK(success, "RoPE in place apply kernel call failed");
}

/*
	q: [bsz, num_qo_heads, total_seq_len, head_dim]
	k: [bsz, num_ko_heads, total_seq_len, head_dim]
	cos_cache: [seq_len, head_dim]
	sin_cache: [seq_len, head_dim]
	len_text_prompt: int
	NOTE (Yilong): last len_text_prompt will be skipped during RoPE
*/
void apply_qk_rope_inplace_cossin_txtlast(torch::Tensor q,
										  torch::Tensor k,
										  torch::Tensor cos_cache,
										  torch::Tensor sin_cache,
										  uint32_t len_text_prompt)
{
	CHECK_INPUT(q);
	CHECK_INPUT(k);
	CHECK_INPUT(cos_cache);
	CHECK_INPUT(sin_cache);

	CHECK_EQ(cos_cache.dtype(), torch::kFloat);
	CHECK_EQ(sin_cache.dtype(), torch::kFloat);

	CHECK_EQ(q.dim(), 4);
	CHECK_EQ(k.dim(), 4);
	CHECK_EQ(cos_cache.dim(), 2);
	CHECK_EQ(sin_cache.dim(), 2);

	const uint32_t bsz = q.size(0);
	const uint32_t num_qo_heads = q.size(1);
	const uint32_t stride_seq_len = q.size(2);
	const uint32_t head_dim = q.size(3);
	const uint32_t num_kv_heads = k.size(1);
	const uint32_t valid_seq_len = stride_seq_len - len_text_prompt;

	assert(valid_seq_len > 0);

	CHECK_EQ(q.size(0), k.size(0));
	CHECK_EQ(q.size(2), k.size(2));
	CHECK_EQ(q.size(3), k.size(3));
	CHECK_EQ(cos_cache.size(0), valid_seq_len);
	CHECK_EQ(cos_cache.size(1), head_dim);
	CHECK_EQ(sin_cache.size(0), valid_seq_len);
	CHECK_EQ(sin_cache.size(1), head_dim);

	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&]
												   {
	DEBUG_CUDA_CALL(ApplyQKRotaryCosSinTXTLASTCacheInPlace<c_type>(static_cast<c_type*>(q.data_ptr()),
									static_cast<c_type*>(k.data_ptr()),
									cos_cache.data_ptr<float>(),
									sin_cache.data_ptr<float>(),
									bsz,
									num_qo_heads,
									num_kv_heads,
									stride_seq_len,
									len_text_prompt,
									head_dim,
									at::cuda::getCurrentCUDAStream()));
	return true; });
	TORCH_CHECK(success, "RoPE Text Last in place apply kernel call failed");
}

/*
	q: [bsz, num_qo_heads, total_seq_len, head_dim]
	k: [bsz, num_ko_heads, total_seq_len, head_dim]
	cos_cache: [seq_len, head_dim // 2]
	sin_cache: [seq_len, head_dim // 2]
	len_text_prompt: int
	NOTE (Yilong): first len_text_prompt will be skipped during RoPE
*/
void apply_qk_rope_inplace_cossin_complex(torch::Tensor q,
										  torch::Tensor k,
										  torch::Tensor cos_cache,
										  torch::Tensor sin_cache,
										  uint32_t len_text_prompt)
{
	CHECK_INPUT(q);
	CHECK_INPUT(k);
	CHECK_INPUT(cos_cache);
	CHECK_INPUT(sin_cache);

	CHECK_EQ(cos_cache.dtype(), torch::kFloat);
	CHECK_EQ(sin_cache.dtype(), torch::kFloat);

	CHECK_EQ(q.dim(), 4);
	CHECK_EQ(k.dim(), 4);
	CHECK_EQ(cos_cache.dim(), 2);
	CHECK_EQ(sin_cache.dim(), 2);

	const uint32_t bsz = q.size(0);
	const uint32_t num_qo_heads = q.size(1);
	const uint32_t stride_seq_len = q.size(2);
	const uint32_t head_dim = q.size(3);

	assert(head_dim % 2 == 0);
	const uint32_t half_head_dim = head_dim / 2;
	const uint32_t num_kv_heads = k.size(1);
	const uint32_t valid_seq_len = stride_seq_len - len_text_prompt;

	assert(valid_seq_len > 0);

	CHECK_EQ(q.size(0), k.size(0));
	CHECK_EQ(q.size(2), k.size(2));
	CHECK_EQ(q.size(3), k.size(3));
	CHECK_EQ(cos_cache.size(0), valid_seq_len);
	CHECK_EQ(cos_cache.size(1), half_head_dim);
	CHECK_EQ(sin_cache.size(0), valid_seq_len);
	CHECK_EQ(sin_cache.size(1), half_head_dim);

	bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&]
												   {
		DEBUG_CUDA_CALL(ApplyQKRotaryCosSinComplexCacheInPlace<c_type>(static_cast<c_type*>(q.data_ptr()),
									static_cast<c_type*>(k.data_ptr()),
									cos_cache.data_ptr<float>(),
									sin_cache.data_ptr<float>(),
									bsz,
									num_qo_heads,
									num_kv_heads,
									stride_seq_len,
									len_text_prompt,
									head_dim,
									at::cuda::getCurrentCUDAStream()));
		return true; });
	TORCH_CHECK(success, "RoPE Complex in place apply kernel call failed");
}