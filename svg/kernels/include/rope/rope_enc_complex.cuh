#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#include "flashinfer/layout.cuh"
#include "flashinfer/math.cuh"
#include "flashinfer/utils.cuh"
#include "flashinfer/vec_dtypes.cuh"

using namespace flashinfer;

template <uint32_t vec_size, uint32_t cos_sin_vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size>
vec_apply_llama_rope_cos_sin_complex_interleave(const T *x,
												const vec_t<float, cos_sin_vec_size> &cos,
												const vec_t<float, cos_sin_vec_size> &sin,
												const uint32_t rotary_dim = vec_size * bdx)
{
	vec_t<float, vec_size> vec, vec_before;
	vec.cast_load(x + threadIdx.x * vec_size);

	if (threadIdx.x * vec_size < rotary_dim)
	{
		vec_before = vec;
#pragma unroll
		for (uint32_t i = 0; i < vec_size; ++i)
		{
			// Adjust index for cos and sin by dividing by 2
			uint32_t cos_sin_idx = floor(i / 2);
			// vec[i] = vec[i] * cos[cos_sin_idx] + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin[cos_sin_idx];

			// Convert to float64 for computation
			double vec_f64 = static_cast<double>(vec_before[i]);
			double vec_other_f64 = static_cast<double>(vec_before[i ^ 1]);
			double cos_f64 = static_cast<double>(cos[cos_sin_idx]);
			double sin_f64 = static_cast<double>(sin[cos_sin_idx]);

			// Perform computation in float64 using the original implementation
			double result_f64 = vec_f64 * cos_f64 + ((i % 2 == 0) ? -vec_other_f64 : vec_other_f64) * sin_f64;

			// Cast back to float32 for storage
			vec[i] = static_cast<T>(result_f64);
		}
	}
	return vec;
}

/*
 * ApplyQKRotaryCosSinCacheInPlaceKernel
 * Apply QKRotary with cosine and sine cache in place.
 * Single batch, Contigous memory layout, row-major.
 * @tparam head_dim: the dimension of the head
 * @tparam vec_size: the size of the vector
 * @tparam bdx: num of threads needed for loading a head
 * @tparam DType: the data type
 *
 * @param q: [bsz, num_qo_heads, seq_len, head_dim]
 * @param k: [bsz, num_kv_heads, seq_len, head_dim]
 * @param cos_cache: [seq_len, head_dim]
 * @param sin_cache: [seq_len, head_dim]
 * kernel launched in:
 *   gridDim(bsz, seq_len+bdy-1/bdy)
 *   blockDim(bdx, bdy)
 */
template <uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType>
__global__ void ApplyQKRotaryCosSinComplexCacheInPlaceKernel(DType *q,
															 DType *k,
															 float *__restrict__ cos_cache,
															 float *__restrict__ sin_cache,
															 uint32_t stride_seq_len,
															 uint32_t skip_seq_len,
															 uint32_t num_qo_heads,
															 uint32_t num_kv_heads)
{
	const uint32_t bx = blockIdx.x, by = blockIdx.y;
	const uint32_t tx = threadIdx.x, ty = threadIdx.y;
	const uint32_t bdy = blockDim.y;

	const uint32_t valid_seq_len = max(0, stride_seq_len - skip_seq_len);
	const uint32_t cur_seq_len = by * bdy + ty;

	const uint32_t cos_sin_vec_size = vec_size / 2;

	vec_t<float, cos_sin_vec_size> cos, sin;
	const uint32_t cos_sin_head_dim = head_dim / 2;
	if (cur_seq_len < valid_seq_len)
	{
		cos.load(cos_cache + cur_seq_len * cos_sin_head_dim + tx * cos_sin_vec_size);
		sin.load(sin_cache + cur_seq_len * cos_sin_head_dim + tx * cos_sin_vec_size);

#pragma unroll 1
		for (uint32_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx)
		{
			DType *q_ptr =
				q + (bx * num_qo_heads + qo_head_idx) * stride_seq_len * head_dim + (cur_seq_len + skip_seq_len) * head_dim;
			vec_t<float, vec_size> q_vec;
			q_vec =
				vec_apply_llama_rope_cos_sin_complex_interleave<vec_size, cos_sin_vec_size, bdx>(q_ptr, cos, sin, head_dim);
			q_vec.cast_store(q_ptr + tx * vec_size);
		}

#pragma unroll 1
		for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx)
		{
			DType *k_ptr =
				k + (bx * num_kv_heads + kv_head_idx) * stride_seq_len * head_dim + (cur_seq_len + skip_seq_len) * head_dim;
			vec_t<float, vec_size> k_vec;
			k_vec =
				vec_apply_llama_rope_cos_sin_complex_interleave<vec_size, cos_sin_vec_size, bdx>(k_ptr, cos, sin, head_dim);
			k_vec.cast_store(k_ptr + tx * vec_size);
		}
	}
}

template <typename DType>
void ApplyQKRotaryCosSinComplexCacheInPlace(DType *q,
											DType *k,
											float *cos_cache,
											float *sin_cache,
											uint32_t bsz,
											uint32_t num_qo_heads,
											uint32_t num_kv_heads,
											uint32_t stride_seq_len,
											uint32_t skip_seq_len,
											uint32_t head_dim,
											cudaStream_t stream)
{
	DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
		constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
		constexpr uint32_t bdx = HEAD_DIM / vec_size;
		uint32_t num_threads = std::max(128U, bdx);
		uint32_t bdy = num_threads / bdx;
		dim3 nblks(bsz, (stride_seq_len - skip_seq_len + bdy - 1) / bdy);
		dim3 nthrs(bdx, bdy);

		ApplyQKRotaryCosSinComplexCacheInPlaceKernel<HEAD_DIM, vec_size, bdx, DType>
			<<<nblks, nthrs, 0, stream>>>(
				q, k, cos_cache, sin_cache, stride_seq_len, skip_seq_len, num_qo_heads, num_kv_heads);
	});
}