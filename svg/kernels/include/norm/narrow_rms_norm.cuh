#pragma once

#include <float.h>

#include "cutlass/cutlass.h"
#include "cutlass/layout/tensor.h"
#include "device_utils.cuh"
#include "norm/narrow_layer_norm.cuh"

#include <iostream>
#include <string_view>
#include <type_traits>
#include <typeinfo>

/*
    * \brief sub-warp reduce rmsnorm
    * \tparam T: data type (all vectorized load by float4)
    * \tparam bdx: blockDim.x, leading dimension, should be 1<=bdx<=32
    * \tparam bdy: folding factor
    */
template <typename T, int bdx, int bdy>
__global__ void rmsnorm_narrow_n_subwarp_reduction(T* data,
													 const T* gamma,
													 float epsilon,
													 const int m,
													 const int n) {
	// input [m,n]
	// gridDim(m+bdy-1/bdy)
	// blockDim(bdx,bdy)

	const int cur_m = blockIdx.x * bdy + threadIdx.y;
	const int tid = threadIdx.x;
	T* input = data + cur_m * n;
	T* output = data + cur_m * n;

	// As there is no inter-warp communication, we use local memory
	float s_variance, local_sum = 0.0f;
	float4 local_val[1];
	const float4* vectorized_input = reinterpret_cast<const float4*>(input);
	float4* vectorized_output = reinterpret_cast<float4*>(output);
	const float4* vectorized_gamma = reinterpret_cast<const float4*>(gamma);

	local_val[0] = cur_m < m ? vectorized_input[tid] : local_val[0];

	T* extracted_local_val = reinterpret_cast<T*>(local_val);
	constexpr int NUM_T_VECTORIZED = sizeof(float4) / sizeof(T);

#pragma UNROLL
	for(int i = 0; i < NUM_T_VECTORIZED; i += 1) {
		local_sum += (static_cast<float>(extracted_local_val[i])) *
					 (static_cast<float>(extracted_local_val[i]));
	}
	local_sum = cur_m < m ? local_sum : 0.0f;

#pragma unroll
	for(int i = bdx / 2; i > 0; i >>= 1) {
		local_sum += __shfl_xor_sync(FINAL_MASK, local_sum, i);
	}
	s_variance = rsqrtf(local_sum / n + epsilon);

	float4 local_gamma[1] = {vectorized_gamma[tid]};
	T* extracted_local_gamma = reinterpret_cast<T*>(local_gamma);
#pragma UNROLL
	for(int i = 0; i < NUM_T_VECTORIZED; i += 1) {
		float tmp = (static_cast<float>(extracted_local_val[i])) * s_variance *
						static_cast<float>(extracted_local_gamma[i]);
		extracted_local_val[i] = static_cast<T>(tmp);
	}

	if(cur_m < m) {
		vectorized_output[tid] = local_val[0];
	}
}

template <typename T>
void narrow_rmsnorm_inplace(cutlass::MatrixCoord tensor_size,
					  T* data,
					  const T* gamma,
					  float epsilon,
					  cudaStream_t stream) {
	const int m = tensor_size.row();
	const int n = tensor_size.column();
	constexpr int NUM_T_VECTORIZED = sizeof(float4) / sizeof(T);
	SWITCH_HEAD_DIM(n, HEAD_DIM, {
		static_assert(HEAD_DIM % NUM_T_VECTORIZED == 0,
						"n should be multiple of NUM_T_VECTORIZED");
		constexpr int bdx = HEAD_DIM / NUM_T_VECTORIZED;
		constexpr int bdy = 32 / bdx;
		static_assert((bdx > 0) && ((bdx & (bdx - 1)) == 0), "bdx should be power of 2");

		dim3 block(bdx, bdy);
		dim3 grid((m + bdy - 1) / bdy);
		rmsnorm_narrow_n_subwarp_reduction<T, bdx, bdy>
			<<<grid, block, 0, stream>>>(data, gamma, epsilon, m, n);
	});
}