/*
  Modified from FlashInfer PyTorch API.
  Check: https://github.com/flashinfer-ai/flashinfer/blob/main/python/csrc/pytorch_extension_utils.h
*/

#pragma once
#include <cuda_fp16.h>
#include <torch/extension.h>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE(pytorch_dtype, c_type, ...) \
	[&]() -> bool {                                                 \
		switch(pytorch_dtype) {                                     \
		case at::ScalarType::Half: {                                \
			using c_type = nv_half;                                 \
			return __VA_ARGS__();                                   \
		}                                                           \
		case at::ScalarType::BFloat16: {                            \
			using c_type = nv_bfloat16;                             \
			return __VA_ARGS__();                                   \
		}                                                           \
		default:                                                    \
			return false;                                           \
		}                                                           \
	}()

inline void check_shape(const torch::Tensor& a,
						const torch::Tensor& b,
						const char* a_name,
						const char* b_name) {
	TORCH_CHECK(
		a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ", a.dim(), " vs ", b.dim());
	for(int i = 0; i < a.dim(); ++i) {
		TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name, ".size(", i, ")");
	}
}

inline constexpr uint32_t pack_u16(uint16_t a, uint16_t b) {
	return (uint32_t(a) << 16) | uint32_t(b);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
	CHECK_CUDA(x);     \
	CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define CHECK_GE(a, b) TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)

#define DEBUG_CUDA_CALL(func, ...)                                                          \
	{                                                                                       \
		(func);                                                                             \
		cudaError_t e = cudaGetLastError();                                                 \
		if(e != cudaSuccess) {                                                              \
			std::ostringstream oss;                                                         \
			oss << "CUDA Error: " << cudaGetErrorString(e) << " (" << e << ") " << __FILE__ \
				<< ": line " << __LINE__ << " at function " << STR(func) << std::endl;      \
			TORCH_CHECK(false, oss.str());                                                  \
		}                                                                                   \
	}