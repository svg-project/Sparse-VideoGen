#include <torch/extension.h>
#include "ops.h"

PYBIND11_MODULE(_kernels, m)
{
	m.def("layer_norm_forward", &layer_norm_forward, "Layer norm with bias and learned weight.");
	m.def("rms_norm_forward", &rms_norm_forward, "RMS norm with bias and learned weight.");
	m.def("apply_qk_rope_inplace_cossin", &apply_qk_rope_inplace_cossin, "Apply QKRotary with cosine and sine cache in place.");
	m.def("apply_qk_rope_inplace_cossin_txtlast", &apply_qk_rope_inplace_cossin_txtlast, "Apply QKRotary with cosine and sine cache in place. But text modality is after video.");
	m.def("apply_qk_rope_inplace_cossin_complex", &apply_qk_rope_inplace_cossin_complex, "Apply QKRotary with cosine and sine cache in place. But the rotary embedding is complex.");
}