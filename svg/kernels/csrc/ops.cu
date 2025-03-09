#include <torch/extension.h>
#include "ops.h"

PYBIND11_MODULE(_kernels, m) {
	m.def("layer_norm_forward", &layer_norm_forward, "Layer norm with bias and learned weight.");
	m.def("apply_qk_rope_inplace_cossin", &apply_qk_rope_inplace_cossin, "Apply QKRotary with cosine and sine cache in place.");
}