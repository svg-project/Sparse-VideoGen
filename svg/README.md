## Directory Structure Overview

### `./kernel/`
Optimized implementations for specific kernels, delivering **5-10× speedup**, including:

- **Fast RoPE**
- **Fast QK-Norm**

**Subdirectories:**
- `include/` — Core optimized implementations  
- `test/` — Benchmarks and correctness validation

**Model Compatibility:**

| Model       | Fast RoPE | Fast QK-Norm |
|-------------|-----------|--------------|
| **CogVideoX** | ✓         | ✓            |
| **Hunyuan**   | ✓         | ✓            |
| **Wan**       | ✓         | ✗            |

---

### `./models/`
Contains modular implementations for supported video generation models:

- **`cog/`** — [CogVideoX](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/cogvideox_transformer_3d.py)  
  Built on Hugging Face Diffusers

- **`hyvideo/`** — [Hunyuan Video](https://github.com/Tencent/HunyuanVideo)  
  Based on the official HunyuanVideo implementation

- **`wan/`** — [Wan 2.1](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py)  
  Hugging Face Diffusers-based variant

- **`wan_orig/`** — [Wan 2.1 (Official)](https://github.com/Wan-Video/Wan2.1)  
  Based on the official Wan2.1 implementation
