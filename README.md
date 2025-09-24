<div align="center" id="sglangtop">
  <img src="assets/Minimal_dark_white_background.png" alt="logo" width="400" margin="10px"></img>
</div>
<h3 align="center">
Accelerate Video Generation with High Pixel-level Fidelity
</h3>

<p align="center">
| <a href="https://svg-project.github.io/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2502.01776"><b>Paper</b></a> | <a href="https://x.com/HaochengXiUCB/status/1899953252327927911"><b>Twitter/X</b></a> |
</p>

## 🔥News🔥
- [2025/09] Sparse VideoGen2 is open-sourced!
- [2025/09] Sparse VideoGen2 is accepted by NeurIPS 2025 as a spotlight!
- [2025/05] Sparse VideoGen is accepted by ICML 2025!
- [2025/04] Wan 2.1 is supported! Both T2V and I2V are accelerated.
- [2025/03] Sparse VideoGen is open-sourced! HunyuanVideo and CogVideoX v1.5 can be accelerated by 2×

## 📚 About
Sparse VideoGen (SVG) is a **training-free framework** that leverages **inherent spatial and temporal sparsity** in the 3D Full Attention operations. Sparse VideoGen's core contributions include:
 - Identifying the **spatial and temporal sparsity patterns** in video diffusion models.
 - Proposing an **Online Profiling Strategy** to dynamically identify these patterns.
 - Implementing an end-to-end generation framework through **efficient algorithm-system co-design**, with **hardware-efficient layout transformation** and **customized kernels**.

## 🎥 Demo
<div style="display: flex; gap: 10px;">
    <img src="assets/video/SparseVideoGenDemo.gif" style="width: 100%;"/>
    <img src="assets/video/Algorithm.gif" style="width: 100%;"/>
</div>


## 🛠️ Installation
Begin by cloning the repository:
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/svg-project/Sparse-VideoGen.git # Do not clone the demo, otherwise is too large
cd Sparse-VideoGen
```

We recommend using CUDA versions 12.4 / 12.8 + PyTorch versions 2.5.1 / 2.6.0
```bash
# 1. Create and activate conda environment
conda create -n SVG python==3.12.9 # or 3.11.9 if have error when installing kernels
conda activate SVG

# 2. Install uv, then install other packages
pip install uv
uv pip install -e .

pip install flash-attn --no-build-isolation

# 4. Install customized kernels. (You might need to upgrade your cmake and CUDA version.)
pip install -U setuptools # Require at least version 77.0.0
git submodule update --init --recursive
cd svg/kernels
pip install -U cmake
bash setup.sh
cd 3rdparty/flashinfer
cp ../../../../assets/patches/modifications.patch ./
git apply modifications.patch
pip install --no-build-isolation --verbose --editable . # Block Sparse Attention with varied block sizes
pip install cuvs-cu12 --extra-index-url=https://pypi.nvidia.com # 
```

## 🚀 Inference Examples
### Wan 2.1
We support running Wan 2.1 inference using diffusers. Please make sure to install the latest version of diffusers.
```bash
pip install git+https://github.com/huggingface/diffusers
```

We support Text-to-Video and Image-to-Video inference of Wan 2.1 model. The running scripts are:
```bash
# Text-to-Video
bash scripts/wan_t2v_inference.sh

# Image-to-Video
bash scripts/wan_i2v_inference.sh
```

Command Line:
```python
# Text-to-Video
python wan_t2v_inference.py \
    --prompt ${prompt} \
    --height 720 \
    --width 1280 \
    --pattern "SVG" \
    --num_sampled_rows 64 \
    --sparsity 0.25 \
    --first_times_fp 0.025 \
    --first_layers_fp 0.075

# Image-to-Video
python wan_i2v_inference.py \
    --prompt "$prompt" \
    --image_path "$image_path" \
    --seed 0 \
    --num_inference_steps 40 \
    --pattern "SVG" \
    --num_sampled_rows 64 \
    --sparsity 0.25 \
    --first_times_fp 0.025 \
    --first_layers_fp 0.075
```
If you want to run 480p video generation, please change the height and weight arguments to 480 and 832.

### HunyuanVideo
To run HunyuanVideo Text-to-Video inference examples, you first need to download the checkpoints under `ckpts` following [the official guide](https://github.com/Tencent/HunyuanVideo/blob/main/ckpts/README.md).
Then, run
```bash
bash scripts/hyvideo_inference.sh
```

Command line:
```python
python3 hyvideo_inference.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --seed 0 \
    --prompt "A cat walks on the grass, realistic style." \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --use-cpu-offload \
    --output_path ./output.mp4 \
    --pattern "SVG" \
    --num_sampled_rows 64 \
    --sparsity 0.2 \
    --first_times_fp 0.055 \
    --first_layers_fp 0.025
```

On a single H100, the generation should takes 14 minutes.

### CogVideoX v1.5
To run CogVideoX v1.5 Image-to-Video inference exmaples, run
```bash
bash scripts/cog_inference.sh
```

Command line:
```python
python3 cog_inference.py \
    --prompt "A bright yellow water taxi glides smoothly across the choppy waters, creating gentle ripples in its wake. The iconic Brooklyn Bridge looms majestically in the background, its intricate web of cables and towering stone arches standing out against the city skyline. The boat, bustling with passengers, offers a lively contrast to the serene, expansive sky dotted with fluffy clouds. As it cruises forward, the vibrant cityscape of New York unfolds, with towering skyscrapers and historic buildings lining the waterfront, capturing the dynamic essence of urban life." \
    --image_path "examples/cog/img/boat.jpg" \
    --output_path "output.mp4"
```

On a single H100, the generation should takes 4 minutes.

## 📑 Open-source Plan
 - [ ] Support FP8 attention
 - [x] Support [Wan 2.1](https://github.com/Wan-Video/Wan2.1)
 - [ ] Support [Cosmos](https://github.com/NVIDIA/Cosmos)

## Efficiency Benchmark
### End-to-End Speedup
## End-to-End Speedup

| Model | Task | Hardware | Resolution | Baseline (min) | SVG (min) | Speedup |
|-------|------|----------|------------|---------------|-----------|---------|
| HunyuanVideo | Text-to-Video | H100 | 720P | 29:57 | 15:38 | 1.91× |
| Wan 2.1 | Text-to-Video | H100 | 720P | 31:35 | 20:51 | 1.51× |
| Wan 2.1 | Text-to-Video | H100 | 480P | 8:05 | 6:11 | 1.32×  |
| Wan 2.1 | Image-to-Video | H100 | 720P | 24:05 | 16:03 | 1.50× |
| HunyuanVideo | Text-to-Video | A100 | 720P | 50:48 | 30:14 | 1.68× |
| Wan 2.1 | Text-to-Video | A100 | 720P | 57:57 | 42:59 | 1.35× |
| Wan 2.1 | Text-to-Video | A100 | 480P | 15:41 | 13:00 | 1.20× |
| Wan 2.1 | Image-to-Video | A100 | 720P | 45:19 | 34:27 | 1.32× |


### Customized Kernels Performance
We evaluate the performance of our customized kernels against the baseline implementations. The following tables show the memory bandwidth (GB/s) comparison for different batch sizes and hidden dimensions:

#### RMSNorm Performance

| Batch Size | Hidden Dim | Diffusers (GB/s) | SVG Customized (GB/s) | Speedup |
|------------|------------|------------------|----------------------|----------|
| 2,097,152  | 32        | 151.36           | 809.69              | 5.35×    |
| 1,048,576  | 64        | 196.54           | 810.61              | 4.12×    |
| 524,288    | 128       | 232.66           | 810.21              | 3.48×    |
| 262,144    | 256       | 252.67           | 810.41              | 3.21×    |

#### LayerNorm Performance

| Batch Size | Hidden Dim | Diffusers (GB/s) | SVG Customized (GB/s) | Speedup |
|------------|------------|------------------|----------------------|----------|
| 2,097,152  | 32        | 45.82            | 808.28              | 17.64×   |
| 1,048,576  | 64        | 91.18            | 805.22              | 8.83×    |
| 524,288    | 128       | 197.89           | 804.29              | 4.06×    |
| 262,144    | 256       | 350.87           | 804.43              | 2.29×    |

Our customized kernels achieve significantly higher memory bandwidth across all configurations, with speedups ranging from 2.29× to 17.64×. The performance improvement is particularly notable for smaller hidden dimensions and larger batch sizes.

### RoPE (Rotary Position Embedding) Performance

| Batch Size | Num Heads | Seq Length | Head Dim | Diffusers (GB/s) | SVG Customized (GB/s) | Speedup |
|------------|-----------|------------|----------|------------------|----------------------|----------|
| 1          | 32        | 1024       | 64      | 17.25           | 158.81              | 9.21×    |
| 1          | 32        | 4096       | 64      | 27.74           | 405.75              | 14.63×   |
| 1          | 32        | 16384      | 64      | 30.86           | 605.89              | 19.63×   |
| 4          | 32        | 1024       | 64      | 27.60           | 475.94              | 17.24×   |
| 4          | 32        | 4096       | 64      | 30.93           | 614.11              | 19.85×   |
| 4          | 32        | 16384      | 64      | 32.41           | 648.36              | 20.00×   |

The RoPE implementation in SVG shows substantial performance improvements over the Diffusers baseline, with speedups ranging from 9.21× to 20.00×. The performance gain is particularly significant for longer sequence lengths and larger batch sizes, demonstrating excellent scaling characteristics.

## 🔗 BibTeX
If you find Sparse VideoGen useful for your research and applications or interesting, please cite our work using BibTeX:
```bibtex
@article{xi2025sparse,
  title={Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity},
  author={Xi, Haocheng and Yang, Shuo and Zhao, Yilong and Xu, Chenfeng and Li, Muyang and Li, Xiuyu and Lin, Yujun and Cai, Han and Zhang, Jintao and Li, Dacheng and others},
  journal={arXiv preprint arXiv:2502.01776},
  year={2025}
}

@article{yang2025sparse,
  title={Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation},
  author={Yang, Shuo and Xi, Haocheng and Zhao, Yilong and Li, Muyang and Zhang, Jintao and Cai, Han and Lin, Yujun and Li, Xiuyu and Xu, Chenfeng and Peng, Kelly and others},
  journal={arXiv preprint arXiv:2505.18875},
  year={2025}
}
```
