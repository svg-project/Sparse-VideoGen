<div align="center" id="sglangtop">
  <img src="assets/Minimal_dark_white_background.png" alt="logo" width="400" margin="10px"></img>
</div>
<h3 align="center">
Accelerate Video Generation with High Pixel-level Fidelity
</h3>

<p align="center">
| <a href="https://svg-project.github.io/"><b>Website</b></a> | <a href="https://arxiv.org/abs/2502.01776"><b>SVG 1 Paper</b></a> | <a href="https://arxiv.org/abs/2505.18875"><b>SVG2 Paper</b></a> | <a href="https://arxiv.org/abs/2603.08982"><b>SVG-EAR Paper</b></a> | <a href="https://x.com/HaochengXiUCB/status/1899953252327927911"><b>SVG 1 Twitter/X</b></a> | <a href="https://x.com/HaochengXiUCB/status/1971219731140182423"><b>SVG 2 Twitter/X</b></a> |
</p>

## 🔥News🔥
- [2026/06] [SVG-EAR](https://arxiv.org/abs/2603.08982) is accepted by ECCV 2026!
- [2025/09] We release [Flash k-Means](https://github.com/svg-project/flash-kmeans), a batched K-Means clustering algorithm implemented with Triton that offers >10x speedup!
- [2025/09] [Sparse VideoGen2](https://arxiv.org/abs/2505.18875) is open-sourced! HunyuanVideo, Wan 2.1 and Cosmos can be accelerated by 2×
- [2025/09] Sparse VideoGen2 is accepted by NeurIPS 2025 as a **spotlight**!
- [2025/05] [Sparse VideoGen](https://arxiv.org/abs/2502.01776) is accepted by ICML 2025!
- [2025/04] Wan 2.1 is supported! Both T2V and I2V are accelerated.
- [2025/03] Sparse VideoGen is open-sourced! HunyuanVideo and CogVideoX v1.5 can be accelerated by 2×

## 📚 About
Sparse VideoGen 1, 2, and EAR are **training-free frameworks** that leverage **inherent sparsity** in the 3D Full Attention operations to accelerate video generation.

Sparse VideoGen 1's core contributions:
 - Identifying the **spatial and temporal sparsity patterns** in video diffusion models.
 - Proposing an **Online Profiling Strategy** to dynamically identify these patterns.
 - Implementing an end-to-end generation framework through **efficient algorithm-system co-design**, with **hardware-efficient layout transformation** and **customized kernels**.

Sparse VideoGen 2's core contributions:
 - Tackles **inaccurate token identification** and **computation waste** in video diffusion.
 - Introduces **semantic-aware** sparse attention with efficient **token permutation**.
 - Provides an end-to-end system design with a **dynamic attention** kernel and **flash k-means** kernel.

SVG-EAR's core contributions:
 - Recovers skipped attention blocks with **parameter-free centroid compensation**.
 - Introduces **error-aware routing** to compute high-error blocks exactly while compensating the rest.

## 🎥 Demo of SVG1
<div style="display: flex; gap: 10px;">
    <img src="assets/video/SparseVideoGenDemo.gif" style="width: 100%;"/>
    <img src="assets/video/Algorithm.gif" style="width: 100%;"/>
</div>

## 🎥 Demo of SVG2
<table border="0" style="width: 100%; text-align: center;">
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/ca4801bb-a94a-4f34-8c67-f63d080536b7"
             width="100%" autoplay loop muted playsinline controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/a030f7f2-6048-4268-b984-ef5027c577d8"
             width="100%" autoplay loop muted playsinline controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/acd186f3-828d-40af-a635-9abbe9fb7962"
             width="100%" autoplay loop muted playsinline controls></video>
    </td>
  </tr>
</table>

## 🎥 Demo of SVG-EAR
<p align="center">
  <img src="assets/video/svg_ear_dense_comparison.gif" width="32%"/>
  <img src="assets/video/svg_ear_wan_t2v_comparison.gif" width="32%"/>
  <img src="assets/video/svg_ear_methodology.gif" width="32%"/>
</p>



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

# 5. Install FlashInfer (standard) and cuVS
cd 3rdparty/flashinfer
pip install --no-build-isolation --verbose --editable .
pip install cuvs-cu12 --extra-index-url=https://pypi.nvidia.com

# Optional: If the FlashInfer monkey patch fails in your environment,
# install the manually patched FlashInfer (block sparse with varied block sizes).
cd 3rdparty/flashinfer
cp ../../../../assets/patches/modifications.patch ./
git apply modifications.patch
pip install --no-build-isolation --verbose --editable . # Block Sparse Attention with varied block sizes
```

You don’t need to install [flash-kmeans](https://github.com/svg-project/flash-kmeans) separately. A copy of flash-kmeans is included in Sparse VideoGen and is used by default.

## 🚀 Inference Examples
### Wan 2.1

We support Text-to-Video and Image-to-Video inference of Wan 2.1 model. The running scripts are:
```bash
# Text-to-Video
# bash scripts/wan/wan_t2v_720p_svg.sh # SVG
bash scripts/wan/wan_t2v_720p_sap.sh # SVG2
bash scripts/wan/wan_t2v_720p_ear.sh # SVG-EAR

# Image-to-Video
# bash scripts/wan/wan_i2v_720p_svg.sh # SVG
bash scripts/wan/wan_i2v_720p_sap.sh # SVG2
bash scripts/wan/wan_i2v_720p_ear.sh # SVG-EAR
```

### HunyuanVideo

The running scripts are:
```bash
# bash scripts/hyvideo/hyvideo_t2v_720p_svg.sh # SVG
bash scripts/hyvideo/hyvideo_t2v_720p_sap.sh # SVG2
bash scripts/hyvideo/hyvideo_t2v_720p_ear.sh # SVG-EAR
```


## 📑 Open-source Plan
 - [ ] Support FP8 attention
 - [x] Support [SVG-EAR](https://arxiv.org/abs/2603.08982)
 - [x] Support [Wan 2.1](https://github.com/Wan-Video/Wan2.1)
 - [x] Support [Cosmos](https://github.com/NVIDIA/Cosmos)

## Efficiency Benchmark
<!-- ### End-to-End Speedup

| Model | Task | Hardware | Resolution | Baseline (min) | SVG (min) | Speedup |
|-------|------|----------|------------|---------------|-----------|---------|
| HunyuanVideo | Text-to-Video | H100 | 720P | 29:57 | 15:38 | 1.91× |
| Wan 2.1 | Text-to-Video | H100 | 720P | 31:35 | 20:51 | 1.51× |
| Wan 2.1 | Text-to-Video | H100 | 480P | 8:05 | 6:11 | 1.32×  |
| Wan 2.1 | Image-to-Video | H100 | 720P | 24:05 | 16:03 | 1.50× |
| HunyuanVideo | Text-to-Video | A100 | 720P | 50:48 | 30:14 | 1.68× |
| Wan 2.1 | Text-to-Video | A100 | 720P | 57:57 | 42:59 | 1.35× |
| Wan 2.1 | Text-to-Video | A100 | 480P | 15:41 | 13:00 | 1.20× |
| Wan 2.1 | Image-to-Video | A100 | 720P | 45:19 | 34:27 | 1.32× | -->


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

@article{zhou2026svgear,
  title={SVG-EAR: Parameter-Free Linear Compensation for Sparse Video Generation via Error-aware Routing},
  author={Zhou, Xuanyi and Mang, Qiuyang and Yang, Shuo and Xi, Haocheng and Zhang, Jintao and Mao, Huanzhi and Gonzalez, Joseph E. and Keutzer, Kurt and Stoica, Ion and Cheung, Alvin},
  journal={arXiv preprint arXiv:2603.08982},
  year={2026},
  note={Accepted to ECCV 2026}
}
```
