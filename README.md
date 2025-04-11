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
conda create -n SVG python==3.10.9
conda activate SVG

# 2. Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Install pip dependencies from CogVideoX and HunyuanVideo
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# 4. (Optional) Install customized kernels for maximized speedup. (You might need to upgrade your cmake and CUDA version.)
git submodule update --init --recursive
cd svg/kernels
bash setup.sh
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

## 🔗 BibTeX
If you find Sparse VideoGen useful for your research and applications or interesting, please cite our work using BibTeX:
```bibtex
@article{xi2025sparse,
  title={Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity},
  author={Xi, Haocheng and Yang, Shuo and Zhao, Yilong and Xu, Chenfeng and Li, Muyang and Li, Xiuyu and Lin, Yujun and Cai, Han and Zhang, Jintao and Li, Dacheng and others},
  journal={arXiv preprint arXiv:2502.01776},
  year={2025}
}
```
