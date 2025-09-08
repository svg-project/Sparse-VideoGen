#!/bin/bash
# Description: This script demonstrates multi-gpu video inference using the HunyuanVideo model

# TFP Values: 
# Set the following values to control the percentage of timesteps using dense attention:
# 35% → 0.07, 30% → 0.055, 25% → 0.04, 20% → 0.033, 15% → 0.02, 10% → 0.015
first_times_fp=0.055
first_layers_fp=0.025

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
hyvideo_inference.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 10 \
    --seed 0 \
    --prompt "A cat walks on the grass, realistic style." \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --output_path ./hunyuan_output_svg_step10_sp4.mp4 \
    --pattern "SVG" \
    --num_sampled_rows 64 \
    --sparsity 0.2 \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp \
    --ulysses-degree 4 \
    --record_attention
