#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

# TFP Values: 
# Set the following values to control the percentage of timesteps using dense attention:
# 35% → 0.07, 30% → 0.055, 25% → 0.04, 20% → 0.033, 15% → 0.02, 10% → 0.015
first_times_fp=0.055
first_layers_fp=0.025

# Sparse VideoGen
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
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp \
    --record_attention

# Dense
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
    --pattern "dense" \