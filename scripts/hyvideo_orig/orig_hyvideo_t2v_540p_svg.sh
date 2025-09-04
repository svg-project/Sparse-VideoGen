#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

# TFP Values: 
# Set the following values to control the percentage of timesteps using dense attention:
# 35% → 0.07, 30% → 0.055, 25% → 0.04, 20% → 0.033, 15% → 0.02, 10% → 0.015
# first_times_fp=0.055
# first_layers_fp=0.025

first_times_fp=0.0
first_layers_fp=0.0

resolution="540p"
resolution_cfg="544 960"
infer_step=50

output_dir="result/hyvideo_orig/t2v/svg"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

# Dense Attention Cfg
dense_attention_cfg="TFP_${first_times_fp}-LFP_${first_layers_fp}"

# Output feature
output_feature="${video_cfg}/${dense_attention_cfg}"

# Sparse VideoGen
python3 orig_hyvideo_inference.py \
    --video-size $resolution_cfg \
    --video-length 129 \
    --infer-steps $infer_step \
    --seed 0 \
    --prompt "A cat walks on the grass, realistic style." \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --use-cpu-offload \
    --output_file "${output_dir}/${output_feature}/cat.mp4" \
    --pattern "SVG" \
    --num_sampled_rows 64 \
    --sparsity 0.2 \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp
