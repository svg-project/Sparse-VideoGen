#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

# TFP Values: 
# Set the following values to control the percentage of timesteps using dense attention:
# 35% → 0.07, 30% → 0.055, 25% → 0.04, 20% → 0.033, 15% → 0.02, 10% → 0.015
# first_times_fp=0.055
# first_layers_fp=0.025

first_times_fp=0.04
first_layers_fp=0.0

resolution="720p"
resolution_cfg="720 1280"
infer_step=50

# Prompt Source
PROMPT_SOURCE="T2V_Hyv_VBench"
prompt_idx=102
prompt="data/hyv_augmented_vbench.txt"


output_dir="result/hyvideo_orig/t2v_${PROMPT_SOURCE}/svg"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

# Dense Attention Cfg
dense_attention_cfg="TFP_${first_times_fp}-LFP_${first_layers_fp}"

# Output feature
output_feature="${video_cfg}/${dense_attention_cfg}"

# Sparse VideoGen
python3 orig_hyvideo_inference.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --seed 0 \
    --prompt_source "${PROMPT_SOURCE}" \
    --prompt_idx "${prompt_idx}" \
    --prompt "${prompt}" \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --use-cpu-offload \
    --output_file "${output_dir}/${output_feature}/${prompt_idx}.mp4" \
    --pattern "SVG" \
    --num_sampled_rows 64 \
    --sparsity 0.2 \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp
