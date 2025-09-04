#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

resolution="720p"
resolution_cfg="720 1280"
infer_step=50

# Prompt Source
PROMPT_SOURCE="T2V_Hyv_VBench"
prompt_idx=102
prompt="data/hyv_augmented_vbench.txt"

output_dir="result/hyvideo_orig/t2v_${PROMPT_SOURCE}/dense"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

output_feature="${video_cfg}"

# Dense
python3 orig_hyvideo_inference.py \
    --video-size ${resolution_cfg} \
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
    --pattern "dense" \