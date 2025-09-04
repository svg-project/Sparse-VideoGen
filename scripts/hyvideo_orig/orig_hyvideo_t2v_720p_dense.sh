#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

resolution="720p"
resolution_cfg="720 1280"
infer_step=50

output_dir="result/hyvideo_orig/t2v/dense"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

output_feature="${video_cfg}"

# Dense
python3 orig_hyvideo_inference.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --seed 0 \
    --prompt "A cat walks on the grass, realistic style." \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --use-cpu-offload \
    --output_file "${output_dir}/${output_feature}/cat.mp4" \
    --pattern "dense" \