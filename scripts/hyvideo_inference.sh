#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

# TFP: 35% = 0.07, 30% = 0.055, 25% = 0.04, 20% = 0.033, 15% = 0.02, 10% = 0.015
# first_times_fp=0.055
first_layers_fp=0.025

for first_times_fp in 0.04
do
    for sparsity in 0.2
    do
        for sample_row in 64
        do
            CUDA_VISIBLE_DEVICES=3 python3 hyvideo_inference.py \
                --video-size 720 1280 \
                --video-length 129 \
                --infer-steps 50 \
                --seed 0 \
                --prompt "镜头从远及近推进，一位肤若凝脂的东方美人立于烟雨蒙蒙的江南庭院。她身着淡青色织金云纹长裙，裙摆随微风轻轻晃动，墨色长发以珍珠玉簪半挽成优雅发髻，几缕青丝随风飘逸。她纤长的手指轻抚栏下的紫藤花，眼神温婉含情，面容似传统工笔画中走出的美人。庭院内假山流水潺潺，古老的青石板路上落满紫藤花瓣，远处粉墙黛瓦与竹影交织。自然光透过薄雾漫射，勾勒出女子完美的侧颜轮廓。摄影以45度仰角缓缓环绕，捕捉其清丽绝伦的气质。超写实质感，电影氛围，中景特写镜头，打造出东方美学的意境。" \
                --embedded-cfg-scale 6.0 \
                --flow-shift 7.0 \
                --flow-reverse \
                --use-cpu-offload \
                --output_path ./output.mp4 \
                --pattern "SVG" \
                --version "v5" \
                --num_sampled_rows $sample_row \
                --sparsity $sparsity \
                --first_times_fp $first_times_fp \
                --first_layers_fp $first_layers_fp \
                --record_attention
        done
    done
done
