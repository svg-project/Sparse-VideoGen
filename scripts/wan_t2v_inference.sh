# Description: This script demonstrates how to inference a video based on HunyuanVideo model
if [ $# -eq 0 ]; then
    echo "Please provide the CUDA device number as an argument."
    echo "Usage: $0 <cuda_device_number>"
    exit 1
fi

# TFP Values: 
# Set the following values to control the percentage of timesteps using dense attention:
# 35% → 0.09, 30% → 0.075, 25% → 0.055, 20% → 0.045, 15% → 0.03, 10% → 0.02
first_times_fp=0.075
first_layers_fp=0.025
sparsity=0.25

CUDA_VISIBLE_DEVICES=$1 python wan_inference.py \
    --file_idx 6 \
    --height 720 \
    --width 1280 \
    --pattern "SVG" \
    --num_sampled_rows 64 \
    --sparsity $sparsity \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp