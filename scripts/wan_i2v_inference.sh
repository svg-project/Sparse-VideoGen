# TFP Values: 
# Set the following values to control the percentage of timesteps using dense attention:
# 35% → 0.09, 30% → 0.075, 25% → 0.055, 20% → 0.045, 15% → 0.03, 10% → 0.02
first_times_fp=0.09
first_layers_fp=0.025
sparsity=0.25

prompt=$(cat /ssd/data/xihaocheng/Sparse-VideoGen/examples/wan/1/prompt.txt)
image_path="examples/wan/1/image.jpg"

python wan_i2v_inference.py \
    --prompt $prompt \
    --image_path $image_path \
    --seed $seed \
    --num_inference_steps 40 \
    --pattern $pattern \
    --num_sampled_rows 64 \
    --sparsity $sparsity \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp