# TFP Values: 
# Set the following values to control the percentage of timesteps using dense attention:
# 35% → 0.09, 30% → 0.075, 25% → 0.055, 20% → 0.045, 15% → 0.03, 10% → 0.02
first_times_fp=0.075
first_layers_fp=0.025
sparsity=0.25

prompt=$(cat examples/wan/3/prompt.txt)

# 720p
python wan_t2v_inference.py \
    --prompt "$prompt" \
    --height 720 \
    --width 1280 \
    --pattern "SVG" \
    --num_sampled_rows 64 \
    --sparsity $sparsity \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp

# 480p
python wan_t2v_inference.py \
    --prompt "$prompt" \
    --height 480 \
    --width 832 \
    --pattern "SVG" \
    --num_sampled_rows 64 \
    --sparsity $sparsity \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp
