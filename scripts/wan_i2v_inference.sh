# TFP Values: 
# Set the following values to control the percentage of timesteps using dense attention:
# 35% → 0.09, 30% → 0.075, 25% → 0.055, 20% → 0.045, 15% → 0.03, 10% → 0.02
first_times_fp=0.09
first_layers_fp=0.025
sparsity=0.25

prompt=$(cat examples/wan/1/prompt.txt)
image_path="examples/wan/1/image.jpg"

python wan_i2v_inference.py \
    --model_id "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers" \
    --prompt "${prompt}" \
    --image_path "${image_path}" \
    --seed 0 \
    --num_inference_steps 40 \
    --pattern "SVG" \
    --num_sampled_rows 64 \
    --sparsity $sparsity \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp

# Dense Attention (Baseline)
python wan_i2v_inference.py \
    --prompt "${prompt}" \
    --image_path "${image_path}" \
    --seed 0 \
    --num_inference_steps 40 \
    --pattern "dense"
