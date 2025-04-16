first_times_fp=0.075
first_layers_fp=0.025
sparsity=0.25

prompt=$(cat examples/wan/1/prompt.txt)
image_path="examples/wan/1/image.jpg"

python orig_wan_generate.py \
    --task i2v-14B \
    --size 1280*720 \
    --ckpt_dir ./Wan2.1-I2V-14B-720P \
    --base_seed 0 \
    --prompt "${prompt}" \
    --image "${image_path}" \
    --pattern "dense"

python orig_wan_generate.py \
    --task i2v-14B \
    --size 1280*720 \
    --ckpt_dir ./Wan2.1-I2V-14B-720P \
    --base_seed 0 \
    --prompt "${prompt}" \
    --image "${image_path}" \
    --pattern "SVG" \
    --num_sampled_rows 64 \
    --sparsity $sparsity \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp \
    --record_attention
        
# huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./Wan2.1-I2V-14B-720P
# huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./Wan2.1-I2V-14B-480P