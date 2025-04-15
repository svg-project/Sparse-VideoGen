# python orig_wan_generate.py \
#     --task t2v-14B \
#     --size 1280*720 \
#     --ckpt_dir ./Wan2.1-T2V-14B \
#     --base_seed 0 \
#     --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
#     --pattern "dense"

first_times_fp=0.075
first_layers_fp=0.025
sparsity=0.25

python orig_wan_generate.py \
    --task t2v-14B \
    --size 1280*720 \
    --ckpt_dir ./Wan2.1-T2V-14B \
    --base_seed 0 \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --pattern "SVG" \
    --num_sampled_rows 64 \
    --sparsity $sparsity \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp \
    --record_attention \
    --sample_steps 50
        
# huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
# huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./Wan2.1-I2V-14B-720P
# huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./Wan2.1-I2V-14B-480P