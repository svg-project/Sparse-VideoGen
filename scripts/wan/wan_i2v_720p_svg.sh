resolution="720p"
infer_step=40

first_times_fp=0.0
first_layers_fp=0.03
attention_backend="flashinfer"

sparsity=0.05

prompt_id=6

prompt=$(cat examples/${prompt_id}/prompt.txt)
image_path="examples/${prompt_id}/image.jpg"

output_dir="result/wan/i2v/svg"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

# Dense Attention Cfg
dense_attention_cfg="TFP_${first_times_fp}-LFP_${first_layers_fp}"

# Sparsity Cfg
sparsity_cfg="Sparsity_${sparsity}"

# Output feature
output_feature="${video_cfg}/${dense_attention_cfg}/${sparsity_cfg}"

python wan_i2v_inference.py \
    --model_id "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers" \
    --prompt "${prompt}" \
    --image_path "${image_path}" \
    --seed 0 \
    --num_inference_steps $infer_step \
    --resolution $resolution \
    --pattern "SVG" \
    --num_sampled_rows 64 \
    --sparsity $sparsity \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp \
    --attention_backend $attention_backend \
    --output_file "${output_dir}/${output_feature}/${prompt_id}-0.mp4"
