resolution="480p"
infer_step=40

first_times_fp=0.3
first_layers_fp=0.03
attention_backend="flashinfer"

sparsity=0.25

prompt=$(cat examples/1/prompt.txt)
image_path="examples/1/image.jpg"

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
    --model_id "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers" \
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
    --output_file "${output_dir}/${output_feature}/0-0.mp4"
