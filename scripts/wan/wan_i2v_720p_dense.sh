resolution="720p"
infer_step=40

prompt=$(cat examples/1/prompt.txt)
image_path="examples/1/image.jpg"

output_dir="result/wan/i2v/dense"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

output_feature="${video_cfg}"

python wan_i2v_inference.py \
    --model_id "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers" \
    --prompt "${prompt}" \
    --image_path "${image_path}" \
    --seed 0 \
    --num_inference_steps $infer_step \
    --resolution $resolution \
    --pattern "dense" \
    --output_file "${output_dir}/${output_feature}/0-0.mp4"
