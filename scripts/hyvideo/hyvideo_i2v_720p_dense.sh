resolution="720p"
infer_step=50

prompt=$(cat examples/wan/3/prompt.txt)
image_path="examples/wan/3/image.jpg"

output_dir="result/hyvideo/i2v/dense"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

output_feature="${video_cfg}"

python hyvideo_i2v_inference.py \
    --model_id "hunyuanvideo-community/HunyuanVideo-I2V" \
    --prompt "${prompt}" \
    --image_path "${image_path}" \
    --seed 0 \
    --num_inference_steps $infer_step \
    --resolution $resolution \
    --pattern "dense" \
    --output_file "${output_dir}/${output_feature}/0-0.mp4"
