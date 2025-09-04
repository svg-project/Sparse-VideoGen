resolution="720p"
infer_step=50

prompt="A cat walks on the grass, realistic"
output_dir="result/hyvideo/t2v/dense"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

output_feature="${video_cfg}"

python hyvideo_t2v_inference.py \
    --model_id "tencent/HunyuanVideo" \
    --prompt "${prompt}" \
    --seed 0 \
    --num_inference_steps $infer_step \
    --resolution $resolution \
    --pattern "dense" \
    --output_file "${output_dir}/${output_feature}/0-0.mp4"
