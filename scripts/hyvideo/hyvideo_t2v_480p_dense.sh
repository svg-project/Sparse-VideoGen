resolution="480p"
infer_step=50

prompt_id=7
prompt=$(cat examples/${prompt_id}/prompt.txt)

output_dir="result/hyvideo/t2v/dense"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

output_feature="${video_cfg}"

python hyvideo_t2v_inference.py \
    --model_id "tencent/HunyuanVideo" \
    --seed 0 \
    --height 480 \
    --width 720 \
    --prompt "${prompt}" \
    --num_inference_steps $infer_step \
    --resolution $resolution \
    --pattern "dense" \
    --output_file "${output_dir}/${output_feature}/${prompt_id}-0.mp4"
