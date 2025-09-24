resolution="720p"
infer_step=50

prompt_id=1

prompt=$(cat examples/${prompt_id}/prompt.txt)

output_dir="result/wan/t2v/dense"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

output_feature="${video_cfg}"

python wan_t2v_inference.py \
    --model_id "Wan-AI/Wan2.1-T2V-14B-Diffusers" \
    --prompt "${prompt}" \
    --height 720 \
    --width 1280 \
    --seed 0 \
    --num_inference_steps $infer_step \
    --pattern "dense" \
    --output_file "${output_dir}/${output_feature}/${prompt_id}-0.mp4"
