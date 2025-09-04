resolution="720p"
infer_step=50

# Prompt Source
PROMPT_SOURCE="T2V_Hyv_VBench"
prompt_idx=102
prompt="data/hyv_augmented_vbench.txt"

output_dir="result/hyvideo/t2v_${PROMPT_SOURCE}/dense"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

output_feature="${video_cfg}"

python hyvideo_t2v_inference.py \
    --model_id "tencent/HunyuanVideo" \
    --seed 0 \
    --height 720 \
    --width 1280 \
    --prompt_source "${PROMPT_SOURCE}" \
    --prompt_idx "${prompt_idx}" \
    --prompt "${prompt}" \
    --num_inference_steps $infer_step \
    --resolution $resolution \
    --pattern "dense" \
    --output_file "${output_dir}/${output_feature}/${prompt_idx}.mp4"
