resolution="720p"
infer_step=35

prompt_id=2

prompt=$(cat examples/${prompt_id}/prompt.txt)

output_dir="result/cosmos/t2v/dense"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

output_feature="${video_cfg}"

python cosmos_t2v_inference.py \
    --model_id "nvidia/Cosmos-1.0-Diffusion-14B-Text2World" \
    --prompt "${prompt}" \
    --height 704 \
    --width 1280 \
    --seed 0 \
    --num_inference_steps $infer_step \
    --pattern "dense" \
    --output_file "${output_dir}/${output_feature}/${prompt_id}-0.mp4"
