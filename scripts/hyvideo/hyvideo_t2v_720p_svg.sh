resolution="720p"
infer_step=50

first_times_fp=0.1
first_layers_fp=0.03

sparsity=0.25

prompt_id=7
prompt=$(cat examples/${prompt_id}/prompt.txt)


output_dir="result/hyvideo/t2v/svg"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

# Dense Attention Cfg
dense_attention_cfg="TFP_${first_times_fp}-LFP_${first_layers_fp}"

# Sparsity Cfg
sparsity_cfg="Sparsity_${sparsity}"

# Output feature
output_feature="${video_cfg}/${dense_attention_cfg}/${sparsity_cfg}"

python hyvideo_t2v_inference.py \
    --model_id "tencent/HunyuanVideo" \
    --seed 0 \
    --height 720 \
    --width 1280 \
    --prompt "${prompt}" \
    --num_inference_steps $infer_step \
    --resolution $resolution \
    --output_file "${output_dir}/${output_feature}/${prompt_id}-0.mp4" \
    --pattern "SVG" \
    --num_sampled_rows 64 \
    --sparsity $sparsity \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp
