resolution="720p"
infer_step=50

first_times_fp=0.3
first_layers_fp=0.03

sparsity=0.25

prompt=$(cat examples/1/prompt.txt)

output_dir="result/wan/t2v/svg"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

# Dense Attention Cfg
dense_attention_cfg="TFP_${first_times_fp}-LFP_${first_layers_fp}"

# Sparsity Cfg
sparsity_cfg="Sparsity_${sparsity}"

# Output feature
output_feature="${video_cfg}/${dense_attention_cfg}/${sparsity_cfg}"

python wan_t2v_inference.py \
    --model_id "Wan-AI/Wan2.1-T2V-14B-Diffusers" \
    --prompt "${prompt}" \
    --height 720 \
    --width 1280 \
    --seed 0 \
    --num_inference_steps $infer_step \
    --pattern "SVG" \
    --num_sampled_rows 64 \
    --sparsity $sparsity \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp \
    --output_file "${output_dir}/${output_feature}/0-0.mp4"
