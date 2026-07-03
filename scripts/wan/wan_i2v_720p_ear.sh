resolution="720p"
infer_step=40

first_times_fp=0.35
first_layers_fp=0.03

prompt_id=1
prompt=$(cat examples/${prompt_id}/prompt.txt)
image_path="examples/${prompt_id}/image.jpg"

# SVG-EAR attention example
qc_kmeans=300
kc_kmeans=1000
top_p_k=0.9
min_kc_ratio=0.10
kmeans_iter_init=50
kmeans_iter_step=2

pattern="EAR"

output_dir="result/wan/i2v/ear"

video_cfg="Step_${infer_step}-Res_${resolution}"
dense_attention_cfg="TFP_${first_times_fp}-LFP_${first_layers_fp}"
centroid_cfg="QC_${qc_kmeans}-KC_${kc_kmeans}-TopP_${top_p_k}"
kmeans_cfg="Init_${kmeans_iter_init}-Step_${kmeans_iter_step}-MinR_${min_kc_ratio}"
output_feature="${video_cfg}/${dense_attention_cfg}/${centroid_cfg}/${kmeans_cfg}"

python wan_i2v_inference.py \
    --model_id "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers" \
    --prompt "${prompt}" \
    --image_path "${image_path}" \
    --seed 0 \
    --num_inference_steps $infer_step \
    --resolution $resolution \
    --pattern $pattern \
    --num_q_centroids $qc_kmeans \
    --num_k_centroids $kc_kmeans \
    --top_p_kmeans $top_p_k \
    --min_kc_ratio $min_kc_ratio \
    --kmeans_iter_init $kmeans_iter_init \
    --kmeans_iter_step $kmeans_iter_step \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp \
    --output_file "${output_dir}/${output_feature}/${prompt_id}-0.mp4" \
    --logging_file "${output_dir}/${output_feature}/${prompt_id}-0.jsonl"
