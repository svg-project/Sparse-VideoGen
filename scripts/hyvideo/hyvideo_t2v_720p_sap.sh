resolution="720p"
infer_step=50

first_times_fp=0.1
first_layers_fp=0.03

prompt_id=7
prompt=$(cat examples/${prompt_id}/prompt.txt)

# Semantic Aware Permutation Attention Example
# Define K-means specific parameters
qc_kmeans=400
kc_kmeans=1000
top_p_k=0.9
min_kc_ratio=0.10
kmeans_iter_init=50
kmeans_iter_step=2

pattern="SAP"
# pattern="dense"

output_dir="result/hyvideo/t2v/sap"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

# Dense Attention Cfg
dense_attention_cfg="TFP_${first_times_fp}-LFP_${first_layers_fp}"

# Centroid cfg
centroid_cfg="QC_${qc_kmeans}-KC_${kc_kmeans}-TopP_${top_p_k}"

# KMeans Cfg
kmeans_cfg="Init_${kmeans_iter_init}-Step_${kmeans_iter_step}-MinR_${min_kc_ratio}"

# Output feature
output_feature="${video_cfg}/${dense_attention_cfg}/${centroid_cfg}/${kmeans_cfg}"


python hyvideo_t2v_inference.py \
    --model_id "tencent/HunyuanVideo" \
    --seed 0 \
    --height 720 \
    --width 1280 \
    --prompt "${prompt}" \
    --num_inference_steps $infer_step \
    --resolution $resolution \
    --output_file "${output_dir}/${output_feature}/${prompt_id}-0.mp4" \
    --logging_file "${output_dir}/${output_feature}/${prompt_id}-0.jsonl" \
    --pattern $pattern \
    --num_q_centroids $qc_kmeans \
    --num_k_centroids $kc_kmeans \
    --top_p_kmeans $top_p_k \
    --min_kc_ratio $min_kc_ratio \
    --kmeans_iter_init $kmeans_iter_init \
    --kmeans_iter_step $kmeans_iter_step \
    --zero_step_kmeans_init \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp

