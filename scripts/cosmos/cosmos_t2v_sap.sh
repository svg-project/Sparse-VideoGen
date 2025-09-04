resolution="720p"
infer_step=35

first_times_fp=0.3
first_layers_fp=0.03

prompt=$(cat examples/3/prompt.txt)

# KMEANS_BLOCK Attention Example
# Define K-means specific parameters
qc_kmeans=100
kc_kmeans=100
top_p_k=0.9
min_kc_ratio=0.10
kmeans_iter_init=50
kmeans_iter_step=2

pattern="SAP"
# pattern="dense"

output_dir="result/cosmos/t2v/sap"

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

python cosmos_t2v_inference.py \
    --model_id "nvidia/Cosmos-1.0-Diffusion-14B-Text2World" \
    --prompt "${prompt}" \
    --height 704 \
    --width 1280 \
    --seed 0 \
    --num_inference_steps $infer_step \
    --pattern $pattern \
    --num_q_centroids $qc_kmeans \
    --num_k_centroids $kc_kmeans \
    --top_p_kmeans $top_p_k \
    --min_kc_ratio $min_kc_ratio \
    --kmeans_iter_init $kmeans_iter_init \
    --kmeans_iter_step $kmeans_iter_step \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp \
    --output_file "${output_dir}/${output_feature}/2-0.mp4" \
    --logging_file "${output_dir}/${output_feature}/2-0.jsonl"