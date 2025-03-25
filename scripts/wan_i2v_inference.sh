# Description: This script demonstrates how to inference a video based on HunyuanVideo model
if [ $# -eq 0 ]; then
    echo "Please provide the CUDA device number as an argument."
    echo "Usage: $0 <cuda_device_number>"
    exit 1
fi

# for file_idx in $(seq 1 30); do # $(seq 1 10); do
#     CUDA_VISIBLE_DEVICES=$1 python wan_i2v_inference.py \
#         --data_path "/ssd/data/xihaocheng/CropVBench/data" \
#         --file_idx $file_idx \
#         --height 720 \
#         --width 1280
# done

# GPUS=(0)
GPUS=(0 1 2)
MAX_JOBS=${#GPUS[@]}
pids=()
task_idx=0

for file_idx in $(seq 1 30); do
    GPU_ID=${GPUS[$((task_idx % MAX_JOBS))]}
    CUDA_VISIBLE_DEVICES=$GPU_ID python wan_i2v_inference.py \
        --data_path "/ssd/data/xihaocheng/CropVBench/data" \
        --file_idx $file_idx \
        --height 720 \
        --width 1280 \
        --num_inference_steps 40 &
    pids+=($!)

    ((task_idx++))

    if [[ ${#pids[@]} -ge $MAX_JOBS ]]; then
        wait "${pids[0]}"
        unset pids[0]
        pids=("${pids[@]}")
    fi
done