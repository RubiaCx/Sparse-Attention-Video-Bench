#! /bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$SCRIPT_DIR/../..
cd $ROOT_DIR

python $ROOT_DIR/scripts/wan_t2v_inference.py \
    --strategy sap \
    --height 720 \
    --width 1280 \
    --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --num_inference_steps 50 \
    --num_frames 81 \
    --warmup_step 1 \
    --num_q_centroids 300 \
    --num_k_centroids 1000 \
    --top_p_kmeans 0.9 \
    --min_kc_ratio 0.10 \
    --kmeans_iter_init 50 \
    --kmeans_iter_step 2 \
    --zero_step_kmeans_init false \
    --first_times_fp 0.2 \
    --first_layers_fp 0.03 \
    --output_dir $ROOT_DIR/outputs/wan1.3b/sap/


