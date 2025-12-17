#! /bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$SCRIPT_DIR/../..
cd $ROOT_DIR

python	$ROOT_DIR/scripts/wan_t2v_inference.py \
    --strategy sparge \
    --height 720 \
    --width 1280 \
    --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --num_inference_steps 50 \
    --num_frames 81 \
    --warmup_step 1 \
    --output_dir $ROOT_DIR/outputs/wan1.3b/sparge/