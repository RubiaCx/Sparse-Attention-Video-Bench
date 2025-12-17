#! /bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$SCRIPT_DIR/../..
cd $ROOT_DIR


export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun \
    --standalone \
	--nproc_per_node 8 \
	$ROOT_DIR/scripts/wan_t2v_inference.py \
    --strategy usp \
    --ulysses-size 1 \
    --ring-size 8 \
    --height 720 \
    --width 1280 \
    --model_id Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --num_inference_steps 50 \
    --num_frames 81 \
    --warmup_step 1 \
    --output_dir $ROOT_DIR/outputs/wan14b/ring/ \
    # --profile