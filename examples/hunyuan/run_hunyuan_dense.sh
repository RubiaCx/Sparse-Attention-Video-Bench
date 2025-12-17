#! /bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$SCRIPT_DIR/../..
cd $ROOT_DIR

python $ROOT_DIR/scripts/hunyuan_t2v_inference.py  \
    --strategy dense \
    --height 720 \
    --width 1280 \
    --num_inference_steps 50 \
    --num_frames 129 \
    --warmup_step 1 \
    --output_dir $ROOT_DIR/outputs/hunyuan/dense/
    