#!/usr/bin/env bash
set -eo pipefail

# Note: This script is assumed to be run from `adaptations/src`

# you need to set the DATA_DIR path, in which the training data is stored
DATA_DIR=../../data

# you also need to set the CHECKPOINT_DIR, to which the checkpoints should be saved
# CHECKPOINT_DIR=

python finetune_visual_features.py \
    --text-dataset $DATA_DIR/datasets/finetune/wikipedia/{train,val}.jsonl \
    --evaluate-every 40 \
    --checkpoint-every 40 \
    --checkpoint-max 1 \
    --checkpoint-dir $CHECKPOINT_DIR/clip-bert-wikipedia \
    --model "clip-bert" \
    --bert-checkpoint "../../models/data/model-weights/clip-bert/mp_rank_00_model_states.pt" \
    --batch-size 32 \
    --tensorboard-logdir "../data/runs/tensorboard/finetune-visual-features-clip-bert-wikipedia" \
    --lr 0.005
