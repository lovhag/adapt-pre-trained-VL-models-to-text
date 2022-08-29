#!/usr/bin/env bash
set -eo pipefail

# Note: This script is assumed to be run from `adaptations/src`

# you need to set the DATA_DIR path, in which the training data is stored
DATA_DIR=../../data

# you also need to set the CHECKPOINT_DIR, to which the checkpoints should be saved
# CHECKPOINT_DIR=

python finetune_visual_features.py \
    --text-dataset $DATA_DIR/datasets/finetune/lxmert/{train,val}.jsonl \
    --evaluate-every 40 \
    --checkpoint-every 40 \
    --checkpoint-max 1 \
    --checkpoint-dir $CHECKPOINT_DIR/clip-bert-lxmert-middle-lr \
    --model "clip-bert" \
    --bert-checkpoint "../../models/data/model-weights/clip-bert/mp_rank_00_model_states.pt" \
    --batch-size 64 \
    --tensorboard-logdir "../data/runs/tensorboard/finetune-visual-features-clip-bert-lxmert-middle-lr" \
    --lr 0.005
