#!/usr/bin/env bash
set -eo pipefail

# Note: This script is assumed to be run from `adaptations/src`

# Generate hostfile
NUM_GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
cat $TMPDIR/mpichnodes | sort | uniq | awk "{print \$0, \"slots=$NUM_GPUS_PER_NODE\"}" > $TMPDIR/hostfile

# For NCCL to with Infiniband...
export NCCL_IB_GID_INDEX=3

# Create environment file for CUDA to work
echo "CUDA_HOME=/apps/Common/Core/CUDAcore/11.1.1" > $TMPDIR/.deepspeed_env
echo "PATH=$PATH" >> $TMPDIR/.deepspeed_env
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> $TMPDIR/.deepspeed_env
echo "CC=gcc" >> $TMPDIR/.deepspeed_env
echo "CXX=g++" >> $TMPDIR/.deepspeed_env

# you need to set the DATA_DIR path, in which the training data is stored
DATA_DIR=../../data

# you also need to set the CHECKPOINT_DIR, to which the checkpoints should be saved
# CHECKPOINT_DIR=

HOME=$TMPDIR deepspeed --hostfile $TMPDIR/hostfile finetune.py \
    --text-dataset $DATA_DIR/lxmert/finetune/{train,val}.jsonl \
    --evaluate-every 40 \
    --checkpoint-every 40 \
    --checkpoint-max 1 \
    --checkpoint-dir $CHECKPOINT_DIR/lxmert-lxmert \
    --deepspeed_config ../data/runs/finetune/lxmert-lxmert/deepspeed.config \
    --model "lxmert"
