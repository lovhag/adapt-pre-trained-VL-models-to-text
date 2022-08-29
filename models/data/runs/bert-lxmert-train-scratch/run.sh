#!/usr/bin/env bash
set -eo pipefail

# Note: This script is assumed to be run from `models/src`

# activate your environment with the modules as specified in `requirements.txt`
source ../../venv/bin/activate

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

DATA_DIR=${PROJECT_ROOT}/data/lxmert

HOME=$TMPDIR deepspeed --hostfile $TMPDIR/hostfile pretrain.py \
    --text-dataset $DATA_DIR/{train_mlm,val_mlm}.jsonl \
    --evaluate-every 1000 \
    --checkpoint-every 1000 \
    --checkpoint-max 1 \
    --checkpoint-dir ../data/runs/bert-lxmert-train-scratch \
    --deepspeed_config ../data/runs/bert-lxmert-train-scratch/deepspeed.config \
    --no-pretrained-weights
