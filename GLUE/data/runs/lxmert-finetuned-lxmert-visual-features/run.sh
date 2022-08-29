#!/usr/bin/env bash

set -eo pipefail

TASK_NAMES=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "stsb" "wnli")

MODEL_NAME="lxmert-finetuned-lxmert-visual-features"

DATA_DIR=GLUE/data
LOGDIR="${DATA_DIR}/logs/${MODEL_NAME}"
TB_LOGDIR="${DATA_DIR}/tb-logs/${MODEL_NAME}" # currently doesn't work - all tb files are put in LOGDIR
CACHE_DIR="${DATA_DIR}/cache"

echo "logdir will be: ${LOGDIR}"
mkdir -p $LOGDIR

# You must set a task ID indicating what task to run, preferably as an argument to this script
# TASK_ID=

python -m GLUE.src.benchmark_model_GLUE \
	--model-name ${MODEL_NAME} \
	--model-path "unc-nlp/lxmert-base-uncased" \
	--visual-features-path "adaptations/data/runs/finetune-visual-features/lxmert-lxmert/best_global_step_features70080.pt" \
	--visual-boxes-path "adaptations/data/runs/finetune-visual-features/lxmert-lxmert/best_global_step_boxes70080.pt" \
	--tokenizer-name bert-base-uncased \
	--task-name ${TASK_NAMES[$TASK_ID]} \
	--train-batch-size 32 \
	--eval-batch-size 64 \
	--epochs 4 \
	--lr 3e-5 \
	--weight-decay 0.01 \
	--cache-dir $CACHE_DIR \
	--logdir $LOGDIR \
	--tb-logdir $LOGDIR \
	--dataloader-num-workers 0 \
	--do-train \
	--do-eval \
	--do-predict \
