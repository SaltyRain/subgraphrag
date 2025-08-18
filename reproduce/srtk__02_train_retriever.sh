#!/bin/bash

# Resolve the root of the project (1 level above this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

INPUT_PATH="results/exp1_preprocess/jaccard-30-04/train.jsonl"
OUTPUT_DIR="results/exp1_preprocess/jaccard-30-04/roberta_bs32_lr2e-5_ce_epoch4/scorer"
LOG_PATH="logs/train/jaccard-30-04-roberta_bs32_lr2e-5_ce_epoch4.log"


#VARIABLES
MODEL_NAME_OR_PATH="FacebookAI/roberta-base" # intfloat/e5-small, FacebookAI/roberta-base
LOSS="cross_entropy" # cross_entropy, contrastive
BATCH_SIZE=16
LEARNING_RATE=2e-5
MAX_EPOCHS=4

# Experiment 1 part 2: SETUP jaccard-15-05 / recall-15-05,
# emb.model: FacebookAI/roberta-base, batch-size 16, learning-rate: 2e-5, loss: cross_entropy, max-epochs: 10

# DOCS: https://github.com/yuancu/subgraph-retrieval-toolkit/tree/docs?tab=readme-ov-file#train-a-retriever
srtk train \
  --train-dataset "$INPUT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --model-name-or-path "$MODEL_NAME_OR_PATH" \
  --loss "$LOSS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --max-epochs "$MAX_EPOCHS" \
  --accelerator gpu \
  --wandb-project subgraphrag \
  --wandb-group roberta \
  | tee "$LOG_PATH"