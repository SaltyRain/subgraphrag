#!/bin/bash

# Resolve the root of the project (1 level above this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

INPUT_PATH="datasets/webqsp-wd/input/webqsp.examples.train.json"
OUTPUT_PATH="outputs/processed/train.jsonl"
MAPPER_PATH="resources/wikimapper/index_enwiki-20190420.db"
INTERMEDIATE_DIR="intermediate/train/"
LOG_PATH="logs/preprocess/prepare_train_dataset.log"

mkdir -p "$(dirname "$OUTPUT_PATH")" "$(dirname "$LOG_PATH")" "$INTERMEDIATE_DIR"

python experiments/datasets-preparation/01_prepare_train_dataset.py \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --mapper_path "$MAPPER_PATH" \
  --intermediate_dir "$INTERMEDIATE_DIR" \
  --log_path "$LOG_PATH"
