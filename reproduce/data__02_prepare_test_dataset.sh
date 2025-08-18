#!/bin/bash

# Resolve the root of the project (1 level above this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

INPUT_PATH="datasets/webqsp-wd/input/webqsp.examples.test.wikidata.json"
OUTPUT_PATH="outputs/processed/test_2.jsonl"
LOG_PATH="logs/preprocess/prepare_test_dataset_2.log"

mkdir -p "$(dirname "$OUTPUT_PATH")" "$(dirname "$LOG_PATH")"

python experiments/datasets-preparation/02_prepare_test_dataset.py \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --log_path "$LOG_PATH"
