#!/bin/bash

set -e
# Resolve the root of the project (1 level above this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

INPUT_PATH="results/exp1_e2e/r-balanced.jsonl"
INTERMEDIATE_DIR="intermediate/subgraphrag/update_labels"
LOG_DIR="logs/subgraphrag/update_labels"

LABELS_MAP_PATH="resources/subgraphrag/labels_map.json"

python experiments/subgraphrag/00_update_labels_map.py \
  --input_path "$INPUT_PATH" \
  --intermediate_dir "$INTERMEDIATE_DIR" \
  --log_dir "$LOG_DIR" \
  --labels_map_path "$LABELS_MAP_PATH"
