#!/bin/bash

# Resolve the root of the project (1 level above this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

INPUT_DIR="outputs/processed/articles"
WORKING_DIR="storage/lightrag"

python experiments/lightrag/01_build_index.py \
  --working_dir "$WORKING_DIR" \
  --input_dir "$INPUT_DIR"
