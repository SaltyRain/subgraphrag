#!/bin/bash

# Resolve the root of the project (1 level above this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

INPUT_PATH="outputs/processed/test.jsonl"
OUTPUT_DIR="outputs/processed/articles"
MAPPER_PATH="resources/wikimapper/index_enwiki-20190420.db"
INTERMEDIATE_DIR="intermediate/fetch_test_wiki_articles/"
LOG_PATH="logs/preprocess/fetch_test_wiki_articles.log"

mkdir -p "$(dirname "$OUTPUT_PATH")" "$(dirname "$LOG_PATH")" "$INTERMEDIATE_DIR"

python experiments/datasets-preparation/03_fetch_test_wiki_articles.py \
  --input_path "$INPUT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --mapper_path "$MAPPER_PATH" \
  --intermediate_dir "$INTERMEDIATE_DIR" \
  --log_path "$LOG_PATH"

