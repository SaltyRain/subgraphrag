#!/bin/bash
set -euo pipefail

# Resolve project root (one level above this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ---- Paths ----
DOCUMENTS_DIR="results/webqsp-wd/articles"
STORAGE_DIR="storage/naiverag"
QUESTIONS_PATH="results/webqsp-wd/test.jsonl"
RESULTS_DIR="results/exp2_baselines/naiverag"
LOG_DIR="logs/naiverag"  # run_pipeline.py creates a timestamped log inside this dir

# ---- Stage toggles ----
REBUILD_INDEX=false         # set true to force rebuild
GENERATE_ANSWERS=false       # set false to skip if answers already exist
RUN_EVALUATION=true         # set false to skip RAGAS

# ---- Retrieval / Eval params ----
SIMILARITY_TOP_K=4
# Chunk size - 1024
RAGAS_BATCH_SIZE=8
ANSWERS_FILENAME="answers.jsonl"

# ---- Flags mapping ----
REBUILD_FLAG=""
[ "$REBUILD_INDEX" = true ] && REBUILD_FLAG="--rebuild_index"

GENERATE_FLAG=""
[ "$GENERATE_ANSWERS" = true ] && GENERATE_FLAG="--generate_answers"

EVAL_FLAG=""
[ "$RUN_EVALUATION" = true ] && EVAL_FLAG="--evaluate"

# ---- Make sure result/log dirs exist ----
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ---- Run pipeline ----
python experiments/naiverag/run_pipeline.py \
  --documents_dir "$DOCUMENTS_DIR" \
  --storage_dir "$STORAGE_DIR" \
  --questions_path "$QUESTIONS_PATH" \
  --results_dir "$RESULTS_DIR" \
  --answers_filename "$ANSWERS_FILENAME" \
  --similarity_top_k "$SIMILARITY_TOP_K" \
  --ragas_batch_size "$RAGAS_BATCH_SIZE" \
  --log_dir "$LOG_DIR" \
  $REBUILD_FLAG \
  $GENERATE_FLAG \
  $EVAL_FLAG
