#!/bin/bash
set -euo pipefail

# Resolve project root (one level above this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ----------- Config (edit as needed) -----------
STORAGE_DIR="storage/lightrag"
QUESTIONS_PATH="outputs/processed/test.jsonl"

# Each mode dir can have its own outputs
MODE="hybrid"   # one of: local | global | hybrid | naive | mix | bypass
OUTPUT_DIR="results/exp2_baselines/lightrag/${MODE}"

# Files inside OUTPUT_DIR
ANSWERS_FILENAME="answers.jsonl"
EVAL_FILENAME="ragas_eval.json"

# Logging
LOG_DIR="logs/lightrag/pipeline"

# Pipeline toggles
GENERATE_ANSWERS=false
EVALUATE=true

# RAGAS / Ollama settings
RAGAS_BATCH_SIZE=8
# If you want to fix judge model (otherwise ragas default LLM is used):
JUDGE_MODEL="${LLM_MODEL_NAME:-}"          # e.g., "llama3.1" or leave empty
EMBED_MODEL="${LLM_EMBEDDING_MODEL_NAME:-nomic-embed-text}"
OLLAMA_HOST="${LLM_BINDING_HOST:-http://localhost:11434}"
# ----------------------------------------------

# Build flags
GEN_ANS_FLAG=""
if [ "${GENERATE_ANSWERS}" = true ]; then
  GEN_ANS_FLAG="--generate_answers"
fi

EVAL_FLAG=""
if [ "${EVALUATE}" = true ]; then
  EVAL_FLAG="--evaluate"
fi

JUDGE_FLAG=()
if [ -n "${JUDGE_MODEL}" ]; then
  JUDGE_FLAG=( --judge_model "${JUDGE_MODEL}" )
fi

python experiments/lightrag/02_run_pipeline.py \
  --storage_dir "${STORAGE_DIR}" \
  --questions_path "${QUESTIONS_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --answers_filename "${ANSWERS_FILENAME}" \
  --eval_filename "${EVAL_FILENAME}" \
  --log_dir "${LOG_DIR}" \
  --mode "${MODE}" \
  --ragas_batch_size "${RAGAS_BATCH_SIZE}" \
  --embed_model "${EMBED_MODEL}" \
  --ollama_host "${OLLAMA_HOST}" \
  "${JUDGE_FLAG[@]}" \
  ${GEN_ANS_FLAG} \
  ${EVAL_FLAG}
