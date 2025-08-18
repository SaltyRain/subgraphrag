#!/bin/bash
set -euo pipefail

# --- neutralize any JSON forcing in env ---
unset OLLAMA_FORMAT
unset OLLAMA_OUTPUT_FORMAT
unset LLM_FORCE_JSON
unset FORMAT

# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

INPUT_PATH="results/exp1_e2e/answers/r-high.jsonl"
OUTPUT_DIR="results/exp1_e2e/eval/r-high"

HOST="http://172.21.112.1:11434"
LLM_MODEL="llama3.1"
EMBED_MODEL="nomic-embed-text"

BATCH_SIZE=12
INCLUDE_RELEVANCY=false
USE_FIRST_GOLD=false
FORCE_JSON=false
NO_EMBEDDINGS=false

mkdir -p "$OUTPUT_DIR"

python experiments/subgraphrag/04_eval.py \
  --input_path "$INPUT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --host "$HOST" \
  --llm_model "$LLM_MODEL" \
  --embed_model "$EMBED_MODEL" \
  --batch_size $BATCH_SIZE \
  $([ "$INCLUDE_RELEVANCY" = true ] && echo "--include_relevancy") \
  $([ "$USE_FIRST_GOLD" = true ] && echo "--use_first_gold") \
  $([ "$FORCE_JSON" = true ] && echo "--force_json_mode") \
  $([ "$NO_EMBEDDINGS" = true ] && echo "--no_embeddings")
