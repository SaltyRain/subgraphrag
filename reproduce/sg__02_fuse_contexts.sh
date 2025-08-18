#!/bin/bash
set -euo pipefail

# Resolve project root (one level above this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ---------------- Inputs/outputs ----------------
LOG_DIR="logs/subgraphrag/fuse_contexts"

# ---------------- Profile knobs -----------------
# Balanced (compact): keep fewer heads, shorter target
# By default: fuse contexts for R-balanced
#INPUT_PATH="results/exp1_e2e/subgraphs/r-balanced.jsonl"
#OUTPUT_PATH="results/exp1_e2e/fused/r-balanced.jsonl"
#MAX_HEADS=6          # number of head snippets per question (None -> drop the flag)
#MAX_WORDS=120        # target word budget for the fused context
#TIMEOUT=60           # LLM request timeout (seconds)

# --- Alternative: R-high (uncomment to switch) ---
 INPUT_PATH="results/exp1_e2e/subgraphs/r-high.jsonl"
 OUTPUT_PATH="results/exp1_e2e/fused/r-high.jsonl"
 MAX_HEADS=10
 MAX_WORDS=140
 TIMEOUT=90

# ---------------- Prep dirs ---------------------
mkdir -p "$(dirname "$OUTPUT_PATH")" "$LOG_DIR"

# ---------------- Run ---------------------------
python experiments/subgraphrag/02_fuse_contexts.py \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --log_dir "$LOG_DIR" \
  --max_heads "$MAX_HEADS" \
  --max_words "$MAX_WORDS" \
  --timeout "$TIMEOUT"
