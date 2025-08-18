#!/bin/bash
set -euo pipefail

# Resolve the root of the project (1 level above this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ============================================================================
# Run final answer generation from fused query-focused contexts
# ============================================================================

# Paths
INPUT_PATH="results/exp1_e2e/fused/r-high.jsonl"   # or switch to r-balanced.jsonl
OUTPUT_PATH="results/exp1_e2e/answers/r-high.jsonl"
LOG_DIR="logs/subgraphrag/generate_answers"

# Timeout for each LLM call (seconds)
TIMEOUT=60

# Create output dir if needed
mkdir -p "$(dirname "$OUTPUT_PATH")"
mkdir -p "$LOG_DIR"

# Run
python experiments/subgraphrag/03_generate_precise_answers.py \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --log_dir "$LOG_DIR" \
  --timeout $TIMEOUT
