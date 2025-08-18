#!/bin/bash

set -euo pipefail

# Resolve the root of the project (1 level above this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ---- Inputs/outputs -------------------------------------------------------
INPUT_PATH="results/exp1_e2e/r-balanced.jsonl"
OUTPUT_PATH="results/exp1_e2e/subgraphs/r-balanced.jsonl"
LABELS_MAP_PATH="resources/subgraphrag/labels_map.json"

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_PATH")"

# ---- Pre-reranking profile -----------------------------------------------
 TOP_K=8
 FRAC_CUTOFF=0.70
 W_SEM=0.55
 W_SEED=0.20
 W_INFO=0.25
 CAP_PER_SUBJECT=2
 TOP_PATHS_PER_HEAD=4
 LAM_FANOUT=0.3



# ---- Run ------------------------------------------------------------------
python experiments/subgraphrag/01_construct_subgraphs.py \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --labels_map_path "$LABELS_MAP_PATH" \
  --top-k "$TOP_K" \
  --frac-cutoff "$FRAC_CUTOFF" \
  --w-sem "$W_SEM" \
  --w-seed "$W_SEED" \
  --w-info "$W_INFO" \
  --cap-per-subject "$CAP_PER_SUBJECT" \
  --top-paths-per-head "$TOP_PATHS_PER_HEAD" \
  --lam-fanout "$LAM_FANOUT" \
  --explosive-preds "${EXPLOSIVE_PREDS[@]}"
