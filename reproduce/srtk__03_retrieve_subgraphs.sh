#!/bin/bash

# Resolve the root of the project (1 level above this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

INPUT_PATH="results/webqsp-wd/test.jsonl"
SCORER_MODEL_PATH="results/exp1_preprocess/jaccard-30-04/roberta_bs32_lr2e-5_ce_epoch4/scorer"
#SCORER_MODEL_PATH="drt/srtk-scorer" # framework-related model (HF) can be used out-of-the-box
OUTPUT_PATH="results/exp1_preprocess/jaccard-30-04/roberta_bs32_lr2e-5_ce_epoch4/bw10_md2.jsonl"



ENDPOINT_URL="http://localhost:1234/api/endpoint/sparql"

srtk retrieve \
    -i "$INPUT_PATH" \
    -o "$OUTPUT_PATH" \
    -kg wikidata \
    -e "$ENDPOINT_URL" \
    -m "$SCORER_MODEL_PATH" \
    --beam-width 10 \
    --max-depth 2 \
    --evaluate \
    --omit-prefixes

