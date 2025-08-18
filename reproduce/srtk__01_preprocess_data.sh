#!/bin/bash

# Resolve the root of the project (1 level above this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

INPUT_PATH="results/webqsp-wd/train.jsonl"
OUTPUT_PATH="results/exp1_preprocess/jaccard-30-04/train.jsonl"
INTERMEDIATE_DIR="results/exp1_preprocess/jaccard-30-04/intermediate"
SPARQL_ENDPOINT="http://localhost:1234/api/endpoint/sparql"
KNOWLEDGE_GRAPH="wikidata"

# Variables
METRIC="recall"
NUM_NEGATIVE=30
POSITIVE_THRESHOLD=0.4

# Observations and experiments:
# Experiment 1: default values -metric jaccard, -n num-negative 15, -p positive-threshold 0.5
# Experiment 2: -metric recall, -n num-negative 15, -p positive-threshold 0.5
# INTERMEDIATE RESULTS: jaccard returned less noisy graphs, recall returned higher 'recall' but more noisy graphs. we take jaccard and trying to optimise it further.
# Experiment 3: -metric jaccard, -n num-negative 30, -p positive-threshold 0.4


# DOCS: https://github.com/yuancu/subgraph-retrieval-toolkit/tree/docs?tab=readme-ov-file#train-a-retriever
srtk preprocess \
  -i "$INPUT_PATH" \
  -o "$OUTPUT_PATH" \
  --intermediate-dir "$INTERMEDIATE_DIR" \
  -e "$SPARQL_ENDPOINT" \
  -kg "$KNOWLEDGE_GRAPH" \
  --search-path \
  --metric "$METRIC" \
  --num-negative "$NUM_NEGATIVE" \
  --positive-threshold "$POSITIVE_THRESHOLD" \
  --omit-prefixes

