#!/usr/bin/env bash
# Build the EGT neighbor map: K = 15 nearest neighbors per training problem, self excluded.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
NUM_NEIGHBORS=15
OUT=data/egt/qwen3_1_7b_thinking/neighbors.json
if [ -e "$OUT" ]; then
  echo "Refusing to overwrite existing neighbor map: $OUT" >&2
  exit 1
fi
exec "${PYTHON:-python3}" scripts/build_similar_questions.py \
  --train-parquet data/verl/deepmath_filtered_level5_7/train.parquet \
  --model "${EGT_EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}" \
  --out "$OUT" --num-neighbors "$NUM_NEIGHBORS" \
  --devices "${EGT_EMBED_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
