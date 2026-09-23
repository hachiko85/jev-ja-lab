#!/usr/bin/env bash
# Trains one Linear(hidden,1) head per (model, primitive), then runs the
# embedding method's full noul/choice/score/summary evaluation.
# Idempotent: skips a head file that already exists, and orchestrate.py's
# resume:true skips a dataset that already has a summary.json.
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS=(
  "e5-small:intfloat/multilingual-e5-small"
  "ruri-v3-310m:cl-nagoya/ruri-v3-310m"
  "ruri-v3-130m:cl-nagoya/ruri-v3-130m"
  "ruri-v3-70m:cl-nagoya/ruri-v3-70m"
  "ruri-v3-30m:cl-nagoya/ruri-v3-30m"
  "bekko-v1-a8m:hotchpotch/bekko-embedding-v1-a8m"
  "bekko-v1-a25m:hotchpotch/bekko-embedding-v1-a25m"
)
PRIMITIVES=(choice score noul)

echo "=== [1/3] training decision heads ==="
for entry in "${MODELS[@]}"; do
  short="${entry%%:*}"
  repo="${entry##*:}"
  for primitive in "${PRIMITIVES[@]}"; do
    output="results/embedding_heads/${short}/${primitive}.pt"
    if [ -f "$output" ]; then
      echo "skip (head exists): $output"
      continue
    fi
    echo "--- train ${short} (${repo}) / ${primitive} ---"
    if ! uv run jev-ja-lab-embedding-train \
      --model "$repo" \
      --primitive "$primitive" \
      --datasets-root datasets \
      --output "$output" \
      --limit 2000 \
      --epochs 5 \
      --val-fraction 0.1 \
      --device cuda:0 \
      --dtype bfloat16; then
      echo "!!! training FAILED for ${short}/${primitive}, continuing with remaining jobs !!!"
    fi
  done
done

echo "=== [2/3] smoke evaluation ==="
uv run jev-ja-lab-eval-workflow --config configs/eval/embedding-series.yaml --phase smoke
smoke_status=$?
if [ $smoke_status -ne 0 ]; then
  echo "!!! smoke phase FAILED (exit $smoke_status); aborting before production !!!"
  exit $smoke_status
fi

echo "=== [3/3] production evaluation ==="
uv run jev-ja-lab-eval-workflow --config configs/eval/embedding-series.yaml --phase production
echo "=== embedding production run finished (exit $?) ==="
