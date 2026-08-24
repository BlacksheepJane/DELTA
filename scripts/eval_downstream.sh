#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 CHECKPOINT_DIR OUTPUT_DIR [GPU_IDS] [delta.evaluate options]" >&2
  exit 2
fi

CHECKPOINT_DIR="$1"
OUTPUT_DIR="$2"
GPU_IDS="${3:-0}"
if [[ "$#" -ge 3 ]]; then
  shift 3
else
  shift 2
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"

TASKS=(openbookqa winogrande hellaswag arc_easy piqa mathqa)
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_IDS must not be empty" >&2
  exit 2
fi

PIDS=()
for INDEX in "${!TASKS[@]}"; do
  TASK="${TASKS[$INDEX]}"
  GPU="${GPUS[$((INDEX % ${#GPUS[@]}))]}"
  CUDA_VISIBLE_DEVICES="$GPU" python -m delta.evaluate \
    --checkpoint "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mode downstream \
    --tasks "$TASK" \
    --device cuda:0 \
    "$@" \
    >"$OUTPUT_DIR/${TASK}.log" 2>&1 &
  PIDS+=("$!")

  if [[ "${#PIDS[@]}" -ge "${#GPUS[@]}" ]]; then
    for PID in "${PIDS[@]}"; do
      wait "$PID"
    done
    PIDS=()
  fi
done

for PID in "${PIDS[@]}"; do
  wait "$PID"
done

echo "Downstream evaluation finished: $OUTPUT_DIR"
