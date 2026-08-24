#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 CHECKPOINT_DIR OUTPUT_DIR [delta.evaluate options]" >&2
  exit 2
fi

CHECKPOINT_DIR="$1"
OUTPUT_DIR="$2"
shift 2
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
exec python -m delta.evaluate \
  --checkpoint "$CHECKPOINT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --mode ppl \
  "$@"
