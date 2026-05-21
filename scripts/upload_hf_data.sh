#!/usr/bin/env bash
# Upload .hf_data to https://huggingface.co/datasets/gwx22/AwareVLN
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ROOT}/.hf_data"
REPO_ID="${REPO_ID:-gwx22/AwareVLN}"

if [[ ! -d "${DATA}/reason/r2r/_anno_cot" ]]; then
  echo "Run first: python3 scripts/package_hf_data.py"
  exit 1
fi

echo "Uploading reason/ (+ README) -> ${REPO_ID} (dataset, resumable)..."
echo "Excluding Human/"
hf upload-large-folder "${REPO_ID}" "${DATA}" \
  --repo-type dataset \
  --num-workers 1 \
  --exclude "Human/**"

echo "Done: https://huggingface.co/datasets/${REPO_ID}"
