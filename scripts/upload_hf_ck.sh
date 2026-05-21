#!/usr/bin/env bash
# Upload checkpoints to https://huggingface.co/gwx22/AwareVLN
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="${ROOT}/.hf_ck_staging"
REPO_ID="${REPO_ID:-gwx22/AwareVLN}"

if [[ ! -d "${STAGING}/navila-llama3-8b-8f/llm" || ! -d "${STAGING}/awarevln/llm" ]]; then
  echo "Staging missing. Run: python3 scripts/upload_hf_ck.py"
  exit 1
fi

echo "Creating repo ${REPO_ID} (skip if exists)..."
hf repo create "${REPO_ID}" --type model 2>/dev/null || true

echo "Uploading ~32GB to ${REPO_ID} (resumable, commits per batch)..."
hf upload-large-folder "${REPO_ID}" "${STAGING}" \
  --repo-type model \
  --num-workers 4

echo "Done: https://huggingface.co/${REPO_ID}"
