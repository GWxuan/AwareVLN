#!/usr/bin/env bash
# Package training data to .hf_data, then upload to Hugging Face dataset repo.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! ps -p "${PACKAGE_PID:-}" >/dev/null 2>&1; then
  echo "=== Packaging training data -> .hf_data ==="
  PYTHONUNBUFFERED=1 python3 scripts/package_hf_data.py 2>&1 | tee .hf_data_package.log
fi

if pgrep -f "upload-large-folder gwx22/AwareVLN .hf_ck_staging" >/dev/null; then
  echo "WARN: model checkpoint upload still running. Wait for it to finish, then run:"
  echo "  bash scripts/upload_hf_data.sh"
  exit 1
fi

bash scripts/upload_hf_data.sh
