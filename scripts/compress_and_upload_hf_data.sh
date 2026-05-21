#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if pgrep -f "upload-large-folder.*\.hf_data" >/dev/null 2>&1; then
  echo "Stop upload first: pkill -f 'upload-large-folder.*hf_data'"
  exit 1
fi

bash scripts/compress_hf_data_frames.sh
bash scripts/upload_hf_data.sh
