#!/usr/bin/env bash
# Compress trajectory frame dirs to videos.tar.gz (four datasets in parallel).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REASON="${ROOT}/.hf_data/reason"
LOG_DIR="${ROOT}"
mkdir -p "$LOG_DIR"

compress_one() {
  local name="$1"
  local dir="${REASON}/${name}"
  local frames="${dir}/videos"
  local archive="${dir}/videos.tar.gz"
  local log="${LOG_DIR}/.hf_data_compress_${name}.log"

  {
    echo "=== $(date) [${name}] start ==="
    if [[ ! -d "$frames" ]]; then
      if [[ -f "$archive" ]]; then
        echo "[${name}] skip: already compressed"
      else
        echo "[${name}] skip: no videos/ and no archive"
      fi
      exit 0
    fi
    if [[ -f "$archive" ]]; then
      echo "[${name}] found existing videos.tar.gz, verifying ..."
      if tar -tzf "$archive" >/dev/null 2>&1; then
        echo "[${name}] archive ok, removing videos/ only"
        rm -rf "$frames"
        echo "[${name}] done ($(du -sh "$archive" | awk '{print $1}'))"
        exit 0
      fi
      echo "[${name}] removing incomplete videos.tar.gz"
      rm -f "$archive"
    fi

    echo "[${name}] compressing videos/ -> videos.tar.gz ..."
    tar -czf "$archive" -C "$dir" videos
    echo "[${name}] removing videos/ ..."
    rm -rf "$frames"
    echo "[${name}] done ($(du -sh "$archive" | awk '{print $1}'))"
    echo "=== $(date) [${name}] finish ==="
  } >"$log" 2>&1
}

echo "=== $(date) parallel compress .hf_data/reason (4 jobs) ===" | tee -a "${LOG_DIR}/.hf_data_compress.log"

pids=()
for name in r2r rxr r2rfollow rxrfollow; do
  compress_one "$name" &
  pids+=($!)
  echo "started ${name} pid=$!"
done

failed=0
for i in "${!pids[@]}"; do
  name=$(echo r2r rxr r2rfollow rxrfollow | cut -d' ' -f$((i + 1)))
  if ! wait "${pids[$i]}"; then
    echo "[${name}] FAILED (see .hf_data_compress_${name}.log)"
    failed=1
  fi
done

cat "${LOG_DIR}"/.hf_data_compress_*.log >> "${LOG_DIR}/.hf_data_compress.log" 2>/dev/null || true

if [[ $failed -ne 0 ]]; then
  echo "=== $(date) some jobs failed ==="
  exit 1
fi
echo "=== $(date) all done ===" | tee -a "${LOG_DIR}/.hf_data_compress.log"
