#!/bin/bash
# Self-healing wrapper: retries the training command on crash (flaky NVML driver),
# relying on per-basin part files to make each retry resume rather than restart.
LOG="$1"; shift
for i in $(seq 1 30); do
  /data0/funato/.venv/bin/python "$@" >> "$LOG" 2>&1
  code=$?
  if [ $code -eq 0 ]; then exit 0; fi
  echo "[retry wrapper] attempt $i exited $code, retrying in 10s" >> "$LOG"
  sleep 10
done
echo "[retry wrapper] gave up after 30 attempts" >> "$LOG"
exit 1
