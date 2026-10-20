#!/bin/bash
# Backfill the 12 RF regional (camels) combos never successfully trained: 2 original
# segfaults (n300/l5/d15/f0.33 and n300/l5/d30/sqrt) plus 10 that were mid-sweep when the
# machine rebooted on 2026-09-14. Run sequentially, one at a time, on a dedicated core
# range to rule out the memory/thread contention that caused the earlier segfaults
# (exit 132/139) under 4-way-concurrent n_jobs=4 execution.
cd /data0/funato/hsp2/99_work/11_RandomForest/RF_regional_modeling-master
PY=/data0/funato/.venv/bin/python3

combos=(
  "300 5 15 0.33"
  "300 5 30 sqrt"
  "500 5 15 0.33"
  "500 5 30 0.33"
  "500 10 15 sqrt"
  "500 10 15 0.33"
  "500 10 30 sqrt"
  "500 10 30 0.33"
  "500 20 15 sqrt"
  "500 20 15 0.33"
  "500 20 30 sqrt"
  "500 20 30 0.33"
)

i=0
for combo in "${combos[@]}"; do
  i=$((i+1))
  read n l d f <<< "$combo"
  echo "[$i/12] n_estimators=$n min_samples_leaf=$l max_depth=$d max_features=$f"
  taskset -c 100-107 "$PY" main_rf_hp.py train \
    --n_estimators "$n" --min_samples_leaf "$l" --max_depth "$d" --max_features "$f" \
    --n_jobs 8 --dataset camels --seed 42
  echo "[$i/12] exit code: $?"
done

echo "Backfill of 12 RF regional combos complete."
