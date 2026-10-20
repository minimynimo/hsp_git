#!/bin/bash
# Backfill the 12 XGB regional (camels) combos that were silently killed (likely OOM)
# during the 8-way-concurrent CPU rerun on 2026-09-09. Run sequentially, one at a time,
# to avoid the memory contention that caused the original failures.
cd /data0/funato/hsp2/99_work/12_XGBoost/XGB_regional_modeling-master
PY=/data0/funato/.venv/bin/python3
export CUDA_VISIBLE_DEVICES=""

combos=(
  "500 7 0.01 1.0 0.7 3"
  "500 7 0.01 1.0 0.7 5"
  "500 7 0.01 1.0 0.8 3"
  "500 7 0.01 1.0 0.8 5"
  "500 7 0.05 0.7 0.7 3"
  "500 7 0.05 0.7 0.7 5"
  "500 7 0.05 0.7 0.8 3"
  "500 7 0.05 0.7 0.8 5"
  "500 7 0.05 0.8 0.7 5"
  "500 7 0.05 0.8 0.8 3"
  "500 7 0.05 0.8 0.8 5"
  "500 7 0.05 1.0 0.7 5"
)

i=0
for combo in "${combos[@]}"; do
  i=$((i+1))
  read n d lr ss cbt mcw <<< "$combo"
  echo "[$i/12] n=$n d=$d lr=$lr ss=$ss cbt=$cbt mcw=$mcw"
  taskset -c 100-103 "$PY" main_xgb_hp.py train \
    --n_estimators "$n" --max_depth "$d" --learning_rate "$lr" \
    --subsample "$ss" --colsample_bytree "$cbt" --min_child_weight "$mcw" \
    --dataset camels --seed 42
done

echo "Backfill of 12 XGB regional combos complete."
