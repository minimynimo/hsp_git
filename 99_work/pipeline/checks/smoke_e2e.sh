#!/bin/bash
# End-to-end smoke test of the whole chain on a tiny basin set (HSP_SMOKE basins, seeds 1-2,
# 2 epochs) in a separate output root. HBV is faked from LSTM output (code-path test only).
set -euo pipefail
export HSP_SMOKE=${HSP_SMOKE:-12} HSP_OUT_ROOT=${HSP_OUT_ROOT:-/data0/funato/hsp2/0_out/hsp_v2_smoke} OMP_NUM_THREADS=4
cd "$(dirname "$0")/.."
P=/data0/funato/.venv/bin/python
$P basins/make_basin_sets.py > /dev/null
for m in RF XGB; do for s in regional individual; do
  $P train/trees.py --model $m --strategy $s --mode tune --n_jobs 8 > /dev/null 2>&1
  for seed in $( [ $s = regional ] && echo 1 2 || echo 1 ); do
    $P train/trees.py --model $m --strategy $s --mode final --seed $seed --n_jobs 8 > /dev/null 2>&1; done
done; done
mkdir -p $HSP_OUT_ROOT/tuning/TCN_regional
echo '{"num_filters": 64, "seq_length": 180, "tag": "f64_w180"}' > $HSP_OUT_ROOT/tuning/TCN_regional/selected.json
for seed in 1 2; do
  $P train/nets.py regional --model LSTM --seed $seed --max_epochs 2 > /dev/null 2>&1
  $P train/nets.py regional --model TCN --seed $seed --num_filters 64 --seq_length 180 --max_epochs 2 > /dev/null 2>&1
  $P train/nets.py adapted --model LSTM --seed $seed > /dev/null 2>&1
  $P train/nets.py adapted --model TCN --seed $seed --regional_tag f64_w180 > /dev/null 2>&1
done
for m in LSTM TCN; do $P train/nets.py individual --model $m --seed 1 --max_epochs 2 > /dev/null 2>&1; done
# fake HBV run (copy of LSTM regional seed 1) to exercise the analysis
H=$HSP_OUT_ROOT/runs/HBV/individual/HBV_individual_seed42; rm -rf $H; mkdir -p $H
cp -r $HSP_OUT_ROOT/runs/LSTM/regional/LSTM_regional_seed1/{sim,metrics.csv,config.json,_DONE} $H/
echo "gauge_id,reason" > $HSP_OUT_ROOT/basins/hbv_failures.csv
$P checks/check_runs.py
$P analysis/eval_set.py
$P analysis/scoring.py
echo SMOKE OK
