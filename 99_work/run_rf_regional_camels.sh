#!/bin/bash
# Final piece of the camels HP re-check production pipeline: RF regional (train -> eval ->
# SHAP), using the hyperparameters selected in SELECTED_HYPERPARAMETERS_camels.md after the
# HP re-check's 24-combo backfill + eval finished on 2026-09-20 (n_estimators=300,
# max_depth=30, min_samples_leaf=20, max_features=sqrt -- median NSE_eva=0.0610, best of 24).
# No adapted stage: RF only has a retired x_-prefixed adapted script (out of scope, per
# earlier instruction to skip x_ code).
#
# Pinned to cores 100-107 (free now that the RF regional HP backfill + eval finished) so it
# doesn't contend with any other pipeline stage or the heavy unrelated jobs elsewhere on this
# shared box.
set -uo pipefail

LOGDIR=/data0/funato/hsp2/bash_logs
TS=$(date +%Y%m%d_%H%M%S)
SEED=42
CPUSET="100-107"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
PY="taskset -c $CPUSET /data0/funato/.venv/bin/python3"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

get_run_dir() {
  grep -oP '(?:successfully (?:saved|calculated and saved) to |Scalers saved to )\K.*(?=/scalers\.json)' "$1" | tail -1
}

log "===================================================================="
log "RF regional (camels) standalone run starting (log timestamp $TS)"
log "===================================================================="

cd /data0/funato/hsp2/hsp_git/99_work/11_RandomForest/RF_regional_modeling-master || exit 1

log "[1] RF regional train (camels) -- n300_d30_l20_fsqrt"
RF_TRAIN_LOG="$LOGDIR/prod_camels_rf_regional_train_$TS.log"
$PY main_rf.py train --dataset camels --seed $SEED \
  --n_estimators 300 --max_depth 30 --max_features sqrt --min_samples_leaf 20 \
  > "$RF_TRAIN_LOG" 2>&1
if [ $? -ne 0 ]; then
  log "[1] RF regional train FAILED -- see $RF_TRAIN_LOG"
  exit 1
fi
RF_RUN_DIR=$(get_run_dir "$RF_TRAIN_LOG")
log "[1] RF regional train done -- run_dir=$RF_RUN_DIR"
if [ -z "$RF_RUN_DIR" ]; then
  log "Could not parse RF run_dir from log -- aborting eval/SHAP"
  exit 1
fi

log "[2] RF regional eval"
$PY eval_regional.py --run_dir "$RF_RUN_DIR" > "$LOGDIR/prod_camels_rf_regional_eval_$TS.log" 2>&1
if [ $? -eq 0 ]; then
  log "[2] RF regional eval done"

  log "[3] RF regional SHAP"
  $PY main_random_forest_shap_regional.py --run_dir "$RF_RUN_DIR" > "$LOGDIR/prod_camels_rf_regional_shap_$TS.log" 2>&1
  if [ $? -eq 0 ]; then log "[3] RF regional SHAP done"; else log "[3] RF regional SHAP FAILED"; fi
else
  log "[2] RF regional eval FAILED -- skipping SHAP"
fi

log "===================================================================="
log "RF regional (camels) standalone run finished. run_dir=$RF_RUN_DIR"
log "===================================================================="
