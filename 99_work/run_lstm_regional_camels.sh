#!/bin/bash
# Standalone LSTM regional stage (train -> eval -> SHAP -> adapted), split out of
# run_camels_hp_selected_production.sh on 2026-09-16 so it runs IN PARALLEL with LSTM
# individual instead of waiting for it. Uses the same camels-selected hyperparameters
# (SELECTED_HYPERPARAMETERS_camels.md) as the original pipeline's stage 12-15.
#
# Pinned to a DIFFERENT 8-core range (8-15) than LSTM individual (0-7) and the RF regional
# HP backfill (100-107), so all three can run concurrently without CPU contention. GPU is
# shared (LSTM individual is using ~5% utilization / <1GB of 49GB), so plenty of headroom
# for a second small LSTM model.
set -uo pipefail

LOGDIR=/data0/funato/hsp2/bash_logs
TS=$(date +%Y%m%d_%H%M%S)
SEED=42
CPUSET="8-15"
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
log "LSTM regional (camels) standalone run starting (log timestamp $TS)"
log "===================================================================="

cd /data0/funato/hsp2/hsp_git/99_work/14_LSTM/ealstm_regional_modeling || exit 1

log "[1] LSTM regional train (camels) -- h64_d0.4_lr0.001_sl365_bs256"
LSTM_TRAIN_LOG="$LOGDIR/prod_camels_lstm_regional_train_$TS.log"
$PY main_lstm_regional.py train --dataset camels --seed $SEED \
  --hidden_size 64 --dropout 0.4 --batch_size 256 --learning_rate 0.001 \
  > "$LSTM_TRAIN_LOG" 2>&1
if [ $? -ne 0 ]; then
  log "[1] LSTM regional train FAILED -- see $LSTM_TRAIN_LOG"
  exit 1
fi
LSTM_RUN_DIR=$(get_run_dir "$LSTM_TRAIN_LOG")
log "[1] LSTM regional train done -- run_dir=$LSTM_RUN_DIR"
if [ -z "$LSTM_RUN_DIR" ]; then
  log "Could not parse LSTM run_dir from log -- aborting eval/SHAP/adapted"
  exit 1
fi

log "[2] LSTM regional eval"
$PY eval.py --run_dir "$LSTM_RUN_DIR" > "$LOGDIR/prod_camels_lstm_regional_eval_$TS.log" 2>&1
if [ $? -eq 0 ]; then
  log "[2] LSTM regional eval done"

  log "[3] LSTM regional SHAP"
  $PY main_lstm_shap_regional.py --run_dir "$LSTM_RUN_DIR" > "$LOGDIR/prod_camels_lstm_regional_shap_$TS.log" 2>&1
  if [ $? -eq 0 ]; then log "[3] LSTM regional SHAP done"; else log "[3] LSTM regional SHAP FAILED"; fi

  log "[4] LSTM adapted"
  $PY main_lstm_adapted.py --run_dir "$LSTM_RUN_DIR" > "$LOGDIR/prod_camels_lstm_adapted_$TS.log" 2>&1
  if [ $? -eq 0 ]; then log "[4] LSTM adapted done"; else log "[4] LSTM adapted FAILED"; fi
else
  log "[2] LSTM regional eval FAILED -- skipping SHAP/adapted"
fi

log "===================================================================="
log "LSTM regional (camels) standalone run finished. run_dir=$LSTM_RUN_DIR"
log "===================================================================="
