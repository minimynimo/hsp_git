#!/bin/bash
# Re-runs the camels regional/adapted EVALUATION (cal + eva + synthetic scenarios) for the
# production RF/XGB/TCN/LSTM runs after 2_MERV/camels/varssim_US was regenerated (2026-10-20)
# with the synthetic scenarios on the current split (no_precip 2001-01-01..2006-12-31, peak
# window 2001-01-01..2001-02-10). The previous varssim_US (Sep 7) still had them on the old
# 1996-2010 eval split, so every ML synthetic prediction was invalid. Base columns (Obs flow,
# Precip, Temp, PET) are unchanged, so the trained models (train 1987-2000) are NOT retrained.
#
# Runs strictly sequentially on 8 pinned cores (shared box), like run_camels_hp_selected_production.sh.
set -uo pipefail

LOGDIR=/data0/funato/hsp2/bash_logs
TS=$(date +%Y%m%d_%H%M%S)
CPUSET="0-7"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
PY="taskset -c $CPUSET /data0/funato/.venv/bin/python3"
W=/data0/funato/hsp2/hsp_git/99_work

RF_RUN_DIR=$W/11_RandomForest/RF_regional_modeling-master/runs_rf/US/rf_run_0920_1807_camels_seed42
XGB_RUN_DIR=$W/12_XGBoost/XGB_regional_modeling-master/runs_xgb/US/xgb_run_0916_0125_camels_seed42
TCN_RUN_DIR=$W/13_TCN/TCN_regional_modeling-master/runs_tcn/tcn_run_0916_0621_camels_seed42
LSTM_RUN_DIR=$W/14_LSTM/ealstm_regional_modeling/runs/run_0916_132426_US_camels_seed42

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
run() {  # run <name> <dir> <cmd...>
  local name=$1 dir=$2; shift 2
  log "[$name] start"
  cd "$dir" || { log "[$name] cannot cd $dir"; return; }
  "$@" > "$LOGDIR/rerun_syn2001_${name}_$TS.log" 2>&1
  if [ $? -eq 0 ]; then log "[$name] done"; else log "[$name] FAILED -- see rerun_syn2001_${name}_$TS.log"; fi
}

run rf_regional_eval  $W/11_RandomForest/RF_regional_modeling-master  $PY eval_regional.py --run_dir "$RF_RUN_DIR"
run xgb_regional_eval $W/12_XGBoost/XGB_regional_modeling-master      env CUDA_VISIBLE_DEVICES="" $PY eval_regional.py --run_dir "$XGB_RUN_DIR"
run tcn_regional_eval $W/13_TCN/TCN_regional_modeling-master          $PY eval_regional.py --run_dir "$TCN_RUN_DIR"
run tcn_adapted       $W/13_TCN/TCN_regional_modeling-master          $PY eval_adapted.py --run_dir "$TCN_RUN_DIR" --model_epoch 50
run lstm_regional_eval $W/14_LSTM/ealstm_regional_modeling            $PY eval.py --run_dir "$LSTM_RUN_DIR"
log "all ML eval reruns finished"
