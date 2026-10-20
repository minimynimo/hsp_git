#!/bin/bash
# Runs the production pipeline (individual / regional+eval / SHAP / adapted) for every
# model/scope whose camels HP re-check is DONE, using the hyperparameters selected in
# SELECTED_HYPERPARAMETERS_camels.md. RF regional is skipped entirely -- its HP re-check
# is still PENDING (backfill running separately, see CAMELS_HP_RECHECK_STATUS.md), so there
# is no selected value to train production with yet.
#
# Per the user's explicit scope for 2026-09-16:
#   - Individual + Regional training for RF-indiv, XGB-indiv+regional, TCN-indiv+regional,
#     LSTM-indiv+regional.
#   - Regional SHAP for XGB/TCN/LSTM, run only AFTER that model's regional training+eval
#     finishes.
#   - Adapted (per-basin fine-tune) for TCN and LSTM ONLY (RF/XGB only have retired
#     x_-prefixed adapted scripts -- not used here).
#   - Runs strictly sequentially (one job at a time) since this box has a documented
#     OOM/segfault history under concurrent heavy jobs, and the RF regional HP backfill
#     (rf_regional_backfill_12.sh, pinned to cores 100-107) is running at the same time.
#     Every step here is pinned to a small CPUSET (see below) -- this is a 128-core SHARED
#     box with other users on it, so no single job should be allowed to spread across most
#     of the machine even when the underlying library (sklearn/xgboost n_jobs=-1, etc.)
#     would happily do so. (2026-09-16: an earlier version of this script only excluded the
#     backfill's 8 cores, i.e. taskset -c 0-95 -- still 96 cores for one job, which drove
#     load average over 95 and drew a complaint. Tightened to 8 cores everywhere below.)
#
# Each step's dataset is 'camels' throughout (--dataset camels / module-level dataset =
# "camels" already default in the individual scripts). Output paths are all
# 0_out/<MODEL>/US/camels/... , kept separate from the old 0_out/<MODEL>/US/caravan/...
# results by the dataset-awareness fixed/verified as part of this same task (see
# ADAPTATION_TO_CAMELS.md and the 2026-09-16 fix to ml_predict_long_format.py).
set -uo pipefail

LOGDIR=/data0/funato/hsp2/bash_logs
TS=$(date +%Y%m%d_%H%M%S)
SEED=42
CPUSET="0-7"
# Belt-and-suspenders alongside taskset: cap thread pools inside numpy/sklearn/xgboost/torch
# so nothing tries to fan out past the 8 cores taskset already restricts it to.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
PY="taskset -c $CPUSET /data0/funato/.venv/bin/python3"
# Set to 1 to skip individual steps already completed in an earlier (unthrottled) run of
# this script -- e.g. after killing and restarting because of excess CPU usage.
SKIP_RF_INDIV="${SKIP_RF_INDIV:-0}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Extracts a run_dir from a training log by looking for the "scalers ... saved to
# .../scalers.json" line every one of RF/XGB/TCN/LSTM's train() prints.
get_run_dir() {
  grep -oP '(?:successfully (?:saved|calculated and saved) to |Scalers saved to )\K.*(?=/scalers\.json)' "$1" | tail -1
}

STATUS_XGB_REGIONAL="not started"
STATUS_TCN_REGIONAL="not started"
STATUS_LSTM_REGIONAL="not started"

log "===================================================================="
log "Camels production pipeline starting (log timestamp $TS)"
log "===================================================================="

### 1/12: RF individual ##############################################################
if [ "$SKIP_RF_INDIV" = "1" ]; then
  log "[1] RF individual (camels) -- SKIPPED (already completed in a prior run of this script)"
else
  log "[1] RF individual (camels) -- n300_d10_l20_fsqrt"
  cd /data0/funato/hsp2/99_work/11_RandomForest || exit 1
  $PY main_random_forest_indiv.py \
    > "$LOGDIR/prod_camels_rf_indiv_$TS.log" 2>&1
  if [ $? -ne 0 ]; then log "[1] RF individual FAILED -- see prod_camels_rf_indiv_$TS.log"; else log "[1] RF individual done"; fi
fi

### 2/12: XGB individual ##############################################################
log "[2] XGB individual (camels) -- n500_d2_lr0.01_ss0.6_cbt1.0_mcw10"
cd /data0/funato/hsp2/hsp_git/99_work/12_XGBoost || exit 1
$PY main_xgboost_indiv.py \
  > "$LOGDIR/prod_camels_xgb_indiv_$TS.log" 2>&1
if [ $? -ne 0 ]; then log "[2] XGB individual FAILED -- see prod_camels_xgb_indiv_$TS.log"; else log "[2] XGB individual done"; fi

### 3-5/12: XGB regional train -> eval -> shap ########################################
log "[3] XGB regional train (camels) -- n1000_d7_lr0.1_ss0.7_cbt0.7_mcw3"
cd /data0/funato/hsp2/hsp_git/99_work/12_XGBoost/XGB_regional_modeling-master || exit 1
XGB_TRAIN_LOG="$LOGDIR/prod_camels_xgb_regional_train_$TS.log"
CUDA_VISIBLE_DEVICES="" $PY main_xgb.py train --dataset camels --seed $SEED \
  --n_estimators 1000 --max_depth 7 --learning_rate 0.1 --subsample 0.7 \
  --colsample_bytree 0.7 --min_child_weight 3 \
  > "$XGB_TRAIN_LOG" 2>&1
if [ $? -eq 0 ]; then
  XGB_RUN_DIR=$(get_run_dir "$XGB_TRAIN_LOG")
  log "[3] XGB regional train done -- run_dir=$XGB_RUN_DIR"
  if [ -n "$XGB_RUN_DIR" ]; then
    log "[4] XGB regional eval"
    $PY eval_regional.py --run_dir "$XGB_RUN_DIR" > "$LOGDIR/prod_camels_xgb_regional_eval_$TS.log" 2>&1
    if [ $? -eq 0 ]; then
      log "[4] XGB regional eval done"
      log "[5] XGB regional SHAP"
      $PY main_xgboost_shap_regional.py --run_dir "$XGB_RUN_DIR" > "$LOGDIR/prod_camels_xgb_regional_shap_$TS.log" 2>&1
      if [ $? -eq 0 ]; then STATUS_XGB_REGIONAL="done (run_dir=$XGB_RUN_DIR)"; log "[5] XGB regional SHAP done"; else STATUS_XGB_REGIONAL="SHAP FAILED (run_dir=$XGB_RUN_DIR)"; log "[5] XGB regional SHAP FAILED"; fi
    else
      STATUS_XGB_REGIONAL="eval FAILED (run_dir=$XGB_RUN_DIR)"; log "[4] XGB regional eval FAILED -- skipping SHAP"
    fi
  else
    STATUS_XGB_REGIONAL="train succeeded but run_dir not found in log -- skipping eval/SHAP"; log "[3] Could not parse XGB run_dir -- skipping eval/SHAP"
  fi
else
  STATUS_XGB_REGIONAL="train FAILED"; log "[3] XGB regional train FAILED -- skipping eval/SHAP"
fi

### 6/12: TCN individual ##############################################################
log "[6] TCN individual (camels) -- f16_k5_d0.4_lr0.0005_w90"
cd /data0/funato/hsp2/hsp_git/99_work/13_TCN || exit 1
$PY main_TCN_indiv.py > "$LOGDIR/prod_camels_tcn_indiv_$TS.log" 2>&1
if [ $? -ne 0 ]; then log "[6] TCN individual FAILED -- see prod_camels_tcn_indiv_$TS.log"; else log "[6] TCN individual done"; fi

### 7-10/12: TCN regional train -> eval -> shap -> adapted ############################
log "[7] TCN regional train (camels) -- f32_k5_d0.2_lr0.0005_w180"
cd /data0/funato/hsp2/hsp_git/99_work/13_TCN/TCN_regional_modeling-master || exit 1
TCN_TRAIN_LOG="$LOGDIR/prod_camels_tcn_regional_train_$TS.log"
$PY main_tcn.py train --dataset camels --seed $SEED \
  --num_filters 32 --dropout 0.2 --kernel_size 5 --learning_rate 0.0005 --seq_length 180 \
  > "$TCN_TRAIN_LOG" 2>&1
if [ $? -eq 0 ]; then
  TCN_RUN_DIR=$(get_run_dir "$TCN_TRAIN_LOG")
  log "[7] TCN regional train done -- run_dir=$TCN_RUN_DIR"
  if [ -n "$TCN_RUN_DIR" ]; then
    log "[8] TCN regional eval"
    $PY eval_regional.py --run_dir "$TCN_RUN_DIR" > "$LOGDIR/prod_camels_tcn_regional_eval_$TS.log" 2>&1
    if [ $? -eq 0 ]; then
      log "[8] TCN regional eval done"
      log "[9] TCN regional SHAP"
      $PY main_tcn_shap_regional.py --run_dir "$TCN_RUN_DIR" --model_epoch 50 > "$LOGDIR/prod_camels_tcn_regional_shap_$TS.log" 2>&1
      if [ $? -eq 0 ]; then log "[9] TCN regional SHAP done"; else log "[9] TCN regional SHAP FAILED"; fi
      log "[10] TCN adapted"
      $PY eval_adapted.py --run_dir "$TCN_RUN_DIR" --model_epoch 50 > "$LOGDIR/prod_camels_tcn_adapted_$TS.log" 2>&1
      if [ $? -eq 0 ]; then STATUS_TCN_REGIONAL="done incl. SHAP+adapted (run_dir=$TCN_RUN_DIR)"; log "[10] TCN adapted done"; else STATUS_TCN_REGIONAL="adapted FAILED (run_dir=$TCN_RUN_DIR)"; log "[10] TCN adapted FAILED"; fi
    else
      STATUS_TCN_REGIONAL="eval FAILED (run_dir=$TCN_RUN_DIR)"; log "[8] TCN regional eval FAILED -- skipping SHAP/adapted"
    fi
  else
    STATUS_TCN_REGIONAL="train succeeded but run_dir not found in log -- skipping eval/SHAP/adapted"; log "[7] Could not parse TCN run_dir -- skipping eval/SHAP/adapted"
  fi
else
  STATUS_TCN_REGIONAL="train FAILED"; log "[7] TCN regional train FAILED -- skipping eval/SHAP/adapted"
fi

### 11/12: LSTM individual #############################################################
log "[11] LSTM individual (camels) -- h32_d0.5_lr0.001_sl270_bs64"
cd /data0/funato/hsp2/hsp_git/99_work/14_LSTM || exit 1
$PY main_LSTM_indiv.py > "$LOGDIR/prod_camels_lstm_indiv_$TS.log" 2>&1
if [ $? -ne 0 ]; then log "[11] LSTM individual FAILED -- see prod_camels_lstm_indiv_$TS.log"; else log "[11] LSTM individual done"; fi

### 12/12: LSTM regional train -> eval -> shap -> adapted ##############################
log "[12] LSTM regional train (camels) -- h64_d0.4_lr0.001_sl365_bs256"
cd /data0/funato/hsp2/hsp_git/99_work/14_LSTM/ealstm_regional_modeling || exit 1
LSTM_TRAIN_LOG="$LOGDIR/prod_camels_lstm_regional_train_$TS.log"
$PY main_lstm_regional.py train --dataset camels --seed $SEED \
  --hidden_size 64 --dropout 0.4 --batch_size 256 --learning_rate 0.001 \
  > "$LSTM_TRAIN_LOG" 2>&1
if [ $? -eq 0 ]; then
  LSTM_RUN_DIR=$(get_run_dir "$LSTM_TRAIN_LOG")
  log "[12] LSTM regional train done -- run_dir=$LSTM_RUN_DIR"
  if [ -n "$LSTM_RUN_DIR" ]; then
    log "[13] LSTM regional eval"
    $PY eval.py --run_dir "$LSTM_RUN_DIR" > "$LOGDIR/prod_camels_lstm_regional_eval_$TS.log" 2>&1
    if [ $? -eq 0 ]; then
      log "[13] LSTM regional eval done"
      log "[14] LSTM regional SHAP"
      $PY main_lstm_shap_regional.py --run_dir "$LSTM_RUN_DIR" > "$LOGDIR/prod_camels_lstm_regional_shap_$TS.log" 2>&1
      if [ $? -eq 0 ]; then log "[14] LSTM regional SHAP done"; else log "[14] LSTM regional SHAP FAILED"; fi
      log "[15] LSTM adapted"
      $PY main_lstm_adapted.py --run_dir "$LSTM_RUN_DIR" > "$LOGDIR/prod_camels_lstm_adapted_$TS.log" 2>&1
      if [ $? -eq 0 ]; then STATUS_LSTM_REGIONAL="done incl. SHAP+adapted (run_dir=$LSTM_RUN_DIR)"; log "[15] LSTM adapted done"; else STATUS_LSTM_REGIONAL="adapted FAILED (run_dir=$LSTM_RUN_DIR)"; log "[15] LSTM adapted FAILED"; fi
    else
      STATUS_LSTM_REGIONAL="eval FAILED (run_dir=$LSTM_RUN_DIR)"; log "[13] LSTM regional eval FAILED -- skipping SHAP/adapted"
    fi
  else
    STATUS_LSTM_REGIONAL="train succeeded but run_dir not found in log -- skipping eval/SHAP/adapted"; log "[12] Could not parse LSTM run_dir -- skipping eval/SHAP/adapted"
  fi
else
  STATUS_LSTM_REGIONAL="train FAILED"; log "[12] LSTM regional train FAILED -- skipping eval/SHAP/adapted"
fi

log "===================================================================="
log "Camels production pipeline finished."
log "  XGB regional : $STATUS_XGB_REGIONAL"
log "  TCN regional : $STATUS_TCN_REGIONAL"
log "  LSTM regional: $STATUS_LSTM_REGIONAL"
log "  (RF individual / XGB individual / TCN individual / LSTM individual: see their own logs above)"
log "  RF regional was skipped -- HP re-check still pending, see CAMELS_HP_RECHECK_STATUS.md"
log "===================================================================="
