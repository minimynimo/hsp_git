#!/bin/bash

# XGBoost Hyperparameter Tuning Script
# Regional: N_estimators[500,1000] × Max_depth[5,7] × Learning_rate[0.05,0.1] = 8 combinations
# Individual: N_estimators[100,300] × Max_depth[3,4] × Learning_rate[0.05,0.1] = 8 combinations

# 

PYTHON=/data0/funato/hsp/.venv/bin/python
REGIONAL_SCRIPT=/data0/funato/hsp/99_work/12_XGBoost/XGB_regional_modeling-master/main_xgb_hp.py
INDIVIDUAL_SCRIPT=/data0/funato/hsp/99_work/12_XGBoost/main_xgboost_indiv_hp.py
LOG_DIR=/data0/funato/hsp/99_work/12_XGBoost/logs

mkdir -p $LOG_DIR

echo "Starting XGBoost Hyperparameter Tuning"
echo "========================================"
echo "Regional combinations: 8"
echo "Individual combinations: 8"
echo "Total jobs: 16"
echo ""

# Regional hyperparameter combinations
# Allocate cores 0-15 for regional jobs (2 cores per job)

job_id=0

for n_est in 500 1000; do
    for max_d in 5 7; do
        for lr in 0.05 0.1; do
            job_id=$((job_id + 1))
            core_start=$(((job_id - 1) * 2))
            core_end=$((core_start + 1))
            
            LOG_FILE="${LOG_DIR}/hp_tuning_regional_job${job_id}_n${n_est}_d${max_d}_lr${lr}.log"
            
            echo "Starting Regional Job ${job_id}/8: n_estimators=${n_est}, max_depth=${max_d}, lr=${lr} (cores ${core_start}-${core_end})"
            
            taskset -c ${core_start}-${core_end} $PYTHON $REGIONAL_SCRIPT train \
                --n_estimators $n_est \
                --max_depth $max_d \
                --learning_rate $lr \
                > $LOG_FILE 2>&1 &
        done
    done
done

echo ""
echo "All 8 regional jobs started (cores 0-15)"
echo ""

# Individual hyperparameter combinations
# Allocate cores 16-31 for individual jobs (2 cores per job)

job_id=0

for n_est in 100 300; do
    for max_d in 3 4; do
        for lr in 0.05 0.1; do
            job_id=$((job_id + 1))
            core_start=$((16 + (job_id - 1) * 2))
            core_end=$((core_start + 1))
            
            LOG_FILE="${LOG_DIR}/hp_tuning_individual_job${job_id}_n${n_est}_d${max_d}_lr${lr}.log"
            
            echo "Starting Individual Job ${job_id}/8: n_estimators=${n_est}, max_depth=${max_d}, lr=${lr} (cores ${core_start}-${core_end})"
            
            taskset -c ${core_start}-${core_end} $PYTHON $INDIVIDUAL_SCRIPT \
                --n_estimators $n_est \
                --max_depth $max_d \
                --learning_rate $lr \
                > $LOG_FILE 2>&1 &
        done
    done
done

echo ""
echo "All 8 individual jobs started (cores 16-31)"
echo ""
echo "Total cores used: 32 (0-31)"
echo "Monitor progress with: tail -f $LOG_DIR/hp_tuning_*.log"
echo ""
echo "To run with GNU parallel instead (optional):"
echo "  Uncomment the parallel sections below"
echo ""

# Optional: Use GNU parallel for better job management
# Uncomment these sections if you have GNU parallel installed
#
# Regional jobs with parallel
# parallel -j 8 --linebuffer "
#     echo 'Regional: n={1}, d={2}, lr={3}';
#     taskset -c {#} $PYTHON $REGIONAL_SCRIPT train --n_estimators {1} --max_depth {2} --learning_rate {3} > ${LOG_DIR}/hp_regional_n{1}_d{2}_lr{3}.log 2>&1
# " ::: 500 1000 ::: 5 7 ::: 0.05 0.1
#
# Individual jobs with parallel
# parallel -j 8 --linebuffer "
#     echo 'Individual: n={1}, d={2}, lr={3}';
#     taskset -c {#} $PYTHON $INDIVIDUAL_SCRIPT --n_estimators {1} --max_depth {2} --learning_rate {3} > ${LOG_DIR}/hp_individual_n{1}_d{2}_lr{3}.log 2>&1
# " ::: 100 300 ::: 3 4 ::: 0.05 0.1

wait
echo "All XGBoost hyperparameter tuning jobs completed!"
