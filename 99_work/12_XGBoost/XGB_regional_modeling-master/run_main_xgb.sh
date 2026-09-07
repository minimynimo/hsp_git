#!/bin/bash

# ============================================
# CONFIGURATION SECTION - Edit parameters here
# ============================================

# Core assignment
cores=1
start_core=127 # Start core (1-indexed) -- moved off 0-79/80-103, which are busy with the
# RF regional HP grid and XGB individual HP grid re-runs as of 2026-08-27.

# Seeds
nseeds=1
firstseed=300

# Static data
# NOTE: this used to default to `false`, and every single saved production run under runs_xgb/
# (8 checked: all US and GB xgb_run_* directories) has no_static=true in cfg.json as a direct
# result -- i.e. the regional model's core cross-basin conditioning mechanism (static catchment
# attributes) was never actually exercised in any real training run, exactly the same bug pattern
# independently found (and fixed) for LSTM (DEBUG_README.md #1) and TCN (DEBUG_README.md #14.1).
# Flipped to `true` so a fresh run actually uses static conditioning by default; the underlying
# int/str basin-index bug that would have silently zeroed out the attributes anyway even with
# static_bool=true (DEBUG_README.md papercode/datautils.py fix) has also been fixed this session.
static_bool=true
# if static_bool is true, use static data; if false, do not use static data , change the log_name and the outfile name accordingly.

# XGBoost Hyperparameters -- selected regional combo, see SELECTED_HYPERPARAMETERS.md.
# subsample/colsample_bytree/min_child_weight were previously not passed by this launcher at
# all (main_xgb.py silently fell back to its own argparse defaults, which didn't include
# min_child_weight until this fix -- see DEBUG_README.md §16.8-adjacent note in main_xgb.py).
N_ESTIMATORS=1000
MAX_DEPTH=7
LEARNING_RATE=0.05
SUBSAMPLE=0.8
COLSAMPLE_BYTREE=0.7
MIN_CHILD_WEIGHT=5

# ============================================
# END CONFIGURATION SECTION
# ============================================

if [ "$static_bool" = true ]; then
    log_name="master_log_XGB_static.log"
else
    log_name="master_log_XGB_nostatic.log"
fi

loc="US"

end_core=$((start_core + cores - 1))  # Calculate end core based on core count

gpucount=-1

# Set default model to "xgb" if not provided
model=${1:-xgb}

output_dir="/data0/funato/hsp2/bash_logs"

mkdir -p "$output_dir" # Create logs directory if it doesn't exist

> "$output_dir/$log_name" # Clear the log file

for (( seed = $firstseed ; seed < $((nseeds+$firstseed)) ; seed++ )); do
    gpucount=$(($gpucount + 1))
    core=$((start_core + (gpucount % cores))) # Ensure core is within start_core to end_core range
    if [ $core -gt $end_core ]; then
        core=$start_core # Wrap around to start_core if exceeding end_core
    fi
    
    echo $seed $gpucount $core >> "$output_dir/$log_name"

    if [ "$model" = "xgb" ]; then
        if [ "$static_bool" = true ]; then
            outfile="/data0/funato/hsp2/0_out/XGB/$loc/reports_static/xgb.$seed.$(date +%Y-%m-%d_%H-%M-%S).out"
        else
            outfile="/data0/funato/hsp2/0_out/XGB/$loc/reports_nostatic/xgb.$seed.$(date +%Y-%m-%d_%H-%M-%S).out"
        fi
        mkdir -p "$(dirname "$outfile")" # Create output directory if it doesn't exist
        start_time=$(date '+%Y-%m-%d %H:%M:%S') # Log start time
        echo "Start Time: $start_time | Seed: $seed" >> "$output_dir/$log_name"

        if [ "$static_bool" = true ]; then  ## static data is used
            echo "Running: taskset -c $core python3 /data0/funato/hsp2/99_work/12_XGBoost/XGB_regional_modeling-master/main_xgb.py train --no_static=False --n_estimators=$N_ESTIMATORS --max_depth=$MAX_DEPTH --learning_rate=$LEARNING_RATE --subsample=$SUBSAMPLE --colsample_bytree=$COLSAMPLE_BYTREE --min_child_weight=$MIN_CHILD_WEIGHT --seed=$seed" >> "$output_dir/$log_name"
            taskset -c $core python3 /data0/funato/hsp2/99_work/12_XGBoost/XGB_regional_modeling-master/main_xgb.py train --no_static=False --n_estimators=$N_ESTIMATORS --max_depth=$MAX_DEPTH --learning_rate=$LEARNING_RATE --subsample=$SUBSAMPLE --colsample_bytree=$COLSAMPLE_BYTREE --min_child_weight=$MIN_CHILD_WEIGHT --seed=$seed > $outfile 2>> "$output_dir/$log_name" &
        else ## no static data is used
            echo "Running: taskset -c $core python3 /data0/funato/hsp2/99_work/12_XGBoost/XGB_regional_modeling-master/main_xgb.py train --no_static=True --n_estimators=$N_ESTIMATORS --max_depth=$MAX_DEPTH --learning_rate=$LEARNING_RATE --subsample=$SUBSAMPLE --colsample_bytree=$COLSAMPLE_BYTREE --min_child_weight=$MIN_CHILD_WEIGHT --seed=$seed" >> "$output_dir/$log_name"
            taskset -c $core python3 /data0/funato/hsp2/99_work/12_XGBoost/XGB_regional_modeling-master/main_xgb.py train --no_static=True --n_estimators=$N_ESTIMATORS --max_depth=$MAX_DEPTH --learning_rate=$LEARNING_RATE --subsample=$SUBSAMPLE --colsample_bytree=$COLSAMPLE_BYTREE --min_child_weight=$MIN_CHILD_WEIGHT --seed=$seed > $outfile 2>> "$output_dir/$log_name" &
        fi
        wait
        end_time=$(date '+%Y-%m-%d %H:%M:%S') # Log end time
        echo "End Time: $end_time | Seed: $seed" >> "$output_dir/$log_name"
    else
        echo "Error: Unsupported model choice '$model'. Supported models: xgb" >> "$output_dir/$log_name"
        exit 1
    fi

    if [ $((gpucount % cores)) -eq $((cores - 1)) ]; then
        wait
    fi
done

###RUN USING####
# chmod +x /data0/funato/hsp2/99_work/12_XGBoost/XGB_regional_modeling-master/run_main_xgb.sh
# nohup /data0/funato/hsp2/99_work/12_XGBoost/XGB_regional_modeling-master/run_main_xgb.sh xgb > /dev/null 2>&1 &

