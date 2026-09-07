#!/bin/bash

# ============================================
# CONFIGURATION SECTION - Edit parameters here
# ============================================

# Core assignment
cores=6
start_core=60 # Start core (1-indexed) -- avoids the XGB individual HP grid (0-89) and the
# TCN/LSTM individual/regional jobs (116-125ish) running concurrently as of 2026-08-27.

# Seeds
nseeds=1
firstseed=300

# Static data
# Was hardcoded to false, which passes --no_static=True on every real run below (see the
# static_bool -> --no_static=True/False mapping) -- the exact same launcher-default bug
# already found and fixed for LSTM (run_lstm_hp_tuning.sh-adjacent), TCN (run_main_tcn.sh),
# and XGBoost (run_main_xgb.sh); see DEBUG_README.md §1/§14.1/§16.1. Static-attribute
# conditioning was therefore never actually exercised by any real RF regional run launched
# through this script. Flipped to true.
static_bool=true
# if static_bool is true, use static data; if false, do not use static data

# Random Forest Hyperparameters -- selected combo from the completed 24/24-combo regional HP
# grid (n_estimators=500, min_samples_leaf=10, max_depth=15, max_features=sqrt;
# NSE_median=0.110 across all 614 basins). See SELECTED_HYPERPARAMETERS.md / main_rf.py's
# matching defaults / DEBUG_README.md §23.
N_ESTIMATORS=500
MAX_DEPTH=15
MAX_FEATURES=sqrt
MAX_SAMPLES=0.8
MIN_SAMPLES_SPLIT=5
MIN_SAMPLES_LEAF=10

# ============================================
# END CONFIGURATION SECTION
# ============================================

if [ "$static_bool" = true ]; then
    log_name="master_log_RF_static.log"
else
    log_name="master_log_RF_nostatic.log"
fi

loc="US"

end_core=$((start_core + cores))  # Calculate end core based on core count

gpucount=-1

# Set default model to "rf" if not provided
model=${1:-rf}

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

    if [ "$model" = "rf" ]; then
        if [ "$static_bool" = true ]; then
            outfile="/data0/funato/hsp2/0_out/RF/$loc/reports_static/rf.$seed.$(date +%Y-%m-%d_%H-%M-%S).out"
        else
            outfile="/data0/funato/hsp2/0_out/RF/$loc/reports_nostatic/rf.$seed.$(date +%Y-%m-%d_%H-%M-%S).out"
        fi
        
        mkdir -p "$(dirname "$outfile")" # Create output directory if it doesn't exist
        start_time=$(date '+%Y-%m-%d %H:%M:%S') # Log start time
        echo "Start Time: $start_time | Seed: $seed" >> "$output_dir/$log_name"

        if [ "$static_bool" = true ]; then
            echo "Running: taskset -c $core python3 /data0/funato/hsp2/99_work/11_RandomForest/RF_regional_modeling-master/main_rf.py train --no_static=False --n_estimators=$N_ESTIMATORS --max_depth=$MAX_DEPTH --max_features=$MAX_FEATURES --max_samples=$MAX_SAMPLES --min_samples_split=$MIN_SAMPLES_SPLIT --min_samples_leaf=$MIN_SAMPLES_LEAF --seed=$seed" >> "$output_dir/$log_name"
            taskset -c $core python3 /data0/funato/hsp2/99_work/11_RandomForest/RF_regional_modeling-master/main_rf.py train --no_static=False --n_estimators=$N_ESTIMATORS --max_depth=$MAX_DEPTH --max_features=$MAX_FEATURES --max_samples=$MAX_SAMPLES --min_samples_split=$MIN_SAMPLES_SPLIT --min_samples_leaf=$MIN_SAMPLES_LEAF --seed=$seed > $outfile 2>> "$output_dir/$log_name" &
        else
            echo "Running: taskset -c $core python3 /data0/funato/hsp2/99_work/11_RandomForest/RF_regional_modeling-master/main_rf.py train --no_static=True --n_estimators=$N_ESTIMATORS --max_depth=$MAX_DEPTH --max_features=$MAX_FEATURES --max_samples=$MAX_SAMPLES --min_samples_split=$MIN_SAMPLES_SPLIT --min_samples_leaf=$MIN_SAMPLES_LEAF --seed=$seed" >> "$output_dir/$log_name"
            taskset -c $core python3 /data0/funato/hsp2/99_work/11_RandomForest/RF_regional_modeling-master/main_rf.py train --no_static=True --n_estimators=$N_ESTIMATORS --max_depth=$MAX_DEPTH --max_features=$MAX_FEATURES --max_samples=$MAX_SAMPLES --min_samples_split=$MIN_SAMPLES_SPLIT --min_samples_leaf=$MIN_SAMPLES_LEAF --seed=$seed > $outfile 2>> "$output_dir/$log_name" &
        fi
        wait # Ensure the process completes before logging end time
        end_time=$(date '+%Y-%m-%d %H:%M:%S') # Log end time
        echo "End Time: $end_time | Seed: $seed" >> "$output_dir/$log_name"
    else
        echo "Error: Unsupported model choice '$model'. Supported models: rf" >> "$output_dir/$log_name"
        exit 1
    fi

    if [ $((gpucount % cores)) -eq $((cores - 1)) ]; then
        wait
    fi
done

###RUN USING####
# chmod +x /data0/funato/hsp2/99_work/11_RandomForest/RF_regional_modeling-master/run_main_rf.sh
# nohup /data0/funato/hsp2/99_work/11_RandomForest/RF_regional_modeling-master/run_main_rf.sh > /dev/null 2>&1 &