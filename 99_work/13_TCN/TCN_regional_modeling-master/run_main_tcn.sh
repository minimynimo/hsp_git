#!/bin/bash

# ============================================
# CONFIGURATION SECTION - Edit parameters here
# ============================================

# Core assignment
cores=3
start_core=124 # Start core (1-indexed) -- moved off 0-79/80-103, which are busy with the
# RF regional HP grid and XGB individual HP grid re-runs as of 2026-08-27.

# Seeds
nseeds=1
firstseed=300

# Static data
# NOTE: this default was previously `false`, which silently disabled the regional
# model's cross-basin conditioning mechanism (static basin attributes) in EVERY
# production run (confirmed via runs_tcn/*/cfg.json - all show no_static=true even
# though main_tcn.py's own argparse default is False). See DEBUG_README.md for the
# LSTM pipeline's identical bug (no_static / EA-LSTM) and how it was found. Default
# flipped to `true` so a canonical run actually exercises static conditioning unless
# explicitly overridden.
static_bool=true
# if static_bool is true, use static data; if false, do not use static data

# TCN Hyperparameters -- selected regional combo from the completed 162/162-combo HP grid
# (154-basin subsample, NSE_median=0.284) -- see SELECTED_HYPERPARAMETERS.md. An earlier pass
# this session mistakenly flagged this grid as broken (catastrophic NSE -99 to -585) based on a
# STALE aggregated CSV generated before the underlying per-combo result files existed/finished;
# re-running the aggregation script (99_work/01_fig/benchmark/hyperparameter_analysis/TCN_hp.py)
# against the actual (fresh, complete) per-combo results showed sane numbers throughout -- see
# DEBUG_README.md's TCN follow-up section for the full writeup. Also fixes the previous
# NUM_CHANNELS="128 256 512" mismatch (a 3-level, non-uniform-width architecture that didn't
# match this project's fixed 6-level uniform-width convention, per HYPERPARAMETERS.md's
# "LAYERS 1 2 4 8 16 32" row / DEBUG_README.md's TCN §15.1 derivation).
NUM_FILTERS=64
NUM_LEVELS=6
BATCH_SIZE=32
EPOCHS=50
KERNEL_SIZE=3
DROPOUT=0.2
LEARNING_RATE=0.0005
# main_tcn.py hardcodes seq_length=270 in GLOBAL_SETTINGS (no --seq_length CLI flag), which
# already matches the selected window size -- no variable/flag needed here.
NUM_WORKERS=2

# 'camels' (default): built directly from raw CAMELS-US, full 1980-2014 record --
# main_tcn.py's DATASET_SETTINGS gives it a 1987-2000 cal / 2001-2006 eval split. 'caravan':
# older Caravan-derived extraction (1987-2006 only, 1990-2000/2001-2006 split, unchanged).
DATASET="camels"

# ============================================
# END CONFIGURATION SECTION
# ============================================

if [ "$static_bool" = true ]; then
    log_name="master_log_TCN_static.log"
else
    log_name="master_log_TCN_nostatic.log"
fi

loc="US"

end_core=$((start_core + cores - 1))  # Calculate end core based on core count

gpucount=-1

# Set default model to "tcn" if not provided
model=${1:-tcn}

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

    if [ "$model" = "tcn" ]; then
        if [ "$static_bool" = true ]; then
            outfile="/data0/funato/hsp2/0_out/TCN/$loc/$DATASET/reports_static/tcn.$seed.$(date +%Y-%m-%d_%H-%M-%S).out"
        else
            outfile="/data0/funato/hsp2/0_out/TCN/$loc/$DATASET/reports_nostatic/tcn.$seed.$(date +%Y-%m-%d_%H-%M-%S).out"
        fi
        mkdir -p "$(dirname "$outfile")" # Create output directory if it doesn't exist
        start_time=$(date '+%Y-%m-%d %H:%M:%S') # Log start time
        echo "Start Time: $start_time | Seed: $seed" >> "$output_dir/$log_name"

        # Build parameter strings from configuration
        TCN_PARAMS="--num_filters=$NUM_FILTERS --num_levels=$NUM_LEVELS"
        OTHER_PARAMS="--batch_size=$BATCH_SIZE --epochs=$EPOCHS --kernel_size=$KERNEL_SIZE --dropout=$DROPOUT --learning_rate=$LEARNING_RATE --seed=$seed --num_workers=$NUM_WORKERS --dataset=$DATASET"


        if [ "$static_bool" = true ]; then ## static data is used
            echo "Running: taskset -c $core python3 /data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/main_tcn.py train --no_static=False $OTHER_PARAMS $TCN_PARAMS" >> "$output_dir/$log_name"

            taskset -c $core python3 /data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/main_tcn.py train --no_static=False $OTHER_PARAMS $TCN_PARAMS > $outfile 2>> "$output_dir/$log_name" &
        else #  no static data is used
            echo "Running: taskset -c $core python3 /data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/main_tcn.py train --no_static=True $OTHER_PARAMS $TCN_PARAMS" >> "$output_dir/$log_name"

            taskset -c $core python3 /data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/main_tcn.py train --no_static=True $OTHER_PARAMS $TCN_PARAMS > $outfile 2>> "$output_dir/$log_name" &
        fi

        wait # Ensure the process completes before logging end time
        end_time=$(date '+%Y-%m-%d %H:%M:%S') # Log end time
        echo "End Time: $end_time | Seed: $seed" >> "$output_dir/$log_name"
    else
        echo "Error: Unsupported model choice '$model'. Supported models: tcn" >> "$output_dir/$log_name"
        exit 1
    fi

    if [ $((gpucount % cores)) -eq $((cores - 1)) ]; then
        wait
    fi
done



###RUN USING####
# chmod +x /data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/run_main_tcn.sh
# nohup /data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/run_main_tcn.sh tcn > /dev/null 2>&1 &
