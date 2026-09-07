#!/bin/bash

# TCN Hyperparameter Tuning Script
# Runs all hyperparameter combinations in parallel using specified cores.
# Combines what used to be two separate scripts (start_tcn_hp_tuning.sh +
# run_tcn_hp_tuning.sh, see x_start_tcn_hp_tuning.sh) into one: pass --background
# (or -b) as a 4th argument to launch detached via nohup exactly like the old
# start_tcn_hp_tuning.sh did; omit it to run in the foreground exactly like the
# old run_tcn_hp_tuning.sh did.
#
# Usage: ./run_tcn_hp_tuning.sh <StartCore> <NumCores> <ModelType> [--background]
# /data0/funato/hsp2/99_work/13_TCN/run_tcn_hp_tuning.sh 0 10 regional  --background            # runs in foreground
# /data0/funato/hsp2/99_work/13_TCN/run_tcn_hp_tuning.sh 20 7 individual --background   # detached, logged, backgrounded

# Note: deliberately NOT using `set -e`. Every failure path in the validation section below
# already calls `exit 1` explicitly, and `-e` combined with `((core_idx++))` further down is a
# known bash footgun: when core_idx is 0, the post-increment expression's value is 0, so the
# arithmetic command itself returns exit status 1 and `-e` would kill the whole script right
# after launching the first job - exactly the bug that made a 20-core sweep silently run only
# job 1. See DEBUG_README.md for how this was diagnosed.
set -uo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <StartCore> <NumCores> <ModelType> [--background]"
    echo "  StartCore: Starting core number (e.g., 0)"
    echo "  NumCores: Number of cores to use (e.g., 8)"
    echo "  ModelType: 'regional' or 'individual'"
    echo "  --background (optional): launch detached via nohup, like the old start_tcn_hp_tuning.sh"
    exit 1
fi

START_CORE=$1
NUM_CORES=$2
MODEL_TYPE=$3
BACKGROUND_FLAG="${4:-}"

# --- CPU validation: make sure the requested core range actually exists and is sane, ---
# --- before pinning any job to it with taskset.                                       ---
if ! [[ "$START_CORE" =~ ^[0-9]+$ ]] || ! [[ "$NUM_CORES" =~ ^[0-9]+$ ]]; then
    echo "Error: StartCore and NumCores must be non-negative integers (got '$START_CORE', '$NUM_CORES')"
    exit 1
fi

if [ "$NUM_CORES" -lt 1 ]; then
    echo "Error: NumCores must be at least 1 (got $NUM_CORES)"
    exit 1
fi

AVAILABLE_CORES=$(nproc --all 2>/dev/null || nproc)
END_CORE=$((START_CORE + NUM_CORES - 1))

if [ "$END_CORE" -ge "$AVAILABLE_CORES" ]; then
    echo "Error: requested cores $START_CORE-$END_CORE exceed the $AVAILABLE_CORES cores available on this machine."
    echo "Lower StartCore/NumCores so that StartCore + NumCores - 1 <= $((AVAILABLE_CORES - 1))."
    exit 1
fi

# --- Background mode: re-exec self detached via nohup, exactly like the old ---
# --- start_tcn_hp_tuning.sh wrapper, then return control to the caller.     ---
if [ "$BACKGROUND_FLAG" == "--background" ] || [ "$BACKGROUND_FLAG" == "-b" ]; then
    SCRIPT_PATH_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    SCRIPT_DIR="$(dirname "$SCRIPT_PATH_SELF")"
    LOG_FILE="${SCRIPT_DIR}/hp_tuning_${MODEL_TYPE}_${START_CORE}_${NUM_CORES}_$(date +%Y%m%d_%H%M%S).log"

    echo "Starting TCN hyperparameter tuning in background..."
    echo "Model Type: $MODEL_TYPE"
    echo "Cores: $START_CORE to $END_CORE (of $AVAILABLE_CORES available)"
    echo "Log file: $LOG_FILE"

    nohup "$SCRIPT_PATH_SELF" "$START_CORE" "$NUM_CORES" "$MODEL_TYPE" > "$LOG_FILE" 2>&1 &

    PID=$!
    echo "Process started with PID: $PID"
    echo "To monitor progress: tail -f $LOG_FILE"
    echo "To check if running: ps -p $PID"
    echo "To stop: kill $PID"
    exit 0
fi

echo "Starting TCN hyperparameter tuning"
echo "Model Type: $MODEL_TYPE"
echo "Using cores: $START_CORE to $END_CORE (of $AVAILABLE_CORES available)"
echo "Total cores: $NUM_CORES"

# Python executable (use virtual environment if available)
PYTHON_CMD="/data0/funato/hsp2/.venv/bin/python"
if [ ! -f "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
fi

# Hyperparameter grids based on model type - must match /data0/funato/hsp2/HYPERPARAMETERS.md's
# TCN section exactly (see DEBUG_README.md for how these were derived/verified). NUM_LEVELS is
# NOT swept: HYPERPARAMETERS.md's "LAYERS 1 2 4 8 16 32" row is identical for individual and
# regional and is exactly the dilation schedule 2**i for i in range(6) - i.e. it's num_levels=6,
# fixed, not an independent grid dimension. Both scripts default num_levels to something else when
# called without --num_levels (backward compat - see main_tcn_hp.py / main_TCN_indiv_hp.py), so it
# is passed explicitly below.
if [ "$MODEL_TYPE" == "regional" ]; then
    # WARNING - memory, not CPU, is the real constraint here: each regional job pools all ~614
    # US basins into memory and uses ~90GB RSS (confirmed empirically - see DEBUG_README.md).
    # NumCores only limits CPU/taskset parallelism, NOT concurrent memory use, so requesting a
    # large NumCores (e.g. 20) WILL try to run that many ~90GB jobs at once and get most of them
    # OOM-killed - this already happened once. On a machine also running other sweeps (RF/XGB
    # etc.), keep NumCores for 'regional' small (3-5) regardless of how many CPU cores are free.
    NUM_FILTERS=(32 64 128)
    KERNEL_SIZE=(3 5 7)
    DROPOUT=(0.1 0.2 0.3)
    LEARNING_RATE=(0.001 0.0005)
    WINDOW_SIZE=(90 180 270)
    NUM_LEVELS=6
    SCRIPT_PATH="/data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/main_tcn_hp.py"
    LOG_DIR="/data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/logs"
elif [ "$MODEL_TYPE" == "individual" ]; then
    NUM_FILTERS=(16 32)
    KERNEL_SIZE=(3 5)
    DROPOUT=(0.3 0.4 0.5)
    LEARNING_RATE=(0.001 0.0005)
    WINDOW_SIZE=(90 180 270)
    NUM_LEVELS=6
    SCRIPT_PATH="/data0/funato/hsp2/99_work/13_TCN/main_TCN_indiv_hp.py"
    LOG_DIR="/data0/funato/hsp2/99_work/13_TCN/logs"
else
    echo "Error: ModelType must be 'regional' or 'individual'"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Generate all hyperparameter combinations (num_levels is fixed, not part of the combination
# tuple, but is still passed to every job explicitly - see run_job below)
combinations=()
for n_filt in "${NUM_FILTERS[@]}"; do
    for k_size in "${KERNEL_SIZE[@]}"; do
        for drop in "${DROPOUT[@]}"; do
            for lr in "${LEARNING_RATE[@]}"; do
                for win in "${WINDOW_SIZE[@]}"; do
                    combinations+=("$n_filt,$k_size,$drop,$lr,$win")
                done
            done
        done
    done
done

TOTAL_COMBINATIONS=${#combinations[@]}
echo "Total hyperparameter combinations: $TOTAL_COMBINATIONS"
echo "num_levels fixed at: $NUM_LEVELS (see HYPERPARAMETERS.md / DEBUG_README.md - not swept)"

# Check if we have enough cores
if [ "$TOTAL_COMBINATIONS" -gt "$NUM_CORES" ]; then
    echo "Warning: More combinations ($TOTAL_COMBINATIONS) than cores ($NUM_CORES)"
    echo "Jobs will be queued and run as cores become available"
fi

# Function to run a single job
run_job() {
    local combo=$1
    local core=$2
    local job_id=$3

    IFS=',' read -r n_filt k_size drop lr win <<< "$combo"

    # Create job-specific log file. learning_rate and window_size are part of the swept grid, so
    # both are in the tag too - a tag missing a swept dimension is exactly the filename-collision
    # bug documented in DEBUG_README.md (§4 / §14.5): two combos differing only in one of them
    # would otherwise silently overwrite each other's log.
    local log_file="${LOG_DIR}/hp_tuning_${MODEL_TYPE}_job${job_id}_f${n_filt}_k${k_size}_d${drop}_lr${lr}_lv${NUM_LEVELS}_w${win}.log"

    echo "[$job_id/$TOTAL_COMBINATIONS] Starting on core $core: num_filters=$n_filt, kernel_size=$k_size, dropout=$drop, learning_rate=$lr, num_levels=$NUM_LEVELS, window=$win"

    # Pin to specific core using taskset
    if [ "$MODEL_TYPE" == "regional" ]; then
        taskset -c $core "$PYTHON_CMD" "$SCRIPT_PATH" train \
            --num_filters $n_filt \
            --kernel_size $k_size \
            --dropout $drop \
            --learning_rate $lr \
            --num_levels $NUM_LEVELS \
            --seq_length $win \
            --seed 42 > "$log_file" 2>&1
    else
        taskset -c $core "$PYTHON_CMD" "$SCRIPT_PATH" \
            --num_filters $n_filt \
            --kernel_size $k_size \
            --dropout $drop \
            --learning_rate $lr \
            --num_levels $NUM_LEVELS \
            --timesteps $win > "$log_file" 2>&1
    fi

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$job_id/$TOTAL_COMBINATIONS] Completed successfully: num_filters=$n_filt, kernel_size=$k_size, dropout=$drop, learning_rate=$lr, window=$win"
    else
        echo "[$job_id/$TOTAL_COMBINATIONS] FAILED (exit code $exit_code): num_filters=$n_filt, kernel_size=$k_size, dropout=$drop, learning_rate=$lr, window=$win"
    fi
}

# Export function and variables for parallel execution
export -f run_job
export PYTHON_CMD
export SCRIPT_PATH
export LOG_DIR
export MODEL_TYPE
export TOTAL_COMBINATIONS
export NUM_LEVELS

# Run jobs in parallel with GNU parallel if available, otherwise use background jobs
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for job management"

    # Create a temporary file with all jobs
    JOBS_FILE=$(mktemp)
    for i in "${!combinations[@]}"; do
        job_id=$((i + 1))
        core=$((START_CORE + (i % NUM_CORES)))
        echo "${combinations[$i]} $core $job_id" >> "$JOBS_FILE"
    done

    # Run with GNU parallel
    cat "$JOBS_FILE" | parallel --colsep ' ' -j $NUM_CORES run_job {1} {2} {3}

    rm "$JOBS_FILE"
else
    echo "GNU parallel not found, using background jobs"

    # Track running jobs
    job_pids=()
    core_idx=0

    for i in "${!combinations[@]}"; do
        job_id=$((i + 1))
        core=$((START_CORE + (core_idx % NUM_CORES)))

        # Run job in background
        run_job "${combinations[$i]}" $core $job_id &
        job_pids+=($!)

        core_idx=$((core_idx + 1))

        # Wait if we've used all cores
        if [ $((core_idx % NUM_CORES)) -eq 0 ]; then
            # Wait for all current jobs to finish
            for pid in "${job_pids[@]}"; do
                wait $pid
            done
            job_pids=()
        fi
    done

    # Wait for remaining jobs
    for pid in "${job_pids[@]}"; do
        wait $pid
    done
fi

echo ""
echo "============================================"
echo "All hyperparameter tuning jobs completed!"
echo "Model Type: $MODEL_TYPE"
echo "Total combinations tested: $TOTAL_COMBINATIONS"
echo "Logs saved in: $LOG_DIR"
echo "============================================"
