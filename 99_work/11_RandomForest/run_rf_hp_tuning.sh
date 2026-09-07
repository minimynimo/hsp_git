#!/bin/bash

# Random Forest Hyperparameter Tuning Script
# Runs all hyperparameter combinations in parallel using specified cores
# Usage: ./run_rf_hp_tuning.sh <StartCore> <NumCores> <ModelType> [ThreadsPerJob]
# nohup /data0/funato/hsp2/99_work/11_RandomForest/run_rf_hp_tuning.sh 0 80 regional 8 &
# nohup /data0/funato/hsp2/99_work/11_RandomForest/run_rf_hp_tuning.sh 40 5 individual &
#
# ThreadsPerJob (regional only, default 1): each RF job's tree-building is
# embarrassingly parallel across trees, so instead of pinning every job to a
# single core, give it a range of ThreadsPerJob cores and pass --n_jobs
# accordingly. NumCores / ThreadsPerJob jobs run concurrently.

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <StartCore> <NumCores> <ModelType> [ThreadsPerJob]"
    echo "  StartCore: Starting core number (e.g., 0)"
    echo "  NumCores: Number of cores to use (e.g., 18)"
    echo "  ModelType: 'regional' or 'individual'"
    echo "  ThreadsPerJob: cores given to each RF job (regional only, default 1)"
    exit 1
fi

START_CORE=$1
NUM_CORES=$2
MODEL_TYPE=$3
THREADS_PER_JOB=${4:-1}
END_CORE=$((START_CORE + NUM_CORES - 1))
NUM_CONCURRENT_JOBS=$((NUM_CORES / THREADS_PER_JOB))

if [ "$NUM_CONCURRENT_JOBS" -lt 1 ]; then
    echo "Error: ThreadsPerJob ($THREADS_PER_JOB) exceeds NumCores ($NUM_CORES)"
    exit 1
fi

echo "Starting RF hyperparameter tuning"
echo "Model Type: $MODEL_TYPE"
echo "Using cores: $START_CORE to $END_CORE"
echo "Total cores: $NUM_CORES"
echo "Threads per job: $THREADS_PER_JOB (=> $NUM_CONCURRENT_JOBS concurrent jobs)"

# Python executable (use virtual environment if available)
PYTHON_CMD="/data0/funato/.venv/bin/python3"
if [ ! -f "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
fi

# Hyperparameter grids based on model type (per /data0/funato/hsp2/HYPERPARAMETERS.md)
if [ "$MODEL_TYPE" == "regional" ]; then # 24
    N_ESTIMATORS=(300 500)
    MIN_SAMPLES_LEAF=(5 10 20)
    MAX_DEPTH=(15 30)
    MAX_FEATURES=("sqrt" 0.33)
    SCRIPT_PATH="/data0/funato/hsp2/99_work/11_RandomForest/RF_regional_modeling-master/main_rf_hp.py"
    LOG_DIR="/data0/funato/hsp2/99_work/11_RandomForest/RF_regional_modeling-master/logs"
elif [ "$MODEL_TYPE" == "individual" ]; then # 54
    N_ESTIMATORS=(100 300 500)
    MIN_SAMPLES_LEAF=(5 10 20)
    MAX_DEPTH=(5 10 15)
    MAX_FEATURES=("sqrt" 0.33)
    SCRIPT_PATH="/data0/funato/hsp2/99_work/11_RandomForest/main_random_forest_indiv_hp.py"
    LOG_DIR="/data0/funato/hsp2/99_work/11_RandomForest/logs"
else
    echo "Error: ModelType must be 'regional' or 'individual'"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# For "regional", the feature matrix (X/y/scalers) is identical across every
# hyperparameter combination -- build and cache it once, single-process, before
# launching the parallel jobs below. Otherwise every one of the (up to 24)
# parallel jobs would redundantly rebuild it from scratch, and racing to build
# the same cache entry simultaneously could corrupt it.
if [ "$MODEL_TYPE" == "regional" ]; then
    echo "Precomputing/caching regional feature matrix (one-time, single process)..."
    (cd "$(dirname "$SCRIPT_PATH")" && "$PYTHON_CMD" precompute_dataset.py)
    if [ $? -ne 0 ]; then
        echo "Error: dataset precompute failed"
        exit 1
    fi
fi

# Generate all hyperparameter combinations
combinations=()
for n_est in "${N_ESTIMATORS[@]}"; do
    for min_leaf in "${MIN_SAMPLES_LEAF[@]}"; do
        for max_d in "${MAX_DEPTH[@]}"; do
            for max_f in "${MAX_FEATURES[@]}"; do
                combinations+=("$n_est,$min_leaf,$max_d,$max_f")
            done
        done
    done
done

TOTAL_COMBINATIONS=${#combinations[@]}
echo "Total hyperparameter combinations: $TOTAL_COMBINATIONS"

# Check if we have enough cores
if [ "$TOTAL_COMBINATIONS" -gt "$NUM_CONCURRENT_JOBS" ]; then
    echo "Warning: More combinations ($TOTAL_COMBINATIONS) than concurrent job slots ($NUM_CONCURRENT_JOBS)"
    echo "Jobs will be queued and run as slots become available"
fi

# Function to run a single job
run_job() {
    local combo=$1
    local core=$2
    local job_id=$3
    
    IFS=',' read -r n_est min_leaf max_d max_f <<< "$combo"
    
    # Sanitize max_features for filename (replace / with underscore)
    local max_f_clean="${max_f//\//_}"
    
    # Create job-specific log file
    local log_file="${LOG_DIR}/hp_tuning_${MODEL_TYPE}_job${job_id}_n${n_est}_l${min_leaf}_d${max_d}_f${max_f_clean}.log"
    
    echo "[$job_id/$TOTAL_COMBINATIONS] Starting on cores $core ($THREADS_PER_JOB threads): n_estimators=$n_est, min_samples_leaf=$min_leaf, max_depth=$max_d, max_features=$max_f"

    # Pin to a core range using taskset; regional jobs get THREADS_PER_JOB
    # threads (RF tree-building parallelizes across trees)
    if [ "$MODEL_TYPE" == "regional" ]; then
        if [ "$max_d" == "None" ]; then
            taskset -c $core "$PYTHON_CMD" "$SCRIPT_PATH" train \
                --n_estimators $n_est \
                --min_samples_leaf $min_leaf \
                --max_features "$max_f" \
                --n_jobs $THREADS_PER_JOB \
                --seed 42 > "$log_file" 2>&1
        else
            taskset -c $core "$PYTHON_CMD" "$SCRIPT_PATH" train \
                --n_estimators $n_est \
                --min_samples_leaf $min_leaf \
                --max_depth $max_d \
                --max_features "$max_f" \
                --n_jobs $THREADS_PER_JOB \
                --seed 42 > "$log_file" 2>&1
        fi
    else
        taskset -c $core "$PYTHON_CMD" "$SCRIPT_PATH" \
            --n_estimators $n_est \
            --min_samples_leaf $min_leaf \
            --max_depth $max_d \
            --max_features "$max_f" > "$log_file" 2>&1
    fi
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$job_id/$TOTAL_COMBINATIONS] Completed successfully: n_estimators=$n_est, min_samples_leaf=$min_leaf, max_depth=$max_d, max_features=$max_f"
    else
        echo "[$job_id/$TOTAL_COMBINATIONS] FAILED (exit code $exit_code): n_estimators=$n_est, min_samples_leaf=$min_leaf, max_depth=$max_d, max_features=$max_f"
    fi
}

# Export function and variables for parallel execution
export -f run_job
export PYTHON_CMD
export SCRIPT_PATH
export LOG_DIR
export MODEL_TYPE
export TOTAL_COMBINATIONS
export THREADS_PER_JOB

# Run jobs in parallel with GNU parallel if available, otherwise use background jobs
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for job management"
    
    # Create a temporary file with all jobs
    JOBS_FILE=$(mktemp)
    for i in "${!combinations[@]}"; do
        job_id=$((i + 1))
        slot=$((i % NUM_CONCURRENT_JOBS))
        core_start=$((START_CORE + slot * THREADS_PER_JOB))
        core_end=$((core_start + THREADS_PER_JOB - 1))
        core="${core_start}-${core_end}"
        echo "${combinations[$i]} $core $job_id" >> "$JOBS_FILE"
    done

    # Run with GNU parallel
    cat "$JOBS_FILE" | parallel --colsep ' ' -j $NUM_CONCURRENT_JOBS run_job {1} {2} {3}

    rm "$JOBS_FILE"
else
    echo "GNU parallel not found, using background jobs"

    # Track running jobs
    job_pids=()
    core_idx=0

    for i in "${!combinations[@]}"; do
        job_id=$((i + 1))
        slot=$((core_idx % NUM_CONCURRENT_JOBS))
        core_start=$((START_CORE + slot * THREADS_PER_JOB))
        core_end=$((core_start + THREADS_PER_JOB - 1))
        core="${core_start}-${core_end}"

        # Run job in background
        run_job "${combinations[$i]}" $core $job_id &
        job_pids+=($!)

        ((core_idx++))

        # Wait if we've used all concurrent job slots
        if [ $((core_idx % NUM_CONCURRENT_JOBS)) -eq 0 ]; then
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
