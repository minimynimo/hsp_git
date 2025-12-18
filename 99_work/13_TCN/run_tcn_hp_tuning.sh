#!/bin/bash

# TCN Hyperparameter Tuning Script
# Runs all hyperparameter combinations in parallel using specified cores
# Usage: ./run_tcn_hp_tuning.sh <StartCore> <NumCores> <ModelType>
# Example: ./run_tcn_hp_tuning.sh 0 8 regional
# Example: ./run_tcn_hp_tuning.sh 8 8 individual

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <StartCore> <NumCores> <ModelType>"
    echo "  StartCore: Starting core number (e.g., 0)"
    echo "  NumCores: Number of cores to use (e.g., 8)"
    echo "  ModelType: 'regional' or 'individual'"
    exit 1
fi

START_CORE=$1
NUM_CORES=$2
MODEL_TYPE=$3
END_CORE=$((START_CORE + NUM_CORES - 1))

echo "Starting TCN hyperparameter tuning"
echo "Model Type: $MODEL_TYPE"
echo "Using cores: $START_CORE to $END_CORE"
echo "Total cores: $NUM_CORES"

# Python executable (use virtual environment if available)
PYTHON_CMD="/data0/funato/hsp/.venv/bin/python"
if [ ! -f "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
fi

# Hyperparameter grids based on model type
if [ "$MODEL_TYPE" == "regional" ]; then
    NUM_FILTERS=(32 64)
    KERNEL_SIZE=(2 4)
    DROPOUT=(0.2 0.4)
    SCRIPT_PATH="/data0/funato/hsp/99_work/13_TCN/TCN_regional_modeling-master/main_tcn_hp.py"
    LOG_DIR="/data0/funato/hsp/99_work/13_TCN/TCN_regional_modeling-master/logs"
elif [ "$MODEL_TYPE" == "individual" ]; then
    NUM_FILTERS=(16 32)
    KERNEL_SIZE=(2 4)
    DROPOUT=(0.2 0.4)
    SCRIPT_PATH="/data0/funato/hsp/99_work/13_TCN/main_TCN_indiv_hp.py"
    LOG_DIR="/data0/funato/hsp/99_work/13_TCN/logs"
else
    echo "Error: ModelType must be 'regional' or 'individual'"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Generate all hyperparameter combinations
combinations=()
for n_filt in "${NUM_FILTERS[@]}"; do
    for k_size in "${KERNEL_SIZE[@]}"; do
        for drop in "${DROPOUT[@]}"; do
            combinations+=("$n_filt,$k_size,$drop")
        done
    done
done

TOTAL_COMBINATIONS=${#combinations[@]}
echo "Total hyperparameter combinations: $TOTAL_COMBINATIONS"

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
    
    IFS=',' read -r n_filt k_size drop <<< "$combo"
    
    # Create job-specific log file
    local log_file="${LOG_DIR}/hp_tuning_${MODEL_TYPE}_job${job_id}_f${n_filt}_k${k_size}_d${drop}.log"
    
    echo "[$job_id/$TOTAL_COMBINATIONS] Starting on core $core: num_filters=$n_filt, kernel_size=$k_size, dropout=$drop"
    
    # Pin to specific core using taskset
    if [ "$MODEL_TYPE" == "regional" ]; then
        taskset -c $core "$PYTHON_CMD" "$SCRIPT_PATH" train \
            --num_filters $n_filt \
            --kernel_size $k_size \
            --dropout $drop \
            --seed 42 > "$log_file" 2>&1
    else
        taskset -c $core "$PYTHON_CMD" "$SCRIPT_PATH" \
            --num_filters $n_filt \
            --kernel_size $k_size \
            --dropout $drop > "$log_file" 2>&1
    fi
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$job_id/$TOTAL_COMBINATIONS] Completed successfully: num_filters=$n_filt, kernel_size=$k_size, dropout=$drop"
    else
        echo "[$job_id/$TOTAL_COMBINATIONS] FAILED (exit code $exit_code): num_filters=$n_filt, kernel_size=$k_size, dropout=$drop"
    fi
}

# Export function and variables for parallel execution
export -f run_job
export PYTHON_CMD
export SCRIPT_PATH
export LOG_DIR
export MODEL_TYPE
export TOTAL_COMBINATIONS

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
        
        ((core_idx++))
        
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
