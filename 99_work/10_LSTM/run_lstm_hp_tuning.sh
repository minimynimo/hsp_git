#!/bin/bash

# LSTM Hyperparameter Tuning Script
# Runs all hyperparameter combinations in parallel using specified cores
# Usage: ./run_lstm_hp_tuning.sh <StartCore> <NumCores> <ModelType>
# Example: ./run_lstm_hp_tuning.sh 0 4 regional
# Example: ./run_lstm_hp_tuning.sh 4 4 individual

# ============================================
# HYPERPARAMETER CONFIGURATION
# Edit these arrays to change the hyperparameter grid
# ============================================

# Regional model hyperparameters
REGIONAL_HIDDEN_SIZE=(256 512) #128
REGIONAL_DROPOUT=(0.2 0.0) #0.4 
REGIONAL_LEARNING_RATE=(0.001 0.0005 0.0001)

# Individual model hyperparameters
INDIVIDUAL_HIDDEN_SIZE=(16 32)
INDIVIDUAL_DROPOUT=(0.2 0.4)
INDIVIDUAL_LEARNING_RATE=(0.001)

# ============================================
# END OF HYPERPARAMETER CONFIGURATION
# ============================================

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <StartCore> <NumCores> <ModelType>"
    echo "  StartCore: Starting core number (e.g., 0)"
    echo "  NumCores: Number of cores to use (e.g., 4)"
    echo "  ModelType: 'regional' or 'individual'"
    exit 1
fi

START_CORE=$1
NUM_CORES=$2
MODEL_TYPE=$3
END_CORE=$((START_CORE + NUM_CORES - 1))

echo "Starting LSTM hyperparameter tuning"
echo "Model Type: $MODEL_TYPE"
echo "Using cores: $START_CORE to $END_CORE"
echo "Total cores: $NUM_CORES"

# Python executable (use virtual environment if available)
PYTHON_CMD="/data0/funato/hsp2/.venv/bin/python"
if [ ! -f "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
fi

# Hyperparameter grids based on model type
if [ "$MODEL_TYPE" == "regional" ]; then
    HIDDEN_SIZE=("${REGIONAL_HIDDEN_SIZE[@]}")
    DROPOUT=("${REGIONAL_DROPOUT[@]}")
    LEARNING_RATE=("${REGIONAL_LEARNING_RATE[@]}")
    SCRIPT_PATH="/data0/funato/hsp2/99_work/10_LSTM/lstm_regional_modeling-master/main_hp.py"
    LOG_DIR="/data0/funato/hsp2/99_work/10_LSTM/lstm_regional_modeling-master/logs"
elif [ "$MODEL_TYPE" == "individual" ]; then
    HIDDEN_SIZE=("${INDIVIDUAL_HIDDEN_SIZE[@]}")
    DROPOUT=("${INDIVIDUAL_DROPOUT[@]}")
    LEARNING_RATE=("${INDIVIDUAL_LEARNING_RATE[@]}")
    SCRIPT_PATH="/data0/funato/hsp2/99_work/10_LSTM/main_LSTM_indiv_hp.py"
    LOG_DIR="/data0/funato/hsp2/99_work/10_LSTM/logs"
else
    echo "Error: ModelType must be 'regional' or 'individual'"
    exit 1
fi

echo "Hyperparameter grid:"
echo "  Hidden sizes: ${HIDDEN_SIZE[@]}"
echo "  Dropout rates: ${DROPOUT[@]}"
echo " Learning rates: ${LEARNING_RATE[@]}"

# Create log directory
mkdir -p "$LOG_DIR"

# Generate all hyperparameter combinations
combinations=()
if [ "$MODEL_TYPE" == "regional" ]; then
    for h_size in "${HIDDEN_SIZE[@]}"; do
        for drop in "${DROPOUT[@]}"; do
            for lr in "${LEARNING_RATE[@]}"; do
                combinations+=("$h_size,$drop,$lr")
            done
        done
    done
else
    # Individual models don't use learning_rate parameter (hardcoded in script)
    for h_size in "${HIDDEN_SIZE[@]}"; do
        for drop in "${DROPOUT[@]}"; do
            combinations+=("$h_size,$drop,fixed")
        done
    done
fi

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
    
    IFS=',' read -r h_size drop lr <<< "$combo"
    
    # Create job-specific log file
    local log_file="${LOG_DIR}/hp_tuning_${MODEL_TYPE}_job${job_id}_h${h_size}_d${drop}_lr${lr}.log"
    
    echo "[$job_id/$TOTAL_COMBINATIONS] Starting on core $core: hidden_size=$h_size, dropout=$drop, learning_rate=$lr"
    echo "DEBUG: h_size='$h_size' drop='$drop' lr='$lr'" >> "$log_file"
    
    # Pin to specific core using taskset
    if [ "$MODEL_TYPE" == "regional" ]; then
        taskset -c $core "$PYTHON_CMD" "$SCRIPT_PATH" train \
            --hidden_size "$h_size" \
            --dropout "$drop" \
            --learning_rate "$lr" \
            --seed 42 >> "$log_file" 2>&1
    else
        # Individual models don't accept --learning_rate (hardcoded at 1e-3)
        taskset -c $core "$PYTHON_CMD" "$SCRIPT_PATH" \
            --hidden_size "$h_size" \
            --dropout "$drop" >> "$log_file" 2>&1
    fi
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$job_id/$TOTAL_COMBINATIONS] Completed successfully: hidden_size=$h_size, dropout=$drop, learning_rate=$lr"
    else
        echo "[$job_id/$TOTAL_COMBINATIONS] FAILED (exit code $exit_code): hidden_size=$h_size, dropout=$drop, learning_rate=$lr"
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
    
    # Run with GNU parallel (colsep uses tab by default, we use space for our fields)
    cat "$JOBS_FILE" | parallel -j $NUM_CORES --colsep ' ' 'run_job {1} {2} {3}'
    
    rm "$JOBS_FILE"
else
    echo "GNU parallel not found, using bash background jobs for parallel execution"
    
    # Run jobs in parallel using background processes
    job_count=0
    for i in "${!combinations[@]}"; do
        job_id=$((i + 1))
        core=$((START_CORE + (i % NUM_CORES)))
        
        # Run job in background
        run_job "${combinations[$i]}" $core $job_id &
        
        job_count=$((job_count + 1))
        
        # Wait if we've reached the max number of parallel jobs
        if [ $((job_count % NUM_CORES)) -eq 0 ]; then
            wait  # Wait for all background jobs to complete
        fi
    done
    
    # Wait for any remaining jobs
    wait
fi

echo ""
echo "============================================"
echo "All hyperparameter tuning jobs completed!"
echo "Model Type: $MODEL_TYPE"
echo "Total combinations tested: $TOTAL_COMBINATIONS"
echo "Logs saved in: $LOG_DIR"
echo "============================================"
