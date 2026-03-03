#!/bin/bash

# Random Forest Regional Hyperparameter Evaluation Script
# This script evaluates all trained RF hyperparameter models
# Based on the combinations from run_rf_hp_tuning.sh

set -e  # Exit on any error

# Default parameters
START_CORE=0
NUM_CORES=8
N_BASINS=""
PYTHON_PATH="/data0/funato/hsp/.venv/bin/python"
RUNS_BASE_DIR="/data0/funato/hsp2/99_work/11_RandomForest/RF_regional_modeling-master/runs_rf"
EVAL_SCRIPT="/data0/funato/hsp2/99_work/11_RandomForest/RF_regional_modeling-master/eval_regional_hp.py"
OUTPUT_BASE="/data0/funato/hsp2/0_out/RF/GB/regional/hp"
LOG_DIR="/data0/funato/hsp2/99_work/11_RandomForest/RF_regional_modeling-master/logs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start_core)
            START_CORE="$2"
            shift 2
            ;;
        --num_cores)
            NUM_CORES="$2"
            shift 2
            ;;
        --n_basins)
            N_BASINS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --start_core    Starting core number (default: 0)"
            echo "  --num_cores     Number of cores to use (default: 8)"
            echo "  --n_basins      Number of basins to process (optional, for testing)"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

END_CORE=$((START_CORE + NUM_CORES - 1))

# Create output directories
mkdir -p "$OUTPUT_BASE/predict"
mkdir -p "$OUTPUT_BASE/results"
mkdir -p "$LOG_DIR"

echo "Starting Random Forest Regional Hyperparameter Evaluation"
echo "=========================================================="
echo "Using cores: $START_CORE to $END_CORE"
echo "Output directory: $OUTPUT_BASE"
echo ""

# Define hyperparameter combinations (matching run_rf_hp_tuning.sh)
N_ESTIMATORS=(100 300 500)
MIN_SAMPLES_LEAF=(2 5)
MAX_FEATURES=("sqrt")

# Generate all combinations
combinations=()
for n_est in "${N_ESTIMATORS[@]}"; do
    for min_leaf in "${MIN_SAMPLES_LEAF[@]}"; do
        for max_f in "${MAX_FEATURES[@]}"; do
            combinations+=("$n_est,$min_leaf,$max_f")
        done
    done
done

TOTAL_COMBINATIONS=${#combinations[@]}
echo "Total hyperparameter combinations to evaluate: $TOTAL_COMBINATIONS"
echo ""

# Function to find the most recent run directory for a given HP combination
find_run_dir() {
    local n_est=$1
    local min_leaf=$2
    local max_f=$3
    
    # Sanitize max_features for directory name
    local max_f_dir
    if [ "$max_f" == "sqrt" ]; then
        max_f_dir="fsqrt"
    else
        max_f_dir="f0.33"
    fi
    
    # Construct HP directory name: n<n>_dNone_l<l>_f<f>
    local hp_dir="n${n_est}_dNone_l${min_leaf}_${max_f_dir}"
    
    # Find most recent run directory inside the HP directory
    local run_dir=$(ls -td "$RUNS_BASE_DIR"/hp/"${hp_dir}"/rf_run_* 2>/dev/null | head -1)
    
    echo "$run_dir"
}

# Function to run evaluation for a single HP combination
run_evaluation() {
    local combo=$1
    local core=$2
    local job_id=$3
    
    IFS=',' read -r n_est min_leaf max_f <<< "$combo"
    
    # Find the run directory
    local run_dir=$(find_run_dir "$n_est" "$min_leaf" "$max_f")
    
    if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
        echo "[$job_id/$TOTAL_COMBINATIONS] ERROR: Run directory not found for n=$n_est, l=$min_leaf, f=$max_f"
        return 1
    fi
    
    # Sanitize max_features for filenames
    local max_f_clean="${max_f//\//_}"
    
    # Create HP string for output naming
    local hp_str="n${n_est}_l${min_leaf}_f${max_f_clean}"
    
    echo "[$job_id/$TOTAL_COMBINATIONS] Evaluating: $hp_str"
    echo "  Run directory: $run_dir"
    
    # Create log file
    local log_file="${LOG_DIR}/eval_hp_${hp_str}.log"
    
    # Build evaluation command (skip synthetic scenarios for HP tuning)
    local cmd="taskset -c $core $PYTHON_PATH $EVAL_SCRIPT --run_dir $run_dir --output_dir $OUTPUT_BASE --hp_string $hp_str --skip_synthetic"
    
    if [[ -n "$N_BASINS" ]]; then
        cmd="$cmd --n_basins $N_BASINS"
    fi
    
    # Run evaluation
    if $cmd > "$log_file" 2>&1; then
        echo "[$job_id/$TOTAL_COMBINATIONS] SUCCESS: $hp_str"
        return 0
    else
        echo "[$job_id/$TOTAL_COMBINATIONS] FAILED: $hp_str (see $log_file)"
        return 1
    fi
}

# Export functions and variables for parallel execution
export -f run_evaluation
export -f find_run_dir
export PYTHON_PATH
export EVAL_SCRIPT
export OUTPUT_BASE
export LOG_DIR
export RUNS_BASE_DIR
export TOTAL_COMBINATIONS
export N_BASINS

# Run evaluations in parallel
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for job management"
    
    # Create temporary file with all jobs
    JOBS_FILE=$(mktemp)
    for i in "${!combinations[@]}"; do
        job_id=$((i + 1))
        core=$((START_CORE + (i % NUM_CORES)))
        echo "${combinations[$i]} $core $job_id" >> "$JOBS_FILE"
    done
    
    # Run with GNU parallel
    cat "$JOBS_FILE" | parallel --colsep ' ' -j $NUM_CORES run_evaluation {1} {2} {3}
    
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
        run_evaluation "${combinations[$i]}" $core $job_id &
        job_pids+=($!)
        
        ((++core_idx))
        
        # Wait if we've used all cores
        if [ $((core_idx % NUM_CORES)) -eq 0 ]; then
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
echo "========================================="
echo "All hyperparameter evaluations completed!"
echo "========================================="
echo ""
echo "Output location:"
echo "  - Predictions: $OUTPUT_BASE/predict/"
echo "  - Results: $OUTPUT_BASE/results/"
echo "  - Logs: $LOG_DIR/eval_hp_*.log"
echo ""
echo "File naming convention: RF_predict_<hp_string>_<period>.csv"
echo "  where hp_string = n<n_estimators>_l<min_samples_leaf>_f<max_features>"
echo ""

###RUN USING####
# chmod +x /data0/funato/hsp2/hsp_git/99_work/11_RandomForest/RF_regional_modeling-master/run_eval_hp.sh
# /data0/funato/hsp2/hsp_git/99_work/11_RandomForest/RF_regional_modeling-master/run_eval_hp.sh --start_core 0 --num_cores 8
# /data0/funato/hsp2/hsp_git/99_work/11_RandomForest/RF_regional_modeling-master/run_eval_hp.sh --start_core 30 --num_cores 2 > output_eval_rf_hp_regional.log 2>&1 &