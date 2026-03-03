#!/bin/bash

# XGBoost Hyperparameter Evaluation Script
# This script evaluates all hyperparameter-tuned XGBoost models

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_PATH="/data0/funato/hsp/.venv/bin/python"
RUNS_BASE="$SCRIPT_DIR/runs_xgb/hp"
OUTPUT_DIR="/data0/funato/hsp2/0_out/XGB/GB/regional/hp"

# Core assignment for parallel execution
START_CORE=1
CORES_PER_JOB=2

# Hyperparameter grid
N_ESTIMATORS=(500 1000)
MAX_DEPTHS=(5 7)
LEARNING_RATES=(0.05 0.1)

# Function to find the run directory for given hyperparameters
find_run_dir() {
    local n=$1
    local d=$2
    local lr=$3
    local hp_dir="n${n}_d${d}_lr${lr}"
    
    # Find most recent run directory inside the HP directory
    local run_dir=$(ls -td "$RUNS_BASE"/"${hp_dir}"/xgb_run_* 2>/dev/null | head -1)
    
    if [[ -z "$run_dir" ]] || [[ ! -d "$run_dir" ]]; then
        echo "WARNING: No run directory found in $RUNS_BASE/${hp_dir}" >&2
        return 1
    fi
    
    # Check if model file exists
    if [[ ! -f "$run_dir/xgb_model.json" ]]; then
        echo "WARNING: No model file found in $run_dir" >&2
        return 1
    fi
    
    echo "$run_dir"
    return 0
}

# Main evaluation loop
echo "========================================="
echo "XGBoost Hyperparameter Evaluation"
echo "========================================="
echo "Base directory: $RUNS_BASE"
echo "Output directory: $OUTPUT_DIR"
echo ""

core_offset=0

for n in "${N_ESTIMATORS[@]}"; do
    for d in "${MAX_DEPTHS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            hp_string="n${n}_d${d}_lr${lr}"
            echo "Processing hyperparameters: $hp_string"
            
            # Find run directory
            if ! run_dir=$(find_run_dir "$n" "$d" "$lr"); then
                echo "Skipping $hp_string - no valid run directory found"
                continue
            fi
            
            echo "  Run directory: $run_dir"
            
            # Calculate core assignment
            start_core=$((START_CORE + core_offset))
            end_core=$((start_core + CORES_PER_JOB - 1))
            
            # Run evaluation in background with core pinning
            (
                taskset -c ${start_core}-${end_core} $PYTHON_PATH "$SCRIPT_DIR/eval_regional_hp.py" \
                    --run_dir "$run_dir" \
                    --output_dir "$OUTPUT_DIR" \
                    --hp_string "$hp_string" \
                    --skip_synthetic
            ) &
            
            echo "  Started evaluation on cores ${start_core}-${end_core}"
            
            # Increment core offset
            core_offset=$((core_offset + CORES_PER_JOB))
        done
    done
done

# Wait for all background jobs to complete
echo ""
echo "Waiting for all evaluations to complete..."
wait

echo ""
echo "========================================="
echo "All hyperparameter evaluations complete!"
echo "========================================="
echo ""
echo "Output location:"
echo "  - Predictions: $OUTPUT_DIR/predict/"
echo "  - Results: $OUTPUT_DIR/results/"
echo ""
echo "File naming convention: XGB_predict_<hp_string>_<period>.csv"
echo "  where hp_string = n<n_estimators>_d<max_depth>_lr<learning_rate>"
echo ""

###RUN USING####
# chmod +x /data0/funato/hsp2/99_work/12_XGBoost/XGB_regional_modeling-master/run_eval_hp.sh
# /data0/funato/hsp2/99_work/12_XGBoost/XGB_regional_modeling-master/run_eval_hp.sh