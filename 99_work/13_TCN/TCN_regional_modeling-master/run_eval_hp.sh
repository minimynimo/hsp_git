#!/bin/bash

# TCN Hyperparameter Evaluation Script
# This script evaluates all hyperparameter-tuned TCN models

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_PATH="/data0/funato/hsp/.venv/bin/python"
RUNS_BASE="$SCRIPT_DIR/runs_tcn/hp"
OUTPUT_DIR="/data0/funato/hsp/0_out/TCN/GB/regional/hp"
MODEL_EPOCH=30

# Core assignment for parallel execution
START_CORE=5
CORES_PER_JOB=3

# Hyperparameter grid
NB_FILTERS=(32 64)
KERNEL_SIZES=(2 4)
DROPOUTS=(0.2 0.4)

# Function to find the run directory for given hyperparameters
find_run_dir() {
    local f=$1
    local k=$2
    local d=$3
    local hp_dir="$RUNS_BASE/f${f}_k${k}_d${d}"
    
    if [[ ! -d "$hp_dir" ]]; then
        echo "WARNING: HP directory not found: $hp_dir" >&2
        return 1
    fi
    
    # Find most recent run directory (sorted by timestamp in dirname)
    local run_dir=$(ls -dt "$hp_dir"/tcn_run_*_seed42 2>/dev/null | head -n1)
    
    if [[ -z "$run_dir" ]]; then
        echo "WARNING: No run directory found in $hp_dir" >&2
        return 1
    fi
    
    # Check if model file exists
    if [[ ! -f "$run_dir/tcn_model_epoch${MODEL_EPOCH}.pt" ]]; then
        echo "WARNING: No model epoch ${MODEL_EPOCH} found in $run_dir" >&2
        return 1
    fi
    
    echo "$run_dir"
    return 0
}

# Main evaluation loop
echo "========================================="
echo "TCN Hyperparameter Evaluation"
echo "========================================="
echo "Base directory: $RUNS_BASE"
echo "Output directory: $OUTPUT_DIR"
echo "Model epoch: $MODEL_EPOCH"
echo ""

core_offset=0

for f in "${NB_FILTERS[@]}"; do
    for k in "${KERNEL_SIZES[@]}"; do
        for d in "${DROPOUTS[@]}"; do
            hp_string="f${f}_k${k}_d${d}"
            echo "Processing hyperparameters: $hp_string"
            
            # Find run directory
            if ! run_dir=$(find_run_dir "$f" "$k" "$d"); then
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
                    --model_epoch $MODEL_EPOCH \
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
echo "Results saved to: $OUTPUT_DIR"
echo "========================================="
echo "    - synthetic/TCN_predict_${PARAM_STRING}_*_eva.csv (synthetic scenarios)"
echo "  - Results: /data0/funato/hsp/0_out/TCN/GB/regional/results/"
echo "    - TCN_results_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - TCN_results_${PARAM_STRING}_eva.csv (evaluation period)"
echo ""


###RUN USING####
# chmod +x /data0/funato/hsp/99_work/13_TCN/TCN_regional_modeling-master/run_eval.sh
