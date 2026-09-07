#!/bin/bash

# LSTM Hyperparameter Evaluation Script
# This script evaluates all hyperparameter-tuned LSTM models

set -e  # Exit on any error

#LOC="GB"  # Location identifier
LOC="US"  # Location identifier

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_PATH="/data0/funato/hsp/.venv/bin/python"
RUNS_BASE="$SCRIPT_DIR/runs"
OUTPUT_DIR="/data0/funato/hsp2/0_out/LSTM/$LOC/regional/hp"
MODEL_EPOCH=30

# Core assignment for parallel execution
START_CORE=3
CORES_PER_JOB=1

# Hyperparameter grid
#HIDDEN_SIZES=(128 256)
#DROPOUTS=(0.2 0.4)

HIDDEN_SIZES=(256 512) #128
DROPOUTS=(0.2 0.0) #0.4 

# Function to find the most recent run directory for given hyperparameters
find_run_dir() {
    local h=$1
    local d=$2
    local hp_dir="$RUNS_BASE/hp/$LOC/h${h}_d${d}_lr0.0005"
    
    if [[ ! -d "$hp_dir" ]]; then
        echo "WARNING: HP directory not found: $hp_dir" >&2
        return 1
    fi
    
    # Find most recent run directory (sorted by timestamp in dirname)
    local run_dir=$(ls -dt "$hp_dir"/run_*_seed42 2>/dev/null | head -n1)
    
    if [[ -z "$run_dir" ]]; then
        echo "WARNING: No run directory found in $hp_dir" >&2
        return 1
    fi
    
    echo "$run_dir"
    return 0
}

# Main evaluation loop
echo "========================================="
echo "LSTM Hyperparameter Evaluation"
echo "========================================="
echo "Base directory: $RUNS_BASE"
echo "Output directory: $OUTPUT_DIR"
echo "Model epoch: $MODEL_EPOCH"
echo ""

core_offset=0

for h in "${HIDDEN_SIZES[@]}"; do
    for d in "${DROPOUTS[@]}"; do
        hp_string="h${h}_d${d}_lr0.0005"
        echo "Processing hyperparameters: $hp_string"
        
        # Find run directory
        if ! run_dir=$(find_run_dir "$h" "$d"); then
            echo "Skipping $hp_string - no run directory found"
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

# Wait for all background jobs to complete
echo ""
echo "Waiting for all evaluations to complete..."
wait

echo ""
echo "========================================="
echo "All hyperparameter evaluations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================="

# nohup /data0/funato/hsp2/99_work/10_LSTM/lstm_regional_modeling-master/run_eval_lstm_hp.sh > output_eval_lstm_hp_regional.log 2>&1 &
