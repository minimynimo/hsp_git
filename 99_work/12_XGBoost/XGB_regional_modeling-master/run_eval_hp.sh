#!/bin/bash

# XGBoost Hyperparameter Evaluation Script
# This script evaluates all hyperparameter-tuned XGBoost models

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_PATH="/data0/funato/hsp/.venv/bin/python"
RUNS_BASE="$SCRIPT_DIR/runs_xgb/hp"
OUTPUT_DIR="/data0/funato/hsp/0_out/XGB/GB/regional/hp"

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
    local hp_dir="$RUNS_BASE/n${n}_d${d}_lr${lr}"
    
    if [[ ! -d "$hp_dir" ]]; then
        echo "WARNING: HP directory not found: $hp_dir" >&2
        return 1
    fi
    
    # Check if model file exists
    if [[ ! -f "$hp_dir/xgb_model.json" ]]; then
        echo "WARNING: No model file found in $hp_dir" >&2
        return 1
    fi
    
    echo "$hp_dir"
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
echo "Results saved to: $OUTPUT_DIR"
echo "========================================="

# Check required arguments
if [[ -z "$RUN_DIR" ]]; then
    echo "Error: --run_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if run directory exists
if [[ ! -d "$RUN_DIR" ]]; then
    echo "Error: Run directory does not exist: $RUN_DIR"
    exit 1
fi

# Check if config file exists
if [[ ! -f "$RUN_DIR/cfg.json" ]]; then
    echo "Error: Config file not found: $RUN_DIR/cfg.json"
    exit 1
fi

# Extract parameters from config file to create naming convention
echo "Extracting model parameters from config..."
N_ESTIMATORS=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json')).get('n_estimators', 100))")
MAX_DEPTH=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json')).get('max_depth', 6))")
LEARNING_RATE=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json')).get('learning_rate', 0.1))")
SUBSAMPLE=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json')).get('subsample', 1.0))")
COLSAMPLE_BYTREE=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json')).get('colsample_bytree', 1.0))")
SEQ_LENGTH=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json')).get('seq_length', 270))")

# Create parameter string for file naming
PARAM_STRING="n${N_ESTIMATORS}_d${MAX_DEPTH}_lr${LEARNING_RATE}_sub${SUBSAMPLE}_col${COLSAMPLE_BYTREE}_w${SEQ_LENGTH}"

echo "Model parameters: $PARAM_STRING"

# Function to run evaluation
run_evaluation() {
    local eval_type=$1
    local script_name=$2
    
    echo ""
    echo "========================================="
    echo "Running $eval_type evaluation..."
    echo "========================================="
    
    # Build command
    local cmd="taskset -c $START_CORE-$END_CORE $PYTHON_PATH $script_name --run_dir $RUN_DIR"
    if [[ -n "$N_BASINS" ]]; then
        cmd="$cmd --n_basins $N_BASINS"
    fi
    
    echo "Command: $cmd"
    echo "Using Cores: $START_CORE to $END_CORE"
    
    # Run the evaluation
    if ! $cmd; then
        echo "Error: $eval_type evaluation failed"
        exit 1
    fi
    
    echo "$eval_type evaluation completed successfully"
}

# Main execution
echo "Starting XGBoost evaluation pipeline..."
echo "Run directory: $RUN_DIR"
if [[ -n "$N_BASINS" ]]; then
    echo "Processing $N_BASINS basins (testing mode)"
fi

# Run regional evaluation
#run_evaluation "regional" "/data0/funato/hsp/99_work/12_XGBoost/XGB_regional_modeling-master/eval_regional.py"

# Run adapted evaluation
run_evaluation "adapted" "/data0/funato/hsp/99_work/12_XGBoost/XGB_regional_modeling-master/eval_adapted.py"

echo ""
echo "========================================="
echo "All evaluations completed successfully!"
echo "========================================="
echo ""
echo "Output files created with parameter string: $PARAM_STRING"
echo ""
echo "Regional evaluation:"
echo "  - Predictions: /data0/funato/hsp/0_out/XGB/GB/regional/predict/"
echo "    - XGB_predict_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - XGB_predict_${PARAM_STRING}_eva.csv (evaluation period)"
echo "    - synthetic/XGB_predict_${PARAM_STRING}_*_eva.csv (synthetic scenarios)"
echo "  - Results: /data0/funato/hsp/0_out/XGB/GB/regional/results/"
echo "    - XGB_results_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - XGB_results_${PARAM_STRING}_eva.csv (evaluation period)"
echo ""
echo "Adapted evaluation:"
echo "  - Predictions: /data0/funato/hsp/0_out/XGB/GB/adapted/predict/"
echo "    - XGB_predict_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - XGB_predict_${PARAM_STRING}_eva.csv (evaluation period)"
echo "  - Results: /data0/funato/hsp/0_out/XGB/GB/adapted/results/"
echo "    - XGB_results_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - XGB_results_${PARAM_STRING}_eva.csv (evaluation period)"
echo ""

###RUN USING####
# chmod +x /data0/funato/hsp/99_work/12_XGBoost/XGB_regional_modeling-master/run_eval.sh
# /data0/funato/hsp/99_work/12_XGBoost/XGB_regional_modeling-master/run_eval.sh --run_dir /data0/funato/hsp/99_work/12_XGBoost/XGB_regional_modeling-master/runs_xgb/xgb_run_1016_1614_seed300 --n_basins 2  # For testing with 2 basins
# nohup /data0/funato/hsp/99_work/12_XGBoost/XGB_regional_modeling-master/run_eval.sh --run_dir /data0/funato/hsp/99_work/12_XGBoost/XGB_regional_modeling-master/runs_xgb/xgb_run_1119_1455_seed300 > output_eval_xgb.log 2>&1 &