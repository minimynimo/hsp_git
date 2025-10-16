#!/bin/bash

# Random Forest Regional and Adapted Evaluation Script
# This script runs both regional and adapted evaluation approaches
# and formats the output to match the expected naming convention

set -e  # Exit on any error

# Default parameters
RUN_DIR=""
N_BASINS=""
PYTHON_PATH="/data0/funato/hsp/.venv/bin/python"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_dir)
            RUN_DIR="$2"
            shift 2
            ;;
        --n_basins)
            N_BASINS="$2"
            shift 2
            ;;
        --python_path)
            PYTHON_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --run_dir <path> [--n_basins <num>] [--python_path <path>]"
            echo ""
            echo "Options:"
            echo "  --run_dir       Path to the trained model directory (required)"
            echo "  --n_basins      Number of basins to process (optional, for testing)"
            echo "  --python_path   Path to Python executable (default: /data0/funato/hsp/.venv/bin/python)"
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
MAX_DEPTH=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json')).get('max_depth', 5))")
MAX_FEATURES=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json')).get('max_features', 0.7))")
MAX_SAMPLES=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json')).get('max_samples', 0.5))")
SEQ_LENGTH=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json')).get('seq_length', 270))")

# Create parameter string for file naming
PARAM_STRING="n${N_ESTIMATORS}_d${MAX_DEPTH}_mf${MAX_FEATURES}_ms${MAX_SAMPLES}_w${SEQ_LENGTH}"

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
    local cmd="$PYTHON_PATH $script_name --run_dir $RUN_DIR"
    if [[ -n "$N_BASINS" ]]; then
        cmd="$cmd --n_basins $N_BASINS"
    fi
    
    echo "Command: $cmd"
    
    # Run the evaluation
    if ! $cmd; then
        echo "Error: $eval_type evaluation failed"
        exit 1
    fi
    
    echo "$eval_type evaluation completed successfully"
}

# Main execution
echo "Starting Random Forest evaluation pipeline..."
echo "Run directory: $RUN_DIR"
if [[ -n "$N_BASINS" ]]; then
    echo "Processing $N_BASINS basins (testing mode)"
fi

# Run regional evaluation
run_evaluation "regional" "/data0/funato/hsp/99_work/11_RandomForest/RF_regional_modeling-master/eval_regional.py"

# Run adapted evaluation
run_evaluation "adapted" "/data0/funato/hsp/99_work/11_RandomForest/RF_regional_modeling-master/eval_adapted.py"

echo ""
echo "========================================="
echo "All evaluations completed successfully!"
echo "========================================="
echo ""
echo "Output files created with parameter string: $PARAM_STRING"
echo ""
echo "Regional evaluation:"
echo "  - Predictions: /data0/funato/hsp/0_out/RF/GB/regional/predict/"
echo "    - RF_predict_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - RF_predict_${PARAM_STRING}_eva.csv (evaluation period)"
echo "    - synthetic/RF_predict_${PARAM_STRING}_*_eva.csv (synthetic scenarios)"
echo "  - Results: /data0/funato/hsp/0_out/RF/GB/regional/results/"
echo "    - RF_results_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - RF_results_${PARAM_STRING}_eva.csv (evaluation period)"
echo ""
echo "Adapted evaluation:"
echo "  - Predictions: /data0/funato/hsp/0_out/RF/GB/adapted/predict/"
echo "    - RF_predict_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - RF_predict_${PARAM_STRING}_eva.csv (evaluation period)"
echo "  - Results: /data0/funato/hsp/0_out/RF/GB/adapted/results/"
echo "    - RF_results_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - RF_results_${PARAM_STRING}_eva.csv (evaluation period)"
echo ""

###RUN USING####
# chmod +x /data0/funato/hsp/99_work/11_RandomForest/RF_regional_modeling-master/run_eval.sh
# /data0/funato/hsp/99_work/11_RandomForest/RF_regional_modeling-master/run_eval.sh --run_dir /data0/funato/hsp/99_work/11_RandomForest/RF_regional_modeling-master/runs_rf/rf_run_1016_1516_seed300 --n_basins 2  # For testing with 2 basins
# /data0/funato/hsp/99_work/11_RandomForest/RF_regional_modeling-master/run_eval.sh --run_dir /data0/funato/hsp/99_work/11_RandomForest/RF_regional_modeling-master/runs_rf/rf_run_1016_1516_seed300