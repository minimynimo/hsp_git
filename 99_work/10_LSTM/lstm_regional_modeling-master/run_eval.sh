#!/bin/bash

# LSTM Regional and Adapted Evaluation Script
# This script runs both regional and adapted evaluation approaches
# and formats the output to match the expected naming convention

set -e  # Exit on any error

# Default parameters
RUN_DIR=""
MODEL_EPOCH=30
N_BASINS=""
PYTHON_PATH="/data0/funato/hsp/.venv/bin/python"

START_CORE=7 # 1 indexed
CORES=1
END_CORE=$((START_CORE + CORES - 1))

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_dir)
            RUN_DIR="$2"
            shift 2
            ;;
        --model_epoch)
            MODEL_EPOCH="$2"
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
            echo "Usage: $0 --run_dir <path> [--model_epoch <epoch>] [--n_basins <num>] [--python_path <path>]"
            echo ""
            echo "Options:"
            echo "  --run_dir       Path to the trained model directory (required)"
            echo "  --model_epoch   Model epoch to use (default: 30)"
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
HIDDEN_SIZE=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json'))['hidden_size'])")
LEARNING_RATE=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json'))['learning_rate'])")
EPOCHS=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json'))['epochs'])")
SEQ_LENGTH=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json'))['seq_length'])")
BATCH_SIZE=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json'))['batch_size'])")
DROPOUT=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json'))['dropout'])")

# Create parameter string for file naming
PARAM_STRING="h${HIDDEN_SIZE}_lr${LEARNING_RATE}_e${EPOCHS}_w${SEQ_LENGTH}_b${BATCH_SIZE}_d${DROPOUT}"

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
    local cmd="taskset -c $START_CORE-$END_CORE $PYTHON_PATH $script_name --run_dir $RUN_DIR --model_epoch $MODEL_EPOCH"
    if [[ -n "$N_BASINS" ]]; then
        cmd="$cmd --n_basins $N_BASINS"
    fi
    
    echo "Command: $cmd"
    echo "Using CPU cores: $START_CORE-$END_CORE"
    
    # Run the evaluation
    if ! $cmd; then
        echo "Error: $eval_type evaluation failed"
        exit 1
    fi
    
    echo "$eval_type evaluation completed successfully"
}

# Main execution
echo "Starting LSTM evaluation pipeline..."
echo "Run directory: $RUN_DIR"
echo "Model epoch: $MODEL_EPOCH"
if [[ -n "$N_BASINS" ]]; then
    echo "Processing $N_BASINS basins (testing mode)"
fi

# Run regional evaluation
run_evaluation "regional" "/data0/funato/hsp/99_work/10_LSTM/lstm_regional_modeling-master/eval_regional.py"

# Run adapted evaluation
#run_evaluation "adapted" "/data0/funato/hsp/99_work/10_LSTM/lstm_regional_modeling-master/eval_adapted.py"

echo ""
echo "========================================="
echo "All evaluations completed successfully!"
echo "========================================="
echo ""
echo "Output files created with parameter string: $PARAM_STRING"
echo ""
echo "Regional evaluation:"
echo "  - Predictions: /data0/funato/hsp/0_out/LSTM/GB/regional/predict/"
echo "    - LSTM_predict_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - LSTM_predict_${PARAM_STRING}_eva.csv (evaluation period)"
echo "    - synthetic/LSTM_predict_${PARAM_STRING}_*_eva.csv (synthetic scenarios)"
echo "  - Results: /data0/funato/hsp/0_out/LSTM/GB/regional/results/"
echo "    - LSTM_results_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - LSTM_results_${PARAM_STRING}_eva.csv (evaluation period)"
echo ""
echo "Adapted evaluation:"
echo "  - Predictions: /data0/funato/hsp/0_out/LSTM/GB/adapted/predict/"
echo "    - LSTM_predict_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - LSTM_predict_${PARAM_STRING}_eva.csv (evaluation period)"
echo "  - Results: /data0/funato/hsp/0_out/LSTM/GB/adapted/results/"
echo "    - LSTM_results_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - LSTM_results_${PARAM_STRING}_eva.csv (evaluation period)"
echo ""

# run using 
# chmod +x /data0/funato/hsp/99_work/10_LSTM/lstm_regional_modeling-master/run_eval.sh
# regional
# nohup /data0/funato/hsp/99_work/10_LSTM/lstm_regional_modeling-master/run_eval.sh --run_dir /data0/funato/hsp/99_work/10_LSTM/lstm_regional_modeling-master/runs/run_2012_0342_seed566531> output_eval_lstm_regional.log 2>&1 &
