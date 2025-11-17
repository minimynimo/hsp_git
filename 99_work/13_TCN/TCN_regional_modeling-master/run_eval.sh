#!/bin/bash

# TCN Regional and Adapted Evaluation Script
# This script runs both regional and adapted evaluation approaches
# and formats the output to match the expected naming convention

set -e  # Exit on any error

# Default parameters
RUN_DIR=""
MODEL_EPOCH=50
N_BASINS=""
PYTHON_PATH="/data0/funato/hsp/.venv/bin/python"

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
            echo "  --model_epoch   Model epoch to use (default: 50)"
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

# Check which parameter format is used
if python3 -c "import json; config=json.load(open('$RUN_DIR/cfg.json')); exit(0 if 'nb_filters' in config else 1)" 2>/dev/null; then
    # New format: nb_filters and nb_stacks
    NB_FILTERS=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json')).get('nb_filters', 128))")
    NB_STACKS=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json')).get('nb_stacks', 3))")
    PARAM_CHANNELS="f${NB_FILTERS}_s${NB_STACKS}"
else
    # Old format: num_channels
    NUM_CHANNELS=$(python3 -c "import json; print('-'.join(map(str, json.load(open('$RUN_DIR/cfg.json')).get('num_channels', [25, 50, 100]))))")
    PARAM_CHANNELS="ch${NUM_CHANNELS}"
fi

KERNEL_SIZE=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json')).get('kernel_size', 2))")
DROPOUT=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json')).get('dropout', 0.2))")
LEARNING_RATE=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json')).get('learning_rate', 0.001))")
SEQ_LENGTH=$(python3 -c "import json; print(json.load(open('$RUN_DIR/cfg.json')).get('seq_length', 30))")

# Create parameter string for file naming
PARAM_STRING="${PARAM_CHANNELS}_k${KERNEL_SIZE}_d${DROPOUT}_lr${LEARNING_RATE}_w${SEQ_LENGTH}"

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
    local cmd="$PYTHON_PATH $script_name --run_dir $RUN_DIR --model_epoch $MODEL_EPOCH"
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
echo "Starting TCN evaluation pipeline..."
echo "Run directory: $RUN_DIR"
echo "Model epoch: $MODEL_EPOCH"
if [[ -n "$N_BASINS" ]]; then
    echo "Processing $N_BASINS basins (testing mode)"
fi

# Run regional evaluation
#run_evaluation "regional" "/data0/funato/hsp/99_work/13_TCN/TCN_regional_modeling-master/eval_regional.py"

# Run adapted evaluation
run_evaluation "adapted" "/data0/funato/hsp/99_work/13_TCN/TCN_regional_modeling-master/eval_adapted.py"

echo ""
echo "========================================="
echo "All evaluations completed successfully!"
echo "========================================="
echo ""
echo "Output files created with parameter string: $PARAM_STRING"
echo ""
echo "Regional evaluation:"
echo "  - Predictions: /data0/funato/hsp/0_out/TCN/GB/regional/predict/"
echo "    - TCN_predict_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - TCN_predict_${PARAM_STRING}_eva.csv (evaluation period)"
echo "    - synthetic/TCN_predict_${PARAM_STRING}_*_eva.csv (synthetic scenarios)"
echo "  - Results: /data0/funato/hsp/0_out/TCN/GB/regional/results/"
echo "    - TCN_results_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - TCN_results_${PARAM_STRING}_eva.csv (evaluation period)"
echo ""
echo "Adapted evaluation:"
echo "  - Predictions: /data0/funato/hsp/0_out/TCN/GB/adapted/predict/"
echo "    - TCN_predict_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - TCN_predict_${PARAM_STRING}_eva.csv (evaluation period)"
echo "  - Results: /data0/funato/hsp/0_out/TCN/GB/adapted/results/"
echo "    - TCN_results_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - TCN_results_${PARAM_STRING}_eva.csv (evaluation period)"
echo ""

###RUN USING####
# chmod +x /data0/funato/hsp/99_work/13_TCN/TCN_regional_modeling-master/run_eval.sh
# /data0/funato/hsp/99_work/13_TCN/TCN_regional_modeling-master/run_eval.sh --run_dir /data0/funato/hsp/99_work/13_TCN/TCN_regional_modeling-master/runs_tcn/tcn_run_1015_1800_seed300 --n_basins 2  # For testing with 2 basins
# /data0/funato/hsp/99_work/13_TCN/TCN_regional_modeling-master/run_eval.sh --run_dir /data0/funato/hsp/99_work/13_TCN/TCN_regional_modeling-master/runs_tcn/tcn_run_1015_1800_seed300