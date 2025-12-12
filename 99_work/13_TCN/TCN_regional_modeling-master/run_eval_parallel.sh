#!/bin/bash

# Parallel TCN Adapted Evaluation Script
# This script runs adapted evaluation across multiple parallel processes
# Each process handles a subset of basins for faster execution

set -e

# Default parameters
RUN_DIR=""
MODEL_EPOCH=30
TOTAL_PROCESSES=15
PYTHON_PATH="/data0/funato/hsp/.venv/bin/python"
STAGE="all"  # Options: finetune, evaluate, all
START_CORE=16  # 0-indexed (for taskset -c)

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
        --total_processes)
            TOTAL_PROCESSES="$2"
            shift 2
            ;;
        --start_core)
            START_CORE="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --python_path)
            PYTHON_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --run_dir <path> [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --run_dir           Path to the trained model directory (required)"
            echo "  --model_epoch       Model epoch to use (default: 30)"
            echo "  --total_processes   Number of parallel processes/cores to use (default: 10)"
            echo "  --start_core        Starting CPU core number, 0-indexed (default: 0)"
            echo "  --stage             Stage to run: finetune, evaluate, or all (default: all)"
            echo "  --python_path       Path to Python executable (default: /data0/funato/hsp/.venv/bin/python)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Example: Run 10 processes on cores 3-12:"
            echo "  $0 --run_dir /path/to/model --start_core 3 --total_processes 10"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$RUN_DIR" ]; then
    echo "Error: --run_dir is required"
    exit 1
fi

if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Run directory does not exist: $RUN_DIR"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/eval_adapted_parallel.py"
OUTPUT_DIR="/data0/funato/hsp/0_out/TCN/GB/adapted"

echo "=========================================="
echo "Parallel TCN Adapted Evaluation"
echo "=========================================="
# Calculate end core
END_CORE=$((START_CORE + TOTAL_PROCESSES - 1))

echo "Run directory: $RUN_DIR"
echo "Model epoch: $MODEL_EPOCH"
echo "Total processes: $TOTAL_PROCESSES"
echo "CPU cores: $START_CORE to $END_CORE (one process per core)"
echo "Stage: $STAGE"
echo "Python: $PYTHON_PATH"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR/predict"
mkdir -p "$OUTPUT_DIR/predict/synthetic"
mkdir -p "$OUTPUT_DIR/results"

# Launch parallel processes
echo "Launching $TOTAL_PROCESSES parallel processes..."

PIDS=()
for ((i=0; i<$TOTAL_PROCESSES; i++)); do
    CORE=$((START_CORE + i))
    LOG_FILE="$OUTPUT_DIR/process_${i}.log"
    echo "Starting process $i on CPU core $CORE (log: $LOG_FILE)"
    
    nohup taskset -c $CORE $PYTHON_PATH "$EVAL_SCRIPT" \
        --run_dir "$RUN_DIR" \
        --model_epoch "$MODEL_EPOCH" \
        --process_id "$i" \
        --total_processes "$TOTAL_PROCESSES" \
        --stage "$STAGE" \
        > "$LOG_FILE" 2>&1 &
    
    PIDS+=($!)
done

echo ""
echo "All processes launched. PIDs: ${PIDS[@]}"
echo "Waiting for all processes to complete..."

# Wait for all processes
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait $PID; then
        echo "Process $i (PID $PID) completed successfully"
    else
        echo "Process $i (PID $PID) failed with exit code $?"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "ERROR: $FAILED process(es) failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "All processes completed successfully!"
echo "=========================================="

# Merge results if evaluation was run
if [ "$STAGE" = "evaluate" ] || [ "$STAGE" = "all" ]; then
    echo ""
    echo "Merging results from parallel processes..."
    
    # Merge cal and eva predictions and results
    for PERIOD in cal eva; do
        echo "Merging $PERIOD period..."
        
        # Merge predictions
        PRED_MERGED="$OUTPUT_DIR/predict/TCN_predict_${PERIOD}.csv"
        FIRST=true
        for ((i=0; i<$TOTAL_PROCESSES; i++)); do
            PRED_FILE="$OUTPUT_DIR/predict/TCN_predict_${PERIOD}_p${i}.csv"
            if [ -f "$PRED_FILE" ]; then
                if $FIRST; then
                    cat "$PRED_FILE" > "$PRED_MERGED"
                    FIRST=false
                else
                    tail -n +2 "$PRED_FILE" >> "$PRED_MERGED"
                fi
            fi
        done
        
        # Sort merged predictions by basin ID
        if [ -f "$PRED_MERGED" ]; then
            $PYTHON_PATH -c "
import pandas as pd
df = pd.read_csv('$PRED_MERGED', index_col='basin')
df.index = df.index.astype(int)
df.sort_index().to_csv('$PRED_MERGED')
print('Sorted predictions: $PRED_MERGED')
"
        fi
        
        # Merge results
        RESULTS_MERGED="$OUTPUT_DIR/results/TCN_results_${PERIOD}.csv"
        FIRST=true
        for ((i=0; i<$TOTAL_PROCESSES; i++)); do
            RESULTS_FILE="$OUTPUT_DIR/results/TCN_results_${PERIOD}_p${i}.csv"
            if [ -f "$RESULTS_FILE" ]; then
                if $FIRST; then
                    cat "$RESULTS_FILE" > "$RESULTS_MERGED"
                    FIRST=false
                else
                    tail -n +2 "$RESULTS_FILE" >> "$RESULTS_MERGED"
                fi
            fi
        done
        
        # Sort merged results
        if [ -f "$RESULTS_MERGED" ]; then
            $PYTHON_PATH -c "
import pandas as pd
df = pd.read_csv('$RESULTS_MERGED', index_col='basin')
df.index = df.index.astype(int)
df.sort_index().to_csv('$RESULTS_MERGED')
print('Sorted results: $RESULTS_MERGED')
"
        fi
        
        # Clean up individual process files
        for ((i=0; i<$TOTAL_PROCESSES; i++)); do
            rm -f "$OUTPUT_DIR/predict/TCN_predict_${PERIOD}_p${i}.csv"
            rm -f "$OUTPUT_DIR/results/TCN_results_${PERIOD}_p${i}.csv"
        done
    done
    
    # Merge synthetic scenario predictions
    echo "Merging synthetic scenarios..."
    
    SCENARIOS="syn_no_precip syn_p_pet syn_peak1_rate0 syn_peak1_rate10 syn_peak1_rate30 syn_peak1_rate50 syn_peak3_rate0 syn_peak3_rate10 syn_peak3_rate30 syn_peak3_rate50 syn_peak5_rate0 syn_peak5_rate10 syn_peak5_rate30 syn_peak5_rate50"
    
    for SCENARIO in $SCENARIOS; do
        echo "  Merging $SCENARIO..."
        PRED_MERGED="$OUTPUT_DIR/predict/synthetic/TCN_predict_${SCENARIO}_eva.csv"
        FIRST=true
        
        for ((i=0; i<$TOTAL_PROCESSES; i++)); do
            PRED_FILE="$OUTPUT_DIR/predict/synthetic/TCN_predict_${SCENARIO}_eva_p${i}.csv"
            if [ -f "$PRED_FILE" ]; then
                if $FIRST; then
                    cat "$PRED_FILE" > "$PRED_MERGED"
                    FIRST=false
                else
                    tail -n +2 "$PRED_FILE" >> "$PRED_MERGED"
                fi
            fi
        done
        
        # Sort merged predictions
        if [ -f "$PRED_MERGED" ]; then
            $PYTHON_PATH -c "
import pandas as pd
df = pd.read_csv('$PRED_MERGED', index_col='basin')
df.index = df.index.astype(int)
df.sort_index().to_csv('$PRED_MERGED')
" 2>/dev/null || true
        fi
        
        # Clean up individual process files
        for ((i=0; i<$TOTAL_PROCESSES; i++)); do
            rm -f "$OUTPUT_DIR/predict/synthetic/TCN_predict_${SCENARIO}_eva_p${i}.csv"
        done
    done
    
    echo ""
    echo "=========================================="
    echo "Results merged successfully!"
    echo "=========================================="
    echo "Predictions: $OUTPUT_DIR/predict/"
    echo "Results: $OUTPUT_DIR/results/"
    echo "Synthetic: $OUTPUT_DIR/predict/synthetic/"
fi

echo ""
echo "Done!"

## RUN using 
# chmod +x /data0/funato/hsp/99_work/13_TCN/TCN_regional_modeling-master/run_eval_parallel.sh
# nohup /data0/funato/hsp/99_work/13_TCN/TCN_regional_modeling-master/run_eval_parallel.sh --run_dir /data0/funato/hsp/99_work/13_TCN/TCN_regional_modeling-master/runs_tcn/tcn_run_1119_1504_seed309 > output_eval_tcn_adapt_parallel.log 2>&1 &