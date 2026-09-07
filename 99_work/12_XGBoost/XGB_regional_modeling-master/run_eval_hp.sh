#!/bin/bash

# XGBoost Hyperparameter Evaluation Script
# This script evaluates all hyperparameter-tuned XGBoost models

###RUN USING####
# chmod +x /data0/funato/hsp2/99_work/12_XGBoost/XGB_regional_modeling-master/run_eval_hp.sh
# nohup /data0/funato/hsp2/99_work/12_XGBoost/XGB_regional_modeling-master/run_eval_hp.sh > /data0/funato/hsp2/run_eval_hp.log 2>&1 &

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_PATH="/data0/funato/.venv/bin/python"
RUNS_BASE="$SCRIPT_DIR/runs_xgb/hp"
OUTPUT_DIR="/data0/funato/hsp2/0_out/XGB/US/regional/hp"

# Core assignment for parallel execution
START_CORE=0
CORES_PER_JOB=5
NUM_CORES=$(nproc)
MAX_PARALLEL=$(( (NUM_CORES - START_CORE) / CORES_PER_JOB ))

# Hyperparameter grid
N_ESTIMATORS=(500 1000)
MAX_DEPTHS=(3 5 7)
LEARNING_RATES=(0.01 0.05 0.1)
SUBSAMPLE=(0.7 0.8 1.0)
COLSAMPLE_BYTREE=(0.7 0.8 1.0)
MIN_CHILD_WEIGHT=(1 3 5)

# Function to find the run directory for given hyperparameters
find_run_dir() {
    local n=$1
    local d=$2
    local lr=$3
    local ss=$4
    local cbt=$5
    local mcw=$6
    local hp_dir="n${n}_d${d}_lr${lr}_ss${ss}_cbt${cbt}_mcw${mcw}"

    # Find most recent run directory (newest first) that actually contains a
    # trained model -- some later retrain attempts for a combo never finished
    # and left a run dir with no xgb_model.json, so we fall back to older dirs.
    local candidates=$(ls -td "$RUNS_BASE"/"${hp_dir}"/xgb_run_* 2>/dev/null)

    if [[ -z "$candidates" ]]; then
        echo "WARNING: No run directory found in $RUNS_BASE/${hp_dir}" >&2
        return 1
    fi

    local run_dir
    for run_dir in $candidates; do
        if [[ -f "$run_dir/xgb_model.json" ]]; then
            echo "$run_dir"
            return 0
        fi
    done

    echo "WARNING: No model file found in any run directory under $RUNS_BASE/${hp_dir}" >&2
    return 1
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
            for ss in "${SUBSAMPLE[@]}"; do
                for cbt in "${COLSAMPLE_BYTREE[@]}"; do
                    for mcw in "${MIN_CHILD_WEIGHT[@]}"; do
                        hp_string="n${n}_d${d}_lr${lr}_ss${ss}_cbt${cbt}_mcw${mcw}"
                        echo "Processing hyperparameters: $hp_string"

                        # Skip if this combination was already evaluated (regardless of which
                        # run_dir/seed/timestamp produced the trained model) -- eval_regional_hp.py
                        # always writes here.
                        eval_output="${OUTPUT_DIR}/results/XGB_results_${hp_string}_eva.csv"
                        if [ -f "$eval_output" ]; then
                            echo "  Already evaluated: $hp_string (skipping)"
                            continue
                        fi

                        # Find run directory
                        if ! run_dir=$(find_run_dir "$n" "$d" "$lr" "$ss" "$cbt" "$mcw"); then
                            echo "Skipping $hp_string - no valid run directory found"
                            continue
                        fi
            
                        echo "  Run directory: $run_dir"
                        
                        # Throttle to the number of jobs that actually fit on the machine
                        while (( $(jobs -r -p | wc -l) >= MAX_PARALLEL )); do
                            wait -n || true
                        done

                        # Calculate core assignment (wrapped to stay within available cores)
                        slot=$((core_offset % MAX_PARALLEL))
                        start_core=$((START_CORE + slot * CORES_PER_JOB))
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
                        core_offset=$((core_offset + 1))
                    done
                done
            done
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

