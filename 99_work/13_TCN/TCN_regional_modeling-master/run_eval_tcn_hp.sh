#!/bin/bash

# TCN Hyperparameter Evaluation Script
# This script evaluates all hyperparameter-tuned TCN models

###RUN USING####
# chmod +x /data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/run_eval_tcn_hp.sh
# nohup /data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/run_eval_tcn_hp.sh > output_eval_tcn_hp_regional.log 2>&1 &

set -e  # Exit on any error

#LOC="GB"  # Location identifier
LOC="US"  # Location identifier

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_PATH="/data0/funato/.venv/bin/python"
RUNS_BASE="$SCRIPT_DIR/runs_tcn"
OUTPUT_DIR="/data0/funato/hsp2/0_out/TCN/$LOC/regional/hp"
MODEL_EPOCH=30

# Core assignment for parallel execution. Jobs are bounded to MAX_CONCURRENT_JOBS and reuse cores
# round-robin (not one taskset range per job) so the grid below (3*3*3*2*3=162 combos) doesn't try
# to pin past the machine's actual core count - see run_tcn_hp_tuning.sh's analogous memory/CPU
# oversubscription warning for why an unbounded launch loop is a real failure mode here.
START_CORE=100
CORES_PER_JOB=5
AVAILABLE_CORES=$(nproc --all 2>/dev/null || nproc)
MAX_CONCURRENT_JOBS=$(( (AVAILABLE_CORES - START_CORE) / CORES_PER_JOB ))
if [ "$MAX_CONCURRENT_JOBS" -lt 1 ]; then
    MAX_CONCURRENT_JOBS=1
fi

# Hyperparameter grid (must match hp_str in main_tcn_hp.py:
# f{num_filters}_k{kernel_size}_d{dropout}_lr{learning_rate}_lv{num_levels}_w{seq_length}_ns{no_static})
NB_FILTERS=(32 64 128)
KERNEL_SIZES=(3 5 7)
DROPOUTS=(0.1 0.2 0.3)
LEARNING_RATES=(0.001 0.0005)
WINDOW_SIZES=(90 180 270)
NUM_LEVELS=6
NO_STATIC=False

# Function to find the run directory for given hyperparameters
find_run_dir() {
    local f=$1
    local k=$2
    local d=$3
    local lr=$4
    local w=$5
    local hp_dir="$RUNS_BASE/hp/$LOC/f${f}_k${k}_d${d}_lr${lr}_lv${NUM_LEVELS}_w${w}_ns${NO_STATIC}"
    
    if [[ ! -d "$hp_dir" ]]; then
        echo "WARNING: HP directory not found: $hp_dir" >&2
        return 1
    fi

    # Search ALL run directories for this combo, newest first, and use the first one that
    # actually has the target epoch checkpoint -- NOT just the single newest by mtime. A newer
    # run_dir can exist without the checkpoint (e.g. a re-run of an already-complete combo that
    # correctly detects "already exists" and exits early after creating only a partial
    # directory), which would otherwise shadow an older, genuinely complete run and cause this
    # combo to be silently skipped even though a valid checkpoint exists. Confirmed this exact
    # scenario for 5/162 US combos (all f32_k3_*) via
    # /data0/funato/hsp2/DEBUG_README.md-style cfg/log cross-referencing.
    local candidate
    for candidate in $(ls -dt "$hp_dir"/tcn_run_*_seed42 2>/dev/null); do
        if [[ -f "${candidate}/tcn_model_epoch${MODEL_EPOCH}.pt" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    echo "WARNING: No run directory in $hp_dir has tcn_model_epoch${MODEL_EPOCH}.pt" >&2
    return 1
}

# Main evaluation loop
echo "========================================="
echo "TCN Hyperparameter Evaluation"
echo "========================================="
echo "Base directory: $RUNS_BASE"
echo "Output directory: $OUTPUT_DIR"
echo "Model epoch: $MODEL_EPOCH"
echo ""

job_slot=0
running_pids=()

for f in "${NB_FILTERS[@]}"; do
    for k in "${KERNEL_SIZES[@]}"; do
        for d in "${DROPOUTS[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                for w in "${WINDOW_SIZES[@]}"; do
                    hp_string="f${f}_k${k}_d${d}_lr${lr}_lv${NUM_LEVELS}_w${w}_ns${NO_STATIC}"
                    echo "Processing hyperparameters: $hp_string"

                    # Skip if this combination was already evaluated (regardless of which
                    # run_dir/seed/timestamp produced the trained model) -- eval_regional_hp.py
                    # always writes here.
                    eval_output="${OUTPUT_DIR}/results/TCN_results_${hp_string}_eva.csv"
                    if [[ -f "$eval_output" ]]; then
                        echo "  Already evaluated: $hp_string (skipping)"
                        continue
                    fi

                    # Find run directory
                    if ! run_dir=$(find_run_dir "$f" "$k" "$d" "$lr" "$w"); then
                        echo "Skipping $hp_string - no valid run directory found"
                        continue
                    fi

                    echo "  Run directory: $run_dir"

                    # Throttle to MAX_CONCURRENT_JOBS before launching another background job
                    if [ "${#running_pids[@]}" -ge "$MAX_CONCURRENT_JOBS" ]; then
                        wait -n || true
                        # Prune finished pids
                        still_running=()
                        for pid in "${running_pids[@]}"; do
                            if kill -0 "$pid" 2>/dev/null; then
                                still_running+=("$pid")
                            fi
                        done
                        running_pids=("${still_running[@]}")
                    fi

                    # Reuse a core slot round-robin instead of an ever-increasing offset
                    slot=$((job_slot % MAX_CONCURRENT_JOBS))
                    start_core=$((START_CORE + slot * CORES_PER_JOB))
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
                    running_pids+=($!)

                    echo "  Started evaluation on cores ${start_core}-${end_core}"

                    job_slot=$((job_slot + 1))
                done
            done
        done
    done
done

# Wait for all background jobs to complete
echo ""
echo "Waiting for all evaluations to complete..."
wait || true

echo ""
echo "========================================="
echo "All hyperparameter evaluations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================="
echo "    - synthetic/TCN_predict_${PARAM_STRING}_*_eva.csv (synthetic scenarios)"
echo "  - Results: /data0/funato/hsp2/0_out/TCN/$LOC/regional/results/"
echo "    - TCN_results_${PARAM_STRING}_cal.csv (calibration period)"
echo "    - TCN_results_${PARAM_STRING}_eva.csv (evaluation period)"
echo ""
