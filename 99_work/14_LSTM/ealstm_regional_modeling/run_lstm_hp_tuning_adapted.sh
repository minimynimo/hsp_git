#!/bin/bash

# Adapted (per-basin fine-tuned true EA-LSTM) LSTM Hyperparameter Tuning Script
#
# Runs main_lstm_adapted.py (this directory) over a ft_epochs x ft_lr grid in parallel, using
# the same taskset + GNU-parallel / bash-background-job pattern as
# run_lstm_hp_tuning_regional.sh and ../run_lstm_hp_tuning.sh.
#
# Unlike regional training, "adapted" always starts from an ALREADY-TRAINED regional checkpoint
# (a run_dir produced by main_lstm_regional.py) and fine-tunes it per basin -- so this script
# takes that run_dir as a required argument rather than training a base model itself. Run
# main_lstm_regional.py (or run_lstm_hp_tuning_regional.sh, then pick a winning run_dir) first.
#
# No --dataset flag here: main_lstm_adapted.py reads loc/dataset ('camels' or 'caravan') back
# from RunDir's own cfg.json, so whichever dataset that regional run was trained on is what
# gets fine-tuned/evaluated here too -- nothing to select at this script's level.
#
# Usage: ./run_lstm_hp_tuning_adapted.sh <StartCore> <NumCores> <RunDir>
# Example (RunDir from a --dataset camels regional run):
#   ./run_lstm_hp_tuning_adapted.sh 0 8 runs/run_0821_134031_US_camels_seed42
#
# CPU-only machine: adapted fine-tuning re-trains a small number of epochs PER BASIN, so cost
# scales with n_basins x ft_epochs, not with the whole basin pool at once like regional training
# does -- still kept small by default below for HP-search speed.

# ============================================
# PARAMETER CONFIGURATION
# Edit these to change the hyperparameter grid / compute budget.
# ============================================

# Grid, grounded in standard transfer-learning practice for fine-tuning a pretrained model
# (small number of epochs, learning rate well below the base-training rate, to adapt to a
# single basin's dynamics without catastrophically forgetting the regional/cross-basin pattern
# the checkpoint already encodes) plus main_lstm_adapted.py's own defaults as the grid center:
#   - ft_epochs: {3,5,10} -- 5 is this script's own default (main_lstm_adapted.py
#     --ft_epochs default); 3 checks whether fewer epochs already capture most of the basin-
#     specific gain, 10 checks whether more fine-tuning keeps helping or starts overfitting the
#     (typically short) per-basin calibration record.
#   - ft_lr: {1e-4,1e-5} -- 1e-4 is main_lstm_adapted.py's own default and matches the base
#     regional model's own fixed Adam LR order-of-magnitude reduced by 10x, a standard
#     fine-tuning LR heuristic; 1e-5 tests a further 10x reduction (gentler adaptation, less
#     risk of overwriting the regional model's learned cross-basin behavior).
# Grid size: 3 x 2 = 6 combinations x ADAPTED_N_BASINS basins x ft_epochs epochs each -- small
# by construction since this reuses one already-trained checkpoint per combination (no base
# training cost), just per-basin fine-tuning.
ADAPTED_FT_EPOCHS=(3 5 10)
ADAPTED_FT_LR=(0.0001 0.00001)

# Compute budget / fixed args passed to every combo. Reduced-scope n_basins for HP *search*
# speed -- edit up toward all 614 basins for a final confirmation run of the winning combo.
ADAPTED_N_BASINS=40
ADAPTED_MODEL_EPOCH=""   # empty = main_lstm_adapted.py's default (uses the run's final epoch)

# ============================================
# END OF PARAMETER CONFIGURATION
# ============================================

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <StartCore> <NumCores> <RunDir>"
    echo "  StartCore: Starting core number (e.g., 0)"
    echo "  NumCores: Number of cores to use (e.g., 8)"
    echo "  RunDir: path to a trained regional run directory (relative to this script's"
    echo "          directory, or absolute), e.g. runs/run_0821_134031_US_seed42 -- produced by"
    echo "          main_lstm_regional.py or run_lstm_hp_tuning_regional.sh"
    exit 1
fi

START_CORE=$1
NUM_CORES=$2
RUN_DIR_ARG=$3
END_CORE=$((START_CORE + NUM_CORES - 1))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/main_lstm_adapted.py"

# Concurrency lock -- see ../run_lstm_hp_tuning.sh for why: overlapping launches of the same HP
# sweep can corrupt shared output via unsynchronized concurrent writers.
LOCK_DIR="${SCRIPT_DIR}/.hp_tuning_adapted.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    LOCK_PID_FILE="${LOCK_DIR}/pid"
    if [ -f "$LOCK_PID_FILE" ] && kill -0 "$(cat "$LOCK_PID_FILE")" 2>/dev/null; then
        echo "Error: another adapted HP-tuning run is already active (PID $(cat "$LOCK_PID_FILE"))."
        echo "Wait for it to finish, or if you're sure it's dead, remove $LOCK_DIR and retry."
        exit 1
    fi
    echo "Stale lock at $LOCK_DIR (owner process no longer running) -- reclaiming it."
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR"
fi
echo $$ > "${LOCK_DIR}/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT
LOG_DIR="${SCRIPT_DIR}/logs/hp_adapted"

# Resolve RunDir to an absolute path (accept either relative-to-this-script or absolute input)
if [[ "$RUN_DIR_ARG" = /* ]]; then
    RUN_DIR="$RUN_DIR_ARG"
else
    RUN_DIR="${SCRIPT_DIR}/${RUN_DIR_ARG}"
fi

if [ ! -f "${RUN_DIR}/cfg.json" ]; then
    echo "Error: no cfg.json found at ${RUN_DIR}/cfg.json"
    echo "RunDir must be a directory produced by main_lstm_regional.py (or its hp variant)."
    exit 1
fi

echo "Starting adapted (per-basin fine-tuned true EA-LSTM) LSTM hyperparameter tuning"
echo "Base regional run_dir: $RUN_DIR"
echo "n_basins: $ADAPTED_N_BASINS"
echo "Using cores: $START_CORE to $END_CORE"
echo "Total cores: $NUM_CORES"
echo "Hyperparameter grid:"
echo "  Fine-tune epochs: ${ADAPTED_FT_EPOCHS[@]}"
echo "  Fine-tune learning rates: ${ADAPTED_FT_LR[@]}"

# Python executable (use the repo's shared venv if available, matching the other hp scripts)
PYTHON_CMD="/data0/funato/hsp2/.venv/bin/python"
if [ ! -f "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
fi

mkdir -p "$LOG_DIR"

# Generate all hyperparameter combinations
combinations=()
for ft_ep in "${ADAPTED_FT_EPOCHS[@]}"; do
    for ft_lr in "${ADAPTED_FT_LR[@]}"; do
        combinations+=("$ft_ep,$ft_lr")
    done
done

TOTAL_COMBINATIONS=${#combinations[@]}
echo "Total hyperparameter combinations: $TOTAL_COMBINATIONS"

if [ "$TOTAL_COMBINATIONS" -gt "$NUM_CORES" ]; then
    echo "Warning: More combinations ($TOTAL_COMBINATIONS) than cores ($NUM_CORES)"
    echo "Jobs will be queued and run as cores become available"
fi

# Function to run a single job. main_lstm_adapted.py's own output tag already encodes
# ft_epochs/ft_lr (_ft{epochs}_ftlr{lr}), so different combos against the same run_dir can never
# collide/overwrite each other's result CSVs.
run_job() {
    local combo=$1
    local core=$2
    local job_id=$3

    IFS=',' read -r ft_ep ft_lr <<< "$combo"

    local log_file="${LOG_DIR}/hp_tuning_adapted_job${job_id}_ft${ft_ep}_ftlr${ft_lr}.log"

    echo "[$job_id/$TOTAL_COMBINATIONS] Starting on core $core: ft_epochs=$ft_ep, ft_lr=$ft_lr"
    echo "DEBUG: ft_ep='$ft_ep' ft_lr='$ft_lr' run_dir='$RUN_DIR'" >> "$log_file"

    local model_epoch_flag=()
    if [ -n "$ADAPTED_MODEL_EPOCH" ]; then
        model_epoch_flag=(--model_epoch "$ADAPTED_MODEL_EPOCH")
    fi

    # GPU is used when available -- each job's own script falls back to CPU automatically if no
    # GPU is free, so no CUDA_VISIBLE_DEVICES override is applied here.
    taskset -c $core "$PYTHON_CMD" "$SCRIPT_PATH" \
        --run_dir "$RUN_DIR" \
        --n_basins "$ADAPTED_N_BASINS" \
        --ft_epochs "$ft_ep" \
        --ft_lr "$ft_lr" \
        "${model_epoch_flag[@]}" >> "$log_file" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$job_id/$TOTAL_COMBINATIONS] Completed successfully: ft_epochs=$ft_ep, ft_lr=$ft_lr"
    else
        echo "[$job_id/$TOTAL_COMBINATIONS] FAILED (exit code $exit_code): ft_epochs=$ft_ep, ft_lr=$ft_lr"
    fi
}

export -f run_job
export PYTHON_CMD
export SCRIPT_PATH
export LOG_DIR
export TOTAL_COMBINATIONS
export RUN_DIR
export ADAPTED_N_BASINS
export ADAPTED_MODEL_EPOCH

if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for job management"

    JOBS_FILE=$(mktemp)
    for i in "${!combinations[@]}"; do
        job_id=$((i + 1))
        core=$((START_CORE + (i % NUM_CORES)))
        echo "${combinations[$i]} $core $job_id" >> "$JOBS_FILE"
    done

    cat "$JOBS_FILE" | parallel -j $NUM_CORES --colsep ' ' 'run_job {1} {2} {3}'

    rm "$JOBS_FILE"
else
    echo "GNU parallel not found, using bash background jobs for parallel execution"

    job_count=0
    for i in "${!combinations[@]}"; do
        job_id=$((i + 1))
        core=$((START_CORE + (i % NUM_CORES)))

        run_job "${combinations[$i]}" $core $job_id &

        job_count=$((job_count + 1))

        if [ $((job_count % NUM_CORES)) -eq 0 ]; then
            wait
        fi
    done

    wait
fi

echo ""
echo "============================================"
echo "All adapted hyperparameter tuning jobs completed!"
echo "Total combinations tested: $TOTAL_COMBINATIONS"
echo "Base regional run_dir: $RUN_DIR"
echo "Logs saved in: $LOG_DIR"
echo "Results (per combo, tagged _ft{epochs}_ftlr{lr}) under:"
echo "  /data0/funato/hsp2/0_out/LSTM/{loc}/{dataset}/adapted/results/  (dataset per RunDir's cfg.json)"
echo "============================================"
