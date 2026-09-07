#!/bin/bash

# Regional (canonical true EA-LSTM) LSTM Hyperparameter Tuning Script
#
# Runs main_lstm_regional_hp.py (this directory) over a hidden_size x dropout x
# learning_rate x seq_length grid in parallel, using the same taskset + GNU-parallel /
# bash-background-job pattern as ../run_lstm_hp_tuning.sh.
#
# This script is PURELY for the canonical ealstm_regional_modeling/main_lstm_regional_hp.py
# (true EA-LSTM, no_static=False/concat_static=False, fixed). It does NOT touch or duplicate
# ../run_lstm_hp_tuning.sh's existing legacy `regional` mode, which still points at
# x_lstm_regional_modeling-master/main_hp.py (concat_static=True by default, NOT a true
# EA-LSTM) -- that remains available as-is for legacy/comparison runs.
#
# Usage: ./run_lstm_hp_tuning_regional.sh <StartCore> <NumCores>
# Example: nohup /data0/funato/hsp2/99_work/14_LSTM/ealstm_regional_modeling/run_lstm_hp_tuning_regional.sh 100 8 > /data0/funato/hsp2/99_work/14_LSTM/ealstm_regional_modeling/logs/hp_tuning_regional_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# CPU-only machine: regional training pools many basins per epoch, so each epoch is much more
# expensive than an individual-model epoch. n_basins/epochs are therefore kept small at the
# script level below (editable) and the HP grid itself is deliberately smaller than the
# individual model's grid in ../run_lstm_hp_tuning.sh.

# ============================================
# PARAMETER CONFIGURATION
# Edit these to change the hyperparameter grid / compute budget.
# ============================================

# Grid matches HYPERPARAMETERS.md's Regional section exactly (that file is now the source of
# truth, superseding the earlier ad hoc literature-derived grid):
#   - hidden_size: {64,128,256}
#   - dropout: {0.2,0.4}
#   - learning_rate: {0.001,0.0005,0.0001}
#   - seq_length: {270,365} -- note seq_length=365 is noticeably more CPU-expensive per epoch
#     than 270 (longer sequences -> more compute per batch), so budget wall-clock accordingly
#     when this dimension is included.
#   - batch_size: {256,512}
# Grid size: 3 x 2 x 3 x 2 x 2 = 72 combinations. No n_basins-cost-based excuse to keep this
# grid artificially small -- HYPERPARAMETERS.md is authoritative; trim REGIONAL_N_BASINS/
# REGIONAL_EPOCHS below instead if wall-clock is too high for your core budget.
REGIONAL_HIDDEN_SIZE=(64 128 256)
REGIONAL_DROPOUT=(0.2 0.4)
REGIONAL_LEARNING_RATE=(0.001 0.0005 0.0001)
REGIONAL_SEQ_LENGTH=(270 365)
BATCH_SIZE=(256 512)

# Compute budget passed to every combo as --n_basins/--epochs. Reduced-scope defaults for HP
# *search* (relative ranking of configs), not a final publishable fit -- edit these up/down to
# trade search fidelity for wall-clock time. Run main_lstm_regional.py directly (all basins, 30
# epochs) for the winning combo's final confirmation run.
REGIONAL_N_BASINS=40
REGIONAL_EPOCHS=10
REGIONAL_LOC="US"
REGIONAL_SEED=42

# ============================================
# END OF PARAMETER CONFIGURATION
# ============================================

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <StartCore> <NumCores>"
    echo "  StartCore: Starting core number (e.g., 0)"
    echo "  NumCores: Number of cores to use (e.g., 8)"
    exit 1
fi

START_CORE=$1
NUM_CORES=$2
END_CORE=$((START_CORE + NUM_CORES - 1))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/main_lstm_regional_hp.py"
LOG_DIR="${SCRIPT_DIR}/logs/hp"

# Concurrency lock -- see ../run_lstm_hp_tuning.sh for why: overlapping launches of the same HP
# sweep can corrupt shared output via unsynchronized concurrent writers.
LOCK_DIR="${SCRIPT_DIR}/.hp_tuning_regional_canonical.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    LOCK_PID_FILE="${LOCK_DIR}/pid"
    if [ -f "$LOCK_PID_FILE" ] && kill -0 "$(cat "$LOCK_PID_FILE")" 2>/dev/null; then
        echo "Error: another regional HP-tuning run is already active (PID $(cat "$LOCK_PID_FILE"))."
        echo "Wait for it to finish, or if you're sure it's dead, remove $LOCK_DIR and retry."
        exit 1
    fi
    echo "Stale lock at $LOCK_DIR (owner process no longer running) -- reclaiming it."
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR"
fi
echo $$ > "${LOCK_DIR}/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT

echo "Starting regional (canonical true EA-LSTM) LSTM hyperparameter tuning"
echo "Location: $REGIONAL_LOC, n_basins: $REGIONAL_N_BASINS, epochs: $REGIONAL_EPOCHS"
echo "Using cores: $START_CORE to $END_CORE"
echo "Total cores: $NUM_CORES"
echo "Hyperparameter grid:"
echo "  Hidden sizes: ${REGIONAL_HIDDEN_SIZE[@]}"
echo "  Dropout rates: ${REGIONAL_DROPOUT[@]}"
echo "  Learning rates: ${REGIONAL_LEARNING_RATE[@]}"
echo "  Seq lengths: ${REGIONAL_SEQ_LENGTH[@]}"
echo "  Batch sizes: ${BATCH_SIZE[@]}"

# Python executable (use the repo's shared venv if available, matching ../run_lstm_hp_tuning.sh)
PYTHON_CMD="/data0/funato/hsp2/.venv/bin/python"
if [ ! -f "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
fi

mkdir -p "$LOG_DIR"

# Generate all hyperparameter combinations
combinations=()
for h_size in "${REGIONAL_HIDDEN_SIZE[@]}"; do
    for drop in "${REGIONAL_DROPOUT[@]}"; do
        for lr in "${REGIONAL_LEARNING_RATE[@]}"; do
            for sl in "${REGIONAL_SEQ_LENGTH[@]}"; do
                for batch_size in "${BATCH_SIZE[@]}"; do
                    combinations+=("$h_size,$drop,$lr,$sl,$batch_size")
                done
            done
        done
    done
done

TOTAL_COMBINATIONS=${#combinations[@]}
echo "Total hyperparameter combinations: $TOTAL_COMBINATIONS"

# Filter out combos that already finished, BEFORE dispatch, so every core spends its
# time on actually-pending work instead of a launch-then-skip round trip. A combo is
# "done" once a runs/hp/{loc}/h{..}_d{..}_lr{..}_sl{..}/*/ dir has all three of
# lstm_model_final.pt, hp_metrics_cal.csv, hp_metrics_eva.csv AND that run's own cfg.json
# batch_size matches the combo being checked.
#
# The batch_size check is NOT optional: main_lstm_regional_hp.py's run-directory naming
# (_hp_str()) does NOT include batch_size -- both batch_size values for the same
# hidden_size/dropout/learning_rate/seq_length land in the SAME runs/hp/{loc}/h.._d.._lr.._sl../
# folder, distinguished only by their own cfg.json. Checking file existence alone (without
# checking cfg.json's batch_size) would find one batch_size's completed run, conclude the whole
# folder is "done", and wrongly skip the OTHER batch_size too -- this exact false-positive was
# found live in the sibling ../run_lstm_hp_tuning.sh (reported 72/72 "already complete" and
# dispatched nothing, while only 61/72 real 5-tuples had an actually-matching completed run).
pending_combinations=()
for combo in "${combinations[@]}"; do
    IFS=',' read -r h_size drop lr sl batch_size <<< "$combo"
    search_root="${SCRIPT_DIR}/runs/hp/${REGIONAL_LOC}/h${h_size}_d${drop}_lr${lr}_sl${sl}"
    done_found=0
    for run_dir in "$search_root"/*/; do
        [ -d "$run_dir" ] || continue
        if [ -f "${run_dir}lstm_model_final.pt" ] && [ -f "${run_dir}hp_metrics_cal.csv" ] && [ -f "${run_dir}hp_metrics_eva.csv" ] && [ -f "${run_dir}cfg.json" ]; then
            run_bs=$("$PYTHON_CMD" -c "import json,sys; print(json.load(open(sys.argv[1])).get('batch_size'))" "${run_dir}cfg.json" 2>/dev/null)
            if [ "$run_bs" == "$batch_size" ]; then
                done_found=1
                break
            fi
        fi
    done
    if [ "$done_found" -eq 1 ]; then
        continue
    fi
    pending_combinations+=("$combo")
done

SKIPPED=$((TOTAL_COMBINATIONS - ${#pending_combinations[@]}))
echo "Already complete: $SKIPPED / $TOTAL_COMBINATIONS"
combinations=("${pending_combinations[@]}")
TOTAL_COMBINATIONS=${#combinations[@]}
echo "Pending combinations to run: $TOTAL_COMBINATIONS"

if [ "$TOTAL_COMBINATIONS" -gt "$NUM_CORES" ]; then
    echo "Warning: More combinations ($TOTAL_COMBINATIONS) than cores ($NUM_CORES)"
    echo "Jobs will be queued and run as cores become available"
fi

# Function to run a single job. main_lstm_regional_hp.py itself derives its output directory
# (runs/hp/{loc}/h{..}_d{..}_lr{..}_sl{..}/run_<timestamp>_seed{..}/) from these same
# hyperparameters, so no --run_dir plumbing is needed here.
run_job() {
    local combo=$1
    local core=$2
    local job_id=$3

    IFS=',' read -r h_size drop lr sl batch_size <<< "$combo"

    local log_file="${LOG_DIR}/hp_tuning_regional_job${job_id}_h${h_size}_d${drop}_lr${lr}_sl${sl}_bs${batch_size}.log"

    echo "[$job_id/$TOTAL_COMBINATIONS] Starting on core $core: hidden_size=$h_size, dropout=$drop, learning_rate=$lr, seq_length=$sl, batch_size=$batch_size"
    echo "DEBUG: h_size='$h_size' drop='$drop' lr='$lr' sl='$sl' batch_size='$batch_size'" >> "$log_file"

    # GPU is used when available -- each job's own script falls back to CPU automatically if no
    # GPU is free, so no CUDA_VISIBLE_DEVICES override is applied here.
    taskset -c $core "$PYTHON_CMD" "$SCRIPT_PATH" \
        --loc "$REGIONAL_LOC" \
        --n_basins "$REGIONAL_N_BASINS" \
        --epochs "$REGIONAL_EPOCHS" \
        --seed "$REGIONAL_SEED" \
        --hidden_size "$h_size" \
        --dropout "$drop" \
        --learning_rate "$lr" \
        --seq_length "$sl" \
        --batch_size "$batch_size" >> "$log_file" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$job_id/$TOTAL_COMBINATIONS] Completed successfully: hidden_size=$h_size, dropout=$drop, learning_rate=$lr, seq_length=$sl, batch_size=$batch_size"
    else
        echo "[$job_id/$TOTAL_COMBINATIONS] FAILED (exit code $exit_code): hidden_size=$h_size, dropout=$drop, learning_rate=$lr, seq_length=$sl, batch_size=$batch_size"
    fi
}

export -f run_job
export PYTHON_CMD
export SCRIPT_PATH
export LOG_DIR
export TOTAL_COMBINATIONS
export REGIONAL_LOC
export REGIONAL_N_BASINS
export REGIONAL_EPOCHS
export REGIONAL_SEED

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

    # Keep all NUM_CORES slots continuously full: `wait -n` returns as soon as ANY
    # one job finishes, freeing just that slot -- unlike a batch `wait` (waits for
    # the WHOLE batch), which leaves already-finished cores idle until the slowest
    # job in that batch completes.
    core_offset=0
    for i in "${!combinations[@]}"; do
        job_id=$((i + 1))
        core=$((START_CORE + (core_offset % NUM_CORES)))

        while (( $(jobs -r -p | wc -l) >= NUM_CORES )); do
            wait -n
        done

        run_job "${combinations[$i]}" $core $job_id &

        core_offset=$((core_offset + 1))
    done

    wait
fi

echo ""
echo "============================================"
echo "All regional hyperparameter tuning jobs completed!"
echo "Total combinations tested: $TOTAL_COMBINATIONS"
echo "Logs saved in: $LOG_DIR"
echo "Run directories under: ${SCRIPT_DIR}/runs/hp/${REGIONAL_LOC}/"
echo "Each run directory contains hp_metrics_cal.csv / hp_metrics_eva.csv for comparison."
echo "============================================"
