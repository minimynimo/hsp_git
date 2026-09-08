#!/bin/bash

# ESN (HYPER bias-correction model) hyperparameter tuning driver, mirroring the convention
# established by /data0/funato/hsp2/99_work/14_LSTM/run_lstm_hp_tuning.sh: hparam_main_HYPER.py
# now runs exactly ONE hyperparameter combination per process (see its own header comment), so
# this script builds the full grid and dispatches each combo to its own core via taskset.
#
# Usage: ./run_hparam_HYPER.sh <StartCore> <NumCores> [background]
#   /data0/funato/hsp2/99_work/03_HYPER/run_hparam_HYPER.sh 0 31
#   /data0/funato/hsp2/99_work/03_HYPER/run_hparam_HYPER.sh 0 60 background
#     -> prints the PID and log file to tail, same pattern as run_lstm_hp_tuning.sh's
#        background self-relaunch.

# ============================================
# BACKGROUND SELF-RELAUNCH
# ============================================
if [ "$3" == "background" ] || [ "$3" == "bg" ]; then
    if [ "$#" -lt 2 ]; then
        echo "Usage: $0 <StartCore> <NumCores> background"
        exit 1
    fi
    SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    BG_LOG_FILE="${SELF_DIR}/hp_tuning_HYPER_${1}_${2}_$(date +%Y%m%d_%H%M%S).log"
    echo "Starting HYPER hyperparameter tuning in background..."
    echo "Cores: $1 to $(($1 + $2 - 1))"
    echo "Log file: $BG_LOG_FILE"
    nohup "${SELF_DIR}/$(basename "${BASH_SOURCE[0]}")" "$1" "$2" > "$BG_LOG_FILE" 2>&1 &
    BG_PID=$!
    echo "Process started with PID: $BG_PID"
    echo "To monitor progress: tail -f $BG_LOG_FILE"
    echo "To check if running: ps -p $BG_PID"
    echo "To stop: kill $BG_PID"
    exit 0
fi

# ============================================
# PARAMETER CONFIGURATION
# Edit these arrays to change the hyperparameter grid. Matches the grid swept internally by
# hparam_main_HYPER.py's nested for-loops. adjacency_density IS a grid dimension: it's passed
# through to ESN() and tagged into file_tag (_ad{adjacency_density}), so each value gets its
# own output dir/files instead of overwriting the others.
# Grid size: 8 x 5 x 3 x 5 x 6 x 1 = 3600 combinations.
# ============================================
RESERVOIR_SIZE=(10 50 100 200 300 400 500 700)
# NOTE: values here must match Python's str(float(x)) exactly (e.g. "0.9", not "0.90") --
# hparam_main_HYPER.py's file_tag is built from the parsed float, and the completion
# pre-filter below constructs the same path string from these literals to check it.
SPECTRAL_RADIUS=(0.85 0.9 0.95 1.0 1.05)
RIDGE_PARAM=(0.1 0.01 0.001)
INPUT_SCALE=(0.1 0.5 1.0 5.0 10.0)
ADJACENCY_DENSITY=(0.05 0.1 0.2 0.3 0.5 1.0)
SPINUP=(1)
LOC="US"
# 'camels' (default): built directly from raw CAMELS-US, full 1980-2014 record --
# hparam_main_HYPER.py sub-splits its own HP-search window (1981-1990/1991-1995) strictly
# inside production's 1981-1995 calibration period. 'caravan': older Caravan-derived
# extraction (1987-2006 only).
DATASET="camels"

# ============================================
# END OF PARAMETER CONFIGURATION
# ============================================

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <StartCore> <NumCores> [background]"
    echo "  StartCore: Starting core number (e.g., 0)"
    echo "  NumCores: Number of cores to use (e.g., 31 for cores 0-30)"
    echo "  [background]: optional 3rd arg, literally 'background' or 'bg', to nohup-launch and"
    echo "                return immediately"
    exit 1
fi

START_CORE=$1
NUM_CORES=$2
END_CORE=$((START_CORE + NUM_CORES - 1))

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SELF_DIR}/hparam_main_HYPER.py"
LOG_DIR="${SELF_DIR}/logs/hp"
mkdir -p "$LOG_DIR"

# ============================================
# CONCURRENCY LOCK
# Refuses a second concurrent launch instead of letting two runs corrupt the same
# output CSVs, following run_lstm_hp_tuning.sh's incident-driven precedent.
# ============================================
LOCK_DIR="${SELF_DIR}/.hp_tuning_HYPER.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    LOCK_PID_FILE="${LOCK_DIR}/pid"
    if [ -f "$LOCK_PID_FILE" ] && kill -0 "$(cat "$LOCK_PID_FILE")" 2>/dev/null; then
        echo "Error: another HYPER HP-tuning run is already active (PID $(cat "$LOCK_PID_FILE"))."
        echo "Wait for it to finish, or if you're sure it's dead, remove $LOCK_DIR and retry."
        exit 1
    fi
    echo "Stale lock at $LOCK_DIR (owner process no longer running) -- reclaiming it."
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR"
fi
echo $$ > "${LOCK_DIR}/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT

echo "Starting HYPER hyperparameter tuning"
echo "Location: $LOC, dataset: $DATASET"
echo "Using cores: $START_CORE to $END_CORE"
echo "Total cores: $NUM_CORES"
echo "Hyperparameter grid:"
echo "  Reservoir sizes: ${RESERVOIR_SIZE[@]}"
echo "  Spectral radii: ${SPECTRAL_RADIUS[@]}"
echo "  Ridge params: ${RIDGE_PARAM[@]}"
echo "  Input scales: ${INPUT_SCALE[@]}"
echo "  Adjacency densities: ${ADJACENCY_DENSITY[@]}"
echo "  Spinup: ${SPINUP[@]}"

PYTHON_CMD="/data0/funato/hsp2/.venv/bin/python"
if [ ! -f "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
fi

# Generate all hyperparameter combinations:
# "reservoir_size,spectral_radius,ridge_param,input_scale,adjacency_density,spinup"
combinations=()
for rs in "${RESERVOIR_SIZE[@]}"; do
    for sr in "${SPECTRAL_RADIUS[@]}"; do
        for rp in "${RIDGE_PARAM[@]}"; do
            for is in "${INPUT_SCALE[@]}"; do
                for ad in "${ADJACENCY_DENSITY[@]}"; do
                    for sp in "${SPINUP[@]}"; do
                        combinations+=("$rs,$sr,$rp,$is,$ad,$sp")
                    done
                done
            done
        done
    done
done

TOTAL_COMBINATIONS=${#combinations[@]}
echo "Total hyperparameter combinations: $TOTAL_COMBINATIONS"

# Filter out combos whose HYPER{file_tag}_log.txt already ends with DONE, before dispatch --
# hparam_main_HYPER.py's own model_name loop does the same check internally (defense in depth),
# but pre-filtering here means an already-finished combo never occupies a core slot at all.
pending_combinations=()
for combo in "${combinations[@]}"; do
    IFS=',' read -r rs sr rp is ad sp <<< "$combo"
    # Must match hparam_main_HYPER.py's file_tag/output_dir construction exactly.
    file_tag="_r${rs}_s${sp}_sr${sr}_rr${rp}_ad${ad}"
    log_file="/data0/funato/hsp2/0_out/HYPER/${LOC}/${DATASET}/HYPER_hp/spin${sp}/reservoir${rs}/HYPER${file_tag}_is${is}/HYPER${file_tag}_log.txt"
    if [ -f "$log_file" ] && tail -n 1 "$log_file" 2>/dev/null | grep -q "^DONE$"; then
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

run_job() {
    local combo=$1
    local core=$2
    local job_id=$3

    local rs sr rp is ad sp
    IFS=',' read -r rs sr rp is ad sp <<< "$combo"

    local log_file="${LOG_DIR}/hp_tuning_HYPER_job${job_id}_r${rs}_sr${sr}_rr${rp}_is${is}_ad${ad}_s${sp}.log"

    echo "[$job_id/$TOTAL_COMBINATIONS] Starting on core $core: reservoir_size=$rs, spectral_radius=$sr, ridge_param=$rp, input_scale=$is, adjacency_density=$ad, spinup=$sp"

    taskset -c $core "$PYTHON_CMD" "$SCRIPT_PATH" \
        --loc "$LOC" \
        --dataset "$DATASET" \
        --reservoir_size "$rs" \
        --spectral_radius "$sr" \
        --ridge_param "$rp" \
        --input_scale "$is" \
        --adjacency_density "$ad" \
        --spinup "$sp" >> "$log_file" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$job_id/$TOTAL_COMBINATIONS] Completed successfully: reservoir_size=$rs, spectral_radius=$sr, ridge_param=$rp, input_scale=$is, adjacency_density=$ad, spinup=$sp"
    else
        echo "[$job_id/$TOTAL_COMBINATIONS] FAILED (exit code $exit_code): reservoir_size=$rs, spectral_radius=$sr, ridge_param=$rp, input_scale=$is, adjacency_density=$ad, spinup=$sp"
    fi
}

export -f run_job
export PYTHON_CMD
export SCRIPT_PATH
export LOG_DIR
export LOC
export DATASET
export TOTAL_COMBINATIONS

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
echo "All HYPER hyperparameter tuning jobs completed!"
echo "Total combinations tested: $TOTAL_COMBINATIONS"
echo "Logs saved in: $LOG_DIR"
echo "============================================"
