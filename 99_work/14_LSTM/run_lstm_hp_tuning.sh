#!/bin/bash

# LSTM Hyperparameter Tuning Script -- individual (main_LSTM_indiv_hp.py) and canonical
# true-EA-LSTM regional (ealstm_regional_modeling/main_lstm_regional_hp.py) HP search, both
# from this one script. No file under x_lstm_regional_modeling-master is used here anymore
# (its main_hp.py had two real bugs that made every combination fail before training even
# started; see the REGIONAL_* grid comment below for details). For a regional-only search with
# more standalone options, ealstm_regional_modeling/run_lstm_hp_tuning_regional.sh calls the
# same canonical script directly.
#
# Runs all hyperparameter combinations in parallel using specified cores, in the foreground by
# default. This is the single script to run for individual/regional LSTM HP tuning -- the
# separate start_lstm_hp_tuning.sh background-launch wrapper has been merged in here (pass
# `background` as an optional 4th argument) so there is only one file to remember to run;
# start_lstm_hp_tuning.sh has been renamed x_start_lstm_hp_tuning.sh (superseded, kept for
# reference only).
#
# Usage: ./run_lstm_hp_tuning.sh <StartCore> <NumCores> <ModelType> [background]
# /data0/funato/hsp2/99_work/14_LSTM/run_lstm_hp_tuning.sh 20 6 regional background
# /data0/funato/hsp2/99_work/14_LSTM/run_lstm_hp_tuning.sh 0 10 individual background
#   -> prints the PID and log file to tail, same as the old start_lstm_hp_tuning.sh did.

# ============================================
# BACKGROUND SELF-RELAUNCH (merged from start_lstm_hp_tuning.sh)
# If "background" is passed as the 4th argument, re-launch this same script via nohup with the
# first 3 arguments only, then exit immediately -- so the actual tuning runs detached from the
# terminal, exactly like start_lstm_hp_tuning.sh used to do as a separate wrapper file.
# ============================================
if [ "$4" == "background" ] || [ "$4" == "bg" ]; then
    if [ "$#" -lt 3 ]; then
        echo "Usage: $0 <StartCore> <NumCores> <ModelType> background"
        exit 1
    fi
    SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    BG_LOG_FILE="${SELF_DIR}/hp_tuning_${3}_${1}_${2}_$(date +%Y%m%d_%H%M%S).log"
    echo "Starting LSTM hyperparameter tuning in background..."
    echo "Model Type: $3"
    echo "Cores: $1 to $(($1 + $2 - 1))"
    echo "Log file: $BG_LOG_FILE"
    nohup "${SELF_DIR}/$(basename "${BASH_SOURCE[0]}")" "$1" "$2" "$3" > "$BG_LOG_FILE" 2>&1 &
    BG_PID=$!
    echo "Process started with PID: $BG_PID"
    echo "To monitor progress: tail -f $BG_LOG_FILE"
    echo "To check if running: ps -p $BG_PID"
    echo "To stop: kill $BG_PID"
    exit 0
fi

# ============================================
# BCPARAMETER CONFIGURATION
# Edit these arrays to change the hyperparameter grid
# ============================================

# Regional model hyperparameters. This now runs the CANONICAL true-EA-LSTM HP-search script,
# ealstm_regional_modeling/main_lstm_regional_hp.py -- NOT the legacy
# x_lstm_regional_modeling-master/main_hp.py, which is no longer used by this script at all.
# (main_hp.py had two separate real bugs surfaced by actually running it: a read-only-array
# crash in its static_stds zero-guard, and -- even after fixing that -- a shape-mismatch crash
# concatenating static attributes into the dynamic sequence, both before any training happened.
# Rather than keep chasing bugs in a script this repo already treats as legacy/non-canonical
# everywhere else, this script now points at the same already-working, already-validated
# canonical script that ealstm_regional_modeling/run_lstm_hp_tuning_regional.sh uses.) Grid
# matches HYPERPARAMETERS.md's Regional section exactly:
#   - hidden_size: {64,128,256}
#   - dropout: {0.2,0.4}
#   - learning_rate: {0.001,0.0005,0.0001}
#   - seq_length: {270,365} (CLI-overridable on this canonical script, unlike the legacy one)
#   - batch_size: {256,512}
# Grid size: 3 x 2 x 3 x 2 x 2 = 72 combinations (full grid). Compute budget (n_basins/epochs) is
# fixed at REGIONAL_N_BASINS/REGIONAL_EPOCHS below, matching run_lstm_hp_tuning_regional.sh's
# defaults.
#
# CAMELS HP re-check (see SELECTED_HYPERPARAMETERS.md "Camels" section): only the
# caravan-selected value + immediate grid neighbor(s) per parameter are tested. Reduced grid
# size: 3 x 2 x 2 x 2 x 2 = 48 combinations. Kept in sync with
# ealstm_regional_modeling/run_lstm_hp_tuning_regional.sh's REGIONAL_* arrays, which is the
# canonical script actually used to launch the regional camels re-check.
REGIONAL_HIDDEN_SIZE=(64 128 256) # unchanged -- caravan-selected=128 is the middle of 3
REGIONAL_DROPOUT=(0.2 0.4) # unchanged -- only 2 values, both always tested
REGIONAL_LEARNING_RATE=(
    0.001
    0.0005
    # 0.0001  # camels re-check: dropped -- caravan-selected=0.001 (highest of 3), neighbor is 0.0005
)
REGIONAL_SEQ_LENGTH=(270 365) # unchanged -- only 2 values, both always tested
REGIONAL_BATCH_SIZE=(256 512) # unchanged -- only 2 values, both always tested
REGIONAL_N_BASINS=40
REGIONAL_EPOCHS=10
REGIONAL_LOC="US"
# 'camels' (default): built directly from raw CAMELS-US, full 1980-2014 record --
# main_lstm_regional_hp.py sub-splits its own HP-search cal/val window (1990-1998/1999-2000)
# strictly inside production's 1987-2000 calibration period. 'caravan': older Caravan-derived
# extraction (1987-2006 only); still the only source for GB.
REGIONAL_DATASET="camels"

# Individual model hyperparameters. Grid now matches HYPERPARAMETERS.md's Individual section
# exactly (that file is the source of truth, superseding the earlier literature-derived grid):
#   - hidden_size: {16,32,64,128}
#   - dropout: {0.4,0.5}
#   - learning_rate: {0.001,0.0005}
#   - seq_length: {270,365}
#   - batch_size: {64,128} (main_LSTM_indiv_hp.py accepts --batch_size, default 256)
# Grid size: 4 x 2 x 2 x 2 x 2 = 64 combinations (full grid).
#
# CAMELS HP re-check (see SELECTED_HYPERPARAMETERS.md "Camels" section): only the
# caravan-selected value + immediate grid neighbor(s) per parameter are tested. Reduced grid
# size: 3 x 2 x 2 x 2 x 2 = 48 combinations.
INDIVIDUAL_HIDDEN_SIZE=(
    16
    32
    64
    # 128  # camels re-check: dropped -- caravan-selected=32 (2nd of 4), neighbors are 16 and 64
)
INDIVIDUAL_DROPOUT=(0.4 0.5) # unchanged -- only 2 values, both always tested
INDIVIDUAL_LEARNING_RATE=(0.001 0.0005) # unchanged -- only 2 values, both always tested
INDIVIDUAL_SEQ_LENGTH=(270 365) # unchanged -- only 2 values, both always tested
INDIVIDUAL_BATCH_SIZE=(64 128) # unchanged -- only 2 values, both always tested
# 'camels' (default): built directly from raw CAMELS-US, full 1980-2014 record --
# main_LSTM_indiv_hp.py sub-splits its own HP-search train/val window (1990-1998/1999-2000)
# strictly inside production's 1987-2000 calibration period. 'caravan': older Caravan-derived
# extraction (1987-2006 only).
INDIVIDUAL_DATASET="camels"

# ============================================
# END OF BCPARAMETER CONFIGURATION
# ============================================

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <StartCore> <NumCores> <ModelType> [background]"
    echo "  StartCore: Starting core number (e.g., 0)"
    echo "  NumCores: Number of cores to use (e.g., 4)"
    echo "  ModelType: 'regional' or 'individual'"
    echo "  [background]: optional 4th arg, literally 'background' or 'bg', to nohup-launch and"
    echo "                return immediately (merged from the old start_lstm_hp_tuning.sh)"
    exit 1
fi

START_CORE=$1
NUM_CORES=$2
MODEL_TYPE=$3
END_CORE=$((START_CORE + NUM_CORES - 1))

# ============================================
# CONCURRENCY LOCK
# A prior run of this script (launched multiple times over one day without realizing an
# earlier launch was still active) had several overlapping instances processing the SAME
# combo simultaneously, each writing to the same output CSV -- corrupting most of the HP grid's
# results (duplicate/garbled rows from unsynchronized concurrent writers). This lock refuses a
# second concurrent launch of the same ModelType instead of silently corrupting output again.
# ============================================
LOCK_DIR="/data0/funato/hsp2/99_work/14_LSTM/.hp_tuning_${MODEL_TYPE}.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    LOCK_PID_FILE="${LOCK_DIR}/pid"
    if [ -f "$LOCK_PID_FILE" ] && kill -0 "$(cat "$LOCK_PID_FILE")" 2>/dev/null; then
        echo "Error: another '$MODEL_TYPE' HP-tuning run is already active (PID $(cat "$LOCK_PID_FILE"))."
        echo "Wait for it to finish, or if you're sure it's dead, remove $LOCK_DIR and retry."
        exit 1
    fi
    echo "Stale lock at $LOCK_DIR (owner process no longer running) -- reclaiming it."
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR"
fi
echo $$ > "${LOCK_DIR}/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT

echo "Starting LSTM hyperparameter tuning"
echo "Model Type: $MODEL_TYPE"
echo "Using cores: $START_CORE to $END_CORE"
echo "Total cores: $NUM_CORES"

# Python executable (use virtual environment if available)
PYTHON_CMD="/data0/funato/hsp2/.venv/bin/python"
if [ ! -f "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
fi

# Hyperparameter grids based on model type
if [ "$MODEL_TYPE" == "regional" ]; then
    HIDDEN_SIZE=("${REGIONAL_HIDDEN_SIZE[@]}")
    DROPOUT=("${REGIONAL_DROPOUT[@]}")
    LEARNING_RATE=("${REGIONAL_LEARNING_RATE[@]}")
    BATCH_SIZE=("${REGIONAL_BATCH_SIZE[@]}")
    SEQ_LENGTH=("${REGIONAL_SEQ_LENGTH[@]}")
    # Canonical true-EA-LSTM hp-search script (no_static=False, concat_static=False, fixed).
    # No file from x_lstm_regional_modeling-master is used by this script anymore.
    SCRIPT_PATH="/data0/funato/hsp2/99_work/14_LSTM/ealstm_regional_modeling/main_lstm_regional_hp.py"
    LOG_DIR="/data0/funato/hsp2/99_work/14_LSTM/ealstm_regional_modeling/logs/hp"
elif [ "$MODEL_TYPE" == "individual" ]; then
    HIDDEN_SIZE=("${INDIVIDUAL_HIDDEN_SIZE[@]}")
    DROPOUT=("${INDIVIDUAL_DROPOUT[@]}")
    LEARNING_RATE=("${INDIVIDUAL_LEARNING_RATE[@]}")
    BATCH_SIZE=("${INDIVIDUAL_BATCH_SIZE[@]}")
    SCRIPT_PATH="/data0/funato/hsp2/99_work/14_LSTM/main_LSTM_indiv_hp.py"
    LOG_DIR="/data0/funato/hsp2/99_work/14_LSTM/logs"
else
    echo "Error: ModelType must be 'regional' or 'individual'"
    exit 1
fi

echo "Hyperparameter grid:"
echo "  Hidden sizes: ${HIDDEN_SIZE[@]}"
echo "  Dropout rates: ${DROPOUT[@]}"
echo " Learning rates: ${LEARNING_RATE[@]}"
echo "  Batch sizes: ${BATCH_SIZE[@]}"

# Create log directory
mkdir -p "$LOG_DIR"

# Generate all hyperparameter combinations
combinations=()
if [ "$MODEL_TYPE" == "regional" ]; then
    # Regional combos are "h,drop,lr,seq_length,batch_size" (main_lstm_regional_hp.py accepts
    # --seq_length, unlike the retired legacy path).
    for h_size in "${HIDDEN_SIZE[@]}"; do
        for drop in "${DROPOUT[@]}"; do
            for lr in "${LEARNING_RATE[@]}"; do
                for sl in "${SEQ_LENGTH[@]}"; do
                    for bs in "${BATCH_SIZE[@]}"; do
                        combinations+=("$h_size,$drop,$lr,$sl,$bs")
                    done
                done
            done
        done
    done
else
    # Individual models sweep hidden_size/dropout/learning_rate/seq_length/batch_size
    # (main_LSTM_indiv_hp.py accepts --hidden_size/--dropout/--learning_rate/--seq_length/
    # --batch_size; see grid comments above).
    for h_size in "${HIDDEN_SIZE[@]}"; do
        for drop in "${DROPOUT[@]}"; do
            for lr in "${LEARNING_RATE[@]}"; do
                for sl in "${INDIVIDUAL_SEQ_LENGTH[@]}"; do
                    for bs in "${BATCH_SIZE[@]}"; do
                        combinations+=("$h_size,$drop,$lr,$sl,$bs")
                    done
                done
            done
        done
    done
fi

TOTAL_COMBINATIONS=${#combinations[@]}
echo "Total hyperparameter combinations: $TOTAL_COMBINATIONS"

# Filter out combos that already finished, BEFORE dispatch, so every core spends its
# time on actually-pending work instead of a launch-then-skip round trip. Individual
# combos are "done" once LSTM_log.txt ends with DONE; regional combos are "done" once
# a runs/hp/{loc}/h{..}_d{..}_lr{..}_sl{..}/*/ dir has all three of
# lstm_model_final.pt, hp_metrics_cal.csv, hp_metrics_eva.csv AND that run's own cfg.json
# batch_size matches the combo being checked.
#
# The batch_size check is NOT optional: main_lstm_regional_hp.py's run-directory naming
# (_hp_str()) does NOT include batch_size -- both batch_size=256 and batch_size=512 runs for
# the same hidden_size/dropout/learning_rate/seq_length land in the SAME runs/hp/{loc}/h.._d.._
# lr.._sl../ folder, distinguished only by their own cfg.json. A version of this check that
# only tested file existence (not cfg.json's batch_size) would find the batch_size=256 run,
# conclude the folder is "done", and wrongly skip dispatching batch_size=512 too -- this exact
# false-positive was found live: the script reported "Already complete: 72/72" and dispatched
# nothing, while an actual per-combo count showed only 61/72 real (h,drop,lr,sl,bs) 5-tuples
# had a genuinely matching completed run. This mirrors main_lstm_regional_hp.py's own
# find_completed_run(), which DOES check batch_size (and every other cfg key) correctly --
# only this bash-level pre-filter had drifted out of sync with it.
pending_combinations=()
for combo in "${combinations[@]}"; do
    IFS=',' read -r h_size drop lr sl bs <<< "$combo"
    if [ "$MODEL_TYPE" == "individual" ]; then
        hp_dir="/data0/funato/hsp2/0_out/LSTM/US/${INDIVIDUAL_DATASET}/individual/hp/h${h_size}_d${drop}_lr${lr}_sl${sl}_b${bs}"
        if [ -f "${hp_dir}/LSTM_log.txt" ] && grep -q "^DONE$" "${hp_dir}/LSTM_log.txt" 2>/dev/null; then
            continue
        fi
    else
        search_root="/data0/funato/hsp2/99_work/14_LSTM/ealstm_regional_modeling/runs/hp/${REGIONAL_LOC}/${REGIONAL_DATASET}/h${h_size}_d${drop}_lr${lr}_sl${sl}"
        done_found=0
        for run_dir in "$search_root"/*/; do
            [ -d "$run_dir" ] || continue
            if [ -f "${run_dir}lstm_model_final.pt" ] && [ -f "${run_dir}hp_metrics_cal.csv" ] && [ -f "${run_dir}hp_metrics_eva.csv" ] && [ -f "${run_dir}cfg.json" ]; then
                run_bs=$("$PYTHON_CMD" -c "import json,sys; print(json.load(open(sys.argv[1])).get('batch_size'))" "${run_dir}cfg.json" 2>/dev/null)
                if [ "$run_bs" == "$bs" ]; then
                    done_found=1
                    break
                fi
            fi
        done
        if [ "$done_found" -eq 1 ]; then
            continue
        fi
    fi
    pending_combinations+=("$combo")
done

SKIPPED=$((TOTAL_COMBINATIONS - ${#pending_combinations[@]}))
echo "Already complete: $SKIPPED / $TOTAL_COMBINATIONS"
combinations=("${pending_combinations[@]}")
TOTAL_COMBINATIONS=${#combinations[@]}
echo "Pending combinations to run: $TOTAL_COMBINATIONS"

# Check if we have enough cores
if [ "$TOTAL_COMBINATIONS" -gt "$NUM_CORES" ]; then
    echo "Warning: More combinations ($TOTAL_COMBINATIONS) than cores ($NUM_CORES)"
    echo "Jobs will be queued and run as cores become available"
fi

# Function to run a single job
run_job() {
    local combo=$1
    local core=$2
    local job_id=$3
    
    # Both individual and regional combos are now "h,drop,lr,seq_length,batch_size".
    local h_size drop lr sl bs
    IFS=',' read -r h_size drop lr sl bs <<< "$combo"

    # Skip combos that already finished on a previous run of this script. Individual runs write
    # their hp output under 0_out/LSTM/US/individual/hp/<hp_str>/ (hp_str built the same way
    # main_LSTM_indiv_hp.py builds it: "h{hidden}_d{dropout}_lr{lr}_sl{seq_length}_b{batch}") and
    # append "DONE" as the last line of LSTM_log.txt only after every basin in the sweep has been
    # processed -- predict/results files exist from the very first basin onward, so their mere
    # presence can't be used as a completeness check the way DONE can. (Regional combos already
    # skip completed runs internally in main_lstm_regional_hp.py, so no check is needed here for
    # that model type.)
    if [ "$MODEL_TYPE" == "individual" ]; then
        local hp_dir="/data0/funato/hsp2/0_out/LSTM/US/${INDIVIDUAL_DATASET}/individual/hp/h${h_size}_d${drop}_lr${lr}_sl${sl}_b${bs}"
        if [ -f "${hp_dir}/LSTM_log.txt" ] && grep -q "^DONE$" "${hp_dir}/LSTM_log.txt" 2>/dev/null; then
            echo "[$job_id/$TOTAL_COMBINATIONS] Skipping (already complete): hidden_size=$h_size, dropout=$drop, learning_rate=$lr, seq_length=$sl, batch_size=$bs"
            return
        fi
    fi

    # Create job-specific log file
    local log_file="${LOG_DIR}/hp_tuning_${MODEL_TYPE}_job${job_id}_h${h_size}_d${drop}_lr${lr}_sl${sl}_bs${bs}.log"

    echo "[$job_id/$TOTAL_COMBINATIONS] Starting on core $core: hidden_size=$h_size, dropout=$drop, learning_rate=$lr, seq_length=$sl, batch_size=$bs"
    echo "DEBUG: h_size='$h_size' drop='$drop' lr='$lr' sl='$sl' bs='$bs'" >> "$log_file"

    # Pin to specific core using taskset. GPU IS used when available: this machine's single
    # shared GPU is sometimes near-full from other users' unrelated jobs, in which case parallel
    # HP-search jobs can OOM (`torch.AcceleratorError: CUDA error: out of memory`) -- that's a
    # real risk on a shared box, not a reason to force CPU unconditionally. Each job's own
    # `main_LSTM_indiv_hp.py`/`main_lstm_regional_hp.py` already falls back to CPU automatically
    # when no GPU is free (`torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`), so
    # no CUDA_VISIBLE_DEVICES override is applied here -- GPU is used whenever it's actually
    # available at launch time.
    if [ "$MODEL_TYPE" == "regional" ]; then
        taskset -c $core "$PYTHON_CMD" "$SCRIPT_PATH" \
            --loc "$REGIONAL_LOC" \
            --dataset "$REGIONAL_DATASET" \
            --n_basins "$REGIONAL_N_BASINS" \
            --epochs "$REGIONAL_EPOCHS" \
            --hidden_size "$h_size" \
            --dropout "$drop" \
            --learning_rate "$lr" \
            --seq_length "$sl" \
            --batch_size "$bs" \
            --seed 42 >> "$log_file" 2>&1
    else
        taskset -c $core "$PYTHON_CMD" "$SCRIPT_PATH" \
            --hidden_size "$h_size" \
            --dropout "$drop" \
            --learning_rate "$lr" \
            --seq_length "$sl" \
            --batch_size "$bs" \
            --dataset "$INDIVIDUAL_DATASET" >> "$log_file" 2>&1
    fi

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$job_id/$TOTAL_COMBINATIONS] Completed successfully: hidden_size=$h_size, dropout=$drop, learning_rate=$lr, seq_length=$sl, batch_size=$bs"
    else
        echo "[$job_id/$TOTAL_COMBINATIONS] FAILED (exit code $exit_code): hidden_size=$h_size, dropout=$drop, learning_rate=$lr, seq_length=$sl, batch_size=$bs"
    fi
}

# Export function and variables for parallel execution
export -f run_job
export PYTHON_CMD
export SCRIPT_PATH
export LOG_DIR
export MODEL_TYPE
export TOTAL_COMBINATIONS
export REGIONAL_LOC
export REGIONAL_DATASET
export REGIONAL_N_BASINS
export REGIONAL_EPOCHS
export INDIVIDUAL_DATASET

# Run jobs in parallel with GNU parallel if available, otherwise use background jobs
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for job management"
    
    # Create a temporary file with all jobs
    JOBS_FILE=$(mktemp)
    for i in "${!combinations[@]}"; do
        job_id=$((i + 1))
        core=$((START_CORE + (i % NUM_CORES)))
        echo "${combinations[$i]} $core $job_id" >> "$JOBS_FILE"
    done
    
    # Run with GNU parallel (colsep uses tab by default, we use space for our fields)
    cat "$JOBS_FILE" | parallel -j $NUM_CORES --colsep ' ' 'run_job {1} {2} {3}'
    
    rm "$JOBS_FILE"
else
    echo "GNU parallel not found, using bash background jobs for parallel execution"

    # Run jobs in parallel using background processes, keeping all NUM_CORES slots
    # continuously full: `wait -n` returns as soon as ANY one job finishes, freeing
    # just that slot for the next combo -- unlike a batch `wait` (waits for the
    # WHOLE batch), which would leave already-finished cores idle until the
    # slowest job in that batch completes.
    core_offset=0
    for i in "${!combinations[@]}"; do
        job_id=$((i + 1))
        core=$((START_CORE + (core_offset % NUM_CORES)))

        while (( $(jobs -r -p | wc -l) >= NUM_CORES )); do
            wait -n
        done

        # Run job in background
        run_job "${combinations[$i]}" $core $job_id &

        core_offset=$((core_offset + 1))
    done

    # Wait for any remaining jobs
    wait
fi

echo ""
echo "============================================"
echo "All hyperparameter tuning jobs completed!"
echo "Model Type: $MODEL_TYPE"
echo "Total combinations tested: $TOTAL_COMBINATIONS"
echo "Logs saved in: $LOG_DIR"
echo "============================================"
