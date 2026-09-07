#!/bin/bash
#
# XGBoost hyperparameter tuning: launch a bounded-parallelism grid search, pinned to a
# specific core range, for either the regional or the individual pipeline.
#
# Usage: ./run_xgb_hp_tuning.sh <StartCore> <NumCores> <ModelType>
#   StartCore: first core to use (e.g. 0)
#   NumCores:  how many cores to use, total (e.g. 10) -- this is a hard budget, not a hint
#   ModelType: 'regional' or 'individual'
#
#   /data0/funato/hsp2/99_work/12_XGBoost/run_xgb_hp_tuning.sh 5 15 regional
#   /data0/funato/hsp2/99_work/12_XGBoost/run_xgb_hp_tuning.sh 23 3 individual
#
# Runs itself in the background (nohup) and returns immediately, printing the PID and log
# file -- this replaces the old start_xgb_hp_tuning.sh + run_xgb_hp_tuning.sh pair, which had
# split the "background launcher" and "actual grid loop" into two files. They're combined here
# because the split was actively misleading: the launcher accepted <StartCore> <NumCores>
# <ModelType> but the inner script ignored all three, always used every core on the machine
# (MAX_PARALLEL_JOBS derived from total `nproc`, not from NumCores) and always ran BOTH the
# regional and individual grids regardless of ModelType. That's why a 10-core regional request
# was launching enough parallel jobs to blow through system memory. Fixed below: parallelism
# and taskset ranges are now derived strictly from StartCore/NumCores, and ModelType selects
# exactly one grid.
#
# Grid is the authoritative one from /data0/funato/hsp2/HYPERPARAMETERS.md (see DEBUG_README.md
# for the correction from the old, narrower grid this script used to sweep):
#
#   Individual: N_ESTIMATORS{500,1000} x SUBSAMPLE{0.5,0.6,0.7} x COLSAMPLE_BY_TREE{0.7,0.8,1.0}
#               x LEARNING_RATE{0.01,0.05,0.1} x MAX_DEPTH{2,3,4} x MIN_CHILD_WEIGHT{5,10,20}
#               = 486 combinations
#   Regional:   N_ESTIMATORS{500,1000} x SUBSAMPLE{0.7,0.8,1.0} x COLSAMPLE_BY_TREE{0.7,0.8,1.0}
#               x LEARNING_RATE{0.01,0.05,0.1} x MAX_DEPTH{3,5,7} x MIN_CHILD_WEIGHT{1,3,5}
#               = 486 combinations

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <StartCore> <NumCores> <ModelType>"
    echo "  StartCore: starting core number (e.g., 0)"
    echo "  NumCores:  number of cores to use, total (e.g., 8) -- hard budget"
    echo "  ModelType: 'regional' or 'individual'"
    exit 1
fi

START_CORE=$1
NUM_CORES=$2
MODEL_TYPE=$3

if [ "$MODEL_TYPE" != "regional" ] && [ "$MODEL_TYPE" != "individual" ]; then
    echo "Error: ModelType must be 'regional' or 'individual', got '$MODEL_TYPE'"
    exit 1
fi

SCRIPT_DIR="/data0/funato/hsp2/99_work/12_XGBoost"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# --- Outer invocation: background the real work and return immediately -------------------
if [ -z "$XGB_HP_TUNING_CHILD" ]; then
    LOG_FILE="${LOG_DIR}/hp_tuning_${MODEL_TYPE}_${START_CORE}_${NUM_CORES}_$(date +%Y%m%d_%H%M%S).log"

    echo "Starting XGBoost hyperparameter tuning in background..."
    echo "Model Type: $MODEL_TYPE"
    echo "Cores: $START_CORE to $((START_CORE + NUM_CORES - 1)) ($NUM_CORES cores)"
    echo "Log file: $LOG_FILE"

    XGB_HP_TUNING_CHILD=1 nohup "$0" "$START_CORE" "$NUM_CORES" "$MODEL_TYPE" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "Process started with PID: $PID"
    echo "To monitor progress: tail -f $LOG_FILE"
    echo "To check if running: ps -p $PID"
    echo "To stop: kill $PID"
    exit 0
fi

# --- Inner invocation: the actual grid search, run in the background as the child ---------
if [ -x "/data0/funato/hsp2/.venv/bin/python3" ]; then
    PYTHON="/data0/funato/hsp2/.venv/bin/python3"
else
    PYTHON="$(command -v python3 || command -v python || true)"
    if [ -z "$PYTHON" ]; then
        echo "Error: No python executable found. Install Python or create the virtualenv at /data0/funato/hsp2/.venv"
        exit 1
    fi
fi
REGIONAL_SCRIPT="${SCRIPT_DIR}/XGB_regional_modeling-master/main_xgb_hp.py"
INDIVIDUAL_SCRIPT="${SCRIPT_DIR}/main_xgboost_indiv_hp.py"

CORES_PER_JOB=2
MAX_PARALLEL_JOBS=$(( NUM_CORES / CORES_PER_JOB ))
if [ "$MAX_PARALLEL_JOBS" -lt 1 ]; then MAX_PARALLEL_JOBS=1; fi

N_ESTIMATORS=(500 1000)
COLSAMPLE_BYTREE=(0.7 0.8 1.0)
LEARNING_RATES=(0.01 0.05 0.1)
REGIONAL_SUBSAMPLE=(0.7 0.8 1.0)
INDIVIDUAL_SUBSAMPLE=(0.5 0.6 0.7)
REGIONAL_MAX_DEPTHS=(3 5 7)
INDIVIDUAL_MAX_DEPTHS=(2 3 4)
REGIONAL_MIN_CHILD_WEIGHT=(1 3 5)
INDIVIDUAL_MIN_CHILD_WEIGHT=(5 10 20)

echo "Starting XGBoost Hyperparameter Tuning ($MODEL_TYPE)"
echo "========================================"
echo "Combinations: 486"
echo "Core budget: $NUM_CORES cores (${START_CORE}-$((START_CORE + NUM_CORES - 1)))"
echo "Max parallel jobs: $MAX_PARALLEL_JOBS (${CORES_PER_JOB} cores/job)"
echo ""

job_id=0
core_offset=0

wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL_JOBS" ]; do
        wait -n
    done
}

if [ "$MODEL_TYPE" = "regional" ]; then
    SCRIPT="$REGIONAL_SCRIPT"
    SUBSAMPLE=("${REGIONAL_SUBSAMPLE[@]}")
    MAX_DEPTHS=("${REGIONAL_MAX_DEPTHS[@]}")
    MIN_CHILD_WEIGHT=("${REGIONAL_MIN_CHILD_WEIGHT[@]}")
    EXTRA_ARGS=(train)
    # main_xgb.py/main_xgb_hp.py default --seed to a random value when omitted, which defeats
    # the completed-run skip check in main_xgb_hp.py's train() (it matches on seed, so two
    # invocations of the same hyperparameter combo would never look like the same run and this
    # script would retrain the entire grid on every restart). Pinned to 42 to match the fixed
    # seed already used by run_rf_hp_tuning.sh/run_tcn_hp_tuning.sh/run_lstm_hp_tuning_regional.sh.
    SEED_ARGS=(--seed 42)
else
    SCRIPT="$INDIVIDUAL_SCRIPT"
    SUBSAMPLE=("${INDIVIDUAL_SUBSAMPLE[@]}")
    MAX_DEPTHS=("${INDIVIDUAL_MAX_DEPTHS[@]}")
    MIN_CHILD_WEIGHT=("${INDIVIDUAL_MIN_CHILD_WEIGHT[@]}")
    EXTRA_ARGS=()
    # main_xgboost_indiv_hp.py has no --seed CLI flag (it hardcodes random_state=42 internally),
    # so nothing is passed here.
    SEED_ARGS=()
fi

for n_est in "${N_ESTIMATORS[@]}"; do
    for max_d in "${MAX_DEPTHS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            for ss in "${SUBSAMPLE[@]}"; do
                for cbt in "${COLSAMPLE_BYTREE[@]}"; do
                    for mcw in "${MIN_CHILD_WEIGHT[@]}"; do
                        wait_for_slot
                        job_id=$((job_id + 1))

                        job_slot=$((core_offset % MAX_PARALLEL_JOBS))
                        core_start=$((START_CORE + job_slot * CORES_PER_JOB))
                        core_end=$((core_start + CORES_PER_JOB - 1))
                        core_offset=$((core_offset + 1))

                        LOG_FILE="${LOG_DIR}/hp_tuning_${MODEL_TYPE}_n${n_est}_d${max_d}_lr${lr}_ss${ss}_cbt${cbt}_mcw${mcw}.log"
                        echo "${MODEL_TYPE} job ${job_id}/486: n=${n_est} d=${max_d} lr=${lr} ss=${ss} cbt=${cbt} mcw=${mcw} (cores ${core_start}-${core_end})"

                        taskset -c ${core_start}-${core_end} $PYTHON $SCRIPT "${EXTRA_ARGS[@]}" \
                            --n_estimators "$n_est" \
                            --max_depth "$max_d" \
                            --learning_rate "$lr" \
                            --subsample "$ss" \
                            --colsample_bytree "$cbt" \
                            --min_child_weight "$mcw" \
                            "${SEED_ARGS[@]}" \
                            > "$LOG_FILE" 2>&1 &
                    done
                done
            done
        done
    done
done

echo ""
echo "Waiting for all ${MODEL_TYPE} jobs to complete..."
wait

echo ""
echo "All XGBoost ${MODEL_TYPE} hyperparameter tuning jobs completed!"
echo "Monitor progress with: tail -f $LOG_DIR/hp_tuning_${MODEL_TYPE}_*.log"
