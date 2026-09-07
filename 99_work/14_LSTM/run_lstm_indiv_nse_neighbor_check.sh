#!/bin/bash
#
# Reduced-grid check: does the SELECTED LSTM Individual hyperparameter combo (hidden_size=32,
# dropout=0.4, learning_rate=0.0005, seq_length=365, batch_size=64 -- picked under the original
# MSE-loss HP grid) still come out on top when trained with NSE loss instead?
#
# Per the user's explicit instruction to limit computational cost: for each hyperparameter
# dimension, only test the selected value plus its ONE adjacent neighbor in the original grid's
# ordered value list (the neighbor toward higher values, unless the selected value was already
# the last/highest in the list, in which case toward lower values). Concretely:
#   hidden_size    [16,32,64,128], selected=32 (not last) -> test {32,64}
#   dropout        [0.4,0.5],      selected=0.4 (not last) -> test {0.4,0.5} (both, only 2 exist)
#   learning_rate  [0.001,0.0005], selected=0.0005 (last)  -> test {0.001,0.0005} (both)
#   seq_length     [270,365],      selected=365 (last)     -> test {270,365} (both)
#   batch_size     [64,128],       selected=64 (not last)  -> test {64,128} (both)
# = 2x2x2x2x2 = 32 combos (half of the original 64-combo MSE grid), each with --loss nse.
#
# Uses the same 154-basin ([::4]) subsample as the original HP grid (main_LSTM_indiv_hp.py's own
# gauge_id_list), same as every other individual/regional HP search in this project.

HIDDEN_SIZE=(32 64)
DROPOUT=(0.4 0.5)
LEARNING_RATE=(0.001 0.0005)
SEQ_LENGTH=(270 365)
BATCH_SIZE=(64 128)

MAX_PARALLEL=4   # kept low: GPU is already at ~99% util / 26GB used from other concurrent work
START_CORE=104
SCRIPT="/data0/funato/hsp2/99_work/14_LSTM/main_LSTM_indiv_hp.py"
LOG_DIR="/data0/funato/hsp2/99_work/14_LSTM/logs"
mkdir -p "$LOG_DIR"

job_id=0
core_offset=0

wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n
    done
}

TOTAL=$(( ${#HIDDEN_SIZE[@]} * ${#DROPOUT[@]} * ${#LEARNING_RATE[@]} * ${#SEQ_LENGTH[@]} * ${#BATCH_SIZE[@]} ))
echo "Running $TOTAL NSE-loss combos (neighbor-of-selected reduced grid), max $MAX_PARALLEL concurrent"

for h in "${HIDDEN_SIZE[@]}"; do
    for d in "${DROPOUT[@]}"; do
        for lr in "${LEARNING_RATE[@]}"; do
            for sl in "${SEQ_LENGTH[@]}"; do
                for b in "${BATCH_SIZE[@]}"; do
                    wait_for_slot
                    job_id=$((job_id + 1))
                    core=$((START_CORE + (core_offset % 8)))
                    core_offset=$((core_offset + 1))
                    LOG_FILE="${LOG_DIR}/nse_neighbor_h${h}_d${d}_lr${lr}_sl${sl}_b${b}.log"
                    echo "[$job_id/$TOTAL] h=$h d=$d lr=$lr sl=$sl b=$b (core $core)"
                    taskset -c "$core" python3 "$SCRIPT" \
                        --hidden_size "$h" --dropout "$d" --learning_rate "$lr" \
                        --seq_length "$sl" --batch_size "$b" --loss nse \
                        > "$LOG_FILE" 2>&1 &
                done
            done
        done
    done
done

wait
echo "All $TOTAL NSE-loss combos finished."
