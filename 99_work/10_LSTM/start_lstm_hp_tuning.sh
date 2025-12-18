#!/bin/bash

# Wrapper script to run LSTM hyperparameter tuning in background
# Usage: ./start_lstm_hp_tuning.sh <StartCore> <NumCores> <ModelType>
# /data0/funato/hsp/99_work/10_LSTM/start_lstm_hp_tuning.sh 24 4 regional
# /data0/funato/hsp/99_work/10_LSTM/start_lstm_hp_tuning.sh 17 2 individual

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <StartCore> <NumCores> <ModelType>"
    echo "  StartCore: Starting core number (e.g., 0)"
    echo "  NumCores: Number of cores to use (e.g., 4)"
    echo "  ModelType: 'regional' or 'individual'"
    exit 1
fi

START_CORE=$1
NUM_CORES=$2
MODEL_TYPE=$3

SCRIPT_DIR="/data0/funato/hsp/99_work/10_LSTM"
LOG_FILE="${SCRIPT_DIR}/hp_tuning_${MODEL_TYPE}_${START_CORE}_${NUM_CORES}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting LSTM hyperparameter tuning in background..."
echo "Model Type: $MODEL_TYPE"
echo "Cores: $START_CORE to $((START_CORE + NUM_CORES - 1))"
echo "Log file: $LOG_FILE"

# Run in background with nohup
nohup ${SCRIPT_DIR}/run_lstm_hp_tuning.sh $START_CORE $NUM_CORES $MODEL_TYPE > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "To monitor progress: tail -f $LOG_FILE"
echo "To check if running: ps -p $PID"
echo "To stop: kill $PID"
