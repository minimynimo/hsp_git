#!/bin/bash

# ============================================
# CONFIGURATION SECTION - Edit parameters here
# ============================================

# Core assignment
cores=1 # KEEP IT 1
start_core=3 # Start core (1-indexed)

# Seeds
nseeds=1
firstseed=300

# Static data
static_bool=false

# LSTM Hyperparameters
HIDDEN_SIZE=256
DROPOUT_RATE=0.2
LEARNING_RATE=0.001
BATCH_SIZE=256

# ============================================
# END CONFIGURATION SECTION
# ============================================


if [ "$static_bool" = true ]; then
    log_name="master_log_LSTM_static.log"
else
    log_name="master_log_LSTM_nostatic.log"
fi


loc="GB"

end_core=$((start_core + cores - 1))  # Calculate end core based on core count

gpucount=1

# Set default model to "lstm" if not provided
model=${1:-lstm}

output_dir="/data0/funato/hsp2/bash_logs"

mkdir -p "$output_dir" # Create logs directory if it doesn't exist

> "$output_dir/$log_name" # Clear the log file


for (( seed = $firstseed ; seed < $((nseeds+$firstseed)) ; seed++ )); do
    gpucount=$(($gpucount + 1))
    core=$((start_core + (gpucount % cores))) # Ensure core is within start_core to end_core range
    if [ $core -gt $end_core ]; then
        core=$start_core # Wrap around to start_core if exceeding end_core
    fi
    gpu=$((gpucount % 3)) # Assign GPU in a round-robin fashion
    echo $seed $gpucount $core $gpu >> "$output_dir/$log_name"

    if [ "$model" = "lstm" ]; then
        if [ "$static_bool" = true ]; then
            #add current date and time to the outfile name
            outfile="/data0/funato/hsp2/0_out/LSTM/GB/reports_static/lstm.$seed.$(date +%Y-%m-%d_%H-%M-%S).out"
        else
            outfile="/data0/funato/hsp2/0_out/LSTM/GB/reports_nostatic/lstm.$seed.$(date +%Y-%m-%d_%H-%M-%S).out"
        fi

        mkdir -p "$(dirname "$outfile")" # Create output directory if it doesn't exist
        start_time=$(date '+%Y-%m-%d %H:%M:%S') # Log start time1
        echo "Start Time: $start_time | Seed: $seed" >> "$output_dir/$log_name"
        
        if [ "$static_bool" = true ]; then  ## static data is used
            echo "Running: taskset -c $core python3 /data0/funato/hsp2/99_work/10_LSTM/lstm_regional_modeling-master/main_lstm.py train --no_static=False --concat_static=False --num_workers=$cores --hidden_size=$HIDDEN_SIZE --dropout=$DROPOUT_RATE --learning_rate=$LEARNING_RATE --batch_size=$BATCH_SIZE --seed=$seed" >> "$output_dir/$log_name"
            taskset -c $core python3 /data0/funato/hsp2/99_work/10_LSTM/lstm_regional_modeling-master/main_lstm.py train --no_static=False --concat_static=False --num_workers=$cores --hidden_size=$HIDDEN_SIZE --dropout=$DROPOUT_RATE --learning_rate=$LEARNING_RATE --batch_size=$BATCH_SIZE --seed=$seed > $outfile 2>> "$output_dir/$log_name" &
        else  ## no static data is used
            echo "Running: taskset -c $core python3 /data0/funato/hsp2/99_work/10_LSTM/lstm_regional_modeling-master/main_lstm.py train --no_static=True --concat_static=False --num_workers=$cores --hidden_size=$HIDDEN_SIZE --dropout=$DROPOUT_RATE --learning_rate=$LEARNING_RATE --batch_size=$BATCH_SIZE --seed=$seed" >> "$output_dir/$log_name"
            taskset -c $core python3 /data0/funato/hsp2/99_work/10_LSTM/lstm_regional_modeling-master/main_lstm.py train --no_static=True --concat_static=False --num_workers=$cores --hidden_size=$HIDDEN_SIZE --dropout=$DROPOUT_RATE --learning_rate=$LEARNING_RATE --batch_size=$BATCH_SIZE --seed=$seed > $outfile 2>> "$output_dir/$log_name" &
        fi

        wait # Ensure the process completes before logging end time
        end_time=$(date '+%Y-%m-%d %H:%M:%S') # Log end time
        echo "End Time: $end_time | Seed: $seed" >> "$output_dir/$log_name"
    else
        echo "Error: Unsupported model choice '$model'. Supported models: lstm" >> "$output_dir/$log_name"
        exit 1
    fi

    if [ $((gpucount % cores)) -eq $((cores - 1)) ]; then
        wait
    fi
done

###RUN USING####
# chmod +x /data0/funato/hsp2/99_work/10_LSTM/lstm_regional_modeling-master/run_main_lstm.sh
# nohup /data0/funato/hsp2/99_work/10_LSTM/lstm_regional_modeling-master/run_main_lstm.sh > /dev/null 2>&1 &
