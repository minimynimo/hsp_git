#!/bin/bash

# ============================================
# CONFIGURATION SECTION - Edit parameters here
# ============================================

# Core assignment
cores=1
start_core=35 # Start core (1-indexed)

# Seeds
nseeds=1
firstseed=309

# Static data
static_bool=false
# if static_bool is true, use static data; if false, do not use static data

# TCN Hyperparameters
NUM_CHANNELS="128 256 512"  # Space-separated list of channel sizes
BATCH_SIZE=32
EPOCHS=50
KERNEL_SIZE=4
NUM_WORKERS=2

# ============================================
# END CONFIGURATION SECTION
# ============================================

if [ "$static_bool" = true ]; then
    log_name="master_log_TCN_static.log"
else
    log_name="master_log_TCN_nostatic.log"
fi

loc="US"

end_core=$((start_core + cores - 1))  # Calculate end core based on core count

gpucount=-1

# Set default model to "tcn" if not provided
model=${1:-tcn}

output_dir="/data0/funato/hsp2/bash_logs"

mkdir -p "$output_dir" # Create logs directory if it doesn't exist

> "$output_dir/$log_name" # Clear the log file

for (( seed = $firstseed ; seed < $((nseeds+$firstseed)) ; seed++ )); do
    gpucount=$(($gpucount + 1))
    core=$((start_core + (gpucount % cores))) # Ensure core is within start_core to end_core range
    if [ $core -gt $end_core ]; then
        core=$start_core # Wrap around to start_core if exceeding end_core
    fi
    
    echo $seed $gpucount $core >> "$output_dir/$log_name"

    if [ "$model" = "tcn" ]; then
        if [ "$static_bool" = true ]; then
            outfile="/data0/funato/hsp2/0_out/TCN/$loc/reports_static/tcn.$seed.$(date +%Y-%m-%d_%H-%M-%S).out"
        else
            outfile="/data0/funato/hsp2/0_out/TCN/$loc/reports_nostatic/tcn.$seed.$(date +%Y-%m-%d_%H-%M-%S).out"
        fi
        mkdir -p "$(dirname "$outfile")" # Create output directory if it doesn't exist
        start_time=$(date '+%Y-%m-%d %H:%M:%S') # Log start time
        echo "Start Time: $start_time | Seed: $seed" >> "$output_dir/$log_name"

        # Build parameter strings from configuration
        TCN_PARAMS="--num_channels $NUM_CHANNELS"
        OTHER_PARAMS="--batch_size=$BATCH_SIZE --epochs=$EPOCHS --kernel_size=$KERNEL_SIZE --seed=$seed --num_workers=$NUM_WORKERS"


        if [ "$static_bool" = true ]; then ## static data is used
            echo "Running: taskset -c $core python3 /data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/main_tcn.py train --no_static=False $OTHER_PARAMS $TCN_PARAMS" >> "$output_dir/$log_name"

            taskset -c $core python3 /data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/main_tcn.py train --no_static=False $OTHER_PARAMS $TCN_PARAMS > $outfile 2>> "$output_dir/$log_name" &
        else #  no static data is used
            echo "Running: taskset -c $core python3 /data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/main_tcn.py train --no_static=True $OTHER_PARAMS $TCN_PARAMS" >> "$output_dir/$log_name"

            taskset -c $core python3 /data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/main_tcn.py train --no_static=True $OTHER_PARAMS $TCN_PARAMS > $outfile 2>> "$output_dir/$log_name" &
        fi

        wait # Ensure the process completes before logging end time
        end_time=$(date '+%Y-%m-%d %H:%M:%S') # Log end time
        echo "End Time: $end_time | Seed: $seed" >> "$output_dir/$log_name"
    else
        echo "Error: Unsupported model choice '$model'. Supported models: tcn" >> "$output_dir/$log_name"
        exit 1
    fi

    if [ $((gpucount % cores)) -eq $((cores - 1)) ]; then
        wait
    fi
done



###RUN USING####
# chmod +x /data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/run_main_tcn.sh
# nohup /data0/funato/hsp2/99_work/13_TCN/TCN_regional_modeling-master/run_main_tcn.sh tcn > /dev/null 2>&1 &
