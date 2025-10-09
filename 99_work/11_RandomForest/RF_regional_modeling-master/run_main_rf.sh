#!/bin/bash

cores=1
start_core=7 # Start core (1-indexed)
###

nseeds=1
firstseed=300

static_bool=false
# if static_bool is true, use static data; if false, do not use static data

if [ "$static_bool" = true ]; then
    log_name="master_log_RF_static.log"
else
    log_name="master_log_RF_nostatic.log"
fi

loc="GB"

end_core=$((start_core + cores - 1))  # Calculate end core based on core count

gpucount=-1

# Set default model to "rf" if not provided
model=${1:-rf}

output_dir="/data0/funato/hsp/bash_logs"

mkdir -p "$output_dir" # Create logs directory if it doesn't exist

> "$output_dir/$log_name" # Clear the log file

for (( seed = $firstseed ; seed < $((nseeds+$firstseed)) ; seed++ )); do
    gpucount=$(($gpucount + 1))
    core=$((start_core + (gpucount % cores))) # Ensure core is within start_core to end_core range
    if [ $core -gt $end_core ]; then
        core=$start_core # Wrap around to start_core if exceeding end_core
    fi
    
    echo $seed $gpucount $core >> "$output_dir/$log_name"

    if [ "$model" = "rf" ]; then
        if [ "$static_bool" = true ]; then
            outfile="/data0/funato/hsp/0_out/RF/GB/reports_static/rf.$seed.$(date +%Y-%m-%d_%H-%M-%S).out"
        else
            outfile="/data0/funato/hsp/0_out/RF/GB/reports_nostatic/rf.$seed.$(date +%Y-%m-%d_%H-%M-%S).out"
        fi
        
        mkdir -p "$(dirname "$outfile")" # Create output directory if it doesn't exist
        start_time=$(date '+%Y-%m-%d %H:%M:%S') # Log start time
        echo "Start Time: $start_time | Seed: $seed" >> "$output_dir/$log_name"

        if [ "$static_bool" = true ]; then
            echo "Running: taskset -c $core python3 /data0/funato/hsp/99_work/11_RandomForest/RF_regional_modeling-master/main_rf.py train --no_static=False --concat_static=True --n_estimators=100 --max_depth=5 --max_features=0.7 --max_samples=0.5 --min_samples_split=2 --seed=$seed" >> "$output_dir/$log_name"
            taskset -c $core python3 /data0/funato/hsp/99_work/11_RandomForest/RF_regional_modeling-master/main_rf.py train --no_static=False --concat_static=True --n_estimators=100 --max_depth=5 --max_features=0.7 --max_samples=0.5 --min_samples_split=2 --seed=$seed > $outfile 2>> "$output_dir/$log_name" &
        else
            echo "Running: taskset -c $core python3 /data0/funato/hsp/99_work/11_RandomForest/RF_regional_modeling-master/main_rf.py train --no_static=True --concat_static=True --n_estimators=100 --max_depth=5 --max_features=0.7 --max_samples=0.5 --min_samples_split=2 --seed=$seed" >> "$output_dir/$log_name"
            taskset -c $core python3 /data0/funato/hsp/99_work/11_RandomForest/RF_regional_modeling-master/main_rf.py train --no_static=True --concat_static=True --n_estimators=100 --max_depth=5 --max_features=0.7 --max_samples=0.5 --min_samples_split=2 --seed=$seed > $outfile 2>> "$output_dir/$log_name" &
        fi
        wait # Ensure the process completes before logging end time
        end_time=$(date '+%Y-%m-%d %H:%M:%S') # Log end time
        echo "End Time: $end_time | Seed: $seed" >> "$output_dir/$log_name"
    else
        echo "Error: Unsupported model choice '$model'. Supported models: rf" >> "$output_dir/$log_name"
        exit 1
    fi

    if [ $((gpucount % cores)) -eq $((cores - 1)) ]; then
        wait
    fi
done

###RUN USING####
# chmod +x /data0/funato/hsp/99_work/11_RandomForest/RF_regional_modeling-master/run_main_rf.sh
# nohup /data0/funato/hsp/99_work/11_RandomForest/RF_regional_modeling-master/run_main_rf.sh > /dev/null 2>&1 &