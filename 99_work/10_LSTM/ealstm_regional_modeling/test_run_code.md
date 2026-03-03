# Train
python3 /data0/funato/hsp2/99_work/10_LSTM/ealstm_regional_modeling/main.py train \
  --camels_root /data0/funato/hsp2/2_MERV/varssim_US \
  --seed 123

# TRAIN BACKGROUND 
nohup python3 /data0/funato/hsp2/99_work/10_LSTM/ealstm_regional_modeling/main.py train --camels_root /data0/funato/hsp2/2_MERV/varssim_US --seed 123 > training.log 2>&1 &


# Evaluate
python3 /data0/funato/hsp2/99_work/10_LSTM/ealstm_regional_modeling/main.py evaluate \
  --camels_root /data0/funato/hsp2/2_MERV/varssim_US \
  --run_dir /data0/funato/hsp2/99_work/10_LSTM/ealstm_regional_modeling/runs/run_2002_1721_seed123

# Evaluate Backgound
nohup python3 /data0/funato/hsp2/99_work/10_LSTM/ealstm_regional_modeling/main.py evaluate \
  --camels_root /data0/funato/hsp2/2_MERV/varssim_US \
  --run_dir /data0/funato/hsp2/99_work/10_LSTM/ealstm_regional_modeling/runs/run_2102_1642_seed123 > evaluate.log 2>&1 &

# Adapted 
python3 /data0/funato/hsp2/99_work/10_LSTM/ealstm_regional_modeling/main.py adapt \
  --camels_root /data0/funato/hsp2/2_MERV/varssim_US \
  --run_dir /data0/funato/hsp2/99_work/10_LSTM/ealstm_regional_modeling/runs/run_2102_1642_seed123 

# Adapted  Background
nohup python3 /data0/funato/hsp2/99_work/10_LSTM/ealstm_regional_modeling/main.py adapt \
  --camels_root /data0/funato/hsp2/2_MERV/varssim_US \
  --run_dir /data0/funato/hsp2/99_work/10_LSTM/ealstm_regional_modeling/runs/run_2102_1642_seed123 > adapted.log 2>&1 &
