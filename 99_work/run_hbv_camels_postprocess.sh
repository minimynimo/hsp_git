#!/bin/bash
# Post-processing for the 2026-10-20 HBV camels recalibration (MATLAB/MARRMoT_v212_HBV/parallel.sh,
# DATA_START/CAL 1987-01-01..2000-12-31, EVAL 2001-01-01..2006-12-31, 41-day synthetic peak
# window identical to the ML forcing). Run ONLY after every MATLAB process has exited:
#   pgrep -f workflow_TEST_cal_parallel_HBV   -> must print nothing
#
# 1. moves the previous (index-shifted, cal 1987-1997) HBV outputs aside
# 2. copies the new MATLAB OutputData_HBV / Parameters_HBV into 0_out/HBV/US/camels
# 3. rebuilds varssim_HBV (create_dataset_HBV.py) and the cal/eva metrics (HBV_benchmark.py)
set -euo pipefail

if pgrep -f workflow_TEST_cal_parallel_HBV > /dev/null; then
  echo "MATLAB HBV processes are still running -- wait until they finish." >&2
  exit 1
fi

M=/data0/funato/hsp2/MATLAB/MARRMoT_v212_HBV
O=/data0/funato/hsp2/0_out/HBV/US/camels
BAK=$O/_bak_cal1987_1997_shifted_20260916
PY="taskset -c 0-7 /data0/funato/.venv/bin/python3"

echo "New outputs: $(ls $M/OutputData_HBV/*_all_Q_outputs.csv | wc -l) basins"
grep -l "FAILED basin" $M/logs/*.log && grep -h "FAILED basin" $M/logs/*.log || echo "No failed basins in MATLAB logs"

mkdir -p "$BAK"
for d in OutputData_HBV Parameters_HBV varssim_HBV results peak_discharge; do
  [ -e "$O/$d" ] && mv "$O/$d" "$BAK/$d"
done
cp -a $M/OutputData_HBV $O/OutputData_HBV
cp -a $M/Parameters_HBV $O/Parameters_HBV

$PY /data0/funato/hsp2/2_MERV/create_dataset_HBV.py
$PY /data0/funato/hsp2/hsp_git/99_work/00_analysis/data/HBV_benchmark.py
echo "HBV post-processing done: $(ls $O/varssim_HBV | wc -l) varssim files"
