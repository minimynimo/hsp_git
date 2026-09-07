#!/usr/bin/env bash
# Full-factorial hyperparameter sweep driver for main_HYPER_hyperparameter.py.
# Every combination of reservoir_size x ridge_param x input_scale x
# spectral_radius x adjacency_density is run as its own single-CPU process
# (pinned via taskset), with up to MAX_JOBS running concurrently.
#
# Before launching each combination, this checks the eva results CSV under
# _out/.../HYPER/kgesoftmax/hp/<hp_tag>/results/ for that combination. If it
# already has one row per expected basin, the combination is considered
# complete and is skipped; otherwise main_HYPER_hyperparameter.py is invoked
# (which always rewrites its output files from scratch, so a partial/failed
# prior run is effectively restarted from the beginning).

# run using
# chmod +x /data0/funato/97_work/01_HYPER/run_HYPER_hyperparameter_sweep.sh
# nohup /data0/funato/97_work/01_HYPER/run_HYPER_hyperparameter_sweep.sh > /dev/null 2>&1 &
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="/data0/funato/.venv/bin/python"
SCRIPT="main_HYPER_hyperparameter.py"
MAX_JOBS="${MAX_JOBS:-60}"
# First CPU core to use; jobs are pinned to cores [START_CORE, START_CORE+MAX_JOBS-1].
# Override to run several sweeps concurrently on disjoint core ranges, e.g.
# START_CORE=25 MAX_JOBS=25 ./run_HYPER_hyperparameter_sweep.sh
START_CORE=0
LOG_DIR="./sweep_logs"
mkdir -p "$LOG_DIR"

# Must match the output-path construction in main_HYPER_hyperparameter.py.
OUT_BASE="/data0/funato/97_work/_out/97_out_glb_MSWEP_MSWX"
BMA_TYPE="kgesoftmax"

# Must match the *_list values in main_HYPER_hyperparameter.py.
RESERVOIR_SIZES=(10 50 100 200 300 500 700) #50 100 200 
RIDGE_PARAMS=(0.1 0.01 0.001 0.0001)
INPUT_SCALES=(0.1 1.0 10.0)
SPECTRAL_RADII=(0.85 0.90 0.95 1.0 1.05 1.10)
ADJACENCY_DENSITIES=(0.05 0.1 0.2 0.3 0.5 1.0)

# main_HYPER_hyperparameter.py builds its output-dir/file tags from Python's
# str(float(...)) (e.g. 0.90 -> "0.9"), not from the literal CLI string, so
# values are normalized the same way before being used to build paths below.
# Cached since the same values recur across the nested loops below.
declare -A SLOT_PID
declare -A FLOAT_CACHE
norm_float() {
  local x="$1"
  if [[ -z "${FLOAT_CACHE[$x]+set}" ]]; then
    FLOAT_CACHE[$x]="$("$PYTHON_BIN" -c "print(float('$x'))")"
  fi
  printf '%s' "${FLOAT_CACHE[$x]}"
}

# Number of basins a complete run is expected to have a result row for.
# Mirrors the load_data()/BMA-index skip logic in main_HYPER_hyperparameter.py
# (basins missing a MARRMoT simulation file or missing from the BMA
# predictions are skipped there too). This doesn't depend on the swept
# hyperparameters, so it's computed once up front rather than per combination.
echo "Computing expected basin count..."
EXPECTED_BASINS=$("$PYTHON_BIN" - <<EOF
import os
import pandas as pd

MET_DATA = "MSWEP_MSWX"
ver = 1
BMA_TYPE = "${BMA_TYPE}"
sim_dir = f"/data0/funato/97_work/200_dataset/data/MARRMoT/glb_GRDC_{MET_DATA}_nocal/v{ver}"
grdc_id_list = pd.read_csv(
    "/data0/funato/97_work/200_dataset/basin_list/grdc/grdc_station_id_list_small_MSWEP_MSWX_3000.csv",
    index_col=None, header=0,
)['grdc_id'].values.tolist()[::4]

output_base = "${OUT_BASE}"
bma_df_cal = pd.read_csv(f'{output_base}/BMA/{BMA_TYPE}/predict/BMA_predict_cal.csv', index_col=0)
bma_df_eva = pd.read_csv(f'{output_base}/BMA/{BMA_TYPE}/predict/BMA_predict_eva.csv', index_col=0)

count = 0
for grdc_id in grdc_id_list:
    if not os.path.exists(f"{sim_dir}/grdc_{grdc_id}.csv"):
        continue
    if f'grdc_{grdc_id}' not in bma_df_cal.index or f'grdc_{grdc_id}' not in bma_df_eva.index:
        continue
    count += 1
print(count)
EOF
)
echo "Expected basins per complete run: $EXPECTED_BASINS"

job_index=0
total=$(( ${#RESERVOIR_SIZES[@]} * ${#RIDGE_PARAMS[@]} * ${#INPUT_SCALES[@]} * ${#SPECTRAL_RADII[@]} * ${#ADJACENCY_DENSITIES[@]} ))
echo "Total combinations: $total (up to $MAX_JOBS concurrent, cores ${START_CORE}-$((START_CORE + MAX_JOBS - 1)))"

for r in "${RESERVOIR_SIZES[@]}"; do
  for rr in "${RIDGE_PARAMS[@]}"; do
    for is in "${INPUT_SCALES[@]}"; do
      for sr in "${SPECTRAL_RADII[@]}"; do
        for ad in "${ADJACENCY_DENSITIES[@]}"; do
          rr_n=$(norm_float "$rr")
          is_n=$(norm_float "$is")
          sr_n=$(norm_float "$sr")
          ad_n=$(norm_float "$ad")

          tag="r${r}_rr${rr_n}_is${is_n}_sr${sr_n}_ad${ad_n}"
          logfile="${LOG_DIR}/${tag}.log"
          results_file="${OUT_BASE}/HYPER/${BMA_TYPE}/hp/${tag}/results/HYPER_results_r${r}_s1_sr${sr_n}_rr${rr_n}_eva.csv"

          # Skip combinations whose eva results file already has a row for
          # every expected basin; otherwise fall through and (re-)run, which
          # overwrites the results/predict/Wout files from scratch.
          if [[ -f "$results_file" ]]; then
            actual_basins=$(( $(wc -l < "$results_file") - 1 ))
            if (( actual_basins >= EXPECTED_BASINS )); then
              echo "[$job_index/$total] SKIP (complete: ${actual_basins}/${EXPECTED_BASINS} basins) $tag"
              job_index=$(( job_index + 1 ))
              continue
            fi
          fi

          # Find a CPU slot whose previous occupant has actually finished.
          # Job runtimes vary hugely across this sweep (reservoir_size spans
          # 10..700), so a simple job_index % MAX_JOBS assignment can pin a
          # new job onto a core that's still busy with an earlier, slower
          # job -- doubling them up on one physical core (each stuck at
          # ~50% CPU) while another core sits idle. Track slot -> pid and
          # only reuse a slot once kill -0 shows that pid is gone.
          slot=-1
          while (( slot < 0 )); do
            for (( c=0; c<MAX_JOBS; c++ )); do
              pid="${SLOT_PID[$c]:-}"
              if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                slot=$c
                break
              fi
            done
            if (( slot < 0 )); then
              wait -n
            fi
          done

          cpu=$(( START_CORE + slot ))
          echo "[$job_index/$total] cpu=$cpu (slot=$slot) $tag"

          # Numpy/BLAS default to multi-threading; pin to 1 thread so each
          # taskset-pinned process actually stays on its single CPU.
          OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "$cpu" "$PYTHON_BIN" "$SCRIPT" \
            --reservoir_size "$r" \
            --ridge_param "$rr" \
            --input_scale "$is" \
            --spectral_radius "$sr" \
            --adjacency_density "$ad" \
            > "$logfile" 2>&1 &
          SLOT_PID[$slot]=$!

          job_index=$(( job_index + 1 ))
        done
      done
    done
  done
done

wait
echo "Sweep complete: $total combinations processed (see log above for skipped vs. run). Logs in $LOG_DIR/"
