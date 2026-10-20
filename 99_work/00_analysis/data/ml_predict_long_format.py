"""Shared writer for the "long format" (per-basin, one row per date) ML regional-prediction
output, matching exactly what `reshape_ML_predict.py` used to produce by transposing the old
basin-by-date wide prediction matrices. Each regional eval script (RF/XGB/TCN/LSTM) now builds
its predictions in memory and calls `save_regional_predictions_long_format` directly, so
`reshape_ML_predict.py` is no longer a required post-processing step.

Output: one CSV per basin at
    {output_root}/{model_name}/{loc}/{dataset}/{regional_subdir}/varssim_{model_name}/varssim_{basin_id}.csv
with Date as the row index, the basin's own forcing/synthetic-precip columns copied in from its
varssim_{loc} input file, and one prediction column per scenario in PRED_DISCHARGE_COLS.

`dataset` ('camels' or 'caravan', matching ADAPTATION_TO_CAMELS.md) selects both the input
varssim directory (2_MERV/{dataset}/varssim_{loc}) and the output path, so camels and caravan
runs never read each other's forcing data or overwrite each other's long-format output. This
was a real gap until 2026-09-16: every caller passed neither `dataset` nor `varssim_dir`, so this
function silently fell back to the old flat 2_MERV/varssim_{loc} path -- which no longer exists
post-adaptation -- and wrote 0 files (a "not found" skip per basin) instead of erroring.
"""

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

# The 14 scenario keys reshape_ML_predict.py wrote as prediction columns: the original ("og")
# prediction plus the 13 synthetic-precipitation scenarios (no_precip + 3 peak lengths x 4 rates).
PRED_DISCHARGE_COLS = [
    "og", "no_precip",
    "peak1_rate0", "peak1_rate10", "peak1_rate30", "peak1_rate50",
    "peak3_rate0", "peak3_rate10", "peak3_rate30", "peak3_rate50",
    "peak5_rate0", "peak5_rate10", "peak5_rate30", "peak5_rate50",
]

# scenario_predictions keys as produced by each eval script's synthetic-scenario loop
# (syn_precip_cols, e.g. "syn_peak1_rate0") map to PRED_DISCHARGE_COLS entries by stripping the
# "syn_" prefix. "syn_p_pet" has no corresponding reshape_ML_predict.py column; kept anyway as a
# bonus "p_pet" column since it costs nothing and the eval scripts already compute it.
def _scenario_key_to_pred_col(scenario_key: str) -> str:
    return scenario_key[len("syn_"):] if scenario_key.startswith("syn_") else scenario_key

# Base columns copied verbatim from each basin's own varssim_{loc}/varssim_{basin}.csv, matching
# reshape_ML_predict.py's column selection exactly (so downstream analysis/figure code that reads
# 0_out/{model}/{loc}/regional/varssim_{model}/varssim_{basin}.csv needs no changes).
BASE_COLS = [
    "Obs flow", "Precip", "Temp", "PET",
    "syn_no_precip", "Temp_at_peak", "PET_at_peak", "Obs_flow_at_peak",
    "syn_peak1_rate0", "syn_peak1_rate10", "syn_peak1_rate30", "syn_peak1_rate50",
    "syn_peak3_rate0", "syn_peak3_rate10", "syn_peak3_rate30", "syn_peak3_rate50",
    "syn_peak5_rate0", "syn_peak5_rate10", "syn_peak5_rate30", "syn_peak5_rate50",
]


def save_regional_predictions_long_format(
    model_name: str,
    loc: str,
    scenario_predictions: Dict[str, Dict[str, pd.Series]],
    basins,
    dataset: str = "camels",
    varssim_dir: Optional[str] = None,
    output_root: str = "/data0/funato/hsp2/0_out",
    regional_subdir: str = "regional",
):
    """Write one per-basin long-format CSV per basin in `basins`.

    Parameters
    ----------
    scenario_predictions : dict
        Maps a scenario key (one of PRED_DISCHARGE_COLS, or a "syn_"-prefixed synthetic-scenario
        key like the eval scripts' own `syn_precip_cols`) to a dict of {basin_id: pd.Series}
        (DatetimeIndex -> predicted discharge). "og" should already be the cal+eva periods
        concatenated into one continuous series per basin.
    basins : iterable of basin id strings to write (normally the full get_basin_list(loc), i.e.
        all 614 US basins).
    dataset : 'camels' (default) or 'caravan' -- pass the evaluated run's own
        cfg.get('dataset', 'camels'), never a hardcoded default, so this matches whichever
        dataset the run was actually trained/evaluated on.
    regional_subdir : output subdirectory name under {loc}/{dataset}/ -- normally "regional",
        but callers with a non-standard regional variant (e.g. LSTM's
        eval_test_duration.py, which must not collide with the main production regional run)
        should pass their own output_base's subdirectory name here.
    """
    if varssim_dir is None:
        varssim_dir = f"/data0/funato/hsp2/2_MERV/{dataset}/varssim_{loc}"

    output_dir = os.path.join(output_root, model_name, loc, dataset, regional_subdir, f"varssim_{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Normalize scenario keys ("syn_peak1_rate0" -> "peak1_rate0") once, up front.
    normalized: Dict[str, Dict[str, pd.Series]] = {}
    for key, per_basin in scenario_predictions.items():
        normalized[_scenario_key_to_pred_col(key)] = per_basin

    n_written = 0
    for basin_id in basins:
        base_file = os.path.join(varssim_dir, f"varssim_{basin_id}.csv")
        if not os.path.exists(base_file):
            print(f"  Skipping {basin_id}: {base_file} not found")
            continue

        base_df = pd.read_csv(base_file, parse_dates=["Date"])
        base_df.set_index("Date", inplace=True)
        base_df.sort_index(inplace=True)

        present_base_cols = [c for c in BASE_COLS if c in base_df.columns]
        out_df = base_df[present_base_cols].copy()

        for pred_col, per_basin in normalized.items():
            series = per_basin.get(basin_id)
            if series is None:
                out_df[pred_col] = np.nan
                continue
            series = series[~series.index.duplicated(keep="last")]
            out_df[pred_col] = series.reindex(out_df.index)

        output_file = os.path.join(output_dir, f"varssim_{basin_id}.csv")
        out_df.to_csv(output_file, index=True, header=True, index_label="Date")
        n_written += 1

    print(f"[{model_name}] Wrote {n_written}/{len(list(basins))} long-format basin prediction files to {output_dir}")
