"""
Shared resume logic for main_LSTM_indiv.py / main_LSTM_indiv_hp.py.

Both scripts loop over ~614 basins, and previously truncated their log file and
rewrote their results CSVs from an empty in-memory list on every launch -- so
killing and restarting the script silently discarded all progress and redid
every basin's LSTM training (100 epochs each in the production script) from
scratch. This lets a restart pick up only the basins that are not yet in the
results CSVs. (Same pattern as 99_work/11_RandomForest/rf_indiv_resume.py; no
shared predict-header helper here because, like TCN, each basin's predict-row
length depends on that basin's own data availability, so the header must be
derived from whichever basin is processed first, not a basin-independent date
range -- see the "not os.path.exists(predict_path)" guard at the call sites.)
"""

import os
from typing import Dict, List, Set, Tuple

import pandas as pd


def load_completed_basins(results_cal_path: str, results_eva_path: str) -> Tuple[Set[str], List[Dict], List[Dict]]:
    """Returns (completed_basin_ids, results_cal_records, results_eva_records).

    A basin only counts as completed if it has a row in BOTH the cal and eva
    results CSVs (a basin present in only one implies a run was killed mid-basin).
    """
    if not (os.path.exists(results_cal_path) and os.path.exists(results_eva_path)):
        return set(), [], []

    # dtype=str is required: gauge_id values like "01022500" have a leading zero, and
    # pd.read_csv's default type inference silently parses an all-digit column as int64,
    # dropping it (-> 1022500). That mismatches the loop's string gauge_id, so the "already
    # completed" skip check never matches and every basin gets reprocessed and duplicated on
    # every resume -- see DEBUG_README.md for the write-up.
    df_cal = pd.read_csv(results_cal_path, dtype={'gauge_id': str})
    df_eva = pd.read_csv(results_eva_path, dtype={'gauge_id': str})

    cal_ids = set(df_cal['gauge_id'].astype(str))
    eva_ids = set(df_eva['gauge_id'].astype(str))
    completed = cal_ids & eva_ids

    results_cal = df_cal[df_cal['gauge_id'].astype(str).isin(completed)].to_dict('records')
    results_eva = df_eva[df_eva['gauge_id'].astype(str).isin(completed)].to_dict('records')
    return completed, results_cal, results_eva
