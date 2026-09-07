"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

from pathlib import Path
from typing import List

import pandas as pd

# ADAPTED for this project: the reference `create_h5_files` (building .h5 caches from the
# CAMELS Maurer-forcing text-file `CamelsTXT` dataset) is not used -- main.py/eval.py build
# training/eval sequences directly from this project's per-basin CSVs
# (/data0/funato/hsp2/2_MERV/varssim_{loc}/varssim_{gauge_id}.csv) via their own HydroDataset /
# EvalDataset classes, so it and its h5py/`papercode.datasets` dependency were removed here.

# Lookup CSV (per location) that lists every basin's gauge_id, used to build the basin list.
FILE_NUM_CSV = {
    'US': '/data0/funato/hsp2/3_gis_data/US/0_data/river_basin/dataset_US/us_file_num.csv',
    'GB': '/data0/funato/hsp2/3_gis_data/GB/0_data/river_basin/dataset_GB/gb_file_num.csv',
    'JP': '/data0/funato/hsp2/3_gis_data/JP/0_data/river_basin/dataset_JP/jp_file_num.csv',
}


def get_basin_list(loc: str = None) -> List:
    """Build the list of gauge_id strings for this project's data.

    ADAPTED for this project: the original CAMELS `data/basin_list.txt` file does not apply
    here. Instead the basin list is read from the `gauge_id` column of `{loc.lower()}_file_num.csv`
    (see `FILE_NUM_CSV`), normalized to a plain string with no `camels_`-style prefix (US's
    lookup CSV stores gauge_id as e.g. `camels_01022500`; GB's is already plain, e.g. `96001`).

    Parameters
    ----------
    loc : str
        One of 'US', 'GB', 'JP'.

    Returns
    -------
    List
        List containing the gauge_id (str) of every basin for `loc`.
    """
    if loc not in FILE_NUM_CSV:
        raise ValueError(f"Unsupported location: {loc}. Supported locations are: {list(FILE_NUM_CSV)}")

    df = pd.read_csv(FILE_NUM_CSV[loc], dtype={'gauge_id': str})
    basins = [str(g).split('_')[-1] for g in df['gauge_id']]
    return basins
