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
# (/data0/funato/hsp2/2_MERV/{dataset}/varssim_{loc}/varssim_{gauge_id}.csv) via their own
# HydroDataset / EvalDataset classes, so it and its h5py/`papercode.datasets` dependency were
# removed here.

# Lookup CSV (per location, per dataset) that lists every basin's gauge_id, used to build the
# basin list. US now has two data sources with separately availability-filtered gauge lists:
#   * 'camels' -- forcing/streamflow built directly from raw CAMELS-US (full 1980-2014 record),
#     see /data0/funato/hsp2/2_MERV/1.create_dataset_plain_camels.py. This is the default.
#   * 'caravan' -- the older Caravan-derived extraction (1987-2006 record only).
# GB/JP only ever had a Caravan-derived extraction, so they keep a single un-nested path.
FILE_NUM_CSV = {
    'US': {
        'camels': '/data0/funato/hsp2/3_gis_data/US/0_data/river_basin/dataset_US/us_camels_file_num.csv',
        'caravan': '/data0/funato/hsp2/3_gis_data/US/0_data/river_basin/dataset_US/us_caravan_file_num.csv',
    },
    'GB': '/data0/funato/hsp2/3_gis_data/GB/0_data/river_basin/dataset_GB/gb_file_num.csv',
    'JP': '/data0/funato/hsp2/3_gis_data/JP/0_data/river_basin/dataset_JP/jp_file_num.csv',
}


def get_basin_list(loc: str = None, dataset: str = 'camels') -> List:
    """Build the list of gauge_id strings for this project's data.

    ADAPTED for this project: the original CAMELS `data/basin_list.txt` file does not apply
    here. Instead the basin list is read from the `gauge_id` column of the per-loc (and, for
    US, per-dataset) file-number CSV in `FILE_NUM_CSV`, normalized to a plain string with no
    `camels_`-style prefix (US's lookup CSVs store gauge_id as e.g. `camels_01022500`; GB's is
    already plain, e.g. `96001`).

    Parameters
    ----------
    loc : str
        One of 'US', 'GB', 'JP'.
    dataset : str
        One of 'camels', 'caravan'. Only affects `loc='US'` (GB/JP have a single source);
        default 'camels' matches this project's current default data source.

    Returns
    -------
    List
        List containing the gauge_id (str) of every basin for `loc`/`dataset`.
    """
    if loc not in FILE_NUM_CSV:
        raise ValueError(f"Unsupported location: {loc}. Supported locations are: {list(FILE_NUM_CSV)}")

    loc_entry = FILE_NUM_CSV[loc]
    if isinstance(loc_entry, dict):
        if dataset not in loc_entry:
            raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets for loc={loc} "
                              f"are: {list(loc_entry)}")
        csv_path = loc_entry[dataset]
    else:
        csv_path = loc_entry

    df = pd.read_csv(csv_path, dtype={'gauge_id': str})
    basins = [str(g).split('_')[-1] for g in df['gauge_id']]
    return basins
