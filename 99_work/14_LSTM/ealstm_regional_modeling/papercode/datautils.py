"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

--------------------------------------------------------------------------------------------
ADAPTED for this project (hsp2 / 99_work/14_LSTM):

The original CAMELS Maurer-forcing / camels_attributes_v2.0 txt-file loaders
(`load_forcing`, `load_discharge`, `add_camels_attributes` reading `camels_*.txt`) do not
apply to this project's data and have been removed. Dynamic forcing + discharge are instead
loaded directly from the per-basin CSVs at
    /data0/funato/hsp2/2_MERV/varssim_{loc}/varssim_{gauge_id}.csv
in main_lstm_regional.py's `load_data_for_basin`, matching the convention already established
in `run_LSTM.py` (individual model) and `x_lstm_regional_modeling-master/main_lstm.py`
(previous regional model).

Static catchment attributes are loaded from the combined CSV at
    /data0/funato/hsp2/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}/<combined csv>
indexed by `gauge_id` (plain string, no `camels_` prefix), restricted to the `VALID_ATTR`
list below.

NOTE on VALID_ATTR: this is now the canonical 27-attribute CAMELS-US catchment-attribute set
listed in /data0/funato/hsp2/CATCHMENT_ATTRIBUTE.md (climate/topo/vegetation/soil/geol), using
the original camels_attributes_v2.0 names (e.g. 'p_seasonality' not 'p_seasonaility',
'frac_snow' not 'frac_snow_daily', 'frac_forest' not 'forest_frac', 'clay_frac' not
'caly_frac', 'carbonate_rocks_frac' not 'carb_rocks_frac' -- the .md file's names are typos of
these). For US, all 27 columns are present in `camels_original_attribute_combined.csv`, built
by joining the raw camels_clim/topo/vege/soil/geol.txt files (supplied at
/data0/funato/hsp2/camels_US/data/basin_attributes/) onto this project's File_num/gauge_id
mapping -- every one of the 667 US basins has all 27 values (no missing data). This replaces
the previous HydroATLAS-derived 18-attribute VALID_ATTR, which did not match
CATCHMENT_ATTRIBUTE.md at all.

GB (camelsgb_attribute_combined.csv) uses CAMELS-GB-native attributes and only happens to
share 10 of these 27 names outright (p_mean, pet_mean, aridity, p_seasonality, frac_snow,
high_prec_freq, high_prec_dur, low_prec_freq, low_prec_dur, elev_mean, soil_depth_pelletier);
the rest (slope_mean, area_gages2, frac_forest, lai_*, gvf_*, soil_depth_statsgo,
soil_porosity, soil_conductivity, max_water_content, sand_frac, silt_frac, clay_frac,
carbonate_rocks_frac, geol_permeability) have no equivalent column there. There is currently
no attribute CSV at all for JP (dataset_JP directory is empty), so JP static attributes /
regional training cannot run until that data is produced. `add_basin_attributes` below does
not guess a GB-specific attribute mapping -- it takes the intersection of VALID_ATTR and
whatever columns actually exist in the CSV for the requested `loc`, and prints a warning
listing what was dropped, so this discrepancy is visible rather than silently wrong.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import sqlite3
from pathlib import Path
from typing import List

import pandas as pd

# Static catchment attributes used for the EA-LSTM's static input gate. This is the 27-attribute
# CAMELS-US set from /data0/funato/hsp2/CATCHMENT_ATTRIBUTE.md (see NOTE above for GB/JP caveats
# and the .md file's naming typos vs. the real camels_attributes_v2.0 names used here).
VALID_ATTR = [
    'p_mean', 'pet_mean', 'aridity', 'p_seasonality', 'frac_snow', 'high_prec_freq',
    'high_prec_dur', 'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2',
    'frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff', 'soil_depth_pelletier',
    'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity', 'max_water_content',
    'sand_frac', 'silt_frac', 'clay_frac', 'carbonate_rocks_frac', 'geol_permeability'
]

# Combined static-attribute CSV per location.
ATTRIBUTE_CSV = {
    'US': '/data0/funato/hsp2/3_gis_data/US/0_data/river_basin/dataset_US/camels_original_attribute_combined.csv',
    'GB': '/data0/funato/hsp2/3_gis_data/GB/0_data/river_basin/dataset_GB/camelsgb_attribute_combined.csv',
    'JP': '/data0/funato/hsp2/3_gis_data/JP/0_data/river_basin/dataset_JP/camels_attribute_combined.csv',
}


def add_basin_attributes(loc: str, db_path: str = None) -> None:
    """Load static catchment attributes for `loc` from the combined CSV and store the columns
    in `VALID_ATTR` (intersected with whatever is actually present) in a sqlite3 table.

    Parameters
    ----------
    loc : str
        One of 'US', 'GB', 'JP'.
    db_path : str, optional
        Path to where the sqlite3 database file should be written.

    Raises
    ------
    RuntimeError
        If no combined attribute CSV exists for `loc`, or it lacks a 'File_num' column.
    """
    if loc not in ATTRIBUTE_CSV:
        raise ValueError(f"Unsupported location: {loc}. Supported locations are: {list(ATTRIBUTE_CSV)}")

    csv_path = Path(ATTRIBUTE_CSV[loc])
    if not csv_path.exists():
        raise RuntimeError(
            f"No combined static-attribute CSV found for loc={loc} at {csv_path}. "
            f"Regional training with static attributes (no_static=False) cannot run for this "
            f"location until it is created."
        )

    df = pd.read_csv(csv_path, encoding='utf-8', dtype={'gauge_id': str})
    if 'gauge_id' not in df.columns:
        raise RuntimeError(f"Expected a 'gauge_id' column in {csv_path}, found: {list(df.columns)}")

    df = df.set_index('gauge_id')

    available = [c for c in VALID_ATTR if c in df.columns]
    missing = [c for c in VALID_ATTR if c not in df.columns]
    if missing:
        print(f"WARNING: add_basin_attributes(loc={loc}): {len(missing)}/{len(VALID_ATTR)} "
              f"VALID_ATTR columns are missing from {csv_path.name} and will be skipped: {missing}. "
              f"Proceeding with the {len(available)} available columns: {available}")
    if not available:
        raise RuntimeError(f"None of VALID_ATTR are present in {csv_path}; cannot build static attributes.")

    df = df[available]

    if db_path is None:
        db_path = str(Path(__file__).absolute().parent.parent / 'data' / 'attributes.db')

    with sqlite3.connect(db_path) as conn:
        df.to_sql('basin_attributes', conn, if_exists='replace')

    print(f"Successfully stored {len(available)} static attributes for {len(df)} basins "
          f"(loc={loc}) in {db_path}.")


def load_attributes(db_path: str, basins: List[str]) -> pd.DataFrame:
    """Load static attributes from the sqlite3 database written by `add_basin_attributes`.

    Parameters
    ----------
    db_path : str
        Path to sqlite3 database file.
    basins : List[str]
        List of gauge_id strings to keep.

    Returns
    -------
    pd.DataFrame
        Attributes, indexed by gauge_id (str).
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col='gauge_id')
    df.index = df.index.astype(str)

    drop_basins = [b for b in df.index if b not in basins]
    df = df.drop(drop_basins, axis=0)

    return df
