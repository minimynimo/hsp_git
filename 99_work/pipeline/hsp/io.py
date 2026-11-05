"""Shared data access: basin sets, per-basin forcing, static attributes, per-basin scalers."""
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd

from config import experiment as E


def load_basin_pool() -> List[str]:
    """The 637 data-available basins (training pool, 1.5)."""
    if E.BASIN_SETS_CSV.exists():
        bs = pd.read_csv(E.BASIN_SETS_CSV, dtype={"gauge_id": str})
        return sorted(bs.loc[bs["in_training_pool"], "gauge_id"])
    df = pd.read_csv(E.POOL_CSV, dtype={"gauge_id": str})
    return sorted({g.split("_")[-1] for g in df["gauge_id"]})


def load_eval_set() -> List[str]:
    """Common evaluation basin set (3.1). Raises if it has not been computed yet."""
    bs = pd.read_csv(E.BASIN_SETS_CSV, dtype={"gauge_id": str})
    if "in_eval_set" not in bs or bs["in_eval_set"].isna().all():
        raise RuntimeError(f"in_eval_set not computed yet in {E.BASIN_SETS_CSV} (run analysis/eval_set.py)")
    return sorted(bs.loc[bs["in_eval_set"].astype(bool), "gauge_id"])


def load_basin_sets() -> pd.DataFrame:
    return pd.read_csv(E.BASIN_SETS_CSV, dtype={"gauge_id": str}).set_index("gauge_id")


@lru_cache(maxsize=None)
def _read_forcing(gauge_id: str) -> pd.DataFrame:
    df = pd.read_csv(E.FORCING_DIR / f"varssim_{gauge_id}.csv", parse_dates=["Date", "Date_at_peak"])
    return df.set_index("Date")


def load_forcing(gauge_id: str) -> pd.DataFrame:
    """Full per-basin forcing table (real record + scenario columns), restricted to days a model
    may see: DATA_START onwards (inputs before 1987 are never used by any model, 1.4)."""
    return _read_forcing(gauge_id).loc[E.DATA_START:]


def scenario_frame(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Forcing table for one scenario with the standard column names (Precip/Temp/PET/Obs flow and
    a `doy_date` column giving the calendar date whose day-of-year the inputs belong to).
    'og' = real forcing. Rows end at the scenario window end."""
    out = pd.DataFrame(index=df.index)
    out["doy_date"] = df.index
    if scenario == "og":
        out[E.DYN_COLS] = df[E.DYN_COLS]
        out[E.Q_COL] = df[E.Q_COL]
        return out.loc[:E.DATA_END]
    end = E.SCENARIO_WINDOW[scenario][1]
    out["Precip"] = df[E.SCENARIO_PRECIP_COL[scenario]]
    if E.is_flood(scenario):
        out["Temp"] = df["Temp_at_peak"]
        out["PET"] = df["PET_at_peak"]
        out[E.Q_COL] = df["Obs_flow_at_peak"]
        out["doy_date"] = df["Date_at_peak"]
    else:
        out["Temp"] = df["Temp"]
        out["PET"] = df["PET"]
        out[E.Q_COL] = df[E.Q_COL]
    return out.loc[:end]


def load_static_attributes(basins: List[str]) -> pd.DataFrame:
    df = pd.read_csv(E.ATTRIBUTE_CSV, dtype={"gauge_id": str}).set_index("gauge_id")
    missing = sorted(set(basins) - set(df.index))
    if missing:
        raise KeyError(f"basins without static attributes: {missing}")
    return df.loc[basins, E.STATIC_ATTRS].astype(float)


def load_camels_signatures() -> pd.DataFrame:
    """CAMELS hydrological signatures + climate (baseflow_index, aridity, frac_snow, ...)."""
    frames = []
    for name in ["hydro", "clim"]:
        f = pd.read_csv(E.CAMELS_ATTR_DIR / f"camels_{name}.txt", sep=";", dtype={"gauge_id": str})
        frames.append(f.set_index("gauge_id"))
    return pd.concat(frames, axis=1)


def basin_scalers(gauge_id: str, df: pd.DataFrame = None) -> Dict:
    """Per-basin mean/std of the dynamic inputs and discharge over the TRAIN period."""
    if df is None:
        df = load_forcing(gauge_id)
    tr = df.loc[E.TRAIN[0]:E.TRAIN[1]]
    return {
        "dyn_mean": tr[E.DYN_COLS].mean().to_numpy(float),
        "dyn_std": tr[E.DYN_COLS].std().to_numpy(float),
        "q_mean": float(tr[E.Q_COL].mean()),
        "q_std": float(tr[E.Q_COL].std()),
    }


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if hasattr(obj, "__fspath__"):
        return str(obj)
    return obj
