"""Tree-model feature table (plan 1.6), shared by individual and regional RF and XGB.

Built from one basin's scenario frame (hsp.io.scenario_frame): Precip, Temp, PET and `doy_date`
(the calendar date the forcing belongs to; in the flood scenario that is the transplanted source
date, so seasonality matches the transplanted T/PET). Observed discharge is never an input.
All rolling windows end on day t (inclusive) and need their full length of history, so with inputs
starting on DATA_START every feature is defined from 1988-01-01 on.
"""
from typing import List

import numpy as np
import pandas as pd

P_LAGS = range(1, 8)
P_PET_SUM_WINDOWS = [3, 7, 14, 30, 90, 180, 365]
T_MEAN_WINDOWS = [7, 30, 90]
WB_WINDOWS = [30, 90, 365]
SNOW_P_WINDOWS = [30, 90, 180]
MELT_WINDOWS = [7, 30]
MAX_WINDOW = max(P_PET_SUM_WINDOWS + WB_WINDOWS + SNOW_P_WINDOWS)

# feature families (used by the SHAP grouping, 8.3)
FAMILIES = {
    "recent_rain": ["P_t"] + [f"P_lag{k}" for k in P_LAGS] + [f"P_sum{w}" for w in [3, 7, 14]],
    "antecedent_rain": [f"P_sum{w}" for w in [30, 90, 180, 365]] + [f"WB_sum{w}" for w in WB_WINDOWS],
    "pet": ["PET_t"] + [f"PET_sum{w}" for w in P_PET_SUM_WINDOWS],
    "temperature": ["T_t"] + [f"T_mean{w}" for w in T_MEAN_WINDOWS],
    "snow": [f"Psnow_sum{w}" for w in SNOW_P_WINDOWS] + [f"melt_sum{w}" for w in MELT_WINDOWS],
    "seasonality": ["doy_sin", "doy_cos"],
}

# static-attribute classes (used by the SHAP grouping, 8.3/8.6), grouped by physical meaning
STATIC_FAMILIES = {
    "climate": ["p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow",
                "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur"],
    "topography": ["elev_mean", "slope_mean", "area_gages2"],
    "vegetation": ["frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff"],
    "soil": ["soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity", "soil_conductivity",
             "max_water_content", "sand_frac", "silt_frac", "clay_frac", "carbonate_rocks_frac",
             "geol_permeability"],
}


def family_map(regional: bool) -> dict:
    """feature name -> family, for the dynamic families (+ static classes if `regional`)."""
    m = {f: fam for fam, feats in FAMILIES.items() for f in feats}
    if regional:
        m.update({f: fam for fam, feats in STATIC_FAMILIES.items() for f in feats})
    return m


def dynamic_feature_names() -> List[str]:
    names = ["P_t", "T_t", "PET_t"] + [f"P_lag{k}" for k in P_LAGS]
    names += [f"P_sum{w}" for w in P_PET_SUM_WINDOWS] + [f"PET_sum{w}" for w in P_PET_SUM_WINDOWS]
    names += [f"T_mean{w}" for w in T_MEAN_WINDOWS] + [f"WB_sum{w}" for w in WB_WINDOWS]
    names += [f"Psnow_sum{w}" for w in SNOW_P_WINDOWS] + [f"melt_sum{w}" for w in MELT_WINDOWS]
    names += ["doy_sin", "doy_cos"]
    return names


def build_dynamic_features(frame: pd.DataFrame) -> pd.DataFrame:
    p, t, pet = frame["Precip"], frame["Temp"], frame["PET"]
    f = {"P_t": p, "T_t": t, "PET_t": pet}
    for k in P_LAGS:
        f[f"P_lag{k}"] = p.shift(k)
    for w in P_PET_SUM_WINDOWS:
        f[f"P_sum{w}"] = p.rolling(w, min_periods=w).sum()
        f[f"PET_sum{w}"] = pet.rolling(w, min_periods=w).sum()
    for w in T_MEAN_WINDOWS:
        f[f"T_mean{w}"] = t.rolling(w, min_periods=w).mean()
    wb = p - pet
    for w in WB_WINDOWS:
        f[f"WB_sum{w}"] = wb.rolling(w, min_periods=w).sum()
    p_snow = p.where(t < 0, 0.0).where(p.notna() & t.notna())
    for w in SNOW_P_WINDOWS:
        f[f"Psnow_sum{w}"] = p_snow.rolling(w, min_periods=w).sum()
    melt = t.clip(lower=0)
    for w in MELT_WINDOWS:
        f[f"melt_sum{w}"] = melt.rolling(w, min_periods=w).sum()
    doy = pd.DatetimeIndex(frame["doy_date"]).dayofyear.to_numpy()
    f["doy_sin"] = pd.Series(np.sin(2 * np.pi * doy / 365.25), index=frame.index)
    f["doy_cos"] = pd.Series(np.cos(2 * np.pi * doy / 365.25), index=frame.index)
    out = pd.DataFrame(f, index=frame.index)[dynamic_feature_names()]
    return out.astype("float32")


def build_features(frame: pd.DataFrame, statics: pd.Series = None) -> pd.DataFrame:
    """Dynamic features (+ static attributes for regional models, constant over time)."""
    X = build_dynamic_features(frame)
    if statics is not None:
        for name, val in statics.items():
            X[name] = np.float32(val)
    return X


def feature_names(regional: bool, static_attrs: List[str]) -> List[str]:
    return dynamic_feature_names() + (list(static_attrs) if regional else [])
