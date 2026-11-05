"""2.3 acceptance check. For every basin in the pool, asserts:
  drought : P = 0 over DROUGHT; T/PET equal the real values; P real before it
  flood   : P = 0 in FLOOD_WINDOW except the storm days; storm values/durations correct; T/PET equal
            the transplanted source values (source peak - 10 d ...); no-storm column P = 0 throughout
  HBV     : every .mat file equals the ML forcing for the same basin and scenario (incl. real dates)
Prints passing / failing counts; exits non-zero if anything fails.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402
from hsp.io import load_basin_pool  # noqa: E402

MAT = {
    "og": E.HBV_MAT_ROOT / "MARRMoT_InputData_US_v212" / "varssim_{g}.mat",
    "no_precip": E.HBV_MAT_ROOT / "MARRMoT_InputData_US_v212_no_precip" / "varssim_{g}_no_precip.mat",
    "flood": E.HBV_MAT_ROOT / "MARRMoT_InputData_US_v212_up" / "varssim_{g}_up_{s}.mat",
}


def load_mat(path):
    m = scipy.io.loadmat(path)["data_MARRMoT_examples"][0, 0]
    dn = m["dates_as_datenum"].ravel()
    dates = pd.to_datetime([pd.Timestamp.fromordinal(int(x) - 366) for x in dn])
    return dates, {k: m[k].ravel() for k in ["precipitation", "temperature",
                                             "potential_evapotranspiration", "streamflow"]}


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-9)


def check(g):
    errs = []
    df = pd.read_csv(E.FORCING_DIR / f"varssim_{g}.csv", parse_dates=["Date", "Date_at_peak"]).set_index("Date")
    if not (df.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        errs.append("forcing dates not continuous")
    dr = df.loc[E.DROUGHT[0]:E.DROUGHT[1]]
    if not (dr["syn_no_precip"] == 0).all():
        errs.append("drought: P != 0")
    pre = df.loc[:E.DROUGHT[0] - pd.Timedelta(days=1)]
    if not same(pre["syn_no_precip"], pre["Precip"]):
        errs.append("drought: P before window not real")

    src = df.loc[E.SOURCE_SEARCH[0]:E.SOURCE_SEARCH[1], "Precip"]
    peak_date, peak = src.idxmax(), src.max()
    win = pd.date_range(*E.FLOOD_WINDOW)
    src_dates = pd.date_range(peak_date - pd.Timedelta(days=E.SOURCE_DAYS_BEFORE_PEAK), periods=len(win))
    w = df.loc[win]
    for col, real in [("Temp_at_peak", "Temp"), ("PET_at_peak", "PET")]:
        if not same(w[col], df.loc[src_dates, real]):
            errs.append(f"flood: {col} != transplanted source values")
    if not (w["Date_at_peak"].to_numpy() == src_dates.to_numpy()).all():
        errs.append("flood: Date_at_peak wrong")
    if (df.loc[STORM := E.STORM_START, "Date_at_peak"] != peak_date):
        errs.append("flood: storm start not aligned with source peak")
    if not (w["syn_nostorm"] == 0).all():
        errs.append("nostorm: P != 0 in window")
    for d in E.STORM_DURATIONS:
        storm = pd.date_range(STORM, periods=d)
        for u in E.STORM_UPLIFTS:
            p = w[f"syn_peak{d}_rate{u}"]
            if not np.allclose(p.loc[storm], peak * (1 + u / 100)):
                errs.append(f"peak{d}_rate{u}: storm value wrong")
            if not (p.drop(storm) == 0).all():
                errs.append(f"peak{d}_rate{u}: P != 0 outside storm")
            if not same(df.loc[:win[0] - pd.Timedelta(days=1), f"syn_peak{d}_rate{u}"],
                        df.loc[:win[0] - pd.Timedelta(days=1), "Precip"]):
                errs.append(f"peak{d}_rate{u}: P before window not real")

    # HBV .mat == ML forcing
    full = df.loc[E.DATA_START:E.DATA_END]
    fl = df.loc[E.DATA_START:E.FLOOD_WINDOW[1]]
    cases = [("og", MAT["og"], full, "Precip", "Temp", "PET", "Obs flow"),
             ("no_precip", MAT["no_precip"], full, "syn_no_precip", "Temp", "PET", "Obs flow")]
    cases += [(s, MAT["flood"], fl, E.SCENARIO_PRECIP_COL[s], "Temp_at_peak", "PET_at_peak", "Obs_flow_at_peak")
              for s in E.FLOOD_SCENARIOS + [E.NOSTORM]]
    for name, tpl, frame, pc, tc, ec, qc in cases:
        path = Path(str(tpl).format(g=g, s=name))
        if not path.exists():
            errs.append(f"HBV {name}: missing {path}")
            continue
        dates, m = load_mat(path)
        if not (dates == frame.index).all() or dates[0] != E.DATA_START:
            errs.append(f"HBV {name}: dates differ")
        for key, col in [("precipitation", pc), ("temperature", tc),
                         ("potential_evapotranspiration", ec), ("streamflow", qc)]:
            if not same(m[key], frame[col]):
                errs.append(f"HBV {name}: {key} != ML {col}")
    return g, errs


if __name__ == "__main__":
    res = Parallel(n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 16)(delayed(check)(g) for g in load_basin_pool())
    fails = [(g, e) for g, e in res if e]
    for g, e in fails:
        print(g, "; ".join(e))
    print(f"verify_forcing: {len(res) - len(fails)} passing, {len(fails)} failing")
    sys.exit(1 if fails else 0)
