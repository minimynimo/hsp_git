"""HBV MATLAB output -> standard run directory (same format as every ML run) + basins/hbv_failures.csv.

Reads <run_dir>/matlab/{Q,calib_log,failed}/ written by MATLAB/MARRMoT_v212_HBV/parallel.sh.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402
from hsp import runio  # noqa: E402
from hsp.io import load_basin_pool, load_forcing  # noqa: E402

rd = E.run_dir("HBV", "individual", E.HBV_OPTIMISER_SEED)
mat = rd / "matlab"
if not (mat / "_MATLAB_FINISHED").exists():
    raise SystemExit(f"HBV MATLAB runs not finished ({mat / '_MATLAB_FINISHED'} missing)")

pool = load_basin_pool()
sims = {c: {} for c in runio.SIM_COLS}
fails = []
for b in pool:
    qf = mat / "Q" / f"varssim_{b}_Q.csv"
    ff = mat / "failed" / f"varssim_{b}_failed.txt"
    if not qf.exists():
        reason = ff.read_text().strip() if ff.exists() else "no output: MATLAB process died before finishing the basin"
        fails.append({"gauge_id": b, "reason": reason})
        continue
    q = pd.read_csv(qf, parse_dates=["Date"]).set_index("Date")
    sims["og"][b] = q["og"].loc[E.TRAIN[0]:E.TEST[1]]
    for s in E.SCENARIOS:
        a, z = E.SCENARIO_WINDOW[s]
        sims[s][b] = q[s].loc[a:z]
        if sims[s][b].isna().any() or len(sims[s][b]) != len(pd.date_range(a, z)):
            raise ValueError(f"{qf}: scenario {s} incomplete")

logs = [pd.read_csv(f, dtype={"basin": str}) for f in sorted((mat / "calib_log").glob("*.csv"))]
calib = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
runio.init_run(rd, {"model": "HBV", "strategy": "individual", "seed": E.HBV_OPTIMISER_SEED,
                    "structure": "MARRMoT m_37_hbv_15p_5s", "objective": "KGE (of_KGE) on TRAIN",
                    "optimiser": {"name": "CMA-ES (my_cmaes)", "PopSize": int(calib["PopSize"].iloc[0]) if len(calib) else None,
                                  "MaxIter": int(calib["MaxIter"].iloc[0]) if len(calib) else None,
                                  "TolX": 1e-3, "TolFun": 1e-2, "TolHistFun": 1e-2, "seed": E.HBV_OPTIMISER_SEED,
                                  "par_ini": "centre of parameter ranges", "S0": "zeros at DATA_START (1987 warm-up)"},
                    "n_evals_median": float(calib["n_evals"].median()) if len(calib) else None,
                    "n_basins_ok": len(sims["og"]), "n_basins_failed": len(fails)})
calib.to_csv(rd / "calib_log.csv", index=False)
summary = runio.finalize(rd, sims)
E.BASIN_DIR.mkdir(exist_ok=True)
pd.DataFrame(fails, columns=["gauge_id", "reason"]).to_csv(E.HBV_FAILURES_CSV, index=False)
print(f"HBV: {len(sims['og'])} basins ok, {len(fails)} failed -> {E.HBV_FAILURES_CSV}\n{summary}")
