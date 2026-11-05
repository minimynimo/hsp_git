"""Phase 5 / 6.6 completeness check: every run of the matrix (hsp.registry) has finished and has the
real-forcing output plus every synthetic scenario, for every basin it should cover, over the right
window. Any missing file or basin stops with an error naming it (never converted to a rank)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402
from hsp import runio  # noqa: E402
from hsp.io import load_basin_pool  # noqa: E402
from hsp.registry import expected_runs  # noqa: E402

pool = set(load_basin_pool())
hbv_failed = set(pd.read_csv(E.HBV_FAILURES_CSV, dtype={"gauge_id": str})["gauge_id"]) if E.HBV_FAILURES_CSV.exists() else set()
errors, n = [], 0
for run in expected_runs():
    if not runio.is_done(run.dir):
        errors.append(f"run not finished: {run.dir}")
        continue
    want = pool - hbv_failed if run.model == "HBV" else pool
    for scen in runio.SIM_COLS:
        f = run.dir / "sim" / f"{scen}.parquet"
        if not f.exists():
            errors.append(f"missing {f}")
            continue
        df = pd.read_parquet(f)
        miss = want - set(df.columns)
        if miss:
            errors.append(f"{f}: {len(miss)} basins missing, e.g. {sorted(miss)[:3]}")
        a, z = (E.TRAIN[0], E.TEST[1]) if scen == "og" else E.SCENARIO_WINDOW[scen]
        if df.index[0] != a or df.index[-1] != z or len(df) != len(pd.date_range(a, z)):
            errors.append(f"{f}: window {df.index[0].date()}..{df.index[-1].date()} != {a.date()}..{z.date()}")
        if df[sorted(want & set(df.columns))].isna().any().any():
            errors.append(f"{f}: NaN in simulated discharge")
        n += 1
if errors:
    print("\n".join(errors))
    raise SystemExit(f"check_runs: {len(errors)} problems")
E.MARK_DIR.mkdir(parents=True, exist_ok=True)
(E.MARK_DIR / "check_runs.done").write_text(f"{n} files ok\n")
print(f"check_runs: all {len(expected_runs())} runs complete, {n} scenario files ok")
