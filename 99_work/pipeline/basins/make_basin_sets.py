"""1.5: basins/basin_sets.csv with in_training_pool, in_531_subset, in_eval_set (filled in later
by analysis/eval_set.py once every run exists, 3.1)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402

pool_df = pd.read_csv(E.POOL_CSV, dtype={"gauge_id": str})
pool = sorted({g.split("_")[-1] for g in pool_df["gauge_id"]})
if E.SMOKE:
    pool = pool[::len(pool) // E.SMOKE][:E.SMOKE]
newman = sorted({l.strip().zfill(8) for l in open(E.NEWMAN_531_LIST) if l.strip()})
missing_531 = sorted(set(newman) - set(pool))

no_forcing = [g for g in pool if not (E.FORCING_DIR / f"varssim_{g}.csv").exists()]
if no_forcing:
    raise SystemExit(f"pool basins without a forcing CSV: {no_forcing}")

out = pd.DataFrame({"gauge_id": pool})
out["in_training_pool"] = True
out["in_531_subset"] = out["gauge_id"].isin(newman)
out["in_eval_set"] = pd.NA
E.BASIN_DIR.mkdir(parents=True, exist_ok=True)
out.to_csv(E.BASIN_SETS_CSV, index=False)
print(f"training pool: {len(pool)} basins; 531-subset flagged: {int(out['in_531_subset'].sum())}; "
      f"{len(missing_531)} of the 531 are not in the pool: {missing_531}")
