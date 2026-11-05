"""3.1 common evaluation basin set: basins with valid test-period output from every run of the
matrix (each seed) and from every model-strategy seed-ensemble mean, plus observed test discharge
to score against. Written as `in_eval_set` into basins/basin_sets.csv; its size is logged."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402
from hsp import runio  # noqa: E402
from hsp.io import load_basin_pool, load_forcing  # noqa: E402
from hsp.metrics import kge_parts  # noqa: E402
from hsp.registry import combos, runs_of  # noqa: E402

pool = load_basin_pool()
obs = pd.DataFrame({b: load_forcing(b)[E.Q_COL] for b in pool}).loc[E.TEST[0]:E.TEST[1]]
ok = pd.Series(True, index=pool)
reasons = {}
for key in combos():
    runs = runs_of(key)
    sims = [runio.load_sim(r.dir, "og").loc[E.TEST[0]:E.TEST[1]] for r in runs]
    ens = sum(s.reindex(columns=pool) for s in sims) / len(sims)
    for label, s in [(f"{r.model}-{r.strategy}-seed{r.seed}", x) for r, x in zip(runs, sims)] + [(f"{key}-ensemble", ens)]:
        s = s.reindex(columns=pool)
        for b in pool:
            col = s[b]
            if col.isna().any() or not np.isfinite(kge_parts(obs[b], col)["KGE"]):
                if ok[b]:
                    reasons[b] = f"no valid test output from {label}"
                ok[b] = False

bs = pd.read_csv(E.BASIN_SETS_CSV, dtype={"gauge_id": str})
bs["in_eval_set"] = bs["gauge_id"].map(ok).fillna(False).astype(bool)
bs["eval_exclusion_reason"] = bs["gauge_id"].map(reasons).fillna("")
bs.to_csv(E.BASIN_SETS_CSV, index=False)
n = int(bs["in_eval_set"].sum())
msg = (f"evaluation set: {n} of {len(pool)} pool basins "
       f"({int((bs['in_eval_set'] & bs['in_531_subset']).sum())} also in the 531 subset); "
       f"excluded {len(pool) - n}: {dict(list(reasons.items())[:20])}")
print(msg)
E.LOG_DIR.mkdir(parents=True, exist_ok=True)
(E.LOG_DIR / "eval_set.log").write_text(msg + "\n")
E.MARK_DIR.mkdir(parents=True, exist_ok=True)
(E.MARK_DIR / "eval_set.done").write_text(f"{n}\n")
