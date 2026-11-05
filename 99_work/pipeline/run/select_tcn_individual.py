"""3.7 re-check: separate hyperparameter search for individual TCN (trained from scratch per basin,
so tuned on its own -- only "adapted" inherits regional's architecture). Scored on the VAL period via
each basin's own best_epoch val_NSE (median over the HP_SEARCH_SUBSET_SIZE-basin subset), mirroring
run/select_tcn.py's two-stage design (filters, then window).

  filters: compare num_filters 16/32/64 at seq_length 90 -> tuning/TCN_individual/filters.json
  final  : compare the best-filter run at seq_length 90 vs 180 -> tuning/TCN_individual/selected.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402
from train.nets import individual_tag  # noqa: E402

OUT = E.OUT_ROOT / "tuning" / "TCN_individual"


class _A:
    """Minimal stand-in for argparse.Namespace, just enough for individual_tag()."""
    def __init__(self, num_filters, dropout, seq_length):
        self.hidden_size = None
        self.num_filters = num_filters
        self.dropout = dropout
        self.seq_length = seq_length


def score(f, w, seed=42):
    d0 = E.TCN_INDIVIDUAL["dropout"]
    tag = individual_tag("TCN", _A(f, None, w if w != 90 else None))
    rd = E.run_dir("TCN", "individual", seed, tag)
    be = pd.read_csv(rd / "best_epochs.csv")
    return {"num_filters": f, "seq_length": w, "run_dir": str(rd), "n_basins": len(be),
            "median_val_NSE": float(be["val_NSE"].median()), "median_best_epoch": float(be["best_epoch"].median())}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    w0 = E.TCN_INDIVIDUAL["seq_length"]
    if sys.argv[1] == "filters":
        rows = [score(f, w0) for f in E.TCN_INDIVIDUAL_FILTER_GRID]
        best = max(rows, key=lambda r: r["median_val_NSE"])
        pd.DataFrame(rows).to_csv(OUT / "filters_grid.csv", index=False)
        (OUT / "filters.json").write_text(json.dumps({"num_filters": best["num_filters"], "grid": rows}, indent=2))
        print(pd.DataFrame(rows))
    else:
        f = json.loads((OUT / "filters.json").read_text())["num_filters"]
        rows = [score(f, w) for w in E.TCN_INDIVIDUAL_SEQ_GRID]
        best = max(rows, key=lambda r: r["median_val_NSE"])
        sel = {"num_filters": f, "seq_length": best["seq_length"], "dropout": E.TCN_INDIVIDUAL["dropout"],
               "seq_comparison": rows, "old_median_val_NSE": None}
        (OUT / "selected.json").write_text(json.dumps(sel, indent=2))
        print(json.dumps(sel, indent=2))
