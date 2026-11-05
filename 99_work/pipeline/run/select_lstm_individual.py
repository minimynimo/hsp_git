"""3.7 re-check: separate hyperparameter search for individual LSTM (trained from scratch per basin,
so tuned on its own -- only "adapted" inherits regional's architecture). Scored on the VAL period via
each basin's own best_epoch val_NSE (median over the HP_SEARCH_SUBSET_SIZE-basin subset), mirroring
run/select_lstm.py's two-stage design (hidden_size, then dropout).

  hidden : compare hidden_size 16/32/64 at the current dropout -> tuning/LSTM_individual/hidden.json
  final  : compare the best-hidden run's dropout (current vs. LSTM_INDIVIDUAL_DROPOUT_ALTERNATIVE)
           -> tuning/LSTM_individual/selected.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402
from train.nets import individual_tag  # noqa: E402

OUT = E.OUT_ROOT / "tuning" / "LSTM_individual"


class _A:
    def __init__(self, hidden_size, dropout):
        self.hidden_size = hidden_size
        self.num_filters = 64
        self.dropout = dropout
        self.seq_length = None


def score(h, d, seed=42):
    tag = individual_tag("LSTM", _A(h, d if d != E.LSTM_INDIVIDUAL["dropout"] else None))
    rd = E.run_dir("LSTM", "individual", seed, tag)
    be = pd.read_csv(rd / "best_epochs.csv")
    return {"hidden_size": h, "dropout": d, "run_dir": str(rd), "n_basins": len(be),
            "median_val_NSE": float(be["val_NSE"].median()), "median_best_epoch": float(be["best_epoch"].median())}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    d0 = E.LSTM_INDIVIDUAL["dropout"]
    if sys.argv[1] == "hidden":
        rows = [score(h, d0) for h in E.LSTM_INDIVIDUAL_HIDDEN_GRID]
        best = max(rows, key=lambda r: r["median_val_NSE"])
        pd.DataFrame(rows).to_csv(OUT / "hidden_grid.csv", index=False)
        (OUT / "hidden.json").write_text(json.dumps({"hidden_size": best["hidden_size"], "grid": rows}, indent=2))
        print(pd.DataFrame(rows))
    else:
        h = json.loads((OUT / "hidden.json").read_text())["hidden_size"]
        rows = [score(h, d0), score(h, E.LSTM_INDIVIDUAL_DROPOUT_ALTERNATIVE)]
        best = max(rows, key=lambda r: r["median_val_NSE"])
        sel = {"hidden_size": h, "dropout": best["dropout"], "dropout_comparison": rows}
        (OUT / "selected.json").write_text(json.dumps(sel, indent=2))
        print(json.dumps(sel, indent=2))
