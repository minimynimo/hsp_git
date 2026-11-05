"""3.5 re-check: hyperparameter search for the regional LSTM under the *current* validation
methodology (train/val split, warm-up, NSE loss), since the production LSTM kept its pre-rewrite
selection unchanged. Mirrors run/select_tcn.py's two-stage design.

  hidden : compare hidden_size 32/64/128 at the current dropout -> tuning/LSTM_regional/hidden.json
  final  : compare the best-hidden run's dropout (current vs. LSTM_DROPOUT_ALTERNATIVE)
           -> tuning/LSTM_regional/selected.json, reported against the production run's own
           calibration NSE/KGE (same basins, same split -- an apples-to-apples baseline, not the
           old pre-rewrite 40-basin HP-search numbers in SELECTED_HYPERPARAMETERS_camels.md).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402

OUT = E.OUT_ROOT / "tuning" / "LSTM_regional"


def score(h, d, seed=42):
    tag = f"h{h}_d{d}_w{E.LSTM_REGIONAL['seq_length']}"
    rd = E.run_dir("LSTM", "regional", seed, tag)
    log = pd.read_csv(rd / "train_log.csv")
    m = pd.read_csv(rd / "metrics.csv", dtype={"basin": str})
    med = m.groupby("period")[["KGE", "NSE"]].median()
    return {"hidden_size": h, "dropout": d, "run_dir": str(rd), "best_val_median_NSE": float(log["val_median_NSE"].max()),
            "best_epoch": int(log.loc[log["val_median_NSE"].idxmax(), "epoch"]), "epochs_run": len(log),
            "cal_median_NSE": float(med.loc["cal", "NSE"]), "cal_median_KGE": float(med.loc["cal", "KGE"]),
            "test_median_NSE": float(med.loc["test", "NSE"]), "test_median_KGE": float(med.loc["test", "KGE"])}


def production_baseline(seed=42):
    rd = E.run_dir("LSTM", "regional", seed, "")
    m = pd.read_csv(rd / "metrics.csv", dtype={"basin": str})
    med = m.groupby("period")[["KGE", "NSE"]].median()
    return {"hidden_size": E.LSTM_REGIONAL["hidden_size"], "dropout": E.LSTM_REGIONAL["dropout"],
            "cal_median_NSE": float(med.loc["cal", "NSE"]), "cal_median_KGE": float(med.loc["cal", "KGE"]),
            "test_median_NSE": float(med.loc["test", "NSE"]), "test_median_KGE": float(med.loc["test", "KGE"])}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    d0 = E.LSTM_REGIONAL["dropout"]
    if sys.argv[1] == "hidden":
        rows = [score(h, d0) for h in E.LSTM_HIDDEN_GRID]
        best = max(rows, key=lambda r: r["best_val_median_NSE"])
        pd.DataFrame(rows).to_csv(OUT / "hidden_grid.csv", index=False)
        (OUT / "hidden.json").write_text(json.dumps({"hidden_size": best["hidden_size"], "grid": rows}, indent=2))
        print(pd.DataFrame(rows))
    else:
        h = json.loads((OUT / "hidden.json").read_text())["hidden_size"]
        rows = [score(h, d0), score(h, E.LSTM_DROPOUT_ALTERNATIVE)]
        best = max(rows, key=lambda r: r["best_val_median_NSE"])
        base = production_baseline()
        sel = {"hidden_size": h, "dropout": best["dropout"], "tag": f"h{h}_d{best['dropout']}_w{E.LSTM_REGIONAL['seq_length']}",
               "dropout_comparison": rows, "production_baseline": base,
               "cal_median_NSE_new": best["cal_median_NSE"], "cal_median_NSE_old": base["cal_median_NSE"],
               "cal_median_KGE_new": best["cal_median_KGE"], "cal_median_KGE_old": base["cal_median_KGE"],
               "test_median_NSE_new": best["test_median_NSE"], "test_median_NSE_old": base["test_median_NSE"],
               "test_median_KGE_new": best["test_median_KGE"], "test_median_KGE_old": base["test_median_KGE"]}
        (OUT / "selected.json").write_text(json.dumps(sel, indent=2))
        print(json.dumps(sel, indent=2))
