"""3.4 selection for the regional TCN, on the VAL period (median NSE over basins, best epoch).

  filters: compare f32/f64/f128 at window 180 -> tuning/TCN_regional/filters.json
  final  : compare the best-filter run at window 180 vs 365 -> tuning/TCN_regional/selected.json,
           with the calibration NSE of the selected run reported against the old regional TCN (0.478).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402

OUT = E.OUT_ROOT / "tuning" / "TCN_regional"
OLD_CAL_NSE, OLD_CAL_KGE = 0.478, 0.626


def score(f, w, seed=42):
    rd = E.run_dir("TCN", "regional", seed, f"f{f}_w{w}")
    log = pd.read_csv(rd / "train_log.csv")
    m = pd.read_csv(rd / "metrics.csv", dtype={"basin": str})
    med = m.groupby("period")[["KGE", "NSE"]].median()
    return {"num_filters": f, "seq_length": w, "run_dir": str(rd), "best_val_median_NSE": float(log["val_median_NSE"].max()),
            "best_epoch": int(log.loc[log["val_median_NSE"].idxmax(), "epoch"]), "epochs_run": len(log),
            "cal_median_NSE": float(med.loc["cal", "NSE"]), "cal_median_KGE": float(med.loc["cal", "KGE"])}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    w0 = E.TCN_REGIONAL["seq_length"]
    if sys.argv[1] == "filters":
        rows = [score(f, w0) for f in E.TCN_FILTER_GRID]
        best = max(rows, key=lambda r: r["best_val_median_NSE"])
        pd.DataFrame(rows).to_csv(OUT / "filters_grid.csv", index=False)
        (OUT / "filters.json").write_text(json.dumps({"num_filters": best["num_filters"], "grid": rows}, indent=2))
        print(pd.DataFrame(rows))
    else:
        f = json.loads((OUT / "filters.json").read_text())["num_filters"]
        rows = [score(f, w0), score(f, E.TCN_SEQ_ALTERNATIVE)]
        best = max(rows, key=lambda r: r["best_val_median_NSE"])
        sel = {"num_filters": f, "seq_length": best["seq_length"], "tag": f"f{f}_w{best['seq_length']}",
               "window_comparison": rows,
               "cal_median_NSE_new": best["cal_median_NSE"], "cal_median_NSE_old": OLD_CAL_NSE,
               "cal_median_KGE_new": best["cal_median_KGE"], "cal_median_KGE_old": OLD_CAL_KGE}
        (OUT / "selected.json").write_text(json.dumps(sel, indent=2))
        print(json.dumps(sel, indent=2))
