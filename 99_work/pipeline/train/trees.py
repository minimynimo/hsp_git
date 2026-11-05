"""RF / XGB, individual and regional (plan 1.6, 3.2, 3.3, 4, 5).

  tune : grid search on the VAL period (fit on TRAIN only); selection = median over basins of VAL NSE.
         Writes OUT_ROOT/tuning/<MODEL>_<strategy>/{grid.csv, selected.json}.
  final: one run per seed with the selected hyperparameters; predicts the real-forcing period and
         every synthetic scenario; writes the standard run directory (hsp.runio).

Targets are per-basin z-scored discharge (TRAIN mean/std); objective = plain squared error.
XGB stops early on the VAL period (no refit on TRAIN+VAL).

RUN e.g.
  python train/trees.py --model XGB --strategy regional --mode tune --n_jobs 32
  python train/trees.py --model XGB --strategy regional --mode final --seed 42 --n_jobs 32
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402
from features.tree_features import build_features, feature_names  # noqa: E402
from hsp.io import (basin_scalers, load_basin_pool, load_forcing, load_static_attributes,  # noqa: E402
                    scenario_frame)
from hsp.metrics import nse  # noqa: E402
from hsp import runio  # noqa: E402

TUNE_DIR = E.OUT_ROOT / "tuning"


# ------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------
def basin_tables(basin, statics, scenarios=("og",)):
    """Features per scenario + z-scored target (og only) for one basin."""
    df = load_forcing(basin)
    sc = basin_scalers(basin, df)
    out = {}
    for s in scenarios:
        fr = scenario_frame(df, s)
        X = build_features(fr, statics)
        if s == "og":
            X = X.loc[E.TRAIN[0]:E.TEST[1]]
        else:
            X = X.loc[E.SCENARIO_WINDOW[s][0]:E.SCENARIO_WINDOW[s][1]]
        if X.isna().any().any():
            raise ValueError(f"{basin}/{s}: NaN features in the prediction window")
        out[s] = X
    y = ((df[E.Q_COL] - sc["q_mean"]) / sc["q_std"]).reindex(out["og"].index).astype("float32")
    return out, y, sc


def split_rows(X, y, period):
    sub = X.loc[period[0]:period[1]]
    yy = y.loc[sub.index]
    m = yy.notna().to_numpy()
    return sub[m], yy[m]


def load_regional(basins, n_jobs):
    attrs = load_static_attributes(basins)
    res = Parallel(n_jobs=n_jobs)(delayed(basin_tables)(b, attrs.loc[b]) for b in basins)
    data = {b: r for b, r in zip(basins, res)}
    parts = {k: [] for k in ["Xtr", "ytr", "Xva", "yva", "bva"]}
    for b in basins:
        X, y, _ = data[b][0]["og"], data[b][1], data[b][2]
        a, ay = split_rows(X, y, E.TRAIN)
        v, vy = split_rows(X, y, E.VAL)
        parts["Xtr"].append(a.to_numpy()); parts["ytr"].append(ay.to_numpy())
        parts["Xva"].append(v.to_numpy()); parts["yva"].append(vy.to_numpy())
        parts["bva"].append(np.full(len(vy), b))
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    return data, cat, attrs


def val_median_nse(y_true, y_pred, basin_of_row):
    df = pd.DataFrame({"b": basin_of_row, "o": y_true, "s": y_pred})
    return float(df.groupby("b").apply(lambda g: nse(g["o"], g["s"]), include_groups=False).median())


# ------------------------------------------------------------------------------------------
# models
# ------------------------------------------------------------------------------------------
def make_model(model, params, seed, n_jobs, device="cpu"):
    if model == "RF":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                     min_samples_leaf=params["min_samples_leaf"],
                                     max_features=params["max_features"],
                                     max_samples=params.get("max_samples", E.RF_MAX_SAMPLES),
                                     random_state=seed, n_jobs=n_jobs)
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=E.XGB_MAX_ROUNDS, learning_rate=params["learning_rate"],
                            max_depth=params["max_depth"], subsample=params["subsample"],
                            colsample_bytree=params["colsample_bytree"],
                            min_child_weight=params.get("min_child_weight", E.XGB_FIXED["min_child_weight"]),
                            objective=E.XGB_FIXED["objective"], tree_method=E.XGB_FIXED["tree_method"],
                            early_stopping_rounds=E.XGB_EARLY_STOPPING_ROUNDS, eval_metric="rmse",
                            random_state=seed, n_jobs=n_jobs, device=device)


def fit(model, est, Xtr, ytr, Xva, yva):
    if model == "XGB":
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return {"best_iteration": int(est.best_iteration), "best_val_rmse_z": float(est.best_score)}
    est.fit(Xtr, ytr)
    return {}


def predict(model, est, X):
    if model == "XGB":
        return est.predict(X, iteration_range=(0, est.best_iteration + 1))
    return est.predict(X)


def grid(model, strategy):
    g = E.RF_GRID if model == "RF" else E.XGB_GRID
    keys = list(g)
    combos = [dict(zip(keys, v)) for v in itertools.product(*[g[k] for k in keys])]
    if model == "RF":
        for c in combos:
            c["n_estimators"] = E.RF_N_ESTIMATORS_TUNE
    return combos


def tag_of(params):
    return "_".join(f"{k}{v}" for k, v in sorted(params.items()))


def selected_params(model, strategy):
    f = TUNE_DIR / f"{model}_{strategy}" / "selected.json"
    if not f.exists():
        raise FileNotFoundError(f"run --mode tune first: {f}")
    p = json.loads(f.read_text())["params"]
    if model == "RF":
        p["n_estimators"] = E.RF_N_ESTIMATORS_FINAL
    return p


# ------------------------------------------------------------------------------------------
# regional
# ------------------------------------------------------------------------------------------
def regional_tune(model, args):
    basins = load_basin_pool()[: args.n_basins]
    _, cat, _ = load_regional(basins, args.n_jobs)
    out = TUNE_DIR / f"{model}_regional"
    out.mkdir(parents=True, exist_ok=True)
    gfile = out / "grid.csv"
    done = pd.read_csv(gfile) if gfile.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    for params in grid(model, "regional"):
        tag = tag_of(params)
        if len(done) and tag in set(done["tag"]):
            continue
        t0 = time.time()
        est = make_model(model, params, 42, args.n_jobs, args.device)
        info = fit(model, est, cat["Xtr"], cat["ytr"], cat["Xva"], cat["yva"])
        score = val_median_nse(cat["yva"], predict(model, est, cat["Xva"]), cat["bva"])
        rows.append({"tag": tag, **{k: str(v) for k, v in params.items()}, "val_median_NSE": score,
                     **info, "fit_seconds": time.time() - t0})
        pd.DataFrame(rows).to_csv(gfile, index=False)
        print(f"[tune {model} regional] {tag}: val median NSE {score:.4f} {info} ({time.time()-t0:.0f}s)", flush=True)
        del est
    write_selection(model, "regional", pd.DataFrame(rows))


def write_selection(model, strategy, df):
    best = df.sort_values("val_median_NSE", ascending=False).iloc[0]
    params = {k: _parse(best[k]) for k in grid(model, strategy)[0] if k != "n_estimators"}
    sel = {"params": params, "val_median_NSE": float(best["val_median_NSE"]), "tag": best["tag"],
           "n_combos": len(df), "selection": "max median VAL NSE over basins"}
    (TUNE_DIR / f"{model}_{strategy}" / "selected.json").write_text(json.dumps(sel, indent=2))
    print(f"[tune {model} {strategy}] selected {sel}")


def _parse(v):
    """Grid values come back from CSV as strings / floats; restore int / float / None / str."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v == "None":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f.is_integer() else f


def regional_final(model, args):
    params = selected_params(model, "regional")
    seed = args.seed
    rd = E.run_dir(model, "regional", seed)
    if runio.is_done(rd) and not args.force:
        print(f"done already: {rd}")
        return
    basins = load_basin_pool()[: args.n_basins]
    data, cat, attrs = load_regional(basins, args.n_jobs)
    names = feature_names(True, E.STATIC_ATTRS)
    runio.init_run(rd, {"model": model, "strategy": "regional", "seed": seed, "params": params,
                        "features": names, "target": "per-basin z-scored discharge (TRAIN mean/std)",
                        "loss": "squared error", "n_basins": len(basins)})
    (rd / "features.txt").write_text("\n".join(names) + "\n")
    t0 = time.time()
    est = make_model(model, params, seed, args.n_jobs, args.device)
    info = fit(model, est, cat["Xtr"], cat["ytr"], cat["Xva"], cat["yva"])
    info["val_median_NSE_z"] = val_median_nse(cat["yva"], predict(model, est, cat["Xva"]), cat["bva"])
    info["fit_seconds"] = time.time() - t0
    del cat
    print(f"[{model} regional seed {seed}] fit {info}", flush=True)
    if args.save_model:
        import joblib
        joblib.dump(est, rd / "model.joblib", compress=3)
    sims = {s: {} for s in runio.SIM_COLS}
    for b in basins:
        tabs, _, sc = data[b]
        for s in runio.SIM_COLS:
            if s not in tabs:
                fr = scenario_frame(load_forcing(b), s)
                X = build_features(fr, attrs.loc[b]).loc[E.SCENARIO_WINDOW[s][0]:E.SCENARIO_WINDOW[s][1]]
            else:
                X = tabs[s]
            q = predict(model, est, X.to_numpy()) * sc["q_std"] + sc["q_mean"]
            sims[s][b] = pd.Series(runio.clip_nonneg(q), index=X.index)
    (rd / "fit_info.json").write_text(json.dumps(info, indent=2))
    summary = runio.finalize(rd, sims)
    gap = summary.loc["cal"] - summary.loc["test"]
    (rd / "run_summary.txt").write_text(
        f"{model} regional seed {seed}\nparams {params}\nfit {info}\nmedian metrics\n{summary}\n"
        f"calibration-test gap (median): KGE {gap['KGE']:.3f}, NSE {gap['NSE']:.3f}\n")
    print((rd / "run_summary.txt").read_text())


# ------------------------------------------------------------------------------------------
# individual
# ------------------------------------------------------------------------------------------
def individual_basin(model, basin, params_list, seed, scenarios):
    tabs, y, sc = basin_tables(basin, None, scenarios)
    Xtr, ytr = split_rows(tabs["og"], y, E.TRAIN)
    Xva, yva = split_rows(tabs["og"], y, E.VAL)
    results = []
    for params in params_list:
        est = make_model(model, params, seed, 1)
        info = fit(model, est, Xtr.to_numpy(), ytr.to_numpy(), Xva.to_numpy(), yva.to_numpy())
        info["val_NSE"] = nse(yva.to_numpy(), predict(model, est, Xva.to_numpy()))
        sims = None
        if len(params_list) == 1:
            sims = {s: pd.Series(runio.clip_nonneg(predict(model, est, X.to_numpy()) * sc["q_std"] + sc["q_mean"]),
                                 index=X.index) for s, X in tabs.items()}
        results.append((info, sims))
    return results


def individual_tune(model, args):
    basins = load_basin_pool()[: args.n_basins]
    combos = grid(model, "individual")
    out = TUNE_DIR / f"{model}_individual"
    out.mkdir(parents=True, exist_ok=True)
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(individual_basin)(model, b, combos, 42, ("og",)) for b in basins)
    rows = []
    for b, rr in zip(basins, res):
        for params, (info, _) in zip(combos, rr):
            rows.append({"basin": b, "tag": tag_of(params), **{k: str(v) for k, v in params.items()}, **info})
    per_basin = pd.DataFrame(rows)
    per_basin.to_csv(out / "grid_per_basin.csv", index=False)
    agg = per_basin.groupby("tag").agg(val_median_NSE=("val_NSE", "median")).reset_index()
    first = per_basin.drop_duplicates("tag").set_index("tag")
    agg = agg.join(first[[k for k in combos[0]]], on="tag")
    agg.to_csv(out / "grid.csv", index=False)
    write_selection(model, "individual", agg)


def individual_final(model, args):
    params = selected_params(model, "individual")
    seed = args.seed
    rd = E.run_dir(model, "individual", seed)
    if runio.is_done(rd) and not args.force:
        print(f"done already: {rd}")
        return
    basins = load_basin_pool()[: args.n_basins]
    runio.init_run(rd, {"model": model, "strategy": "individual", "seed": seed, "params": params,
                        "features": feature_names(False, []), "loss": "squared error",
                        "target": "per-basin z-scored discharge (TRAIN mean/std)", "n_basins": len(basins)})
    (rd / "features.txt").write_text("\n".join(feature_names(False, [])) + "\n")
    todo = [b for b in basins if not runio.part_done(rd, b)]

    def one(b):
        (info, sims), = individual_basin(model, b, [params], seed, tuple(runio.SIM_COLS))
        runio.save_part(rd, b, sims, info)
        return b

    Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(one)(b) for b in todo)
    summary = runio.finalize(rd)
    infos = pd.DataFrame([{"basin": f.stem, **json.loads(f.read_text())} for f in (rd / "parts").glob("*.json")])
    infos.to_csv(rd / "fit_info_per_basin.csv", index=False)
    gap = summary.loc["cal"] - summary.loc["test"]
    (rd / "run_summary.txt").write_text(
        f"{model} individual seed {seed}\nparams {params}\nmedian metrics\n{summary}\n"
        f"calibration-test gap (median): KGE {gap['KGE']:.3f}, NSE {gap['NSE']:.3f}\n")
    print((rd / "run_summary.txt").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["RF", "XGB"], required=True)
    ap.add_argument("--strategy", choices=["regional", "individual"], required=True)
    ap.add_argument("--mode", choices=["tune", "final"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_basins", type=int, default=None, help="smoke tests only")
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    fn = {("regional", "tune"): regional_tune, ("regional", "final"): regional_final,
          ("individual", "tune"): individual_tune, ("individual", "final"): individual_final}
    fn[(a.strategy, a.mode)](a.model, a)
