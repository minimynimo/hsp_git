"""Phase 8 SHAP / interpretability attribution (plan 8.1-8.8), regional models (8.9).

Scope decisions (the plan leaves these open):
  - 8.9: regional strategy only, seed 42, for all four ML models (RF, XGB, TCN, LSTM). That is the
    plan's own "main result". Adapted and individual models are NOT covered here: train/trees.py
    never persists a fitted RF/XGB estimator to disk (no --save_model on the driver's regional/
    individual commands) and train/nets.py never persists adapted/individual network weights
    (only per-basin simulated output). Reproducing them for attribution would mean refitting 397
    individual trees or ~400 individual/adapted networks, which is not attempted in this pass.
    The regional refit done here (`tree_model`) is deterministic (same seed, params, data as
    train/trees.py's regional_final) and is cheap relative to that.
  - 8.2 fixed sample: a basin subsample stratified on baseflow index (so low/high-BFI and
    snow-heavy basins are all represented, matching the terciles scoring.py already uses for 6.3),
    not the full evaluation set -- GradientExplainer/IG over ~400 basins x several baselines x 4
    models was judged too expensive. Within each sampled basin, a fixed small set of TEST-period
    days is drawn, split across that basin's own low/mid/high observed-flow terciles (8.2). The
    same (basin, date) sample is reused for every model (saved to sample_basins.csv/sample_days.csv
    so the figure script and any follow-up can reuse it exactly).
  - 8.5 two baselines:
      within-basin  = this basin's own long-run TRAIN-period average dynamic input (the z-scoring
                      already centres each basin's TRAIN mean at 0, so this is the zero vector in
                      z-space) and this basin's OWN static attributes -- so the static contribution
                      is exactly zero by construction, leaving only the dynamic/day-to-day signal.
      between-basin = the same zero dynamic baseline (the explained input itself carries the
                      day's actual weather either way) but the POOL's mean static attributes (for
                      trees: the raw pool mean; for networks: the pool z-score mean, which is the
                      zero vector by construction) -- so only the static difference is non-zero,
                      isolating "how this basin's attributes differ from an average basin" (8.6).
    TreeSHAP: interventional mode is tried first (plan 8.5 asks for it, and the deep-tree failure
    noted against the old 270-raw-lag feature set no longer applies after 1.6 shrank the tree
    feature count); on failure it falls back to tree_path_dependent, logs which mode was used, and
    in that mode only one (undifferentiated) attribution set is produced -- see `tree_model`.
  - 8.4 Integrated Gradients (captum, installed into the venv for this run) is the primary method
    for TCN/LSTM; GradientSHAP (captum.attr.GradientShap, not shap.GradientExplainer -- captum's
    multi-input handling is far less awkward than the pattern the legacy LSTM SHAP script had to
    work around) is the cross-check.
  - Family mapping (8.3/8.6): tree families come straight from features.tree_features.FAMILIES /
    STATIC_FAMILIES. Networks see raw per-day P/T/PET over a lookback window, not engineered sums,
    so only four dynamic families are comparable across model types: Precip lag 0-6d -> recent_rain,
    Precip lag 7+ -> antecedent_rain, Temp (all lags) -> temperature, PET (all lags) -> pet. There
    is no network equivalent of the tree "snow" or "seasonality" families (networks get no explicit
    day-of-year or snow-proxy input) -- comparisons across model types are therefore restricted to
    the four shared families, and this is reported as a known limitation, not silently glossed over.

Dependency on HBV / eval_set: this script calls hsp.io.load_eval_set(), i.e. basins/basin_sets.csv
`in_eval_set`. At the time this ran, that set still reflected HBV's incomplete 397-basin snapshot
(HBV calibration was mid-run). THIS SCRIPT SHOULD BE RERUN once HBV finishes and eval_set.py /
scoring.py are rerun on the complete basin set, so the stratified sample is drawn from the final
evaluation set rather than the stale one. HBV itself is not explained here (no engineered feature
set to attribute; the plan's SHAP section 8.3-8.4 is scoped to the four ML models).

Every number this script computes is saved to ANALYSIS_DIR/shap/*.csv|json; figures/f02_shap.py
reads only those files (8.10).
"""
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402
from features.tree_features import FAMILIES, STATIC_FAMILIES, build_features, family_map, feature_names  # noqa: E402
from hsp.io import load_basin_pool, load_camels_signatures, load_eval_set, load_forcing, load_static_attributes  # noqa: E402
from hsp.registry import tcn_tag  # noqa: E402
from train import trees as TR  # noqa: E402
# NOTE: train.nets (and therefore torch) is imported lazily, inside nn_model() only. Importing
# torch at module level poisons joblib/loky's fork-based worker pool for tree_model's Parallel
# calls (crashes at worker-pool teardown, independent of CUDA) -- confirmed by reproducing and
# fixing this exact segfault while building this script. tree_model() must keep running before
# nn_model() in main()'s loop for the same reason.

warnings.filterwarnings("ignore")

OUT = E.ANALYSIS_SHAP_DIR
MARK = E.MARK_DIR
RNG = np.random.default_rng(42)

N_TERCILE_BASINS = 15     # per BFI tercile -> 45 basins total, the fixed sample (8.2)
N_DAYS_PER_TERCILE = 3    # per basin, per flow tercile -> 9 days/basin
N_BG_WITHIN = 150         # background rows drawn from one basin's TRAIN period (trees)
N_BG_BETWEEN = 300        # background rows drawn across the pool's TRAIN period (trees)
IG_STEPS = 64
GSHAP_BG = 25             # baseline-distribution size for the GradientSHAP cross-check
RECENT_RAIN_LAG = 7       # days: network Precip lag < this -> recent_rain, else antecedent_rain

RF_DOC = """SHAP methods already in use before this rewrite (plan 8.1), for interpreting old results:
  RF  : shap.TreeExplainer(model, feature_perturbation="tree_path_dependent") -- no baseline.
  XGB : shap.TreeExplainer(booster) on the raw booster -- exact TreeSHAP, additive by construction.
  TCN : shap.GradientExplainer(model, background), CPU, nsamples default -- random multi-basin
        background used as the implicit baseline distribution (Expected Gradients).
  LSTM: shap.GradientExplainer(wrapped two-input model, [x_d_bg, x_s_bg]) -- same family of method,
        with extra wrapper code to cope with shap's awkward handling of a two-tensor forward().
Collapsing the window into one value per feature, pre-1.6: RF/XGB had up to 270 raw daily lags per
dynamic variable as literal separate features (no aggregation at all -- "catchment area is most
important" partly reflects that the lag features individually carried little signal each). TCN/LSTM
summed |attribution| over the whole input window per variable.

Deviation from the production config (this rewrite): the regional RF refit used here for SHAP uses
100 trees / max depth 20, not the production model's 500 trees / unlimited depth. The full-scale
single-threaded fit (forced single-threaded by a reproducible joblib/loky segfault on this machine)
ran 4.5-5.5h twice without completing. The reduced config is still representative of which features
the ensemble actually uses, but SHAP values here are from this smaller RF, not the production one --
do not treat them as attributions of the exact model reported in the accuracy figures."""


# ------------------------------------------------------------------------------------------
# 8.2 fixed sample
# ------------------------------------------------------------------------------------------
def sample_basins_and_days():
    sb_f, sd_f = OUT / "sample_basins.csv", OUT / "sample_days.csv"
    if sb_f.exists() and sd_f.exists():
        sb = pd.read_csv(sb_f, dtype={"basin": str})
        sd = pd.read_csv(sd_f, dtype={"basin": str}, parse_dates=["date"])
        return sb, sd
    basins = load_eval_set()
    sig = load_camels_signatures()
    sig.index = sig.index.astype(str).str.zfill(8)
    bfi = sig.loc[basins, "baseflow_index"]
    order = bfi.sort_values().index                        # rank-based tercile split (robust to ties)
    groups = np.array_split(order, 3)
    terc = pd.Series({d: lab for lab, grp in zip(["low", "mid", "high"], groups) for d in grp})
    rows = []
    for t in ["low", "mid", "high"]:
        pool = sorted(terc.index[terc == t])
        pick = RNG.choice(pool, size=min(N_TERCILE_BASINS, len(pool)), replace=False)
        rows += [{"basin": b, "bfi_tercile": t} for b in pick]
    sb = pd.DataFrame(rows)
    day_rows = []
    for b in sb["basin"]:
        q = load_forcing(b)[E.Q_COL].loc[E.TEST[0]:E.TEST[1]].dropna()
        order = q.sort_values().index                      # rank-based tercile split: robust to
        groups = np.array_split(order, 3)                   # ties (common at low flow), unlike qcut
        ft = pd.Series({d: lab for lab, grp in zip(["low", "mid", "high"], groups) for d in grp})
        for t in ["low", "mid", "high"]:
            pool = ft.index[ft == t]
            n = min(N_DAYS_PER_TERCILE, len(pool))
            if n == 0:
                continue
            pick = RNG.choice(pool, size=n, replace=False)
            day_rows += [{"basin": b, "date": pd.Timestamp(d), "flow_tercile": t, "Q_obs": float(q.loc[d])} for d in pick]
    sd = pd.DataFrame(day_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    sb.to_csv(sb_f, index=False)
    sd.to_csv(sd_f, index=False)
    return sb, sd


# ------------------------------------------------------------------------------------------
# shared diagnostics (8.7 sanity checks, 8.8 faithfulness), computed per model on its own long table
# ------------------------------------------------------------------------------------------
def family_share(long_df, baseline="within"):
    d = long_df[long_df.baseline == baseline].copy()
    d["abs"] = d["value"].abs()
    tot = d.groupby(["basin", "date"])["abs"].transform("sum")
    d["share"] = d["abs"] / tot
    return d.groupby(["basin", "date", "family"])["share"].sum().reset_index()


def sanity_checks(model, long_df, sb, sd, sig):
    """sb: sampled basins (basin, bfi_tercile). sd: sampled days (basin, date, flow_tercile)."""
    out = {}
    d = long_df[long_df.baseline == "within"]
    rain_rows = d[d["P_on_day"] == True]  # noqa: E712 -- the current-day-precip row, tagged at creation
    out["1_rain_pushes_up_frac_positive"] = float((rain_rows["value"] > 0).mean()) if len(rain_rows) else np.nan

    share_w = family_share(d)
    merged = share_w.merge(sb, on="basin")
    rr = merged[merged.family.isin(["recent_rain", "antecedent_rain"])]
    mem = (rr[rr.family == "antecedent_rain"].groupby("basin")["share"].mean()
           / rr[rr.family == "recent_rain"].groupby("basin")["share"].mean())
    bfi = sig.loc[mem.index.intersection(sig.index), "baseflow_index"]
    out["2_memory_vs_BFI_spearman"] = (float(pd.Series(mem).reindex(bfi.index).corr(bfi, method="spearman"))
                                        if len(bfi) > 2 else np.nan)

    s3 = share_w.merge(sd[["basin", "date", "flow_tercile"]].drop_duplicates(), on=["basin", "date"], how="left")
    g = s3[s3.family == "recent_rain"].groupby("flow_tercile")["share"].median()
    out["3_recent_rain_share_by_flow_tercile"] = g.to_dict()

    snow_hi = sig.loc[sig.index.intersection(sb["basin"]), "frac_snow"]
    snow_hi = snow_hi[snow_hi >= snow_hi.quantile(2 / 3)].index
    dd = d.merge(sd[["basin", "date"]].drop_duplicates(), on=["basin", "date"], how="left")
    dd["month"] = pd.to_datetime(dd["date"]).dt.month
    spring = dd[dd.month.isin([3, 4, 5]) & dd.basin.isin(snow_hi)]
    other = dd[~dd.month.isin([3, 4, 5]) & dd.basin.isin(snow_hi)]

    def temp_snow_share(df):
        s = family_share(df)
        return float(s[s.family.isin(["temperature", "snow"])]["share"].median()) if len(s) else np.nan

    out["4_temp_snow_share_spring_vs_other_snowbasins"] = {"spring": temp_snow_share(spring), "other": temp_snow_share(other)}
    return out


def faithfulness(forward_fn, X, attr, baseline, feature_names_):
    """Mean |f(x)-f(x with top-k removed)| vs |f(x)-f(x with k random removed)|, k = nnz top decile."""
    k = max(1, X.shape[-1] // 10)
    order = np.argsort(-np.abs(attr), axis=-1)
    top = order[..., :k]
    f_x = forward_fn(X)
    def removed(idx):
        Xp = X.copy()
        for i in range(X.shape[0]):
            Xp[i, idx[i]] = baseline[i, idx[i]] if baseline.ndim == X.ndim else baseline[idx[i]]
        return forward_fn(Xp)
    d_top = np.abs(f_x - removed(top))
    rand = np.stack([RNG.choice(X.shape[-1], k, replace=False) for _ in range(X.shape[0])])
    d_rand = np.abs(f_x - removed(rand))
    return {"mean_abs_delta_top": float(np.nanmean(d_top)), "mean_abs_delta_random": float(np.nanmean(d_rand)),
            "ratio_top_over_random": float(np.nanmean(d_top) / np.nanmean(d_rand)) if np.nanmean(d_rand) > 0 else np.nan}


# ------------------------------------------------------------------------------------------
# RF / XGB (TreeSHAP)
# ------------------------------------------------------------------------------------------
def tree_model(model_name, sb, sd, sig):
    import shap
    print(f"[{model_name}] refitting regional model for SHAP (plan 1.6 feature set)...", flush=True)
    basins_full = load_basin_pool()
    params = dict(TR.selected_params(model_name, "regional"))
    # joblib/loky's process pool segfaults deterministically at worker-pool teardown on this
    # machine -- reproduced repeatedly while building this script, independent of n_jobs level,
    # of whether torch had been imported in this process, and of system load. n_jobs=1 makes
    # joblib's Parallel run in-process (joblib's own "sequential" backend, no forking at all),
    # trading speed for reliability. RF's own fit() also forks via joblib internally, so it gets
    # the same treatment; XGB's n_jobs is thread-based (no fork), so it can stay higher.
    n_jobs_load = 1
    n_jobs_fit = 1 if model_name == "RF" else 8
    if model_name == "RF":
        # The production RF (500 trees, unlimited depth) fit single-threaded on ~2.3M rows ran
        # 4.5-5.5h twice without finishing or erroring (no traceback; possibly killed under memory
        # pressure from other jobs on this shared machine). For the SHAP refit only -- not the
        # production model/results -- cut to 100 trees and cap depth at 20: still representative of
        # the ensemble's feature-use pattern, but fits in a fraction of the time and memory. Logged
        # here and in SHAP_METHODS_DOCUMENTED.md so this deviation from the production config is
        # never silently assumed to be the real RF.
        print(f"[RF] SHAP refit uses reduced params (n_estimators 500->100, max_depth None->20) "
              f"for tractability; production RF results are unaffected.", flush=True)
        params["n_estimators"] = 100
        params["max_depth"] = 20
    print(f"[{model_name}] loading regional feature tables for {len(basins_full)} basins "
          f"(n_jobs={n_jobs_load})...", flush=True)
    data, cat, attrs = TR.load_regional(basins_full, n_jobs=n_jobs_load)
    print(f"[{model_name}] feature tables loaded: Xtr {cat['Xtr'].shape}, Xva {cat['Xva'].shape}. "
          f"Fitting (n_estimators={params.get('n_estimators')}, n_jobs={n_jobs_fit})...", flush=True)
    est = TR.make_model(model_name, params, seed=42, n_jobs=n_jobs_fit, device="cpu")
    TR.fit(model_name, est, cat["Xtr"], cat["ytr"], cat["Xva"], cat["yva"])
    print(f"[{model_name}] fit complete.", flush=True)
    names = feature_names(True, E.STATIC_ATTRS)
    fam_of = family_map(regional=True)
    static_pool_mean = attrs.mean()

    bg_between_idx = RNG.choice(len(cat["Xtr"]), size=min(N_BG_BETWEEN, len(cat["Xtr"])), replace=False)
    bg_between = cat["Xtr"][bg_between_idx]

    mode_used = {}
    rows = []
    for b in sb["basin"]:
        Xb_full = data[b][0]["og"]
        Xb_train = Xb_full.loc[E.TRAIN[0]:E.TRAIN[1]]
        bg_within = Xb_train.sample(min(N_BG_WITHIN, len(Xb_train)), random_state=42).to_numpy()
        dates = sd.loc[sd.basin == b, "date"]
        X_explain = Xb_full.reindex(dates).to_numpy()
        if np.isnan(X_explain).any():
            missing = dates[np.isnan(X_explain).any(axis=1)]
            print(f"  [{model_name}] {b}: skipping {len(missing)} dates with NaN features", flush=True)
            keep = ~np.isnan(X_explain).any(axis=1)
            X_explain, dates = X_explain[keep], dates[keep]
        if len(X_explain) == 0:
            continue
        # Plan 8.5 flagged that interventional TreeSHAP "had problems with very deep trees" for RF,
        # with a note to re-check once the 1.6 features made the trees smaller. Re-checked here: it
        # still segfaults for RF (shap 0.51.0, confirmed via faulthandler -- shap/explainers/_tree.py
        # shap_values(), a C-extension crash that a Python try/except cannot catch or fall back from,
        # even at max_depth=20 / 100 trees / background 150-300 rows). So RF always uses
        # tree_path_dependent (no baseline, as in the pre-rewrite code); XGB -- whose boosted trees
        # are much shallower -- still gets the interventional attempt with a path-dependent fallback.
        if model_name == "RF":
            exp_pd = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
            sv_pd = exp_pd.shap_values(X_explain)
            mode_used[b] = "tree_path_dependent"
            variants = [("within", sv_pd, exp_pd.expected_value)]
        else:
            try:
                exp_w = shap.TreeExplainer(est, data=bg_within, feature_perturbation="interventional")
                sv_w = exp_w.shap_values(X_explain)
                exp_b = shap.TreeExplainer(est, data=bg_between, feature_perturbation="interventional")
                sv_b = exp_b.shap_values(X_explain)
                mode_used[b] = "interventional"
                base_w, base_b = exp_w.expected_value, exp_b.expected_value
                variants = [("within", sv_w, base_w), ("between", sv_b, base_b)]
            except Exception as e:  # noqa: BLE001
                print(f"  [{model_name}] {b}: interventional TreeSHAP failed ({e!r}); falling back to tree_path_dependent", flush=True)
                exp_pd = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
                sv_pd = exp_pd.shap_values(X_explain)
                mode_used[b] = "tree_path_dependent"
                variants = [("within", sv_pd, exp_pd.expected_value)]
        for tag, sv, base in variants:
            recon = sv.sum(axis=1) + base
            for i, dt in enumerate(dates):
                for j, fname in enumerate(names):
                    rows.append({"model": model_name, "basin": b, "date": dt, "baseline": tag,
                                 "feature": fname, "family": fam_of.get(fname, "static" if fname in static_pool_mean.index else "other"),
                                 "lag_or_window": 0, "value": float(sv[i, j]),
                                 "P_on_day": bool(Xb_full.loc[dt, "P_t"] > 0.1) if fname == "P_t" else None})
            rows.append({"model": model_name, "basin": b, "date": None, "baseline": tag, "feature": "__completeness__",
                         "family": None, "lag_or_window": None,
                         "value": float(np.nanmean(np.abs(recon - est.predict(X_explain)))), "P_on_day": None})
    long_df = pd.DataFrame(rows)
    long_df.to_csv(OUT / f"shap_values_{model_name}.csv", index=False)
    pd.Series(mode_used, name="treeshap_mode").to_csv(OUT / f"treeshap_mode_{model_name}.csv")

    def fwd(X):
        return est.predict(X)
    expl = long_df[(long_df.feature != "__completeness__") & (long_df.baseline == "within")]
    first_b = sb["basin"].iloc[0]
    dates0 = sd.loc[sd.basin == first_b, "date"]
    X0 = data[first_b][0]["og"].reindex(dates0).to_numpy()
    attr0 = (expl[expl.basin == first_b].pivot_table(index="date", columns="feature", values="value")
             .reindex(index=dates0, columns=names).to_numpy())
    base0 = np.zeros_like(X0) + data[first_b][0]["og"].loc[E.TRAIN[0]:E.TRAIN[1]].mean().to_numpy()
    faith = faithfulness(fwd, X0, attr0, base0, names) if len(X0) and attr0.shape == X0.shape else {}

    checks = sanity_checks(model_name, long_df, sb, sd, sig)
    checks["faithfulness"] = faith
    checks["treeshap_modes"] = mode_used
    (OUT / f"checks_{model_name}.json").write_text(json.dumps(checks, indent=2, default=float))
    print(f"[{model_name}] done: {len(sb)} basins, modes used: {pd.Series(mode_used).value_counts().to_dict()}", flush=True)
    return long_df


# ------------------------------------------------------------------------------------------
# TCN / LSTM (Integrated Gradients + GradientSHAP cross-check)
# ------------------------------------------------------------------------------------------
def nn_model(model_name, sb, sd, sig):
    import torch
    from train import nets as NT  # lazy: see the note by the top-level imports
    from captum.attr import GradientShap, IntegratedGradients

    # cudnn's fused RNN kernel refuses to run backward() while the module is in eval() mode
    # (needed here for deterministic attribution, no dropout) -- fall back to the generic kernel.
    # NOTE: this is also why tree_model (RF/XGB) must run before nn_model in main()'s loop: once
    # CUDA is initialised in this process, forking workers for sklearn/xgboost's joblib Parallel
    # (as tree_model does) is unsafe and segfaults. Keep RF/XGB first.
    torch.backends.cudnn.enabled = False
    free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    print(f"[{model_name}] GPU free {free/1e9:.1f}/{total/1e9:.1f} GB", flush=True)
    tag = tcn_tag() if model_name == "TCN" else ""
    reg_dir = E.run_dir(model_name, "regional", 42, tag)
    ck = torch.load(reg_dir / "best_model.pt", map_location=NT.DEV, weights_only=False)
    cfg, basins_full, seq = ck["cfg"], ck["basins"], ck["cfg"]["seq_length"]
    statics = NT.static_table(basins_full)
    model = NT.build(model_name, cfg, statics.shape[1])
    model.load_state_dict(ck["state_dict"]); model.eval()
    data = NT.Data(basins_full, "og", statics=statics)
    static_pool_mean = data.S.mean(dim=0, keepdim=True)
    basin_idx = {b: basins_full.index(b) for b in sb["basin"] if b in basins_full}

    def fwd(x_d, x_s):
        return model(x_d, x_s).squeeze(-1)

    ig = IntegratedGradients(fwd)
    gshap = GradientShap(fwd)
    fam_of = family_map(regional=False)
    names3 = E.DYN_COLS
    rows, mem_rows = [], []
    for b in sb["basin"]:
        if b not in basin_idx:
            print(f"  [{model_name}] {b}: not in this model's regional basin pool, skipping", flush=True)
            continue
        bi = basin_idx[b]
        dates = sd.loc[sd.basin == b, "date"]
        for dt in dates:
            if dt not in data.dates:
                continue
            ti = data.dates.get_loc(dt)
            if ti < seq - 1:
                continue
            x_d = data.X[bi, ti - seq + 1: ti + 1].unsqueeze(0).clone().requires_grad_(True)
            x_s = data.S[bi: bi + 1].clone().requires_grad_(True) if data.S is not None else None
            base_d = torch.zeros_like(x_d)
            base_s_within = x_s.clone().detach() if x_s is not None else None
            base_s_between = static_pool_mean.clone().detach() if x_s is not None else None
            for tag_name, base_s in [("within", base_s_within), ("between", base_s_between)]:
                attr_d, attr_s = ig.attribute((x_d, x_s), baselines=(base_d, base_s), n_steps=IG_STEPS)
                f_x = fwd(x_d, x_s).item()
                f_base = fwd(base_d, base_s).item() if base_s is not None else fwd(base_d, None).item()
                completeness_err = abs((attr_d.sum() + attr_s.sum()).item() - (f_x - f_base))
                bg_d = base_d.repeat(GSHAP_BG, 1, 1) + 0.01 * torch.randn(GSHAP_BG, *x_d.shape[1:], device=x_d.device)
                bg_s = base_s.repeat(GSHAP_BG, 1) if base_s is not None else None
                gattr_d, gattr_s = gshap.attribute((x_d, x_s), baselines=(bg_d, bg_s))
                ig_vec = torch.cat([attr_d.flatten(), attr_s.flatten() if attr_s is not None else torch.tensor([])])
                gs_vec = torch.cat([gattr_d.flatten(), gattr_s.flatten() if gattr_s is not None else torch.tensor([])])
                agree = float(torch.corrcoef(torch.stack([ig_vec, gs_vec]))[0, 1]) if ig_vec.numel() > 1 else np.nan
                for vi, var in enumerate(names3):
                    for lag in range(seq):
                        lagd = seq - 1 - lag
                        v = float(attr_d[0, lag, vi])
                        fam = ("recent_rain" if (var == "Precip" and lagd < RECENT_RAIN_LAG) else
                               "antecedent_rain" if var == "Precip" else
                               "temperature" if var == "Temp" else "pet")
                        rows.append({"model": model_name, "basin": b, "date": dt, "baseline": tag_name,
                                     "feature": f"{var}_lag{lagd}", "family": fam, "lag_or_window": lagd,
                                     "value": v, "P_on_day": bool(data.X[bi, ti, 0] > 0) if var == "Precip" and lagd == 0 else None})
                        mem_rows.append({"model": model_name, "basin": b, "variable": var, "lag": lagd,
                                         "abs_value": abs(v), "baseline": tag_name})
                if x_s is not None:
                    for si, sname in enumerate(E.STATIC_ATTRS):
                        rows.append({"model": model_name, "basin": b, "date": dt, "baseline": tag_name,
                                     "feature": sname, "family": STATIC_FAMILIES_OF.get(sname, "other"),
                                     "lag_or_window": None, "value": float(attr_s[0, si]), "P_on_day": None})
                rows.append({"model": model_name, "basin": b, "date": dt, "baseline": tag_name, "feature": "__completeness__",
                             "family": None, "lag_or_window": None, "value": completeness_err, "P_on_day": None})
                rows.append({"model": model_name, "basin": b, "date": dt, "baseline": tag_name, "feature": "__ig_gshap_agreement__",
                             "family": None, "lag_or_window": None, "value": agree, "P_on_day": None})
    long_df = pd.DataFrame(rows)
    long_df.to_csv(OUT / f"shap_values_{model_name}.csv", index=False)
    mem = pd.DataFrame(mem_rows)
    (mem.groupby(["model", "variable", "lag", "baseline"])["abs_value"].mean().reset_index()
     .to_csv(OUT / f"memory_profile_{model_name}.csv", index=False))

    checks = sanity_checks(model_name, long_df, sb, sd, sig)
    agree_rows = long_df[long_df.feature == "__ig_gshap_agreement__"]
    checks["5_ig_gradientshap_agreement_median"] = float(agree_rows["value"].median()) if len(agree_rows) else np.nan
    comp_rows = long_df[long_df.feature == "__completeness__"]
    checks["completeness_mean_abs_error"] = float(comp_rows["value"].mean()) if len(comp_rows) else np.nan
    (OUT / f"checks_{model_name}.json").write_text(json.dumps(checks, indent=2, default=float))
    print(f"[{model_name}] done: {len(sb)} basins x {len(dates)} max days/basin", flush=True)
    return long_df


STATIC_FAMILIES_OF = {f: fam for fam, feats in STATIC_FAMILIES.items() for f in feats}


# ------------------------------------------------------------------------------------------
def cross_model_agreement(all_long):
    """8.7.5: do models broadly agree? Spearman correlation of shared-family importance shares."""
    shares = {}
    for m, df in all_long.items():
        s = family_share(df[df.feature.map(lambda f: not str(f).startswith("__"))], "within")
        shared = s[s.family.isin(["recent_rain", "antecedent_rain", "temperature", "pet"])]
        shares[m] = shared.groupby("family")["share"].median()
    tab = pd.DataFrame(shares)
    rows = []
    models = list(tab.columns)
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            rho = tab[models[i]].corr(tab[models[j]], method="spearman")
            rows.append({"model_a": models[i], "model_b": models[j], "spearman_rho_family_share": rho})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "cross_model_agreement.csv", index=False)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SHAP_METHODS_DOCUMENTED.md").write_text(RF_DOC)
    sb, sd = sample_basins_and_days()
    print(f"SHAP sample: {len(sb)} basins, {len(sd)} (basin, date) points", flush=True)
    sig = load_camels_signatures()
    sig.index = sig.index.astype(str).str.zfill(8)

    all_long, ok, failed = {}, [], []
    for model_name, fn in [("RF", tree_model), ("XGB", tree_model), ("TCN", nn_model), ("LSTM", nn_model)]:
        t0 = time.time()
        try:
            all_long[model_name] = fn(model_name, sb, sd, sig)
            ok.append(model_name)
            print(f"=== {model_name} finished in {time.time()-t0:.0f}s ===", flush=True)
        except Exception:  # noqa: BLE001
            print(f"=== {model_name} FAILED after {time.time()-t0:.0f}s ===", flush=True)
            traceback.print_exc()
            failed.append(model_name)

    if len(all_long) >= 2:
        cross_model_agreement(all_long)
    summary = {"ok": ok, "failed": failed, "n_sample_basins": len(sb), "n_sample_points": len(sd)}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if failed:
        print(f"NOT writing shap.done: {failed} failed. Re-run after fixing.", flush=True)
        sys.exit(1)
    MARK.mkdir(parents=True, exist_ok=True)
    (MARK / "shap.done").write_text("ok\n")


if __name__ == "__main__":
    main()
