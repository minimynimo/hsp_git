"""Every score of the study (plan 2.5-2.7, 3.3, 6.1-6.6), computed here and nowhere else. Figure
scripts only read the files this module writes to ANALYSIS_DIR.

Basins: the common evaluation set (3.1) only. Period: TEST unless stated.
Seeds: headline accuracy uses the seed-ensemble-mean hydrograph; strategy comparisons and the
plausibility dimensions use per-seed scores (median over seeds), so models with fewer seeds
(individual) are not penalised by missing ensemble smoothing (1.2).
Missing output for any run, scenario or basin raises an error naming the file (6.6).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402
from hsp import runio  # noqa: E402
from hsp.io import load_basin_sets, load_camels_signatures, load_eval_set, load_forcing  # noqa: E402
from hsp.metrics import kge_parts  # noqa: E402
from hsp.registry import combos, runs_of  # noqa: E402

OUT_ACC = E.ANALYSIS_ACCURACY_DIR
OUT_MASS = E.ANALYSIS_MASS_DIR
OUT_DROUGHT = E.ANALYSIS_DROUGHT_DIR
OUT_FLOOD = E.ANALYSIS_FLOOD_DIR
OUT_SCORE = E.ANALYSIS_SCORING_DIR
N_BOOT = 1000
RNG = np.random.default_rng(42)
Q_FLOOR = 1e-3          # mm/d, floor for log(Q) in recession fits
K_CLIP = (0.1, 1e4)     # days, recession constants outside this are clipped
DRY_P = 0.1             # mm/d, a day with P <= DRY_P is dry
REC_DAYS = (2, 30)      # days since rain over which recession constants are fitted (obs and synthetic)
MIN_DRY_SPELL = 5
OBS_COVERAGE = 0.9      # observed water balance needs this fraction of test days observed


# ------------------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------------------
def require(df, basins, what):
    miss = sorted(set(basins) - set(df.columns))
    if miss:
        raise RuntimeError(f"{what}: {len(miss)} evaluation basins missing, e.g. {miss[:5]}")
    return df[basins]


def load_combo(key, scenario, basins):
    """{seed: DataFrame dates x basins} for one model-strategy and scenario."""
    out = {}
    for r in runs_of(key):
        f = r.dir / "sim" / f"{scenario}.parquet"
        out[r.seed] = require(runio.load_sim(r.dir, scenario), basins, str(f))
    return out


def ensemble(per_seed):
    return sum(per_seed.values()) / len(per_seed)


# ------------------------------------------------------------------------------------------
# 6.1 accuracy
# ------------------------------------------------------------------------------------------
def basin_metrics(sim, obs, period):
    a, z = E.PERIODS[period]
    rows = {b: kge_parts(obs[b].loc[a:z], sim[b].reindex(obs.loc[a:z].index)) for b in sim.columns}
    return pd.DataFrame(rows).T


def iqr(x):
    return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))


def accuracy(basins, obs, subset_531):
    per_basin, summary, seedtab = [], [], []
    for key in combos():
        sims = load_combo(key, "og", basins)
        ens = ensemble(sims)
        for period in E.PERIODS:
            m_ens = basin_metrics(ens, obs, period)
            seed_m = {s: basin_metrics(x, obs, period) for s, x in sims.items()}
            per_seed_kge = pd.DataFrame({s: v["KGE"] for s, v in seed_m.items()})
            per_seed_nse = pd.DataFrame({s: v["NSE"] for s, v in seed_m.items()})
            for b in basins:
                per_basin.append({"combo": key, "period": period, "basin": b, "KGE_ens": m_ens.loc[b, "KGE"],
                                  "NSE_ens": m_ens.loc[b, "NSE"], "r_ens": m_ens.loc[b, "r"],
                                  "alpha_ens": m_ens.loc[b, "alpha"], "beta_ens": m_ens.loc[b, "beta"],
                                  "KGE_perseed": per_seed_kge.loc[b].median(), "NSE_perseed": per_seed_nse.loc[b].median()})
            for s in sims:
                seedtab.append({"combo": key, "period": period, "seed": s,
                                "median_KGE": seed_m[s]["KGE"].median(), "median_NSE": seed_m[s]["NSE"].median()})
    pb = pd.DataFrame(per_basin)
    pb.to_csv(OUT_ACC / "accuracy_per_basin.csv", index=False)
    st = pd.DataFrame(seedtab)
    st.to_csv(OUT_ACC / "accuracy_per_seed.csv", index=False)

    def summarise(df, tag):
        rows = []
        for (key, period), g in df.groupby(["combo", "period"], sort=False):
            ss = st[(st.combo == key) & (st.period == period)]
            rows.append({"combo": key, "period": period, "n_basins": len(g), "n_seeds": len(ss),
                         "KGE_median_ens": g["KGE_ens"].median(), "KGE_IQR_ens": iqr(g["KGE_ens"]),
                         "NSE_median_ens": g["NSE_ens"].median(), "NSE_IQR_ens": iqr(g["NSE_ens"]),
                         "KGE_median_perseed": g["KGE_perseed"].median(), "NSE_median_perseed": g["NSE_perseed"].median(),
                         "seed_median_KGE_min": ss["median_KGE"].min(), "seed_median_KGE_max": ss["median_KGE"].max(),
                         "seed_median_KGE_std": ss["median_KGE"].std(), "seed_median_NSE_min": ss["median_NSE"].min(),
                         "seed_median_NSE_max": ss["median_NSE"].max()})
        out = pd.DataFrame(rows)
        piv = out.pivot(index="combo", columns="period", values="KGE_median_ens")
        out["cal_test_gap_KGE"] = out["combo"].map(piv["cal"] - piv["test"])
        pivn = out.pivot(index="combo", columns="period", values="NSE_median_ens")
        out["cal_test_gap_NSE"] = out["combo"].map(pivn["cal"] - pivn["test"])
        out.to_csv(OUT_ACC / f"accuracy_summary{tag}.csv", index=False)
        return out

    summ = summarise(pb, "")
    summarise(pb[pb["basin"].isin(subset_531)], "_531subset")
    wilcoxon(pb[pb.period == "test"])
    drop = rf_drop_rule(summ)
    return pb, summ, drop


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def wilcoxon(test):
    for col, tag in [("KGE_perseed", "perseed"), ("KGE_ens", "ensemble")]:
        wide = test.pivot(index="basin", columns="combo", values=col)
        rows = []
        for a, b in itertools.combinations(combos(), 2):
            d = (wide[a] - wide[b]).dropna()
            res = stats.wilcoxon(d) if (d != 0).any() else None
            rows.append({"combo_a": a, "combo_b": b, "n": len(d), "median_paired_diff_KGE": d.median(),
                         "statistic": res.statistic if res else np.nan, "p": res.pvalue if res else 1.0})
        df = pd.DataFrame(rows)
        df["p_holm"] = holm(df["p"].to_numpy())
        df.to_csv(OUT_ACC / f"wilcoxon_KGE_test_{tag}.csv", index=False)


def rf_drop_rule(summ):
    """3.3 rule fixed before results: regional RF median test KGE < 0.40 -> Supporting Information.
    Returns the drop decision; logged into RF-regional's own run_summary.txt, not a standalone file
    (every run so far has KGE well above the threshold, so there has been nothing to surface)."""
    row = summ[(summ.combo == "RF-regional") & (summ.period == "test")].iloc[0]
    kge = float(row["KGE_median_ens"])
    drop = kge < E.RF_DROP_RULE_KGE
    txt = (f"RF drop rule (plan 3.3): regional RF median test KGE (seed-ensemble mean, evaluation set) = {kge:.3f} "
           f"(per-seed: {row['KGE_median_perseed']:.3f}); threshold {E.RF_DROP_RULE_KGE}. Outcome: "
           + ("RF moves to the Supporting Information and is excluded from the main ranking." if drop
              else "RF stays in the main results and ranking."))
    for r in runs_of("RF-regional"):
        with open(r.dir / "run_summary.txt", "a") as f:
            f.write(txt + "\n")
    return bool(drop)
    print(txt)


# ------------------------------------------------------------------------------------------
# 6.2 mass balance
# ------------------------------------------------------------------------------------------
def water_balance(q, P, PET, days_mask):
    """Flags over the same test days for model and observation (2.5: one period for both)."""
    qs, ps = q[days_mask].sum(), P[days_mask].sum()
    n = int(days_mask.sum())
    et = (ps - qs) / n
    return {"runoff_ratio": qs / ps, "flag_runoff_gt_P": qs > ps, "implied_ET": et,
            "flag_ET_neg": et < 0, "flag_ET_gt_PET": et > PET[days_mask].mean(),
            "flag_any": (qs > ps) or (et < 0) or (et > PET[days_mask].mean())}


def mass(basins, obs, forc):
    a, z = E.TEST
    test_days = pd.date_range(a, z)
    obs_t = obs.loc[a:z]
    rows_obs = {}
    for b in basins:
        m = obs_t[b].notna().to_numpy()
        P, PET = forc[b]["Precip"].loc[a:z].to_numpy(), forc[b]["PET"].loc[a:z].to_numpy()
        assert len(P) == len(test_days) == len(obs_t)          # identical date ranges (2.5)
        if m.mean() < OBS_COVERAGE:
            rows_obs[b] = {"obs_coverage": m.mean(), "flag_any": np.nan}
            continue
        rows_obs[b] = {"obs_coverage": m.mean(), **water_balance(obs_t[b].to_numpy(), P, PET, m)}
    obs_wb = pd.DataFrame(rows_obs).T
    obs_wb.index.name = "basin"
    obs_wb.to_csv(OUT_MASS / "mass_observed_per_basin.csv")
    obs_ok = obs_wb["flag_any"].astype(float) == 0

    per_basin = []
    for key in combos():
        for seed, sim in load_combo(key, "og", basins).items():
            for b in basins:
                m = obs_t[b].notna().to_numpy()
                s = sim[b].loc[a:z].to_numpy()
                vol = (s[m].sum() - obs_t[b].to_numpy()[m].sum()) / obs_t[b].to_numpy()[m].sum()
                P, PET = forc[b]["Precip"].loc[a:z].to_numpy(), forc[b]["PET"].loc[a:z].to_numpy()
                wb = water_balance(s, P, PET, m if m.mean() >= OBS_COVERAGE else np.ones_like(m, bool))
                per_basin.append({"combo": key, "seed": seed, "basin": b, "volume_error": vol,
                                  "abs_volume_error": abs(vol), **wb, "obs_ok": bool(obs_ok.get(b, False))})
    pb = pd.DataFrame(per_basin)
    pb.to_csv(OUT_MASS / "mass_per_basin.csv", index=False)
    return pb, obs_wb


def mass_primary(pb_seed):
    """Fraction of basins with a model water-balance violation, among basins whose observed
    discharge does not violate the same checks (6.2b)."""
    g = pb_seed[pb_seed["obs_ok"]]
    return float(g["flag_any"].astype(float).mean())


# ------------------------------------------------------------------------------------------
# 6.3 drought
# ------------------------------------------------------------------------------------------
def k_from_log_slope(slope):
    k = -1.0 / slope if slope < 0 else np.inf
    return float(np.clip(k, *K_CLIP))


def observed_k(q, p):
    """Recession constant from observed dry spells (>= MIN_DRY_SPELL dry days), days REC_DAYS after the
    last rain, pooled with spell-specific intercepts: ln Q = c_spell - d / k."""
    dry = (p <= DRY_P).to_numpy()
    qv = q.to_numpy()
    xs, ys = [], []
    i, n = 0, len(dry)
    while i < n:
        if not dry[i]:
            i += 1
            continue
        j = i
        while j < n and dry[j]:
            j += 1
        if j - i >= MIN_DRY_SPELL:
            d = np.arange(1, j - i + 1)            # day 1 = first dry day after rain
            sel = (d >= REC_DAYS[0]) & (d <= REC_DAYS[1])
            seg = qv[i:j][sel]
            dd = d[sel]
            ok = np.isfinite(seg) & (seg > 0)
            if ok.sum() >= 3:
                ly = np.log(seg[ok])
                xs.append(dd[ok] - dd[ok].mean())
                ys.append(ly - ly.mean())
        i = j
    if not xs:
        return np.nan, 0
    x, y = np.concatenate(xs), np.concatenate(ys)
    return k_from_log_slope(np.sum(x * y) / np.sum(x * x)), len(xs)


def synthetic_k(q):
    """Same days-since-rain range on the drought run: P = 0 from TEST[0], so day 1 = TEST[0]."""
    d = np.arange(1, len(q) + 1)
    sel = (d >= REC_DAYS[0]) & (d <= REC_DAYS[1])
    y = np.log(np.maximum(q.to_numpy()[sel], Q_FLOOR))
    x = d[sel]
    return k_from_log_slope(np.polyfit(x, y, 1)[0])


def drought(basins, obs, forc, sig):
    pre = (E.DATA_START, E.VAL[1])          # observed dry spells 1987-2000
    k_obs = {}
    for b in basins:
        k_obs[b], nseg = observed_k(obs[b].loc[pre[0]:pre[1]], forc[b]["Precip"].loc[pre[0]:pre[1]])
    snow_hi = sig.loc[basins, "frac_snow"] >= sig.loc[basins, "frac_snow"].quantile(2 / 3)
    rows = []
    for key in combos():
        for seed, sim in load_combo(key, "no_precip", basins).items():
            for b in basins:
                q = sim[b]
                q1 = q.loc[E.DROUGHT[0]:E.DROUGHT[0] + pd.Timedelta(days=364)]
                dq = np.diff(q1.to_numpy())
                months = q1.index[1:].month
                keep = ~np.isin(months, [3, 4, 5, 6, 7]) if snow_hi[b] else np.ones(len(dq), bool)
                ks = synthetic_k(q)
                rows.append({"combo": key, "seed": seed, "basin": b, "k_syn": ks, "k_obs": k_obs[b],
                             "abs_log_ratio": abs(np.log(ks / k_obs[b])) if np.isfinite(k_obs[b]) else np.nan,
                             "frac_decreasing_90d": float(np.mean(dq[:89][keep[:89]] <= 0)),
                             "frac_decreasing_365d": float(np.mean(dq[keep] <= 0)),
                             "meanQ_2002_2006": float(q.loc["2002-01-01":E.DROUGHT[1]].mean()),
                             "snow_melt_season_excluded": bool(snow_hi[b])})
    pb = pd.DataFrame(rows)
    for col, name in [("baseflow_index", "BFI"), ("aridity", "aridity"), ("frac_snow", "frac_snow")]:
        terc = pd.qcut(sig.loc[basins, col], 3, labels=["low", "mid", "high"])
        pb[f"{name}_tercile"] = pb["basin"].map(terc).astype(str)
    pb.to_csv(OUT_DROUGHT / "drought_per_basin.csv", index=False)
    strata = []
    for strat in ["BFI_tercile", "aridity_tercile", "frac_snow_tercile"]:
        g = pb.groupby(["combo", strat])
        s = g.agg(median_abs_log_ratio=("abs_log_ratio", "median"), median_k_syn=("k_syn", "median"),
                  median_k_obs=("k_obs", "median"), frac_decreasing_90d=("frac_decreasing_90d", "median"),
                  meanQ_2002_2006=("meanQ_2002_2006", "median")).reset_index().rename(columns={strat: "stratum"})
        s.insert(1, "strata", strat)
        strata.append(s)
    pd.concat(strata).to_csv(OUT_DROUGHT / "drought_strata.csv", index=False)
    # Only pick an example basin from ones every plotted model (regional + HBV) actually calibrated
    # reasonably well on -- an attribute-extreme basin where e.g. HBV calibrated badly (seen: KGE_cal
    # 0.21 on the first high-BFI pick) produces a misleading example, since one line's collapse is
    # that basin's bad fit, not a general high-BFI behaviour.
    acc = pd.read_csv(E.ANALYSIS_ACCURACY_DIR / "accuracy_per_basin.csv", dtype={"basin": str})
    shown = [k for k in combos() if k.split("-", 1)[1] == "regional" or k.split("-", 1)[0] == "HBV"]
    cal_ok = acc[(acc.combo.isin(shown)) & (acc.period == "cal")]
    well_calibrated = set(cal_ok.groupby("basin")["KGE_ens"].min()[lambda x: x > 0.3].index) & set(basins)
    s_all = sig.loc[basins]
    s = sig.loc[[b for b in basins if b in well_calibrated]]
    if len(s) < 10:  # fallback if the filter is too strict for some reason -- never leave it empty
        s = s_all
    examples = {"high_BFI": s["baseflow_index"].idxmax(), "low_BFI": s["baseflow_index"].idxmin(),
                "snow": s["frac_snow"].idxmax()}
    (OUT_DROUGHT / "drought_example_basins.json").write_text(json.dumps(examples))
    return pb


def drought_primary(pb_seed):
    return float(pb_seed["abs_log_ratio"].median())


# ------------------------------------------------------------------------------------------
# 6.4 flood
# ------------------------------------------------------------------------------------------
def flood(basins, obs, forc, sig):
    s0 = E.STORM_START
    win = pd.date_range(s0, E.FLOOD_WINDOW[1])
    ups = np.array(E.STORM_UPLIFTS) / 100.0
    rows = []
    for key in combos():
        per = {sc: load_combo(key, sc, basins) for sc in E.FLOOD_SCENARIOS + [E.NOSTORM]}
        og = load_combo(key, "og", basins)
        for seed in per[E.NOSTORM]:
            base = per[E.NOSTORM][seed].loc[win]
            for b in basins:
                fb = forc[b]
                src = fb.loc[E.SOURCE_SEARCH[0]:E.SOURCE_SEARCH[1], "Precip"]
                src_date, src_peak = src.idxmax(), float(src.max())
                # 6.4.7 baseline: real source event in the normal test run vs observed discharge
                ev = slice(src_date - pd.Timedelta(days=1), src_date + pd.Timedelta(days=7))
                o_pk, s_pk = obs[b].loc[ev].max(), og[seed][b].loc[ev].max()
                for d in E.STORM_DURATIONS:
                    peaks, rcs, inc, frac, lag = [], [], [], [], []
                    for u in E.STORM_UPLIFTS:
                        dq = (per[f"peak{d}_rate{u}"][seed][b].loc[win] - base[b]).to_numpy()
                        pk = dq.max()
                        peaks.append(pk)
                        total = dq.sum()
                        rcs.append(total / (d * src_peak * (1 + u / 100)))
                        inc.append(dq[-7:].mean() > 0.01 * pk if pk > 0 else True)
                        cs = np.cumsum(dq)
                        frac.append([cs[min(n, len(cs)) - 1] / total if total != 0 else np.nan for n in (30, 90, 180, 365)])
                        lag.append(float(np.argmax(dq) - (d - 1) / 2))   # days from storm P centroid
                    peaks = np.array(peaks)
                    elast = (np.polyfit(np.log1p(ups), np.log(peaks), 1)[0] if (peaks > 0).all() else np.nan)
                    mono = bool(np.all(np.diff(peaks) > 0))
                    incomplete = bool(any(inc))
                    under_cap = bool(np.all(np.array(rcs) <= 1))
                    fr = np.array(frac)
                    rows.append({"combo": key, "seed": seed, "basin": b, "duration": d, "elasticity": elast,
                                 "monotonic": mono, "max_runoff_coef": float(np.max(rcs)), "under_cap": under_cap,
                                 "incomplete": incomplete,
                                 "pass": (mono and under_cap) if not incomplete else np.nan,
                                 **{f"frac_dQ_day{n}": float(np.nanmedian(fr[:, i])) for i, n in enumerate((30, 90, 180, 365))},
                                 "lag_days": float(np.median(lag)), "peak_dQ_rate0": float(peaks[0]),
                                 "baseline_peak_rel_error": (s_pk - o_pk) / o_pk if o_pk and np.isfinite(o_pk) else np.nan})
    pb = pd.DataFrame(rows)
    pb["BFI"] = pb["basin"].map(sig["baseflow_index"])
    pb["frac_snow"] = pb["basin"].map(sig["frac_snow"])
    pb.to_csv(OUT_FLOOD / "flood_per_basin.csv", index=False)
    g = pb.groupby(["combo", "duration"])
    summ = g.agg(elasticity_median=("elasticity", "median"),
                 elasticity_IQR=("elasticity", iqr), elasticity_nan=("elasticity", lambda x: int(x.isna().sum())),
                 frac_monotonic=("monotonic", "mean"), frac_over_cap=("under_cap", lambda x: 1 - x.mean()),
                 n_incomplete=("incomplete", "sum"), lag_median=("lag_days", "median"),
                 frac_day30=("frac_dQ_day30", "median"), frac_day90=("frac_dQ_day90", "median"),
                 frac_day180=("frac_dQ_day180", "median"), frac_day365=("frac_dQ_day365", "median"),
                 baseline_peak_rel_error_median=("baseline_peak_rel_error", "median")).reset_index()
    # n_incomplete counted per seed run; report per seed (mean over seeds)
    nseed = pb.groupby("combo")["seed"].nunique()
    summ["n_incomplete"] = summ["n_incomplete"] / summ["combo"].map(nseed)
    corr = []
    for key, gg in pb.groupby("combo"):
        for col in ["frac_dQ_day30", "frac_dQ_day90", "lag_days"]:
            for att in ["BFI", "frac_snow"]:
                r = stats.spearmanr(gg[col], gg[att], nan_policy="omit")
                corr.append({"combo": key, "metric": col, "attribute": att, "spearman_rho": r.statistic, "p": r.pvalue})
    pd.DataFrame(corr).to_csv(OUT_FLOOD / "flood_response_vs_attributes.csv", index=False)
    summ.to_csv(OUT_FLOOD / "flood_summary.csv", index=False)
    return pb


def flood_primary(pb_seed):
    """Fraction of basin x duration cases that are monotonic at every uplift step and under the
    runoff-coefficient cap, among cases not flagged incomplete."""
    return float(pb_seed["pass"].dropna().astype(float).mean())


# ------------------------------------------------------------------------------------------
# 6.5 composite
# ------------------------------------------------------------------------------------------
DIMS = {"mass": (mass_primary, "low"), "drought": (drought_primary, "low"), "flood": (flood_primary, "high")}


def combo_values(tables, basins_sample=None):
    """Per-seed dimension values, median over seeds, for every combo."""
    out = {}
    for key in combos():
        vals = {}
        for dim, (fn, _) in DIMS.items():
            t = tables[dim]
            t = t[t.combo == key]
            if basins_sample is not None:
                t = basins_sample.merge(t, on="basin")          # keeps duplicates of resampled basins
            vals[dim] = float(np.median([fn(g) for _, g in t.groupby("seed")]))
        out[key] = vals
    return pd.DataFrame(out).T


def normalise(v):
    n = pd.DataFrame(index=v.index)
    for dim, (_, direction) in DIMS.items():
        x = v[dim]
        rng = x.max() - x.min()
        z = (x - x.min()) / rng if rng > 0 else pd.Series(1.0, index=x.index)
        n[dim] = 1 - z if direction == "low" else z
    return n


def composite(tables, basins, acc_summary, main_combos):
    raw = combo_values(tables)
    norm = normalise(raw)
    norm_main = normalise(raw.loc[main_combos])
    comp = norm.mean(axis=1)
    # bootstrap over basins
    boot_raw, boot_comp, first = [], [], {k: 0 for k in raw.index}
    bdf = pd.DataFrame({"basin": basins})
    for i in range(N_BOOT):
        sample = bdf.sample(len(bdf), replace=True, random_state=int(RNG.integers(1 << 31)))
        v = combo_values(tables, sample)
        nb = normalise(v)
        c = nb.mean(axis=1)
        boot_raw.append(v)
        boot_comp.append(c)
        first[c.loc[main_combos].idxmax()] += 1
    br = pd.concat(boot_raw, keys=range(N_BOOT))
    bc = pd.concat(boot_comp, axis=1)
    acc = acc_summary[acc_summary.period == "test"].set_index("combo")
    rows = []
    for k in raw.index:
        row = {"combo": k, "in_main_ranking": k in main_combos,
               "test_KGE_median_ens": acc.loc[k, "KGE_median_ens"], "test_KGE_median_perseed": acc.loc[k, "KGE_median_perseed"]}
        for dim in DIMS:
            x = br.xs(k, level=1)[dim]
            row.update({f"{dim}_raw": raw.loc[k, dim], f"{dim}_raw_p05": x.quantile(.05), f"{dim}_raw_p95": x.quantile(.95),
                        f"{dim}_norm": norm.loc[k, dim]})
        row.update({"composite": comp[k], "composite_p05": bc.loc[k].quantile(.05), "composite_p95": bc.loc[k].quantile(.95),
                    "composite_main_only": norm_main.mean(axis=1).get(k, np.nan),
                    "frac_bootstrap_rank_first": first[k] / N_BOOT})
        rows.append(row)
    tab = pd.DataFrame(rows).sort_values("composite", ascending=False)
    tab.to_csv(OUT_SCORE / "composite_table.csv", index=False)
    # weight sweep on a 0.05 grid (main-ranking combos, normalised among all combos)
    sweep = []
    g = np.round(np.arange(0, 1.0001, 0.05), 2)
    for wm in g:
        for wd in g:
            wf = round(1 - wm - wd, 2)
            if wf < -1e-9:
                continue
            c = (norm["mass"] * wm + norm["drought"] * wd + norm["flood"] * wf).loc[main_combos]
            sweep.append({"w_mass": wm, "w_drought": wd, "w_flood": wf, "best": c.idxmax(), **c.to_dict()})
    pd.DataFrame(sweep).to_csv(OUT_SCORE / "composite_weight_sweep.csv", index=False)
    return tab


# ------------------------------------------------------------------------------------------
def main():
    for d in [OUT_ACC, OUT_MASS, OUT_DROUGHT, OUT_FLOOD, OUT_SCORE]:
        d.mkdir(parents=True, exist_ok=True)
    basins = load_eval_set()
    bs = load_basin_sets()
    subset_531 = [b for b in basins if bs.loc[b, "in_531_subset"]]
    sig = load_camels_signatures()
    sig.index = sig.index.astype(str).str.zfill(8)
    forc = {b: load_forcing(b) for b in basins}
    obs = pd.DataFrame({b: forc[b][E.Q_COL] for b in basins})
    print(f"scoring on {len(basins)} evaluation basins")
    pb_acc, acc_summary, rf_drop = accuracy(basins, obs, subset_531)
    print("accuracy done", flush=True)
    m, _ = mass(basins, obs, forc)
    print("mass done", flush=True)
    d = drought(basins, obs, forc, sig)
    print("drought done", flush=True)
    f = flood(basins, obs, forc, sig)
    print("flood done", flush=True)
    main_combos = [k for k in combos() if not (rf_drop and k.startswith("RF-"))]
    tab = composite({"mass": m, "drought": d, "flood": f}, basins, acc_summary, main_combos)
    print(tab.to_string())
    E.MARK_DIR.mkdir(parents=True, exist_ok=True)
    (E.MARK_DIR / "scoring.done").write_text("ok\n")


if __name__ == "__main__":
    main()
