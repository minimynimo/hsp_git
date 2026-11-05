"""Phase 7 figures f01, f03-f06 (house print style, /data0/funato/_style). Read-only: every number comes
from analysis/scoring.py outputs (ANALYSIS_DIR) or from saved run outputs; nothing is recomputed
beyond plotting. Legend labels come from each run's saved config (plan 2.4)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/data0/funato/_style")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import viz_style as vs  # noqa: E402

from config import experiment as E  # noqa: E402
from hsp import runio  # noqa: E402
from hsp.io import load_eval_set, load_forcing  # noqa: E402
from hsp.registry import combos, runs_of  # noqa: E402

FIG = E.FIG_DIR
MODEL_ORDER = ["LSTM", "TCN", "XGB", "RF", "HBV"]
STRAT_LS = {"individual": ":", "regional": "-", "adapted": "--"}
STRAT_MK = {"individual": "o", "regional": "s", "adapted": "^"}
STRAT_SHORT = {"individual": "ind", "regional": "reg", "adapted": "ada"}

ctx = vs.apply_style()
vs.set_models(MODEL_ORDER)
# These figures are drawn larger than house print size for legibility (same deviation as the
# titles); ctx["lw"]=1.0 reads as too thin at that scale, so these figures use explicit widths.
BIG_LW = 2.2
BIG_LW_THICK = 3.0


def split(key):
    return key.split("-", 1)


def label(key):
    """Legend label: model + strategy, no hyperparameter tag (plan 2.4 showed the tag in run_label()
    before this; dropped per user request -- the model/strategy name is all the legend needs)."""
    m, s = split(key)
    return "HBV (per-basin calibrated)" if m == "HBV" else f"{m} {s}"


def short(key):
    m, s = split(key)
    return "HBV" if m == "HBV" else f"{m}-{STRAT_SHORT[s]}"


def save(fig, stem, data=None, data_dir=None):
    """data_dir: the analysis/<type> folder the figure's companion CSV belongs in -- figures live in
    1_fig/ (PNG only), their underlying numbers stay with the rest of that analysis type's output."""
    vs.savefig(fig, stem, FIG)
    plt.close(fig)
    if data is not None:
        data_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(data_dir / f"{stem}.csv", index=False)


def per_seed_metrics(key, basins):
    out = []
    for r in runs_of(key):
        m = pd.read_csv(r.dir / "metrics.csv", dtype={"basin": str})
        out.append(m[m.basin.isin(basins)].assign(seed=r.seed))
    return pd.concat(out)


def ecdf(x, grid):
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    return np.searchsorted(x, grid, side="right") / len(x)


# Panel titles deviate from house style ("no panel titles/suptitle") at the user's explicit request,
# so every panel is self-explanatory without a separate caption. Uses vs.panel_label(..., title=...),
# which draws "**a** Title" left-aligned in one line -- never collides, unlike a separate centered title.
def panel_titles(axes, titles):
    for i, (a, t) in enumerate(zip(axes, titles)):
        vs.panel_label(a, i, title=t)


# ------------------------------------------------------------------------------------------
def f01(basins):
    pb = pd.read_csv(E.ANALYSIS_ACCURACY_DIR / "accuracy_per_basin.csv", dtype={"basin": str})
    # Larger-than-house-print canvas (deviates from the "full" 160mm print width, same as the
    # titling deviation): 11 categories / dense legends need the extra room to not overlap.
    fig, axs = plt.subplots(3, 3, figsize=(16, 15))
    grid = np.linspace(0, 1, 400)
    rows, title_grid = [], {}
    for j, strat in enumerate(["individual", "regional", "adapted"]):
        for i, metric in enumerate(["KGE", "NSE"]):
            ax = axs[i, j]
            keys = [k for k in combos() if split(k)[1] == strat or split(k)[0] == "HBV"]
            for key in keys:
                m = split(key)[0]
                ens = pb[(pb.combo == key) & (pb.period == "test")][f"{metric}_ens"]
                ps = per_seed_metrics(key, basins)
                ps = ps[ps.period == "test"]
                curves = np.array([ecdf(g[metric], grid) for _, g in ps.groupby("seed")])
                c = vs.model_color(m)
                if len(curves) > 1:
                    ax.fill_between(grid, curves.min(0), curves.max(0), color=c, alpha=0.2, lw=0, step="post")
                # Panel title already says the strategy (individual/regional/adapted); the legend only
                # needs the model name, not "TCN regional" repeated in every row of every panel.
                mlab = "HBV" if m == "HBV" else m
                lab = f"{mlab} ({np.median(ens):.2f})"
                ax.plot(grid, ecdf(ens, grid), color=c, ls="-" if m != "HBV" else ":", lw=BIG_LW,
                        drawstyle="steps-post", label=lab, zorder=3)
                rows.append({"panel": f"{metric}-{strat}", "combo": key, "median_ens": float(np.median(ens)),
                             "n_seeds": len(curves)})
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=f"{metric} (test)" if i == 1 else "",
                   ylabel=vs.LABELS["cdf"] if j == 0 else "")
            ax.legend(loc="upper left", fontsize=ctx["legend"] - 1, frameon=False, handlelength=1.2)
            title_grid[i, j] = f"{metric} — {strat}"
    for a in axs[2, :]:
        a.remove()
    # Test is the headline result; calibration-vs-test (overfitting) is shown separately in f01b, not
    # mixed in here.
    ax = fig.add_subplot(3, 1, 3)
    keys = combos()
    pos = np.arange(len(keys))
    data = [pb[(pb.combo == k) & (pb.period == "test")]["KGE_ens"].dropna() for k in keys]
    bp = ax.boxplot(data, positions=pos, widths=0.5, patch_artist=True, showfliers=False, showmeans=True,
                    meanprops=dict(marker="D", mfc="white", mec="black", ms=ctx["ms"]),
                    medianprops=dict(color="black", lw=BIG_LW), whiskerprops=dict(lw=ctx["lw_axes"] * 1.5),
                    capprops=dict(lw=ctx["lw_axes"] * 1.5))
    for b, k in zip(bp["boxes"], keys):
        c = vs.model_color(split(k)[0])
        b.set(facecolor=c, alpha=0.5, edgecolor=c, lw=BIG_LW)
    ax.plot([], [], marker="D", mfc="white", mec="black", ms=ctx["ms"], lw=0, label="mean")
    ax.plot([], [], color="black", lw=BIG_LW, label="median")
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=ctx["legend"])
    ax.set_xticks(pos, [short(k) for k in keys], rotation=45, ha="right")
    ax.set(ylabel="KGE (test)", ylim=(0, 1))
    axes = [axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2], ax]
    titles = [title_grid[0, 0], title_grid[0, 1], title_grid[0, 2], title_grid[1, 0], title_grid[1, 1],
              title_grid[1, 2], "KGE by model & strategy (test)"]
    panel_titles(axes, titles)
    fig.suptitle("Figure 1 — Accuracy (test period 2001–2006): KGE/NSE CDFs by strategy; calibration-vs-test "
                 "comparison is in f01b.\nShaded band = range across the 5 individual-seed models' own CDFs "
                 "(a different curve from the solid ensemble-mean line, so the line can sit outside the band)",
                 fontsize=ctx["legend"] + 1, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "f01_accuracy_cdf_box", pd.DataFrame(rows), E.ANALYSIS_ACCURACY_DIR)


def f03():
    pb = pd.read_csv(E.ANALYSIS_MASS_DIR / "mass_per_basin.csv", dtype={"basin": str})
    obs = pd.read_csv(E.ANALYSIS_MASS_DIR / "mass_observed_per_basin.csv", dtype={"basin": str})
    keys = combos()
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    ax = axs[0]
    ens = pb.groupby(["combo", "basin"])["volume_error"].median().reset_index()
    data = [100 * ens[ens.combo == k]["volume_error"] for k in keys]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, showmeans=True,
                    meanprops=dict(marker="D", mfc="white", mec="black", ms=ctx["ms"] * 0.8),
                    medianprops=dict(color="black", lw=ctx["lw"]))
    for b, k in zip(bp["boxes"], keys):
        c = vs.model_color(split(k)[0])
        b.set(facecolor=(*matplotlib.colors.to_rgb(c), 0.3), edgecolor=c)
    ax.axhline(0, color=vs.OBS, lw=ctx["lw_axes"])
    ax.set_xticks(range(1, len(keys) + 1), [short(k) for k in keys], rotation=45, ha="right")
    ax.set(ylabel="Volume error [%]")
    t0 = "Volume error per basin (test period)"
    ax = axs[1]
    frac = pb[pb.obs_ok].groupby(["combo", "seed"])["flag_any"].mean().groupby("combo").median().reindex(keys)
    allfrac = pb.groupby(["combo", "seed"])["flag_any"].mean().groupby("combo").median().reindex(keys)
    x = np.arange(len(keys))
    ax.bar(x - 0.2, 100 * frac, 0.4, color=[vs.model_color(split(k)[0]) for k in keys],
           label="model violates (basins where\nobserved data itself is OK)")
    ax.bar(x + 0.2, 100 * allfrac, 0.4, color="white", edgecolor=[vs.model_color(split(k)[0]) for k in keys],
           label="model violates (all basins,\nincl. ones with bad observed data)")
    of = 100 * obs["flag_any"].astype(float).mean()
    ax.axhline(of, color=vs.OBS, lw=ctx["lw"], ls="--",
               label=f"reference: observed data itself\nviolates in {of:.0f}% of basins")
    ax.set_xticks(x, [short(k) for k in keys], rotation=45, ha="right")
    ax.set(ylabel="Basins violating water balance [%]", ylim=(0, max(100 * allfrac.max(), of) * 1.35))
    ax.legend(loc="upper left", frameon=False, fontsize=ctx["legend"] - 1.5)
    panel_titles(axs, [t0, "Water-balance violations (ΣQsim>ΣP or ET out of range)"])
    fig.suptitle("Figure 3 — Mass balance: volume error (accuracy) vs. water-balance consistency (conservation)",
                 fontsize=ctx["legend"] + 1.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "f03_mass_balance", pd.DataFrame({"combo": keys, "violation_frac_obs_ok": frac.values,
                                                "violation_frac_all": allfrac.values, "observed_frac": of / 100}),
         E.ANALYSIS_MASS_DIR)


def f04(basins):
    d = pd.read_csv(E.ANALYSIS_DROUGHT_DIR / "drought_per_basin.csv", dtype={"basin": str})
    ex = json.loads((E.ANALYSIS_DROUGHT_DIR / "drought_example_basins.json").read_text())
    keys = combos()
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    med = d.groupby(["combo", "basin"])[["k_syn", "k_obs"]].median().reset_index()
    for k in keys:
        if split(k)[1] not in ("regional", "individual") or (split(k)[1] == "individual" and split(k)[0] != "HBV"):
            continue
        g = med[med.combo == k]
        ax.scatter(g.k_obs, g.k_syn, s=4, color=vs.model_color(split(k)[0]), alpha=0.6, lw=0, label=short(k))
    lim = [1, 1e4]
    ax.plot(lim, lim, color=vs.OBS, lw=ctx["lw_axes"])
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=(0.1, 1e4), xlabel="Observed k [d]", ylabel="Synthetic k [d]")
    ax.legend(loc="upper left", frameon=False, fontsize=ctx["legend"] - 1.5, markerscale=2)
    axes, titles = [ax], ["Synthetic vs. observed k"]
    for j, strat in enumerate(["BFI_tercile", "aridity_tercile"]):
        ax = fig.add_subplot(gs[0, j + 1])
        x = np.arange(3)
        w = 0.8 / len(keys)
        for i, k in enumerate(keys):
            g = d[d.combo == k].groupby(["seed", strat])["abs_log_ratio"].median().groupby(strat).median()
            g = g.reindex(["low", "mid", "high"])
            ax.bar(x - 0.4 + (i + 0.5) * w, g.values, w, color=vs.model_color(split(k)[0]),
                   alpha={"individual": 0.45, "regional": 1, "adapted": 0.75}[split(k)[1]])
        ax.set_xticks(x, ["low", "mid", "high"])
        ax.set(xlabel=strat.replace("_tercile", " tercile").replace("BFI", "Baseflow index"),
               ylabel="Median |ln(k_syn / k_obs)|" if j == 0 else "")
        titles.append(f"k error by {'BFI' if j == 0 else 'aridity'} tercile")
        axes.append(ax)
    tab = []
    # HBV gets a distinct dashed line (not just a different color) since its magenta can read close
    # to LSTM's red once lines overlap at this density; every panel gets its own legend so none of
    # the three reads as a mystery without the others.
    for j, (name, b) in enumerate(ex.items()):
        ax = fig.add_subplot(gs[1, j])
        for k in [k for k in keys if split(k)[1] == "regional" or split(k)[0] == "HBV"]:
            m = split(k)[0]
            r = runs_of(k)[0]
            q = runio.load_sim(r.dir, "no_precip")[b]
            ax.plot(q.index, q.values, color=vs.model_color(m), lw=BIG_LW if m != "HBV" else BIG_LW * 0.9,
                    ls="--" if m == "HBV" else "-", label=m, zorder=5 if m == "HBV" else 3,
                    alpha=0.95)
            tab.append(pd.DataFrame({"basin": b, "combo": k, "date": q.index, "Q": q.values}))
        ax.set_yscale("log")
        ax.set(ylim=(1e-3, None), ylabel="Q [mm d$^{-1}$]" if j == 0 else "")
        ax.text(0.98, 0.95, f"{b} ({name.replace('_', ' ')})", transform=ax.transAxes, ha="right", va="top",
                fontsize=ctx["legend"])
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="lower left", frameon=False, fontsize=ctx["legend"] - 1, ncol=2)
        titles.append(f"{name.replace('_', ' ')} basin (drought)")
        axes.append(ax)
    panel_titles(axes, titles)
    fig.suptitle("Figure 4 — Drought scenario: recession behaviour under zero precipitation (2001–2006)",
                 fontsize=ctx["legend"] + 1.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "f04_drought", pd.concat(tab) if tab else None, E.ANALYSIS_DROUGHT_DIR)


def f05():
    f = pd.read_csv(E.ANALYSIS_FLOOD_DIR / "flood_per_basin.csv", dtype={"basin": str})
    s = pd.read_csv(E.ANALYSIS_FLOOD_DIR / "flood_summary.csv")
    keys = combos()
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax = axs[0, 0]
    data = [f[(f.combo == k) & (f.duration == 3)]["elasticity"].dropna() for k in keys]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color="black"))
    for b, k in zip(bp["boxes"], keys):
        c = vs.model_color(split(k)[0])
        b.set(facecolor=(*matplotlib.colors.to_rgb(c), 0.3), edgecolor=c)
    ax.axhline(1, color=vs.OBS, lw=ctx["lw_axes"], ls="--")
    ax.set_xticks(range(1, len(keys) + 1), [short(k) for k in keys], rotation=45, ha="right")
    ax.set(ylabel="Peak elasticity of ΔQ (3-day storm)")
    titles = ["Peak ΔQ sensitivity to storm uplift"]
    ax = axs[0, 1]
    x = np.arange(len(keys))
    ss = s.groupby("combo")[["frac_monotonic", "frac_over_cap", "n_incomplete"]].mean().reindex(keys)
    nb = f.groupby(["combo"])["basin"].nunique().reindex(keys)
    cols = [vs.model_color(split(k)[0]) for k in keys]
    ax.bar(x - 0.27, 100 * ss.frac_monotonic, 0.27, color=cols, label="monotonic")
    ax.bar(x, 100 * ss.frac_over_cap, 0.27, color="white", edgecolor=cols, label="runoff coefficient > 1")
    ax.bar(x + 0.27, 100 * ss.n_incomplete / nb, 0.27, color=cols, alpha=0.35, label="incomplete in 365 d")
    ax.set_xticks(x, [short(k) for k in keys], rotation=45, ha="right")
    ax.set(ylabel="Basins [%]", ylim=(0, 115))
    ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=ctx["legend"] - 1.5)
    titles.append("Flags: monotonic / runoff cap / incomplete")
    ax = axs[1, 0]
    reg = [k for k in keys if split(k)[1] == "regional" or split(k)[0] == "HBV"]
    bins = np.linspace(0, 1, 6)
    for k in reg:
        g = f[(f.combo == k) & (f.duration == 3)]
        mids, vals = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = g[(g.BFI >= lo) & (g.BFI < hi)]["frac_dQ_day30"]
            if len(sel) >= 5:
                mids.append((lo + hi) / 2)
                vals.append(sel.median())
        ax.plot(mids, vals, color=vs.model_color(split(k)[0]), marker="o", ms=ctx["ms"], lw=ctx["lw"], label=split(k)[0])
    ax.set(xlabel="Baseflow index", ylabel="Fraction of ΣΔQ by day 30")
    ax.legend(loc="lower left", frameon=False)
    titles.append("Storm-response speed vs. BFI (regional)")
    ax = axs[1, 1]
    # Lag is measured in whole days, so many combos land on the exact same (duration, lag) point --
    # a small per-combo horizontal jitter keeps every line visible instead of stacked invisibly.
    for i, k in enumerate(keys):
        m, st = split(k)
        g = s[s.combo == k].sort_values("duration")
        jit = (i - (len(keys) - 1) / 2) * 0.06
        ax.plot(g.duration + jit, g.lag_median, color=vs.model_color(m), ls=STRAT_LS[st], marker=STRAT_MK[st],
                ms=ctx["ms"], lw=ctx["lw"], alpha=0.85)
    ax.set(xlabel="Storm duration [d]", ylabel="Lag of ΔQ peak [d]", xticks=E.STORM_DURATIONS)
    mh = [Line2D([], [], color=vs.model_color(m), lw=ctx["lw"]) for m in MODEL_ORDER]
    sh = [Line2D([], [], color="k", ls=STRAT_LS[t], marker=STRAT_MK[t], ms=ctx["ms"]) for t in STRAT_LS]
    leg1 = ax.legend(mh, MODEL_ORDER, loc="upper left", frameon=False, fontsize=ctx["legend"] - 1.5, title="model")
    ax.add_artist(leg1)
    ax.legend(sh, list(STRAT_LS), loc="lower right", frameon=False, fontsize=ctx["legend"] - 1.5, title="strategy")
    titles.append("Lag of peak ΔQ vs. storm duration (jittered)")
    panel_titles(axs.flat, titles)
    fig.suptitle("Figure 5 — Flood scenario: storm-response sensitivity (ΔQ = storm run − no-storm run)",
                 fontsize=ctx["legend"] + 1.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "f05_flood", s, E.ANALYSIS_FLOOD_DIR)


def f06():
    t = pd.read_csv(E.ANALYSIS_SCORING_DIR / "composite_table.csv")
    sw = pd.read_csv(E.ANALYSIS_SCORING_DIR / "composite_weight_sweep.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))
    ax = axs[0]
    for _, r in t.iterrows():
        m, st = split(r.combo)
        c = vs.model_color(m)
        ax.errorbar(r.test_KGE_median_ens, r.composite, yerr=[[r.composite - r.composite_p05], [r.composite_p95 - r.composite]],
                    fmt=STRAT_MK[st], color=c, mfc=c if r.in_main_ranking else "white", ms=ctx["ms"] * 1.3,
                    lw=ctx["lw_axes"], capsize=2)
        ax.annotate(short(r.combo), (r.test_KGE_median_ens, r.composite), xytext=(3, 3), textcoords="offset points",
                    fontsize=ctx["annot"])
    ax.set(xlabel="Median KGE (test)", ylabel="Composite plausibility")
    t0 = "Accuracy vs. composite plausibility (90% CI)"
    ax = axs[1]
    main = t[t.in_main_ranking].combo.tolist()
    # ternary coordinates: mass at bottom-left, drought bottom-right, flood top
    xx = sw.w_drought + 0.5 * sw.w_flood
    yy = np.sqrt(3) / 2 * sw.w_flood
    best = sw.best
    for k in main:
        sel = best == k
        if sel.any():
            m, st = split(k)
            ax.scatter(xx[sel], yy[sel], s=14, marker=STRAT_MK[st], color=vs.model_color(m), label=short(k), lw=0)
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color=vs.OBS, lw=ctx["lw_axes"])
    for (px, py, lab, ha) in [(0, -0.06, "mass", "center"), (1, -0.06, "drought", "center"),
                              (0.5, np.sqrt(3) / 2 + 0.03, "flood", "center")]:
        ax.text(px, py, lab, ha=ha, fontsize=ctx["legend"])
    ax.scatter([1 / 3 + 1 / 6], [np.sqrt(3) / 6], marker="x", color=vs.OBS, s=20)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper left", bbox_to_anchor=(0.72, 1.0), frameon=False, fontsize=ctx["legend"] - 1)
    panel_titles(axs, [t0, "Best model by weighting (x = equal weights)"])
    fig.suptitle("Figure 6 — Accuracy vs. plausibility: composite score and weight sensitivity",
                 fontsize=ctx["legend"] + 1.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "f06_accuracy_vs_plausibility", t, E.ANALYSIS_SCORING_DIR)


def f01b(metric):
    """Benchmark-style box + CDF per metric, calibration vs. test side by side (one figure per metric,
    every model x strategy as its own box/line) -- the large, single-metric layout from the reference
    scripts (99_work/01_fig/benchmark/f01_fig_*.py), as opposed to f01's compact multi-metric grid."""
    pb = pd.read_csv(E.ANALYSIS_ACCURACY_DIR / "accuracy_per_basin.csv", dtype={"basin": str})
    keys = combos()
    grid = np.linspace(0, 1, 400)
    fig, axs = plt.subplots(2, 2, figsize=(12, 11))
    for col, period in enumerate(["cal", "test"]):
        ax = axs[0, col]
        data = [pb[(pb.combo == k) & (pb.period == period)][f"{metric}_ens"].dropna() for k in keys]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, showmeans=True,
                        meanprops=dict(marker="D", mfc="white", mec="black", ms=ctx["ms"]),
                        medianprops=dict(color="black", lw=BIG_LW))
        for b, k in zip(bp["boxes"], keys):
            c = vs.model_color(split(k)[0])
            b.set(facecolor=(*matplotlib.colors.to_rgb(c), 0.4), edgecolor=c, lw=BIG_LW)
        ax.set_xticks(range(1, len(keys) + 1), [short(k) for k in keys], rotation=45, ha="right")
        ax.set(ylabel=metric if col == 0 else "", ylim=(0, 1))
        ax = axs[1, col]
        for k in keys:
            m, st = split(k)
            g = pb[(pb.combo == k) & (pb.period == period)][f"{metric}_ens"]
            ax.plot(grid, ecdf(g, grid), color=vs.model_color(m), ls=STRAT_LS[st] if m != "HBV" else ":",
                    lw=BIG_LW, drawstyle="steps-post", zorder=3 if st == "regional" else 2)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=metric,
               ylabel=vs.LABELS["cdf"] if col == 0 else "")
        mh = [Line2D([], [], color=vs.model_color(m), lw=BIG_LW) for m in MODEL_ORDER]
        sh = [Line2D([], [], color="k", ls=STRAT_LS[s], lw=BIG_LW) for s in STRAT_LS]
        leg1 = ax.legend(mh, MODEL_ORDER, loc="upper left", frameon=False, fontsize=ctx["legend"],
                         handlelength=1.6, title="model", title_fontsize=ctx["legend"])
        ax.add_artist(leg1)
        ax.legend(sh, list(STRAT_LS), loc="upper left", bbox_to_anchor=(0, 0.62), frameon=False,
                  fontsize=ctx["legend"], handlelength=1.8, title="strategy", title_fontsize=ctx["legend"])
    period_name = {"cal": "Calibration 1988-1997", "test": "Test 2001-2006"}
    titles = [f"Box — {period_name[p]}" for p in ["cal", "test"]] + \
             [f"CDF — {period_name[p]}" for p in ["cal", "test"]]
    panel_titles(axs.flat, titles)
    fig.suptitle(f"Figure 1b — {metric} benchmark: every model × strategy, calibration vs. test",
                 fontsize=ctx["legend"] + 1.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, f"f01b_benchmark_{metric}", pb[pb.combo.isin(keys)][["combo", "period", "basin", f"{metric}_ens"]], E.ANALYSIS_ACCURACY_DIR)


def f01c(metric):
    """One subplot per model (reference style: 99_work/01_fig/benchmark/f01_fig_by_benchmark_ML_
    regional_individual_adapted.py) -- within a single model's panel, individual/regional/adapted
    are directly comparable without having to track color across the other models."""
    pb = pd.read_csv(E.ANALYSIS_ACCURACY_DIR / "accuracy_per_basin.csv", dtype={"basin": str})
    grid = np.linspace(0, 1, 400)
    models = ["RF", "XGB", "TCN", "LSTM"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    rows = []
    for ax, m in zip(axs.flat, models):
        for s in E.STRATEGIES[m]:
            k = f"{m}-{s}"
            g = pb[(pb.combo == k) & (pb.period == "test")][f"{metric}_ens"]
            med = float(np.median(g))
            ax.plot(grid, ecdf(g, grid), color=vs.model_color(m), ls=STRAT_LS[s], lw=BIG_LW,
                    drawstyle="steps-post", label=f"{s} ({med:.2f})")
            rows.append({"model": m, "strategy": s, "median_test": med})
        if "HBV-individual" in combos():
            hbv = pb[(pb.combo == "HBV-individual") & (pb.period == "test")][f"{metric}_ens"]
            ax.plot(grid, ecdf(hbv, grid), color=vs.OBS, ls=(0, (1, 1)), lw=ctx["lw_axes"] * 1.5,
                    drawstyle="steps-post", label=f"HBV ref. ({np.median(hbv):.2f})")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=f"{metric} (test)", ylabel=vs.LABELS["cdf"])
        ax.legend(loc="upper left", frameon=False, fontsize=ctx["legend"] - 1)
    panel_titles(axs.flat, models)
    fig.suptitle(f"Figure 1c — {metric}: individual vs. regional vs. adapted, one panel per model",
                 fontsize=ctx["legend"] + 1.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, f"f01c_strategy_by_model_{metric}", pd.DataFrame(rows), E.ANALYSIS_ACCURACY_DIR)


if __name__ == "__main__":
    basins = load_eval_set()
    FIG.mkdir(parents=True, exist_ok=True)
    f01(basins)
    f01b("KGE")
    f01b("NSE")
    f01c("KGE")
    f01c("NSE")
    f03()
    f04(basins)
    f05()
    f06()
    E.MARK_DIR.mkdir(parents=True, exist_ok=True)
    (E.MARK_DIR / "figures.done").write_text("ok\n")
    print(f"figures written to {FIG}")
