"""f02: SHAP / interpretability figure (plan 8.10). Read-only: every number comes from
analysis/shap_attribution.py's saved files (ANALYSIS_DIR/shap); nothing is recomputed here."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/data0/funato/_style")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import viz_style as vs  # noqa: E402
from config import experiment as E  # noqa: E402

A = E.ANALYSIS_SHAP_DIR
FIG = E.FIG_DIR
MODELS = ["LSTM", "TCN", "XGB", "RF"]
SHARED_FAMILIES = ["recent_rain", "antecedent_rain", "temperature", "pet"]

ctx = vs.apply_style()
vs.set_models(MODELS)


def save(fig, stem, data=None):
    vs.savefig(fig, stem, FIG)
    plt.close(fig)
    if data is not None:
        A.mkdir(parents=True, exist_ok=True)
        data.to_csv(A / f"{stem}.csv", index=False)


def load_long(model):
    f = A / f"shap_values_{model}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, dtype={"basin": str})
    return df[~df["feature"].astype(str).str.startswith("__")]


def family_share(df, baseline="within"):
    d = df[df.baseline == baseline].copy()
    d["absval"] = d["value"].abs()
    fam = d.groupby(["basin", "date", "family"], as_index=False)["absval"].sum()
    tot = d.groupby(["basin", "date"], as_index=False)["absval"].sum().rename(columns={"absval": "tot"})
    out = fam.merge(tot, on=["basin", "date"])
    out["share"] = out["absval"] / out["tot"]
    return out[["basin", "date", "family", "share"]]


def panel_family_importance(ax, models):
    rows = []
    x = np.arange(len(SHARED_FAMILIES))
    w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        df = load_long(m)
        if df is None:
            continue
        s = family_share(df)
        med = s.groupby("family")["share"].median().reindex(SHARED_FAMILIES)
        ax.bar(x - 0.4 + (i + 0.5) * w, 100 * med.values, w, color=vs.model_color(m), label=m)
        for fam, v in med.items():
            rows.append({"model": m, "family": fam, "median_share_pct": 100 * v})
    ax.set_xticks(x, [f.replace("_", " ") for f in SHARED_FAMILIES], rotation=20, ha="right")
    ax.set(ylabel="Median |attribution| share [%]\n(within-basin baseline)")
    ax.legend(loc="upper right", frameon=False, fontsize=ctx["legend"] - 1.5)
    return pd.DataFrame(rows)


def panel_memory_profile(ax, models):
    rows = []
    for m in models:
        f = A / f"memory_profile_{m}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[(df.baseline == "within") & (df.variable == "Precip")]
        if not len(df):
            continue
        prof = df.groupby("lag")["abs_value"].mean().sort_index()
        ax.plot(prof.index, prof.values / prof.values.max(), color=vs.model_color(m), lw=ctx["lw"], label=m)
        rows.append(df.assign(model=m))
    ax.set(xlabel="Days before prediction day", ylabel="Mean |IG attribution|\n(normalised, Precip)")
    ax.legend(loc="upper right", frameon=False, fontsize=ctx["legend"] - 1.5)
    return pd.concat(rows) if rows else pd.DataFrame()


def panel_static_vs_dynamic(ax, models):
    rows = []
    x = np.arange(len(models))
    static_share, dynamic_share = [], []
    for m in models:
        df = load_long(m)
        if df is None:
            static_share.append(np.nan); dynamic_share.append(np.nan)
            continue
        between = df[df.baseline == "between"]
        tot = between.groupby(["basin", "date"])["value"].apply(lambda v: v.abs().sum())
        is_static = ~between["family"].isin(list(set(df["family"].dropna())
                                                     & {"recent_rain", "antecedent_rain", "temperature", "pet", "snow", "seasonality"}))
        stat = between[is_static].groupby(["basin", "date"])["value"].apply(lambda v: v.abs().sum())
        frac = float((stat / tot.reindex(stat.index)).median()) if len(stat) else np.nan
        static_share.append(100 * frac)
        dynamic_share.append(100 * (1 - frac) if np.isfinite(frac) else np.nan)
        rows.append({"model": m, "static_share_pct_between_basin": 100 * frac if np.isfinite(frac) else np.nan})
    ax.bar(x - 0.2, static_share, 0.4, color=[vs.model_color(m) for m in models], label="static (between-basin baseline)")
    ax.bar(x + 0.2, dynamic_share, 0.4, color="white", edgecolor=[vs.model_color(m) for m in models], label="dynamic")
    ax.set_xticks(x, models)
    ax.set(ylabel="Share of |attribution| [%]")
    ax.legend(loc="upper right", frameon=False, fontsize=ctx["legend"] - 1.5)
    ax.text(0.98, 0.02, "RF: no between-basin baseline (tree_path_dependent mode)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=ctx["legend"] - 2, style="italic")
    return pd.DataFrame(rows)


def panel_checks_summary(ax, models):
    rows = []
    for m in models:
        f = A / f"checks_{m}.json"
        if not f.exists():
            continue
        c = json.loads(f.read_text())
        rows.append({"model": m,
                     "rain_pushes_up": c.get("1_rain_pushes_up_frac_positive"),
                     "memory_vs_BFI_rho": c.get("2_memory_vs_BFI_spearman"),
                     "ig_gshap_agreement": c.get("5_ig_gradientshap_agreement_median", np.nan),
                     "faithfulness_ratio": c.get("faithfulness", {}).get("ratio_top_over_random", np.nan)})
    tab = pd.DataFrame(rows)
    ax.axis("off")
    if len(tab):
        ax.table(cellText=np.round(tab.set_index("model").to_numpy().astype(float), 2),
                  rowLabels=tab["model"], colLabels=[c.replace("_", " ") for c in tab.columns[1:]],
                  loc="center", cellLoc="center")
    return tab


def _variable_shares_for_row(fam, feat):
    """Map one SHAP row onto the raw variable(s) it is actually built from (Precip/Temp/PET) --
    the one input representation every model (tree or network) shares, so RF/XGB/TCN/LSTM can be
    compared with no family-definition mismatch and no n/a cells.
    Convention (user-specified): snow proxies split by construction, not lumped as one family --
    Psnow_sum is rainfall filtered by T<0 (features/tree_features.py:74) -> counts as Precip;
    melt_sum is a pure degree-day function of T (:77, no P term at all) -> counts as Temp. WB_sum
    (antecedent_rain family) is literally P - PET (:71-73) -- an equal mix, not attributable to one
    side, so its attribution is split 50/50 Precip/PET. doy_sin/doy_cos (seasonality) has no raw-
    variable equivalent at all (not derived from P, T or PET) and is excluded here; it is reported,
    trees-only, in the secondary full-family breakdown (panel_tree_family_heatmap) instead."""
    feat = str(feat)
    if fam == "temperature":
        return {"Temp": 1.0}
    if fam == "pet":
        return {"PET": 1.0}
    if fam == "snow":
        return {"Temp": 1.0} if feat.startswith("melt_sum") else {"Precip": 1.0}
    if fam == "antecedent_rain" and feat.startswith("WB_sum"):
        return {"Precip": 0.5, "PET": 0.5}
    if fam in ("recent_rain", "antecedent_rain"):
        return {"Precip": 1.0}
    return {}  # seasonality, static, or unrecognised -- no raw-variable mapping


def raw_variable_share(df, baseline="within"):
    """Every dynamic SHAP row collapsed onto {Precip, Temp, PET} via _variable_shares_for_row."""
    d = df[df.baseline == baseline].copy()
    d = d[d["family"].isin(["recent_rain", "antecedent_rain", "temperature", "pet", "snow"])]
    if not len(d):
        return pd.DataFrame(columns=["basin", "date", "variable", "share"])
    exploded = []
    for (fam, feat), grp in d.groupby(["family", "feature"], sort=False):
        for var, w in _variable_shares_for_row(fam, feat).items():
            exploded.append(grp[["basin", "date", "value"]].assign(variable=var, absval=(grp["value"] * w).abs()))
    e = pd.concat(exploded, ignore_index=True)
    fam_sum = e.groupby(["basin", "date", "variable"], as_index=False)["absval"].sum()
    tot = fam_sum.groupby(["basin", "date"], as_index=False)["absval"].sum().rename(columns={"absval": "tot"})
    out = fam_sum.merge(tot, on=["basin", "date"])
    out["share"] = out["absval"] / out["tot"]
    return out[["basin", "date", "variable", "share"]]


def _draw_heatmap(ax, mat, rownames, colnames, divider_after=None):
    vmax = np.nanmax(mat) if np.isfinite(mat).any() else 1
    # Grey-hatched, not blank-white: every missing cell has a reason (RF has no between-basin
    # baseline at all -- interventional TreeSHAP segfaults on RF's deep trees, so static importance
    # cannot be computed for RF by any convention) and should read as "deliberately not applicable",
    # never "data missing by accident".
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isfinite(mat[i, j]):
                ax.add_patch(matplotlib.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="0.88",
                                                           edgecolor="white", hatch="///", lw=0.5))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                color = "black" if mat[i, j] > vmax * 0.5 else "white"
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", color=color,
                        fontsize=ctx["annot"] + 4, fontweight="bold")
            else:
                ax.text(j, i, "n/a", ha="center", va="center", color="0.5", fontsize=ctx["annot"] + 1,
                        style="italic")
    ax.set_xticks(range(len(colnames)), colnames, fontsize=ctx["legend"] + 2)
    ax.set_yticks(range(len(rownames)), [r.replace("_", " ") for r in rownames], fontsize=ctx["legend"] + 2)
    ax.grid(False)
    if divider_after is not None:
        ax.axhline(divider_after - 0.5, color="black", lw=1.5)
    cb = plt.colorbar(im, ax=ax, label="Median |attribution| share [%]", shrink=0.85)
    cb.ax.tick_params(labelsize=ctx["legend"])
    cb.set_label("Median |attribution| share [%]", fontsize=ctx["legend"] + 1)


def panel_raw_variable_heatmap(ax, models):
    """Primary cross-model comparison (user-specified, replacing the family-level heatmap): every
    model collapsed onto the one input representation it shares with every other model -- raw
    Precip/Temp/PET (dynamic, within-basin baseline) + the static attribute classes (between-basin
    baseline, same named features for trees and networks already -- plan 1.6). No family-definition
    mismatch, so no n/a cells except RF's static row (RF has no between-basin baseline at all, a
    separate, pre-existing technical limitation -- see panel_tree_family_heatmap's note)."""
    raw_vars = ["Precip", "Temp", "PET"]
    stat_fams = ["climate", "topography", "vegetation", "soil"]
    rows = []
    mat = np.full((len(raw_vars) + len(stat_fams), len(models)), np.nan)
    for j, m in enumerate(models):
        df = load_long(m)
        if df is None:
            continue
        s = raw_variable_share(df, "within")
        med = s.groupby("variable")["share"].median()
        for i, v in enumerate(raw_vars):
            if v in med.index:
                mat[i, j] = 100 * med[v]
                rows.append({"model": m, "variable": v, "baseline": "within", "median_share_pct": mat[i, j]})
        if "between" in df.baseline.unique():
            fs = family_share(df, "between")
            for i, fam in enumerate(stat_fams):
                v = fs[fs.family == fam]["share"]
                if len(v):
                    mat[len(raw_vars) + i, j] = 100 * v.median()
                    rows.append({"model": m, "family": fam, "baseline": "between",
                                "median_share_pct": mat[len(raw_vars) + i, j]})
    _draw_heatmap(ax, mat, raw_vars + stat_fams, models, divider_after=len(raw_vars))
    return pd.DataFrame(rows)


def panel_tree_family_heatmap(ax, tree_models):
    """Secondary, trees-only: the full engineered-family breakdown (including snow and seasonality,
    which have no raw-variable or network equivalent -- see panel_raw_variable_heatmap's docstring).
    Dynamic families from the within-basin baseline, statics from the between-basin baseline."""
    dyn_fams = ["recent_rain", "antecedent_rain", "temperature", "pet", "snow", "seasonality"]
    stat_fams = ["climate", "topography", "vegetation", "soil"]
    rows = []
    mat = np.full((len(dyn_fams) + len(stat_fams), len(tree_models)), np.nan)
    for j, m in enumerate(tree_models):
        df = load_long(m)
        if df is None:
            continue
        s = family_share(df, "within")
        for i, fam in enumerate(dyn_fams):
            v = s[s.family == fam]["share"]
            if len(v):
                mat[i, j] = 100 * v.median()
                rows.append({"model": m, "family": fam, "baseline": "within", "median_share_pct": mat[i, j]})
        if "between" in df.baseline.unique():
            fs = family_share(df, "between")
            for i, fam in enumerate(stat_fams):
                v = fs[fs.family == fam]["share"]
                if len(v):
                    mat[len(dyn_fams) + i, j] = 100 * v.median()
                    rows.append({"model": m, "family": fam, "baseline": "between",
                                "median_share_pct": mat[len(dyn_fams) + i, j]})
    _draw_heatmap(ax, mat, dyn_fams + stat_fams, tree_models, divider_after=len(dyn_fams))
    return pd.DataFrame(rows)


def make_heatmap_figure(models):
    fig, ax = plt.subplots(figsize=(12, 10))
    d = panel_raw_variable_heatmap(ax, models)
    vs.panel_label(ax, 0, title="Raw-variable importance: Precip/Temp/PET (within-basin) "
                                "+ static classes (between-basin) -- same inputs, every model")
    fig.suptitle("Figure 2b — SHAP importance across all models, common raw-variable basis",
                 fontsize=ctx["legend"] + 1.5, fontweight="bold", y=1.02)
    save(fig, "f02b_shap_family_heatmap", d)

    tree_models = [m for m in models if m in ("RF", "XGB")]
    if tree_models:
        fig2, ax2 = plt.subplots(figsize=(8, 10))
        d2 = panel_tree_family_heatmap(ax2, tree_models)
        vs.panel_label(ax2, 0, title="Tree models only: full engineered-feature family breakdown\n"
                                     "(snow, seasonality -- no raw-variable or network equivalent)")
        fig2.suptitle("Figure 2c — SHAP family importance, trees-only detail",
                      fontsize=ctx["legend"] + 1.5, fontweight="bold", y=1.02)
        save(fig2, "f02c_shap_tree_family_heatmap", d2)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 11))
    models = [m for m in MODELS if (A / f"shap_values_{m}.csv").exists()]
    if not models:
        raise FileNotFoundError(f"no SHAP outputs found in {A} -- run analysis/shap_attribution.py first")
    d1 = panel_family_importance(axs[0, 0], models)
    d2 = panel_memory_profile(axs[0, 1], models)
    d3 = panel_static_vs_dynamic(axs[1, 0], models)
    d4 = panel_checks_summary(axs[1, 1], models)
    # Titles deviate from house style ("no panel titles/suptitle") at the user's explicit request,
    # matching f01/f03-f06.
    titles = ["Dynamic-input importance by family (within-basin)",
              "Memory length: how far back Precip still matters (IG)",
              "Static vs. dynamic importance",
              "Sanity checks (8.7) and faithfulness ratio (8.8)"]
    for i, (a, t) in enumerate(zip(axs.flat, titles)):
        vs.panel_label(a, i, title=t)
    fig.suptitle("Figure 2 — SHAP/Integrated-Gradients attribution (regional models, 45-basin sample)",
                 fontsize=ctx["legend"] + 1.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "f02_shap_attribution", d1)
    for name, d in [("f02_memory_profile", d2), ("f02_static_vs_dynamic", d3), ("f02_checks_summary", d4)]:
        if len(d):
            A.mkdir(parents=True, exist_ok=True)
            d.to_csv(A / f"{name}.csv", index=False)
    make_heatmap_figure(models)
    E.MARK_DIR.mkdir(parents=True, exist_ok=True)
    (E.MARK_DIR / "shap_figure.done").write_text("ok\n")
    print(f"f02 written to {FIG} for models: {models}")


if __name__ == "__main__":
    main()
