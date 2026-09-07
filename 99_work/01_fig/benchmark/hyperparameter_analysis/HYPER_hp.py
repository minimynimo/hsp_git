import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

output_dir = "/data0/funato/hsp2/0_out/1_fig/US/benchmark/hyperparameter_analysis/HYPER"
os.makedirs(output_dir, exist_ok=True)

loc = "US"
benchmark_list = ["KGE", "NSE", "RMSE", "MAE"]
TOP_N = 5

# Grid must match run_hparam_HYPER.sh exactly (see its "PARAMETER CONFIGURATION" section).
# SPINUP has only one value there (1) so it isn't a real grid dimension -- kept fixed below
# rather than swept, to keep labels/legend readable.
RESERVOIR_SIZE = [10, 50, 100, 200, 300, 400, 500, 700]
SPECTRAL_RADIUS = [0.85, 0.9, 0.95, 1.0, 1.05]
RIDGE_PARAM = [0.1, 0.01, 0.001]
INPUT_SCALE = [0.1, 0.5, 1.0, 5.0, 10.0]
ADJACENCY_DENSITY = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
SPINUP = 1

df = pd.DataFrame()

for RESERVOIR in RESERVOIR_SIZE:
    for SPECTRAL in SPECTRAL_RADIUS:
        for RIDGE in RIDGE_PARAM:
            for SCALE in INPUT_SCALE:
                for ADJACENCY in ADJACENCY_DENSITY:
                    # Must match hparam_main_HYPER.py's file_tag/output_dir construction
                    # exactly: file_tag excludes input_scale, but the run directory name
                    # has "_is{input_scale}" appended after file_tag.
                    file_tag = f"_r{RESERVOIR}_s{SPINUP}_sr{SPECTRAL}_rr{RIDGE}_ad{ADJACENCY}"
                    run_dir = (
                        f"/data0/funato/hsp2/0_out/HYPER/{loc}/HYPER_hp/spin{SPINUP}/"
                        f"reservoir{RESERVOIR}/HYPER{file_tag}_is{SCALE}"
                    )
                    input_path = os.path.join(
                        run_dir, "results", f"HYPER_results{file_tag}_eva.csv",
                    )

                    if not os.path.exists(input_path):
                        continue

                    try:
                        temp_df = pd.read_csv(input_path)
                        temp_df.columns = temp_df.columns.str.strip()

                        temp_df["RESERVOIR_SIZE"] = RESERVOIR
                        temp_df["SPECTRAL_RADIUS"] = SPECTRAL
                        temp_df["RIDGE_PARAM"] = RIDGE
                        temp_df["INPUT_SCALE"] = SCALE
                        temp_df["ADJACENCY_DENSITY"] = ADJACENCY

                        df = pd.concat([df, temp_df], axis=0)

                    except Exception as e:
                        print(f"Error reading {input_path}: {e}")

if df.empty:
    raise SystemExit("Warning: No HYPER hyperparameter results found. Nothing to analyze.")

param_cols = ["RESERVOIR_SIZE", "SPECTRAL_RADIUS", "RIDGE_PARAM", "INPUT_SCALE", "ADJACENCY_DENSITY"]

agg_metrics = ["KGE", "NSE", "RMSE"]
metric_cols = {}
for m in agg_metrics:
    for col in df.columns:
        if m in col:
            metric_cols[m] = col
            break

agg_kwargs = {}
for m in agg_metrics:
    if m in metric_cols:
        agg_kwargs[f"{m}_median"] = (metric_cols[m], "median")
        agg_kwargs[f"{m}_mean"] = (metric_cols[m], "mean")
agg_kwargs["n_basins"] = (metric_cols[agg_metrics[0]], "count")

agg_df = df.groupby(param_cols).agg(**agg_kwargs).reset_index()
agg_df = agg_df.sort_values("NSE_median", ascending=False)

output_path = os.path.join(output_dir, f"HYPER_{loc}_hp_eva.csv")
agg_df.to_csv(output_path, index=False)
print(f"Saved per-combination median/mean results (ranked by NSE median) to {output_path}")
total_combos = (
    len(RESERVOIR_SIZE) * len(SPECTRAL_RADIUS) * len(RIDGE_PARAM)
    * len(INPUT_SCALE) * len(ADJACENCY_DENSITY)
)
print(f"Found {len(agg_df)} / {total_combos} hyperparameter combinations")

# --- CDF per hyperparameter combination ---
topN = agg_df.head(TOP_N)
topN_keys = [tuple(row[c] for c in param_cols) for _, row in topN.iterrows()]
topN_colors = plt.cm.tab10.colors[:TOP_N]


def fmt_label(key):
    rs, sr, rp, isc, ad = key
    return f"r={rs}, sr={sr}, rr={rp}, is={isc}, ad={ad}"


for benchmark in benchmark_list:
    target_col = None
    for col in df.columns:
        if benchmark in col:
            target_col = col
            break

    if target_col is None:
        print(f"  Warning: Could not find column containing '{benchmark}'. Available: {df.columns.tolist()}")
        continue

    plt.figure(figsize=(10, 7))

    for name, group in df.groupby(param_cols):
        values = np.sort(group[target_col].dropna().values)
        if len(values) == 0:
            continue
        y = np.arange(1, len(values) + 1) / len(values)

        if name in topN_keys:
            rank = topN_keys.index(name)
            plt.plot(values, y, drawstyle='steps-post', color=topN_colors[rank],
                      linewidth=2.2, label=f"#{rank+1}: {fmt_label(name)}", zorder=5)
        else:
            plt.plot(values, y, drawstyle='steps-post', color='gray', alpha=0.15,
                      linewidth=0.8, zorder=1)

    plt.title(f"CDF of {benchmark} across hyperparameter combinations -- HYPER (ESN)")
    plt.xlabel(benchmark)
    plt.ylabel("Cumulative probability (across basins)")
    if benchmark in ("KGE", "NSE"):
        plt.xlim(-1, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title=f"Top {TOP_N} (by NSE median)", fontsize=8, loc='best')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"HYPER_{loc}_{benchmark}_cdf.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved CDF plot to {plot_path}")

print("Hyperparameter analysis completed.")
