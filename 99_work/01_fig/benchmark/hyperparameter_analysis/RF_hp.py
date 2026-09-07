import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

MODEL_TYPES = ["regional", "individual"]

loc = "US"

output_dir = f"/data0/funato/hsp2/0_out/1_fig/{loc}/benchmark/hyperparameter_analysis/RF/"
os.makedirs(output_dir, exist_ok=True)

benchmark_list = ["KGE", "NSE", "RMSE", "MAE"]

for MODEL_TYPE in MODEL_TYPES:
    df = pd.DataFrame() # to save the results of all of the hyperparameter combinations

    # Hyperparameter grids -- must match run_rf_hp_tuning.sh's ACTUAL regional grid exactly
    # (24 combos: N_ESTIMATORS(300,500) x MIN_SAMPLES_LEAF(5,10,20) x MAX_DEPTH(15,30) x
    # MAX_FEATURES(sqrt,0.33)). Previously listed a wider nominal 81-combo grid (n_estimators up
    # to 1000, max_depth "None", max_features 0.5, min_samples_leaf up to 10 but missing 20) that
    # was never what actually got trained -- silently dropped the 8 real min_samples_leaf=20
    # combos from every aggregation. Same bug independently found in run_eval_hp.sh -- see
    # DEBUG_README.md §23.
    if MODEL_TYPE == "regional":
        N_ESTIMATORS = [300, 500]
        MIN_SAMPLES_LEAF = [5, 10, 20]
        MAX_DEPTH = [15, 30]
        MAX_FEATURES = ["sqrt", 0.33]
    elif MODEL_TYPE == "individual":
        N_ESTIMATORS = [100, 300, 500]
        MIN_SAMPLES_LEAF = [5, 10, 20]
        MAX_DEPTH = [5, 10, 15]
        MAX_FEATURES = ["sqrt", 0.33]

    print(f"--- Processing Model: {MODEL_TYPE} ---")

    for N_ESTIMATOR in N_ESTIMATORS:
        for MIN_SAMPLE_LEAF in MIN_SAMPLES_LEAF:
            for MAX_DEPT in MAX_DEPTH:
                for MAX_FEATUR in MAX_FEATURES:

                    if MODEL_TYPE == "regional":
                        input_path = f"/data0/funato/hsp2/0_out/RF/{loc}/{MODEL_TYPE}/hp/results/RF_results_n{N_ESTIMATOR}_d{MAX_DEPT}_l{MIN_SAMPLE_LEAF}_f{MAX_FEATUR}_eva.csv"
                    elif MODEL_TYPE == "individual":
                        input_path = f"/data0/funato/hsp2/0_out/RF/{loc}/{MODEL_TYPE}/hp/n{N_ESTIMATOR}_d{MAX_DEPT}_l{MIN_SAMPLE_LEAF}_f{MAX_FEATUR}/results/RF_results_eva.csv"

                    if not os.path.exists(input_path):
                        continue

                    try:
                        temp_df = pd.read_csv(input_path, index_col=0)

                        # Strip whitespace from column names to handle accidental spaces
                        temp_df.columns = temp_df.columns.str.strip()

                        temp_df["N_ESTIMATOR"] = N_ESTIMATOR
                        temp_df["MIN_SAMPLE_LEAF"] = MIN_SAMPLE_LEAF
                        temp_df["MAX_DEPT"] = MAX_DEPT
                        temp_df["MAX_FEATUR"] = MAX_FEATUR

                        df = pd.concat([df, temp_df], axis=0)

                    except Exception as e:
                        print(f"Error reading {input_path}: {e}")

    if df.empty:
        print(f"Warning: No data found for model {MODEL_TYPE}. Skipping...")
        continue

    param_cols = ["N_ESTIMATOR", "MIN_SAMPLE_LEAF", "MAX_DEPT", "MAX_FEATUR"]

    # Aggregate per-basin results down to one row per hyperparameter combination: median/mean
    # of KGE, NSE, RMSE across basins, ranked from highest to lowest median NSE.
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

    output_path = os.path.join(output_dir, f"RF_{loc}_{MODEL_TYPE}_hp_eva.csv")
    agg_df.to_csv(output_path, index=False)
    print(f"Saved per-combination median/mean results (ranked by NSE median) to {output_path}")

    # --- CDF per hyperparameter combination ---
    # Every combination gets its own empirical CDF line (thin grey, unlabeled) so the spread
    # across the whole grid is visible at a glance; the top 5 combinations by median NSE (from
    # agg_df above) are drawn in color and labeled with their hyperparameter values.
    top5 = agg_df.head(5)
    top5_keys = [tuple(row[c] for c in param_cols) for _, row in top5.iterrows()]
    top5_colors = plt.cm.tab10.colors[:5]

    def fmt_label(key):
        n, l, d, f = key
        return f"n={n}, l={l}, d={d}, f={f}"

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

            if name in top5_keys:
                rank = top5_keys.index(name)
                plt.plot(values, y, drawstyle='steps-post', color=top5_colors[rank],
                          linewidth=2.2, label=f"#{rank+1}: {fmt_label(name)}", zorder=5)
            else:
                plt.plot(values, y, drawstyle='steps-post', color='gray', alpha=0.15,
                          linewidth=0.8, zorder=1)

        plt.title(f"CDF of {benchmark} across hyperparameter combinations -- {MODEL_TYPE.capitalize()} RF"
                  + (" (partial -- HP search still running)" if MODEL_TYPE == "regional" else ""))
        plt.xlabel(benchmark)
        plt.ylabel("Cumulative probability (across basins)")
        if benchmark in ("KGE", "NSE"):
            plt.xlim(-1, 1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title="Top 5 (by NSE median)", fontsize=8, loc='best')
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"RF_{loc}_{MODEL_TYPE}_{benchmark}_cdf.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved CDF plot to {plot_path}")

print("Hyperparameter analysis completed.")
