import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

MODEL_TYPES = ["regional", "individual"]

loc = "US"

# Updated output directory for XGB
output_dir = f"/data0/funato/hsp2/0_out/1_fig/{loc}/benchmark/hyperparameter_analysis/XGB/"
os.makedirs(output_dir, exist_ok=True)

benchmark_list = ["KGE", "NSE", "RMSE", "MAE"]

for MODEL_TYPE in MODEL_TYPES:
    df = pd.DataFrame()

    # --- 1. Define Hyperparameters based on model type ---
    # Grid per /data0/funato/hsp2/HYPERPARAMETERS.md (corrected this session -- see
    # DEBUG_README.md; previously this only matched the old, narrower grid that
    # run_xgb_hp_tuning.sh used to sweep before that script was fixed to match
    # HYPERPARAMETERS.md exactly).
    N_ESTIMATORS = [500, 1000]
    COLSAMPLE_BYTREE = [0.7, 0.8, 1.0]
    LEARNING_RATES = [0.01, 0.05, 0.1]
    if MODEL_TYPE == "regional":
        SUBSAMPLE = [0.7, 0.8, 1.0]
        MAX_DEPTHS = [3, 5, 7]
        MIN_CHILD_WEIGHT = [1, 3, 5]
    elif MODEL_TYPE == "individual":
        SUBSAMPLE = [0.5, 0.6, 0.7]
        MAX_DEPTHS = [2, 3, 4]
        MIN_CHILD_WEIGHT = [5, 10, 20]

    print(f"--- Processing Model: {MODEL_TYPE} ---")

    # --- 2. Iterate through Hyperparameters ---
    for N_EST in N_ESTIMATORS:
        for MAX_D in MAX_DEPTHS:
            for LR in LEARNING_RATES:
                for SS in SUBSAMPLE:
                    for CBT in COLSAMPLE_BYTREE:
                        for MCW in MIN_CHILD_WEIGHT:

                            hp_str = f"n{N_EST}_d{MAX_D}_lr{LR}_ss{SS}_cbt{CBT}_mcw{MCW}"

                            # --- 3. Construct File Paths ---
                            # Regional: eval_regional_hp.py writes to <output_dir>/results/
                            # XGB_results_<hp_string>_eva.csv (run via run_eval_hp.sh, whose
                            # OUTPUT_DIR is .../regional/hp).
                            # Individual: main_xgboost_indiv_hp.py's own hp_str *is* the output
                            # subdirectory name: .../individual/hp/<hp_str>/results/XGB_results_eva.csv
                            if MODEL_TYPE == "regional":
                                input_path = (
                                    f"/data0/funato/hsp2/0_out/XGB/{loc}/{MODEL_TYPE}/hp/results/XGB_results_{hp_str}_eva.csv"
                                )
                            elif MODEL_TYPE == "individual":
                                input_path = (
                                    f"/data0/funato/hsp2/0_out/XGB/{loc}/{MODEL_TYPE}/hp/"
                                    f"{hp_str}/results/XGB_results_eva.csv"
                                )

                            # Check existence
                            if not os.path.exists(input_path):
                                # Optional: Print missing paths for debugging if needed
                                # print(f"  Skipping (not found): {input_path}")
                                continue

                            try:
                                temp_df = pd.read_csv(input_path, index_col=0)

                                # Strip whitespace from column names
                                temp_df.columns = temp_df.columns.str.strip()

                                # Add hyperparameter columns for grouping later
                                temp_df["N_ESTIMATORS"] = N_EST
                                temp_df["MAX_DEPTH"] = MAX_D
                                temp_df["LEARNING_RATE"] = LR
                                temp_df["SUBSAMPLE"] = SS
                                temp_df["COLSAMPLE_BYTREE"] = CBT
                                temp_df["MIN_CHILD_WEIGHT"] = MCW

                                df = pd.concat([df, temp_df], axis=0)

                            except Exception as e:
                                print(f"Error reading {input_path}: {e}")

    # Check if data was collected
    if df.empty:
        print(f"Warning: No data found for model {MODEL_TYPE}. Skipping...")
        continue

    param_cols = ["N_ESTIMATORS", "MAX_DEPTH", "LEARNING_RATE", "SUBSAMPLE", "COLSAMPLE_BYTREE", "MIN_CHILD_WEIGHT"]

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

    output_path = os.path.join(output_dir, f"XGB_{loc}_{MODEL_TYPE}_hp_eva.csv")
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
        n, d, lr, ss, cbt, mcw = key
        return f"n={n}, d={d}, lr={lr}, ss={ss}, cbt={cbt}, mcw={mcw}"

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
                plt.plot(values, y, drawstyle='steps-post', color='gray', alpha=0.1,
                          linewidth=0.6, zorder=1)

        plt.title(f"CDF of {benchmark} across hyperparameter combinations -- {MODEL_TYPE.capitalize()} XGB")
        plt.xlabel(benchmark)
        plt.ylabel("Cumulative probability (across basins)")
        if benchmark in ("KGE", "NSE"):
            plt.xlim(-1, 1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title="Top 5 (by NSE median)", fontsize=7, loc='best')
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"XGB_{loc}_{MODEL_TYPE}_{benchmark}_cdf.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved CDF plot to {plot_path}")

print("Hyperparameter analysis completed.")