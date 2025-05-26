import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import os

output_dir =  "/data0/funato/0_out/0_fig/US/benchmark/arid_predictions"
os.makedirs(output_dir, exist_ok=True)

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

fs = 13


log_exp_RC_path_cal = "/data0/funato/0_out/99_out/US/RC_700_0.001_scaled_exp/results/RC_results_r700_s1_sr0.4_rr0.001_cal.csv"
log_2_RC_path_cal = "/data0/funato/0_out/99_out/US/RC_700_0.001_scaled_log2/results/RC_results_r700_s1_sr0.4_rr0.001_cal.csv"
norm_RC_path_cal = "/data0/funato/0_out/99_out/US/RC_700_0.001/results/RC_results_r700_s1_sr0.4_rr0.001_cal.csv"
log_exp_RCHBMA_path_cal = "/data0/funato/0_out/99_out/US/RCHBMA_700_0.001_scaled_exp/results/RCHBMA_results_r700_s1_sr0.4_rr0.001_cal.csv"
log_2_RCHBMA_path_cal = "/data0/funato/0_out/99_out/US/RCHBMA_700_0.001_scaled_log2/results/RCHBMA_results_r700_s1_sr0.4_rr0.001_cal.csv"
norm_RCHBMA_path_cal = "/data0/funato/0_out/99_out/US/RCHBMA_700_0.001/results/RCHBMA_results_r700_s1_sr0.4_rr0.001_cal.csv"

log_exp_RC_path_eva = "/data0/funato/0_out/99_out/US/RC_700_0.001_scaled_exp/results/RC_results_r700_s1_sr0.4_rr0.001_eva.csv"
log_2_RC_path_eva = "/data0/funato/0_out/99_out/US/RC_700_0.001_scaled_log2/results/RC_results_r700_s1_sr0.4_rr0.001_eva.csv"
norm_RC_path_eva = "/data0/funato/0_out/99_out/US/RC_700_0.001/results/RC_results_r700_s1_sr0.4_rr0.001_eva.csv"
log_exp_RCHBMA_path_eva = "/data0/funato/0_out/99_out/US/RCHBMA_700_0.001_scaled_exp/results/RCHBMA_results_r700_s1_sr0.4_rr0.001_eva.csv"
log_2_RCHBMA_path_eva = "/data0/funato/0_out/99_out/US/RCHBMA_700_0.001_scaled_log2/results/RCHBMA_results_r700_s1_sr0.4_rr0.001_eva.csv"
norm_RCHBMA_path_eva = "/data0/funato/0_out/99_out/US/RCHBMA_700_0.001/results/RCHBMA_results_r700_s1_sr0.4_rr0.001_eva.csv"

# Load data
cal_files = {
    "log_exp_RC": log_exp_RC_path_cal,
    "log_2_RC": log_2_RC_path_cal,
    "norm_RC": norm_RC_path_cal,
    "log_exp_RCHBMA": log_exp_RCHBMA_path_cal,
    "log_2_RCHBMA": log_2_RCHBMA_path_cal,
    "norm_RCHBMA": norm_RCHBMA_path_cal,
}

eva_files = {
    "log_exp_RC": log_exp_RC_path_eva,
    "log_2_RC": log_2_RC_path_eva,
    "norm_RC": norm_RC_path_eva,
    "log_exp_RCHBMA": log_exp_RCHBMA_path_eva,
    "log_2_RCHBMA": log_2_RCHBMA_path_eva,
    "norm_RCHBMA": norm_RCHBMA_path_eva,
}

for benchmark in benchmark_list:

    RC_column_cal = f"RC_r700_s1_sr0.4_rr0.001_{benchmark}_cal"
    RC_column_eva = f"RC_r700_s1_sr0.4_rr0.001_{benchmark}_eva"
    RCHBMA_column_cal = f"RCHBMA_r700_s1_sr0.4_rr0.001_{benchmark}_cal"
    RCHBMA_column_eva = f"RCHBMA_r700_s1_sr0.4_rr0.001_{benchmark}_eva"

    # Function to load and extract NSE column
    def load_BMK_data(file_paths):
        data = []
        for label, path in file_paths.items():
            df = pd.read_csv(path)
            if RC_column_cal in df.columns:
                data.append(pd.DataFrame({f"{benchmark}": df[RC_column_cal], "Model": label}))
            elif RCHBMA_column_cal in df.columns:
                data.append(pd.DataFrame({f"{benchmark}": df[RCHBMA_column_cal], "Model": label}))
            elif RC_column_eva in df.columns:
                data.append(pd.DataFrame({f"{benchmark}": df[RC_column_eva], "Model": label}))
            elif RCHBMA_column_eva in df.columns:
                data.append(pd.DataFrame({f"{benchmark}": df[RCHBMA_column_eva], "Model": label}))
            else:
                print(df.columns)
                print(f"Warning: {label} does not contain {benchmark} data.")
        return pd.concat(data, ignore_index=True)

    # Load calibration and evaluation data
    cal_data = load_BMK_data(cal_files)
    eva_data = load_BMK_data(eva_files)

    # Create box plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    sns.boxplot(data=cal_data, x="Model", y=f"{benchmark}", ax=axes[0])
    axes[0].set_title("Calibration (Cal)", fontsize=fs)
    axes[0].tick_params(axis="x", rotation=45, labelsize=fs)
    axes[0].tick_params(axis="y", labelsize=fs)
    if benchmark == "RMSE" or benchmark == "MAE":
        axes[0].set_ylim(0, 10)
    else:
        axes[0].set_ylim(-10, 1)
    axes[0].set_xlabel(axes[0].get_xlabel(), fontsize=fs)
    axes[0].set_ylabel(axes[0].get_ylabel(), fontsize=fs)

    sns.boxplot(data=eva_data, x="Model", y=f"{benchmark}", ax=axes[1])
    axes[1].set_title("Evaluation (Eva)", fontsize=fs)
    axes[1].tick_params(axis="x", rotation=45, labelsize=fs)
    axes[1].tick_params(axis="y", labelsize=fs)
    if benchmark == "RMSE" or benchmark == "MAE":
        axes[1].set_ylim(0, 10)
    else:
        axes[1].set_ylim(-10, 1)
    axes[1].set_xlabel(axes[1].get_xlabel(), fontsize=fs)
    axes[1].set_ylabel(axes[1].get_ylabel(), fontsize=fs)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"benchmark_boxplot_{benchmark}.png"))
    plt.close()

print("Done")
