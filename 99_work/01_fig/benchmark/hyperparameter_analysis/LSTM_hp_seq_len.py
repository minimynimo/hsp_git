import pandas as pd
import os
import matplotlib.pyplot as plt

MODEL_TYPES = ["regional", "individual"]

# Updated output directory for LSTM
output_dir = "/data0/funato/hsp2/0_out/1_fig/benchmark/hyperparameter_analysis/LSTM/GB/sequence_length/"
os.makedirs(output_dir, exist_ok=True)

benchmark_list = ["KGE", "NSE", "RMSE", "MAE"]

for MODEL_TYPE in MODEL_TYPES:
    df = pd.DataFrame() 

    # --- 1. Define Hyperparameters based on model type ---
    if MODEL_TYPE == "regional":
        HIDDEN_SIZES = [128, 256] ##256
        DROPOUTS = [0.2, 0.4] ##0.2
    elif MODEL_TYPE == "individual":
        HIDDEN_SIZES = [32, 64]  # Changed to 32, 64
        DROPOUTS = [0.2] # Only d=0.2 for cleaner comparison

    print(f"--- Processing Model: {MODEL_TYPE} ---")

    # --- 2. Iterate through Hyperparameters ---
    for HIDDEN_SIZE in HIDDEN_SIZES:
        for DROPOUT in DROPOUTS:
            
            # --- 3. Construct File Paths ---
            if MODEL_TYPE == "regional":
                # Pattern: /data0/funato/hsp2/0_out/LSTM/GB/regional/hp/results/LSTM_results_h128_d0.4_cal.csv
                input_path = f"/data0/funato/hsp2/0_out/LSTM/GB/{MODEL_TYPE}/hp/results/LSTM_results_h{HIDDEN_SIZE}_d{DROPOUT}_cal.csv"
            
            elif MODEL_TYPE == "individual":
                # Pattern for individual models with different sequence lengths
                input_path_30 = (
                    f"/data0/funato/hsp2/0_out/LSTM/GB/individual_30/hp/h{HIDDEN_SIZE}_d{DROPOUT}/results/LSTM_results_h{HIDDEN_SIZE}_lr0.001_e20_w30_b256_d{DROPOUT}_eva.csv"
                )
                input_path_90 = (
                    f"/data0/funato/hsp2/0_out/LSTM/GB/individual/results/LSTM_results_h{HIDDEN_SIZE}_lr0.001_e100_w90_b256_d{DROPOUT}_eva.csv"
                )
                input_path_270 = (
                    f"/data0/funato/hsp2/0_out/LSTM/GB/individual_270/results/LSTM_results_h{HIDDEN_SIZE}_lr0.001_e100_w270_b256_d{DROPOUT}_eva.csv"
                )

            # Build list of candidate input paths depending on model type
            input_paths = []
            if MODEL_TYPE == "regional":
                input_paths = [input_path]
            else:
                input_paths = [input_path_30, input_path_90, input_path_270]

            # Keep only paths that exist
            existing_paths = [p for p in input_paths if p and os.path.exists(p)]
            if not existing_paths:
                continue

            # Read available files and concatenate
            temp_dfs = []
            for p in existing_paths:
                try:
                    tdf = pd.read_csv(p, index_col=0)
                    # Add dataset source tracking
                    if "individual_30" in p:
                        tdf["DATASET"] = "seq_30"
                    elif "individual_270" in p:
                        tdf["DATASET"] = "seq_270"
                    elif "individual" in p:
                        tdf["DATASET"] = "seq_90"
                    else:
                        tdf["DATASET"] = "regional"
                    temp_dfs.append(tdf)
                except Exception as e:
                    print(f"Error reading {p}: {e}")

            if not temp_dfs:
                continue

            temp_df = pd.concat(temp_dfs, axis=0)

            # Strip whitespace from column names
            temp_df.columns = temp_df.columns.str.strip()

            # Add hyperparameter columns for grouping later
            temp_df["HIDDEN_SIZE"] = HIDDEN_SIZE
            temp_df["DROPOUT"] = DROPOUT

            df = pd.concat([df, temp_df], axis=0)

    # Check if data was collected
    if df.empty:
        print(f"Warning: No data found for model {MODEL_TYPE}. Skipping...")
        continue

    # Save combined CSV
    output_path = os.path.join(output_dir, f"LSTM_GB_{MODEL_TYPE}_hp_eva.csv")
    df.to_csv(output_path)
    print(f"Saved combined results to {output_path}")

    # --- 4. Generate Boxplots ---
    for benchmark in benchmark_list:
        
        # Dynamic column search (finds 'KGE' inside 'LSTM_KGE_eva' or 'KGE')
        target_col = None
        for col in df.columns:
            if benchmark in col:
                target_col = col
                break
        
        if target_col is None:
            print(f"  Warning: Could not find column containing '{benchmark}'. Available: {df.columns.tolist()}")
            continue

        # Prepare data for Boxplot
        plot_data = []
        labels = []
        
        # Group by the LSTM hyperparameters AND dataset source
        if MODEL_TYPE == "individual":
            groups = df.groupby(["HIDDEN_SIZE", "DROPOUT", "DATASET"])
        else:
            groups = df.groupby(["HIDDEN_SIZE", "DROPOUT"])

        for name, group in groups:
            # name is a tuple: (hidden, dropout) or (hidden, dropout, dataset)
            if MODEL_TYPE == "individual":
                label_str = f"h={name[0]}, d={name[1]}\n{name[2]}"
            else:
                label_str = f"h={name[0]}, d={name[1]}"
            labels.append(label_str)
            
            values = group[target_col].dropna().values
            plot_data.append(values)

        if not plot_data:
            print(f"  No valid data for benchmark {benchmark}")
            continue

        # Plotting
        fig_width = 14 if MODEL_TYPE == "individual" else 10
        plt.figure(figsize=(fig_width, 8))
        
        plt.boxplot(plot_data, patch_artist=True, 
                    boxprops=dict(facecolor="lightgreen", color="green"), # Changed color for LSTM
                    medianprops=dict(color="black"))
        
        plt.title(f"Hyperparameter Analysis for {MODEL_TYPE.capitalize()} Model (LSTM) - {benchmark}")
        plt.ylabel(benchmark)
        plt.xlabel("Hyperparameter Combination")
        
        plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"LSTM_GB_{MODEL_TYPE}_{benchmark}_hp_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved boxplot to {plot_path}")

print("Hyperparameter analysis completed.")