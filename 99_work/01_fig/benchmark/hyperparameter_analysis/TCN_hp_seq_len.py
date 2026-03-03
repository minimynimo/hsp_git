import pandas as pd
import os
import matplotlib.pyplot as plt

MODEL_TYPES = ["regional", "individual"]

# Updated output directory for TCN
output_dir = "/data0/funato/hsp2/0_out/1_fig/benchmark/hyperparameter_analysis/TCN/GB/sequence_length/"
os.makedirs(output_dir, exist_ok=True)

benchmark_list = ["KGE", "NSE", "RMSE", "MAE"]

benchmark_limit = {
    "KGE": {'min': -2, 'max': 1.0},
    "NSE": {'min': -1, 'max': 1.0},
    "RMSE": {'min': 0, 'max': 100},
    "MAE": {'min': 0, 'max': 50},   
}

for MODEL_TYPE in MODEL_TYPES:
    df = pd.DataFrame() 

    # --- 1. Define Hyperparameters based on model type ---
    if MODEL_TYPE == "regional":
        NUM_FILTERS = [32, 64] #64??
        KERNEL_SIZES = [2, 4] #4
        DROPOUTS = [0.2, 0.4] #0.4
    elif MODEL_TYPE == "individual":
        NUM_FILTERS = [32, 64, 128] # Changed to 32, 64
        KERNEL_SIZES = [4] # Only k=4 for 6 boxes
        DROPOUTS = [0.2] # Only d=0.2 for 6 boxes

    print(f"--- Processing Model: {MODEL_TYPE} ---")

    # --- 2. Iterate through Hyperparameters ---
    for NUM_FILTER in NUM_FILTERS:
        for KERNEL_SIZE in KERNEL_SIZES:
            for DROPOUT in DROPOUTS:
                
                # --- 3. Construct File Paths ---
                if MODEL_TYPE == "regional":
                    # Pattern: /data0/funato/hsp2/0_out/TCN/GB/regional/hp/results/TCN_results_f32_k2_d0.2_cal.csv
                    input_path = (
                        f"/data0/funato/hsp2/0_out/TCN/GB/{MODEL_TYPE}/hp/results/"
                        f"TCN_results_f{NUM_FILTER}_k{KERNEL_SIZE}_d{DROPOUT}_cal.csv"
                    )
                
                elif MODEL_TYPE == "individual":
                    # Pattern: /data0/funato/hsp2/0_out/TCN/GB/individual_2/results/TCN_results_f16_k2_d0.4_eva.csv
                    input_path_90 = (
                        f"/data0/funato/hsp2/0_out/TCN/GB/{MODEL_TYPE}_2/results/"
                        f"TCN_results_f{NUM_FILTER}_k{KERNEL_SIZE}_d{DROPOUT}_eva.csv")
                    input_path_30 = (
                        f"/data0/funato/hsp2/0_out/TCN/GB/{MODEL_TYPE}_30/results/"
                        f"TCN_results_f{NUM_FILTER}_k{KERNEL_SIZE}_d{DROPOUT}_eva.csv"
                    )

                # Build list of candidate input paths depending on model type
                input_paths = []
                if MODEL_TYPE == "regional":
                    input_paths = [input_path]
                else:
                    input_paths = [input_path_90, input_path_30]

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
                        if "individual_2" in p:
                            tdf["DATASET"] = "seq_90"
                        elif "individual_30" in p:
                            tdf["DATASET"] = "seq_30"
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
                temp_df["NUM_FILTER"] = NUM_FILTER
                temp_df["KERNEL_SIZE"] = KERNEL_SIZE
                temp_df["DROPOUT"] = DROPOUT

                df = pd.concat([df, temp_df], axis=0)

    # Check if data was collected
    if df.empty:
        print(f"Warning: No data found for model {MODEL_TYPE}. Skipping...")
        continue

    # Save combined CSV
    output_path = os.path.join(output_dir, f"TCN_GB_{MODEL_TYPE}_hp_eva.csv")
    df.to_csv(output_path)
    print(f"Saved combined results to {output_path}")

    # --- 4. Generate Boxplots ---
    for benchmark in benchmark_list:
        
        # Dynamic column search (finds 'KGE' inside 'TCN_KGE_eva' or 'KGE')
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
        
        # Group by the TCN hyperparameters AND dataset source
        if MODEL_TYPE == "individual":
            groups = df.groupby(["NUM_FILTER", "KERNEL_SIZE", "DROPOUT", "DATASET"])
        else:
            groups = df.groupby(["NUM_FILTER", "KERNEL_SIZE", "DROPOUT"])

        for name, group in groups:
            # name is a tuple: (num_filter, kernel_size, dropout) or (num_filter, kernel_size, dropout, dataset)
            if MODEL_TYPE == "individual":
                label_str = f"f={name[0]}, k={name[1]}, d={name[2]}\n{name[3]}"
            else:
                label_str = f"f={name[0]}, k={name[1]}, d={name[2]}"
            labels.append(label_str)
            
            values = group[target_col].dropna().values
            plot_data.append(values)

        if not plot_data:
            print(f"  No valid data for benchmark {benchmark}")
            continue

        # Plotting
        fig_width = 16 if MODEL_TYPE == "individual" else 12
        plt.figure(figsize=(fig_width, 8))
        
        plt.boxplot(plot_data, patch_artist=True, 
                    boxprops=dict(facecolor="orange", color="darkorange"), # TCN distinct color
                    medianprops=dict(color="black"))
        
        plt.title(f"Hyperparameter Analysis for {MODEL_TYPE.capitalize()} Model (TCN) - {benchmark}")
        plt.ylabel(benchmark)
        plt.xlabel("Hyperparameter Combination")
        
        plt.xticks(range(1, len(labels) + 1), labels, rotation=45, fontsize=9)
        plt.ylim(benchmark_limit[benchmark]['min'], benchmark_limit[benchmark]['max'])
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"TCN_GB_{MODEL_TYPE}_{benchmark}_hp_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved boxplot to {plot_path}")

print("Hyperparameter analysis completed.")