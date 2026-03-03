import pandas as pd
import os
import matplotlib.pyplot as plt

MODEL_TYPES = ["regional", "individual"]

# Updated output directory for XGB
output_dir = "/data0/funato/hsp2/0_out/1_fig/benchmark/hyperparameter_analysis/XGB/GB/"
os.makedirs(output_dir, exist_ok=True)

benchmark_list = ["KGE", "NSE", "RMSE", "MAE"]

for MODEL_TYPE in MODEL_TYPES:
    df = pd.DataFrame() 

    # --- 1. Define Hyperparameters based on model type ---
    if MODEL_TYPE == "regional":
        N_ESTIMATORS = [500, 1000] #any  # keep the original model
        MAX_DEPTHS = [5, 7] #any
        LEARNING_RATES = [0.05, 0.1] #any
    elif MODEL_TYPE == "individual":
        N_ESTIMATORS = [100, 300] #100
        MAX_DEPTHS = [3, 4] ##3 or 4
        LEARNING_RATES = [0.05, 0.1] ##0.05

    print(f"--- Processing Model: {MODEL_TYPE} ---")

    # --- 2. Iterate through Hyperparameters ---
    for N_EST in N_ESTIMATORS:
        for MAX_D in MAX_DEPTHS:
            for LR in LEARNING_RATES:
                
                # --- 3. Construct File Paths ---
                # Based on the user provided individual path: .../n300_d3_lr0.1/results/XGB_results_eva.csv
                # I am assuming a similar structure for Regional given the loops, 
                # OR standardizing to the 'hp' folder structure used in previous steps.
                
                if MODEL_TYPE == "regional":
                    input_path = (
                        f"/data0/funato/hsp2/0_out/XGB/GB/{MODEL_TYPE}/hp/results/XGB_results_n{N_EST}_d{MAX_D}_lr{LR}_eva.csv"
                    )
                    
                    # Fallback: specific check if the user meant the file contains the HP info
                    # or if the static file example was just one instance.
                
                elif MODEL_TYPE == "individual":
                    # Pattern: /data0/funato/hsp2/0_out/XGB/GB/individual/hp/n300_d3_lr0.1/results/XGB_results_eva.csv
                    input_path = (
                        f"/data0/funato/hsp2/0_out/XGB/GB/{MODEL_TYPE}/hp/"
                        f"n{N_EST}_d{MAX_D}_lr{LR}/results/XGB_results_eva.csv"
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

                    df = pd.concat([df, temp_df], axis=0)
                    
                except Exception as e:
                    print(f"Error reading {input_path}: {e}")

    # Check if data was collected
    if df.empty:
        print(f"Warning: No data found for model {MODEL_TYPE}. Skipping...")
        continue

    # Save combined CSV
    output_path = os.path.join(output_dir, f"XGB_GB_{MODEL_TYPE}_hp_eva.csv")
    df.to_csv(output_path)
    print(f"Saved combined results to {output_path}")

    # --- 4. Generate Boxplots ---
    for benchmark in benchmark_list:
        
        # Dynamic column search (finds 'KGE' inside 'XGB_KGE_eva' or 'KGE')
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
        
        # Group by the XGB hyperparameters
        groups = df.groupby(["N_ESTIMATORS", "MAX_DEPTH", "LEARNING_RATE"])

        for name, group in groups:
            # name is a tuple: (n_est, max_d, lr)
            label_str = f"n={name[0]}, d={name[1]}, lr={name[2]}"
            labels.append(label_str)
            
            values = group[target_col].dropna().values
            plot_data.append(values)

        if not plot_data:
            print(f"  No valid data for benchmark {benchmark}")
            continue

        # Plotting
        plt.figure(figsize=(12, 8))
        
        plt.boxplot(plot_data, patch_artist=True, 
                    boxprops=dict(facecolor="mediumpurple", color="purple"), # XGB distinct color
                    medianprops=dict(color="yellow"))
        
        plt.title(f"Hyperparameter Analysis for {MODEL_TYPE.capitalize()} Model (XGB) - {benchmark}")
        plt.ylabel(benchmark)
        plt.xlabel("Hyperparameter Combination")
        
        plt.xticks(range(1, len(labels) + 1), labels, rotation=45, fontsize=9)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"XGB_GB_{MODEL_TYPE}_{benchmark}_hp_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved boxplot to {plot_path}")

print("Hyperparameter analysis completed.")