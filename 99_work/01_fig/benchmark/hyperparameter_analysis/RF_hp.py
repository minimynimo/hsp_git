import pandas as pd
import os
import matplotlib.pyplot as plt

MODEL_TYPES = ["regional", "individual"]

output_dir = "/data0/funato/hsp/0_out/1_fig/benchmark/hyperparameter_analysis/RF/GB/"
os.makedirs(output_dir, exist_ok=True)

benchmark_list = ["KGE", "NSE", "RMSE", "MAE"]

for MODEL_TYPE in MODEL_TYPES:
    df = pd.DataFrame() # to save the results of all of the hyperparameter combinations

    # Hyperparameter grids based on model type
    if MODEL_TYPE  == "regional":
        N_ESTIMATORS = [100, 300, 500] #any ##keep the originally ran model 
        MIN_SAMPLES_LEAF = [2, 5] #any
        MAX_DEPTH = ["None"]
        MAX_FEATURES = ["sqrt"] #0.7
    elif MODEL_TYPE  == "individual":
        N_ESTIMATORS = [100, 300, 500] #100
        MIN_SAMPLES_LEAF = [5, 10] #10
        MAX_DEPTH = [10, 20] #20
        MAX_FEATURES = ["sqrt"]

    print(f"--- Processing Model: {MODEL_TYPE} ---")

    for N_ESTIMATOR in N_ESTIMATORS:
        for MIN_SAMPLE_LEAF in MIN_SAMPLES_LEAF:
            for MAX_DEPT in MAX_DEPTH:
                for MAX_FEATUR in MAX_FEATURES:
                    
                    if MODEL_TYPE == "regional":
                        input_path = f"/data0/funato/hsp/0_out/RF/GB/{MODEL_TYPE}/hp/results/RF_results_n{N_ESTIMATOR}_l{MIN_SAMPLE_LEAF}_f{MAX_FEATUR}_eva.csv"
                    elif MODEL_TYPE == "individual":
                        if MAX_FEATUR == "sqrt":
                            input_path = f"/data0/funato/hsp/0_out/RF/GB/{MODEL_TYPE}/hp/n{N_ESTIMATOR}_d{MAX_DEPT}_l{MIN_SAMPLE_LEAF}_fsqrt/results/RF_results_eva.csv"
                        else:  # MAX_FEATUR == "1/3"
                            input_path = f"/data0/funato/hsp/0_out/RF/GB/{MODEL_TYPE}/hp/n{N_ESTIMATOR}_d{MAX_DEPT}_l{MIN_SAMPLE_LEAF}_f1/3/results/RF_results_eva.csv"

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

    output_path = os.path.join(output_dir, f"RF_GB_{MODEL_TYPE}_hp_eva.csv")
    df.to_csv(output_path)
    print(f"Saved combined results to {output_path}")

    # Generate Boxplots for each benchmark
    for benchmark in benchmark_list:
        
        # --- FIX: Search for the column that contains the benchmark name ---
        target_col = None
        for col in df.columns:
            # We check if 'KGE' is inside 'RF_KGE_eva' or matches 'KGE' exactly
            if benchmark in col:
                # Optional: Ensure we don't pick up partial matches if needed, 
                # but usually "KGE" is unique enough.
                target_col = col
                break
        
        if target_col is None:
            print(f"  Warning: Could not find column containing '{benchmark}'. Available: {df.columns.tolist()}")
            continue
        # -------------------------------------------------------------------

        # Prepare data for Boxplot (Group by params)
        plot_data = []
        labels = []
        
        # Helper column for consistent sorting
        df['sort_helper'] = df['MAX_DEPT'].astype(str)
        groups = df.groupby(["N_ESTIMATOR", "MIN_SAMPLE_LEAF", "sort_helper", "MAX_FEATUR"])

        for name, group in groups:
            # name is a tuple: (n, l, d, f)
            label_str = f"n={name[0]}, l={name[1]}, d={name[2]}, f={name[3]}"
            labels.append(label_str)
            
            # Get values using the detected target_col
            values = group[target_col].dropna().values
            plot_data.append(values)

        if not plot_data:
            print(f"  No valid data for benchmark {benchmark}")
            continue

        # Plotting
        plt.figure(figsize=(14, 8))
        
        plt.boxplot(plot_data, patch_artist=True, 
                    boxprops=dict(facecolor="lightblue", color="blue"),
                    medianprops=dict(color="red"))
        
        plt.title(f"Hyperparameter Analysis for {MODEL_TYPE.capitalize()} Model - {benchmark}")
        plt.ylabel(benchmark)
        plt.xlabel("Hyperparameter Combination")
        
        plt.xticks(range(1, len(labels) + 1), labels, rotation=90, fontsize=8)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"RF_GB_{MODEL_TYPE}_{benchmark}_hp_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved boxplot to {plot_path}")

print("Hyperparameter analysis completed.")