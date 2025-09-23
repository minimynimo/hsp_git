import pandas as pd

PCA = False
#PCA = True

benchmark_list = ["KGE", "NSE", "E1", "VE", "d", "RMSE", "MAE"]
train_basin_int = 50
if PCA:
    # Define the file path
    for pc in range(1,4):
        file_path = f'/data0/funato/0_out/99_out/JP/BC-PCA-lasso_PUB_kfold_200/test_basin/results/BC-PCA-lasso_PUB_results_rev_PC{pc}_eva.csv'

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Calculate the median of each column
        medians = df.median()
        medians.index = benchmark_list

        # Print the medians
        print(medians)
        
else:
    file_path = f'/data0/funato/hsp/0_out/TCN/GB/shap/regional/results/TCN_results_eva.csv'

    # Read the CSV file
    df = pd.read_csv(file_path, index_col=0, header=0)

    # Calculate the median of each column
    medians = df.median()
    medians.index = benchmark_list

    # Print the medians
    print(file_path)
    print(medians)

print("\n")
# Print NSE KGE E1 and RMSE for .2f
print("Median values for NSE, KGE, E1 and RMSE:")
print(f"{medians['NSE']:.2f} {medians['KGE']:.2f} {medians['E1']:.2f} {medians['RMSE']:.2f}")

