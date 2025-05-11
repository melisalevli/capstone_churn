import pandas as pd
import os


input_folder = "data"
output_file = "data/combined_model_results.csv"


resampling_methods = [
    "SMOTE",
    "SMOTEENN",
    "ADASYN",
    "ROS",
    "RUS"   
]

all_dfs = []


for method in resampling_methods:
    file_path = os.path.join(input_folder, f"run_all_results_{method}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df["Resampling"] = method
        all_dfs.append(df)
    else:
        print(f"❗ File not found: {file_path}")


if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined results saved: {output_file}")
else:
    print("No CSV files were found.")
