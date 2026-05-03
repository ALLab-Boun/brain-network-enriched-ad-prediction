import os
import pandas as pd
import argparse

def collect_summary_dfs(root_folder):
    """
    Walk through all subdirectories and collect all summary_dfs CSVs.
    Returns a list of DataFrames.
    """
    dfs = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Process all CSVs inside
        for file in filenames:
            if file.endswith("mean_std_results.csv"): 
                full_path = os.path.join(dirpath, file)
                try:
                    df = pd.read_csv(full_path, index_col=0)
                    # Only keep Mean and Std columns
                    df = df[["Mean", "Std"]]
                    dfs.append(df)
                except Exception as e:
                    print(f"Could not read: {full_path} → {e}")
    return dfs


def aggregate_summaries(dfs):
    """
    dfs: list of DataFrames (each with Mean, Std columns for same metrics).
    Returns a final aggregated DataFrame.
    """
    if not dfs:
        raise ValueError("No summary_dfs found.")

    # Concatenate along a new axis
    combined = pd.concat(dfs, axis=0, keys=range(len(dfs)))
    # Multi-index: (df_id, metric_name)

    # Group by metric name (index level 1)
    grouped = combined.groupby(level=1)

    # Aggregate:
    final = pd.DataFrame({
        "Mean_of_Means": grouped["Mean"].mean(),
        "Std_of_Means": grouped["Mean"].std(),
        "Mean_of_Stds": grouped["Std"].mean(),
        "Std_of_Stds": grouped["Std"].std(),
    })

    return final, combined


if __name__ == "__main__":
    # set root as current directory for easier usage
    parser = argparse.ArgumentParser(description="Aggregate summary_dfs from multiple runs.")
    parser.add_argument(
        "--root_folder",
        type=str,
        default=".",
        help="Root folder to search for summary_dfs CSV files. Defaults to current directory."
    )
    args = parser.parse_args()
    ROOT = args.root_folder

    dfs = collect_summary_dfs(ROOT)
    print(f"Found {len(dfs)} summary_dfs files.")

    final_df, combined_df = aggregate_summaries(dfs)

    save_path = os.path.join(ROOT, f"aggregated_summary.csv")
    final_df.to_csv(save_path)
    combined_df.to_csv(os.path.join(ROOT, f"combined_summary.csv"))

    print("Saved aggregated results to:", save_path)
    print(final_df)
