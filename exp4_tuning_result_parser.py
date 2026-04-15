#!/usr/bin/env python3
import os
import json
import glob
import pandas as pd

# Root folder containing all hyperparameter tuning run directories
# OUTPUT_ROOT = r"/content/drive/MyDrive/thesis_gnn_results/mind_graph_exps/tuning_stratified/gnn/unified_model_first_tuning_run"
OUTPUT_ROOT = r"./drive/MyDrive/thesis_gnn_results/mind_graph_exps/tuning_stratified/adjgnn/5e-5"

# Path for the aggregated summary output
SUMMARY_PATH = os.path.join(os.path.dirname(OUTPUT_ROOT), "unified_aggregated_summary.csv")


def read_json(path):
    """Safely read a JSON file and return a dictionary."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def extract_best_epoch_metrics(run_dir):
    """
    Single-run version:
    Read epoch_metrics.csv and return the metrics for the best test_f1_weighted.
    """
    epoch_path = os.path.join(run_dir, "training_logs_plots", "epoch_metrics.csv")
    if not os.path.exists(epoch_path):
        return None, None

    df = pd.read_csv(epoch_path)
    if "test_f1_weighted" not in df.columns:
        print(f"[Warning] 'test_f1_weighted' column not found in {epoch_path}")
        return None, None

    best_row = df.loc[df["test_f1_weighted"].idxmax()]

    return {
        "best_epoch": int(best_row["epoch"]),
        "best_test_f1_weighted": float(best_row["test_f1_weighted"]),
        "best_test_acc": float(best_row.get("test_acc", float("nan"))),
        "best_test_loss": float(best_row.get("test_loss", float("nan"))),
    }, best_row.to_dict()


# def extract_best_epoch_metrics_5foldcv(
#     run_dir,
#     selection_metric="test_f1_weighted",
#     selection_mode="max",
#     expected_num_folds=5,
# ):
#     """
#     For a 5-fold CV run:
#     - finds fold*_epoch_metrics.csv files
#     - computes mean and std of common metrics across folds for each epoch
#     - picks the best epoch based on mean(selection_metric)
#     - returns mean/std metrics at that selected epoch
#     """
#     pattern = os.path.join(run_dir, "training_logs_plots", "fold*_epoch_metrics.csv")
#     fold_paths = sorted(glob.glob(pattern))

#     if not fold_paths:
#         print(f"[Warning] No fold*_epoch_metrics.csv files found in {run_dir}")
#         return None, None

#     if len(fold_paths) != expected_num_folds:
#         print(
#             f"[Warning] Expected {expected_num_folds} fold files but found {len(fold_paths)} in {run_dir}"
#         )

#     fold_dfs = []
#     for fold_path in fold_paths:
#         df = pd.read_csv(fold_path)

#         if "epoch" not in df.columns:
#             print(f"[Warning] 'epoch' column not found in {fold_path}")
#             return None, None

#         if selection_metric not in df.columns:
#             print(f"[Warning] '{selection_metric}' column not found in {fold_path}")
#             return None, None

#         # keep only numeric columns + epoch
#         numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
#         if "epoch" not in numeric_cols:
#             numeric_cols = ["epoch"] + numeric_cols

#         df = df[numeric_cols].copy()
#         fold_dfs.append(df)

#     # Merge all folds on common epochs
#     merged = fold_dfs[0].copy().add_suffix("_fold1")
#     merged = merged.rename(columns={"epoch_fold1": "epoch"})

#     for i, df in enumerate(fold_dfs[1:], start=2):
#         df_i = df.copy().add_suffix(f"_fold{i}")
#         df_i = df_i.rename(columns={f"epoch_fold{i}": "epoch"})
#         merged = pd.merge(merged, df_i, on="epoch", how="inner")

#     if merged.empty:
#         print(f"[Warning] No common epochs across folds in {run_dir}")
#         return None, None

#     # metrics common to all folds
#     common_metrics = set(fold_dfs[0].columns) - {"epoch"}
#     for df in fold_dfs[1:]:
#         common_metrics &= (set(df.columns) - {"epoch"})
#     common_metrics = sorted(common_metrics)

#     # Compute mean and std across folds for each epoch
#     stats_df = pd.DataFrame()
#     stats_df["epoch"] = merged["epoch"]

#     num_folds = len(fold_dfs)

#     for metric in common_metrics:
#         fold_metric_cols = [f"{metric}_fold{i}" for i in range(1, num_folds + 1)]
#         stats_df[f"avg_{metric}"] = merged[fold_metric_cols].mean(axis=1)
#         stats_df[f"std_{metric}"] = merged[fold_metric_cols].std(axis=1, ddof=1)

#     # Select best epoch based on average selection metric
#     avg_selection_col = f"avg_{selection_metric}"
#     if avg_selection_col not in stats_df.columns:
#         print(f"[Warning] Averaged selection column '{avg_selection_col}' not found")
#         return None, None

#     if selection_mode == "max":
#         best_idx = stats_df[avg_selection_col].idxmax()
#     elif selection_mode == "min":
#         best_idx = stats_df[avg_selection_col].idxmin()
#     else:
#         raise ValueError("selection_mode must be either 'max' or 'min'")

#     best_row = stats_df.loc[best_idx].to_dict()

#     best_epoch_dict = {
#         "best_epoch": int(best_row["epoch"]),
#         f"best_avg_{selection_metric}": float(best_row[f"avg_{selection_metric}"]),
#         f"best_std_{selection_metric}": float(best_row[f"std_{selection_metric}"]),
#         "num_folds_found": num_folds,
#     }

#     # Add a few commonly used metrics if available
#     for metric in ["test_f1_weighted", "test_acc", "test_loss", "val_f1", "val_loss"]:
#         avg_col = f"avg_{metric}"
#         std_col = f"std_{metric}"

#         if avg_col in best_row:
#             best_epoch_dict[f"best_{avg_col}"] = float(best_row[avg_col])
#         if std_col in best_row:
#             best_epoch_dict[f"best_{std_col}"] = float(best_row[std_col])

#     return best_epoch_dict, best_row

import os
import pandas as pd


def extract_best_epoch_metrics_5foldcv(
    run_dir,
    selection_metric="test_f1_weighted",
    selection_mode="max",
):
    """
    For a 5-fold CV run with a precomputed average_epoch_metrics.csv:
    - reads training_logs_plots/average_epoch_metrics.csv
    - expects columns such as:
          epoch
          test_f1_weighted_mean
          test_f1_weighted_std
          test_acc_mean
          test_acc_std
          ...
    - picks the best epoch based on <selection_metric>_mean
    - returns:
          1) a compact summary dict
          2) the full selected row as a dict
    """
    avg_path = os.path.join(run_dir, "training_logs_plots", "average_epoch_metrics.csv")

    if not os.path.exists(avg_path):
        print(f"[Warning] average_epoch_metrics.csv not found in {avg_path}")
        return None, None

    df = pd.read_csv(avg_path)

    if "epoch" not in df.columns:
        print(f"[Warning] 'epoch' column not found in {avg_path}")
        return None, None

    selection_mean_col = f"{selection_metric}_mean"
    selection_std_col = f"{selection_metric}_std"

    if selection_mean_col not in df.columns:
        print(f"[Warning] Selection column '{selection_mean_col}' not found in {avg_path}")
        return None, None

    if df.empty:
        print(f"[Warning] average_epoch_metrics.csv is empty in {avg_path}")
        return None, None

    if selection_mode == "max":
        best_idx = df[selection_mean_col].idxmax()
    elif selection_mode == "min":
        best_idx = df[selection_mean_col].idxmin()
    else:
        raise ValueError("selection_mode must be either 'max' or 'min'")

    best_row = df.loc[best_idx].to_dict()

    best_epoch_dict = {
        "best_epoch": int(best_row["epoch"]),
        f"best_{selection_mean_col}": float(best_row[selection_mean_col]),
    }

    if selection_std_col in best_row and pd.notna(best_row[selection_std_col]):
        best_epoch_dict[f"best_{selection_std_col}"] = float(best_row[selection_std_col])

    # Add a few commonly used metrics if available
    for metric in ["test_f1_weighted", "test_acc", "test_loss", "val_f1", "val_loss"]:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col in best_row and pd.notna(best_row[mean_col]):
            best_epoch_dict[f"best_{mean_col}"] = float(best_row[mean_col])

        if std_col in best_row and pd.notna(best_row[std_col]):
            best_epoch_dict[f"best_{std_col}"] = float(best_row[std_col])

    return best_epoch_dict, best_row

def main():
    """
    Collect hyperparameters, best epochs, and metrics from all runs into one summary.
    """
    run_dirs = [
        os.path.join(OUTPUT_ROOT, d)
        for d in os.listdir(OUTPUT_ROOT)
        if os.path.isdir(os.path.join(OUTPUT_ROOT, d))
    ]

    print(f"Found {len(run_dirs)} run folders under {OUTPUT_ROOT}")
    summaries = []

    for run_dir in sorted(run_dirs):
        hyperparams = read_json(os.path.join(run_dir, "hyperparams.json"))
        print(hyperparams)

        best_epoch_dict, best_row = extract_best_epoch_metrics_5foldcv(
            run_dir,
            selection_metric="test_f1_weighted",
            selection_mode="max",
            # expected_num_folds=5,
        )

        print(f"Best epoch metrics: {best_epoch_dict}")

        summary = {"run_dir": run_dir}
        summary.update(hyperparams)

        if best_epoch_dict:
            summary.update(best_epoch_dict)

        if best_row:
            summary.update(best_row)

        summaries.append(summary)

    if not summaries:
        print("No runs with valid data found.")
        return

    df = pd.DataFrame(summaries)

    if "best_test_f1_weighted_mean" in df.columns:
        df = df.sort_values(by="best_test_f1_weighted_mean", ascending=False)
    elif "best_test_loss_mean" in df.columns:
        df = df.sort_values(by="best_test_loss_mean", ascending=True)

    df.to_csv(SUMMARY_PATH, index=False)
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()