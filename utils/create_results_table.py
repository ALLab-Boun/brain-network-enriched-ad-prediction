from pathlib import Path
import pandas as pd


# -----------------------------
# CONFIG
# -----------------------------
root_dir = Path(
    # r"./drive/MyDrive/crosssectional_experiments_lt_conversion"
    r"C:\Users\efeka\Documents\thesis_results\thesis_results\2026_05_03\crosssectional_experiments_adni"
)

#     r"/content/drive/MyDrive/temporal_experiments")
output_excel = root_dir / "combined_results_table_smcipmci12m.xlsx"

# Metrics to include.
# Set to None to include all metrics found in the CSV files.
metrics_to_include = [
    "ACC",
    "Accuracy",
    "AUC",
    "AUPRC",
    "BALANCED_ACC",
    "CONVERSION_RECALL",
    "F1_macro",
    "F1_weighted",
    "PRECISION_CLASS_0",
    "PRECISION_CLASS_1",
    "RECALL_CLASS_0",
    "RECALL_CLASS_1",
]

# summary_filename = "aggregated_summary.csv"
# summary_filename = "aggregated_summary_converters.csv"
summary_filename = "aggregated_summary_smci_pmci_12m.csv"


# -----------------------------
# BUILD COMBINED TABLE
# -----------------------------
rows = []

for subdir in sorted(root_dir.iterdir()):
    if not subdir.is_dir():
        continue

    csv_path = subdir / summary_filename

    if not csv_path.exists():
        print(f"Skipping {subdir.name}: no {summary_filename}")
        continue

    # The metric names are in the first column, usually saved as an unnamed index.
    df = pd.read_csv(csv_path, index_col="Metric")

    row = {"Model": subdir.name}

    selected_metrics = metrics_to_include if metrics_to_include is not None else df.index.tolist()

    for metric in selected_metrics:
        if metric not in df.index:
            print(f"Warning: {metric} missing in {subdir.name}")
            row[f"{metric} mean (std)"] = pd.NA
            continue

        mean_val = df.loc[metric, "Mean_of_Means"]
        std_val = df.loc[metric, "Mean_of_Stds"]

        if pd.isna(mean_val) or pd.isna(std_val):
            row[f"{metric}\nmean (std)"] = pd.NA
        else:
            row[f"{metric}\nmean (std)"] = f"{mean_val * 100:.2f} ({std_val * 100:.2f})"

    for metric in selected_metrics:
        if metric not in df.index:
            row[f"{metric} seed std"] = pd.NA
            continue

        seed_std = df.loc[metric, "Std_of_Means"]

        if pd.isna(seed_std):
            row[f"{metric} seed std"] = pd.NA
        else:
            row[f"{metric} seed std"] = f"{seed_std * 100:.2f}"

    rows.append(row)


combined_df = pd.DataFrame(rows)

# Optional: sort by numeric prefix in folder name, e.g. 1_morphmlp, 2_morphgnn, ...
def get_numeric_prefix(model_name):
    try:
        return int(model_name.split("_")[0])
    except Exception:
        return 10**9

combined_df["_sort_key"] = combined_df["Model"].apply(get_numeric_prefix)
combined_df = combined_df.sort_values("_sort_key").drop(columns="_sort_key")

# Save to Excel
combined_df.to_excel(output_excel, index=False)

print(f"Saved combined table to:\n{output_excel}")
print(combined_df.head())