from pathlib import Path
import pandas as pd


# -----------------------------
# CONFIG
# -----------------------------
root_dir = Path(
    r"C:\dev\GitHub\graph-based-dementia-prediction\results_baseline_models\cs_trad_next_visit_lastvisitadfilled_5xseed"
)

output_excel = root_dir / "combined_results_table_nvis_lastvisitadfilled_converter.xlsx"

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

# Traditional model summary files inside each subdirectory.
# Pattern matches files like:
# aggregated_summary_Logistic_Regression.csv
# aggregated_summary_Random_Forest.csv
# aggregated_summary_SVM_RBF.csv
# aggregated_summary_XGBoost.csv
summary_glob_pattern = "aggregated_summary_converters_*.csv"
        # summary_glob_pattern = "aggregated_summary_*_smci_pmci_24m.csv"
# summary_glob_pattern = "aggregated_summary_*.csv"

# -----------------------------
# HELPERS
# -----------------------------
def get_numeric_prefix(name):
    """
    For sorting names like:
        1_morph_lt_conv
        2_adj_lt_conv
    """
    try:
        return int(name.split("_")[0])
    except Exception:
        return 10**9


def extract_model_name_from_summary_file(csv_path):
    """
    Converts:
        aggregated_summary_Logistic_Regression.csv
    into:
        Logistic_Regression
    """
    stem = csv_path.stem  # aggregated_summary_Logistic_Regression
    prefix = "aggregated_summary_"

    if stem.startswith(prefix):
        return stem[len(prefix):]

    return stem


# -----------------------------
# BUILD COMBINED TABLE
# -----------------------------
rows = []

for experiment_dir in sorted(root_dir.iterdir(), key=lambda p: get_numeric_prefix(p.name)):
    if not experiment_dir.is_dir():
        continue

    summary_files = sorted(experiment_dir.glob(summary_glob_pattern))

    if not summary_files:
        print(f"Skipping {experiment_dir.name}: no files matching {summary_glob_pattern}")
        continue

    for csv_path in summary_files:
        traditional_model_name = extract_model_name_from_summary_file(csv_path)

        # Final row name combines experiment subdirectory + traditional model.
        combined_model_name = f"{experiment_dir.name}_{traditional_model_name}"

        # Metric names are in the first column, usually saved as an unnamed index.
        
        
        # df = pd.read_csv(csv_path, index_col=0)
        df = pd.read_csv(csv_path, index_col="Metric")

        row = {
            "Experiment": experiment_dir.name,
            "Traditional_Model": traditional_model_name,
            "Model": combined_model_name,
        }

        selected_metrics = metrics_to_include if metrics_to_include is not None else df.index.tolist()

        # Main formatted metric columns: mean (std)
        for metric in selected_metrics:
            if metric not in df.index:
                print(f"Warning: {metric} missing in {combined_model_name}")
                row[f"{metric} mean (std)"] = pd.NA
                continue

            mean_val = df.loc[metric, "Mean_of_Means"]
            std_val = df.loc[metric, "Mean_of_Stds"]

            if pd.isna(mean_val) or pd.isna(std_val):
                row[f"{metric} mean (std)"] = pd.NA
            else:
                row[f"{metric}\nmean (std)"] = f"{mean_val * 100:.2f} ({std_val * 100:.2f})"

        # Optional seed-variability columns
        for metric in selected_metrics:
            if metric not in df.index:
                row[f"{metric} seed std"] = pd.NA
                continue

            if "Std_of_Means" not in df.columns:
                row[f"{metric} seed std"] = pd.NA
                continue

            seed_std = df.loc[metric, "Std_of_Means"]

            if pd.isna(seed_std):
                row[f"{metric} seed std"] = pd.NA
            else:
                row[f"{metric} seed std"] = f"{seed_std * 100:.2f}"

        rows.append(row)


combined_df = pd.DataFrame(rows)

if combined_df.empty:
    raise RuntimeError(
        f"No rows were created. Check root_dir and whether files matching "
        f"{summary_glob_pattern} exist inside the subdirectories."
    )

# Sort by experiment numeric prefix, then traditional model name.
combined_df["_sort_key"] = combined_df["Experiment"].apply(get_numeric_prefix)
combined_df = combined_df.sort_values(
    by=["_sort_key", "Traditional_Model"]
).drop(columns="_sort_key")

# Save to Excel.
combined_df.to_excel(output_excel, index=False)

print(f"Saved combined table to:\n{output_excel}")
print(combined_df.head())