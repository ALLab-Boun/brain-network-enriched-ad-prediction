import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Plot losses + metrics (works for single or multiple folds)
def plot_fold_curves(epoch_metrics, out_path, fold_idx=None):
    """
    epoch_metrics: list[dict] as you already collect in epoch_metrics.append({...})
    out_path: where to save png
    fold_idx: optional (for title)
    """
    df = pd.DataFrame(epoch_metrics)
    if df.empty:
        return

    # Metrics you asked to include (only plot those that actually exist in df)
    metrics_to_plot = [
        # losses
        ("train_loss", "Train Loss"),
        ("test_loss", "Test Loss"),

        # train metrics requested
        ("train_f1_weighted", "Train F1 (weighted)"),
        ("train_f1_macro", "Train F1 (macro)"),
        ("train_auc", "Train AUC"),
        ("train_conv_recall", "Train Conversion Recall"),

        # test metrics requested
        ("test_f1_weighted", "Test F1 (weighted)"),
        ("test_f1_macro", "Test F1 (macro)"),
        ("test_auc", "Test AUC"),
        ("test_class0_precision", "Test Class 0 Precision"),
        ("test_class1_precision", "Test Class 1 Precision"),
        ("test_class0_recall", "Test Class 0 Recall"),        ("test_class1_recall", "Test Class 1 Recall"),
        ("test_conv_recall", "Test Conversion Recall"),

        # early stopping metrics requested
        ("es_loss", "Early Stopping Loss"),
        ("es_f1_weighted", "Early Stopping F1 (weighted)"),
        ("es_f1_macro", "Early Stopping F1 (macro)"),
        ("es_auc", "Early Stopping AUC"),
        ("es_class0_precision", "Early Stopping Class 0 Precision"),
        ("es_class1_precision", "Early Stopping Class 1 Precision"),
        ("es_class0_recall", "Early Stopping Class 0 Recall"),        ("es_class1_recall", "Early Stopping Class 1 Recall"),
        ("es_conv_recall", "Early Stopping Conversion Recall"),
    ]

    # Keep only columns that are present
    present = [(col, label) for col, label in metrics_to_plot if col in df.columns]
    if not present:
        return

    x = df["epoch"].values if "epoch" in df.columns else np.arange(1, len(df) + 1)

    # Layout
    n_plots = len(present)
    ncols = 3
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    title = "Training Curves" if fold_idx is None else f"Training Curves (Fold {fold_idx})"
    fig.suptitle(title)

    for ax_i, (col, label) in enumerate(present):
        ax = axes[ax_i]
        ax.plot(x, df[col].values, label=label)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.2)

    # Hide any unused axes
    for j in range(len(present), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_all_folds_losses(
    all_epoch_metrics,
    out_path,
    train_loss_key="train_loss",
    test_loss_key="test_loss",
    validation_loss_key=None,
    sharey=True,
    dpi=200,
):
    """
    Plot ONLY losses for all folds at once.

    Parameters
    ----------
    all_epoch_metrics : list
        Either:
          - [epoch_metrics] for single-fold, where epoch_metrics is list[dict], or
          - [fold1_epoch_metrics, fold2_epoch_metrics, ...] for multi-fold,
            where each fold_epoch_metrics is list[dict].
        This matches your current `all_epoch_metrics` structure.

    out_path : str
        Path to save the figure (e.g., os.path.join(results_dir, "loss_curves.png"))

    train_loss_key, test_loss_key : str
        Column names inside each epoch dict.

    sharey : bool
        Share y-axis across folds (useful for comparison).

    dpi : int
        Save dpi.

    Notes
    -----
    - Works for 1 fold or N folds (e.g., 5).
    - Handles matplotlib axes shape for single subplot correctly.
    - Skips missing loss keys gracefully.
    """
    if not all_epoch_metrics or len(all_epoch_metrics) == 0:
        raise ValueError("all_epoch_metrics is empty.")

    n_folds = len(all_epoch_metrics)

    # Create subplots: 1 row, n_folds columns
    fig, axes = plt.subplots(1, n_folds, figsize=(5 * n_folds, 3), sharey=sharey)
    if n_folds == 1:
        axes = [axes]  # normalize to list for consistent indexing

    for i in range(n_folds):
        fold_metrics = all_epoch_metrics[i]
        df = pd.DataFrame(fold_metrics)

        # x-axis: epoch if present, else 1..N
        if "epoch" in df.columns:
            x = df["epoch"].values
        else:
            x = np.arange(1, len(df) + 1)

        ax = axes[i]
        plotted_any = False

        if train_loss_key in df.columns:
            y_train = pd.to_numeric(df[train_loss_key], errors="coerce").values
            if not np.all(np.isnan(y_train)):
                ax.plot(x, y_train, label="Train Loss")
                plotted_any = True

        if test_loss_key in df.columns:
            y_test = pd.to_numeric(df[test_loss_key], errors="coerce").values
            if not np.all(np.isnan(y_test)):
                ax.plot(x, y_test, label="Test Loss", linestyle="--")
                plotted_any = True

        if validation_loss_key and validation_loss_key in df.columns:
            y_val = pd.to_numeric(df[validation_loss_key], errors="coerce").values
            if not np.all(np.isnan(y_val)):
                ax.plot(x, y_val, label="Validation Loss", linestyle=":")
                plotted_any = True

        ax.set_title(f"Fold {i+1}")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.2)

        if i == 0:
            ax.set_ylabel("Loss")

        if not plotted_any:
            ax.text(0.5, 0.5, "No loss data", ha="center", va="center", transform=ax.transAxes)

    # Put legend on first axis if it has lines, otherwise on the last axis that has lines
    legend_ax = None
    for ax in axes:
        if len(ax.lines) > 0:
            legend_ax = ax
            break
    if legend_ax is not None:
        legend_ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)