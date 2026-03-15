import os
from collections import Counter
import numpy as np
import torch

def _counter_from_labels(labels):
    c = Counter(labels)
    # return sorted dict for stable printing
    return {k: c.get(k, 0) for k in sorted(c.keys())}

def _pct_str(counter_dict):
    total = sum(counter_dict.values())
    if total == 0:
        return "(n=0)"
    parts = []
    for k in sorted(counter_dict.keys()):
        parts.append(f"{k}: {counter_dict[k]} ({100.0*counter_dict[k]/total:.1f}%)")
    return f"(n={total}) " + ", ".join(parts)

def _get_labels_from_files(files, filename_to_data, label_attr="y"):
    labels = []
    missing = 0
    for f in files:
        d = filename_to_data.get(f, None)
        if d is None:
            missing += 1
            continue
        if not hasattr(d, label_attr):
            raise AttributeError(f"Data object for {f} is missing `{label_attr}`.")
        # y is typically shape [1] long tensor in your code
        y = getattr(d, label_attr)
        if torch.is_tensor(y):
            labels.append(int(y.item()))
        else:
            labels.append(int(y))
    return labels, missing

def print_cv_class_distributions(
    splits,
    filename_to_data,
    fold_start_index=1,
    out_path=None,                 # e.g. os.path.join(results_dir, "class_distributions.txt")
    label_attr="y",                # use "y" (recommended)
    class_names=None               # optional dict like {0:"CN/MCI", 1:"AD"} etc.
):
    lines = []
    header = "Per-fold class distributions"
    lines.append(header)
    lines.append("=" * len(header))

    for fold, split in enumerate(splits):
        fold_id = fold + fold_start_index

        train_files = split.get("train_files", [])
        val_files   = split.get("val_files", [])  # may not exist
        test_files  = split.get("test_files", [])

        train_labels, miss_tr = _get_labels_from_files(train_files, filename_to_data, label_attr=label_attr)
        val_labels,   miss_va = _get_labels_from_files(val_files,   filename_to_data, label_attr=label_attr)
        test_labels,  miss_te = _get_labels_from_files(test_files,  filename_to_data, label_attr=label_attr)

        train_ctr = _counter_from_labels(train_labels)
        val_ctr   = _counter_from_labels(val_labels) if len(val_files) > 0 else {}
        trval_ctr = _counter_from_labels(train_labels + val_labels)
        test_ctr  = _counter_from_labels(test_labels)

        lines.append(f"\nFold {fold_id}:")
        lines.append(f"  Train-only      {_pct_str(train_ctr)}" + (f" | missing_files={miss_tr}" if miss_tr else ""))
        if len(val_files) > 0:
            lines.append(f"  Val-only        {_pct_str(val_ctr)}"   + (f" | missing_files={miss_va}" if miss_va else ""))
        else:
            lines.append(f"  Val-only        (no val_files key in split)")
        lines.append(f"  Train+Val used  {_pct_str(trval_ctr)}")
        lines.append(f"  Test            {_pct_str(test_ctr)}"  + (f" | missing_files={miss_te}" if miss_te else ""))

        # Optional: pretty class name mapping
        if class_names is not None:
            def map_ctr(ctr):
                return {class_names.get(k, str(k)): v for k, v in ctr.items()}
            lines.append(f"  (mapped) Train+Val: {map_ctr(trval_ctr)} | Test: {map_ctr(test_ctr)}")

    text = "\n".join(lines)
    print(text)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\nSaved class distribution report to: {out_path}")

def save_model_summary(model, filepath="model_summary.txt"):
    lines = []
    lines.append(f"\nModel Summary for {model.__class__.__name__}:\n")
    
    for name, module in model.named_children():
        lines.append(f"  {name}: {module.__class__.__name__} -> {module}")
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    lines.append(f"\nTotal parameters: {total_params}")
    lines.append(f"Trainable parameters: {trainable_params}")
    
    # Save to file
    with open(filepath, "w") as f:
        for line in lines:
            f.write(line + "\n")
    
    print(f"Model summary saved to {filepath}")

