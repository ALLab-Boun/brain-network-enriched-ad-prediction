#!/usr/bin/env python3
import os
import sys
import time
import json
import uuid
import itertools
import subprocess
import datetime
from pathlib import Path


# =========================================================
# Config
# =========================================================
# BASE_CMD = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     # "exp4_main_deterministic.py", # for cross sectional models
# ]

BASE_CMD = [
    r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
    "exp4_temporal.py", # for temporal models
    "--include_gnn",
    "--gnn_dropout", "0.2",
    "--gnn_hidden_dim", "64",
    "--edge_threshold", "1.0",
    "--gnn_num_layers", "1",
    "--gnn_layer", "gcn",
    "--gnn_use_pre_mlp",
    "--gnn_cnn_input_add_flattened_node_features",
    "--gnn_add_output_skip",
    "--gnn_layer_connectivity", "skipsum",
    "--gnn_norm_type", "layernorm",
    "--gnn_readout", "cnn",
]

OUTPUT_ROOT = "./drive/MyDrive/thesis_gnn_results/mind_graph_exps/temporal_current_visit_tuning/morph_mlp"
DATASET_PATH = "./data/adni/CT_Vol_graphs_complete_features_filtered_negative.pt"
CROSS_VAL_PKL_PATH = "./data/adni/splits/tuning_cv_splits.pkl"

# How many runs to keep active simultaneously
MAX_PARALLEL = 1

# Keep 5 seconds between launches
LAUNCH_STAGGER_SECONDS = 5

# Optional:
# - Set to [] or None for no explicit GPU pinning
# - Example: GPU_IDS = [0, 1]
GPU_IDS = None

# # -----------------------------
# # Cortex MLP param grid
# # -----------------------------
# param_grid = {
#     "lr": [5e-5, 5e-4],
#     "batch_size": [64, 128],
#     "epochs": [50],
#     "include_cortex_mlp": [True],
#     "dropout": [0.0, 0.2, 0.4],
#     "cortex_mlp_dropout": [0.0, 0.2, 0.4],
#     "cortex_mlp_hidden_dim": [32, 64, 128, 256],
#     "cortex_mlp_use_residual": [True],
#     "cortex_mlp_activation": ["leakyrelu"],
#     "cortex_mlp_use_layernorm": [True],
#     "cortex_mlp_num_layers": [1, 2, 3],

#     "weight_decay": [5e-2],
# }

# -----------------------------
# GNN param grid
# -----------------------------
# param_grid = {
#     "include_gnn": [True],

#     "lr": [ 5e-4],
#     "batch_size": [64],
#     "epochs": [50],
#     "dropout": [0.0, 0.2, 0.5],

#     # architecture
#     "gnn_hidden_dim": [128],
#     "gnn_num_layers": [2],
#     "gnn_layer": ["gcn"],
#     "gnn_layer_connectivity": ["stack", "skipsum", "skipcat"],

#     # regularization
#     "gnn_dropout": [0.0, 0.2, 0.5],
#     # "gnn_norm_type": ["layernorm", "graphnorm"],

#     # graph preprocessing
#     "edge_threshold": [1.0],

#     # # feature augmentations
#     # "add_adj_row_as_node_feature": [False, True],
#     # "separate_adj_features_instead_of_concat": [False],
#     # "add_weighted_degree_as_node_feature": [False, True],

#     # model options
#     "gnn_use_pre_mlp": [True, False],
#     "gnn_cnn_input_add_flattened_node_features": [True, False],
#     "gnn_add_output_skip": [True, False],
# }



# Adjacency CNN param grid
# param_grid = {
#     "include_cnn": [True],
    
#     # Fundamental training params
#     "lr": [5e-5],
#     "batch_size": [64],
#     "epochs": [125],
#     "dropout": [0, 0.2, 0.4],

#     # Architecture Capacity: lightweight to high
#     "adj_cnn_conv_channels": [
#         (16, 64, 128),
#         (32, 128, 256),
#         (32, 256, 512),
#     ],

#     "adj_cnn_dropout": [0.0, 0.2],
    
#     # Receptive Fields: Wide vs Local
#     "adj_cnn_kernel_sizes": [
#         [7, 5, 3],
#         [3, 3, 3],
#     ],
    
#     # Downsampling Strategies
#     "adj_cnn_strides": [
#         [1, 1, 1], # Rely mostly on pooling
#         [2, 2, 1], # Aggressive downsampling early
#     ],
    
#     # Pooling behaviors
#     "adj_cnn_pool_types": [
#         ["max", "max", "avg"],
#     ],
#     "adj_cnn_pool_kernel_sizes": [
#         [2, 2, 2],
#         [4, 4, 3],
#     ],
    
#     "adj_cnn_norm_type": ["batch", "group"],
#     "adj_cnn_readout": ["flatten", "gap_gmp"],
#     "weight_decay": [5e-2],
# }


# -----------------------------
# Cortex Transformer param grid
# -----------------------------
# param_grid = {
#     "include_transformer": [True],

#     # Fundamental training params
#     "lr": [5e-5, 5e-4],
#     "batch_size": [64],
#     "epochs": [30],
#     "dropout": [0.0, 0.2, 0.4],

#     # Architecture Capacity
#     "cortex_transformer_hidden_dim": [128],
#     "cortex_transformer_num_layers": [2],
#     "cortex_transformer_num_heads": [4],
    
#     # Regularization
#     "cort_transformer_dropout": [0.0, 0.2, 0.4],
#     "weight_decay": [5e-2],

#     # Positional Encoding
#     "pos_encoding_type": ["lpe", "none", "learnable"],

#     "add_laplacian_pe": [True],
# #     # "add_adj_row_as_node_feature": [False, True],
# #     # "separate_adj_features_instead_of_concat": [False],
#     # Connectivity & Feature aug
#     "cortex_transformer_cnn_input_add_flattened_node_features": [True, False],
#     "cortex_transformer_add_output_skip": [True, False],
# }


# -----------------------------
# GNN  POOL param grid
# -----------------------------
# param_grid = {
#     "include_gnn": [True],

#     "lr": [ 5e-4],
#     "batch_size": [64],
#     "epochs": [1],
#     "dropout": [0.2],

#     # architecture
#     "gnn_hidden_dim": [64, 128, 512],
#     "gnn_num_layers": [1, 2, 5],
#     "gnn_layer": ["gcn"],
#     "gnn_layer_connectivity": ["stack"], #, "skipsum", "skipcat"],
#     "gnn_readout": ["pool"],
#     "gnn_graph_pool": ["mean", "max", "sum", "mean_max"],

#     # regularization
#     "gnn_dropout": [0.2],
#     # "gnn_norm_type": ["layernorm", "graphnorm"],

#     # graph preprocessing
#     "edge_threshold": [1.0],

#     # # feature augmentations
#     # "add_adj_row_as_node_feature": [False, True],
#     # "separate_adj_features_instead_of_concat": [False],
#     # "add_weighted_degree_as_node_feature": [False, True],

#     # model options
#     "gnn_use_pre_mlp": [True],
#     "gnn_add_output_skip": [True],
# }


# # # -----------------------------
# # # GNN param grid adjacency
# # # -----------------------------
# param_grid = {
#     "include_gnn": [True],

#     "lr": [ 5e-5],
#     "batch_size": [64],
#     "epochs": [150],
#     "dropout": [0.2, 0.4],

#     # architecture
#     "gnn_hidden_dim": [256, 128, 64],
#     "gnn_num_layers": [2, 1],
#     "gnn_layer": ["gcn"],
#     "gnn_layer_connectivity": ["skipsum"],

#     # regularization
#     "gnn_dropout": [0.2, 0.4],
#     # "gnn_norm_type": ["layernorm", "graphnorm"],

#     # graph preprocessing
#     "edge_threshold": [1.0],

#     # # feature augmentations
#     "add_adj_row_as_node_feature": [True],
#     "separate_adj_features_instead_of_concat": [True],

#     # model options
#     "gnn_use_pre_mlp": [True],
#     "gnn_cnn_input_add_flattened_node_features": [True],
#     "gnn_add_output_skip": [True],
# }

# # -----------------------------
# # # Adj Cortex Transformer param grid
# # # -----------------------------
# param_grid = {
#     "include_transformer": [True],

#     # Fundamental training params
#     "lr": [5e-5, 5e-4],
#     "batch_size": [64],
#     "epochs": [90],
#     "dropout": [0.4],

#     # Architecture Capacity
#     "cortex_transformer_hidden_dim": [64,128,256],
#     "cortex_transformer_num_layers": [1,2],
#     "cortex_transformer_num_heads": [1,4,8],
    
#     # Regularization
#     "cort_transformer_dropout": [0.4],
#     "weight_decay": [5e-2],

#     # Positional Encoding
#     "pos_encoding_type": [ "none", "learnable"],

#     "add_laplacian_pe": [True],
#     "add_adj_row_as_node_feature": [ True],
#     "separate_adj_features_instead_of_concat": [True],
#     # Connectivity & Feature aug
#     "cortex_transformer_cnn_input_add_flattened_node_features": [True],
#     "cortex_transformer_add_output_skip": [ True, False],
# }


# # -----------------------------
# # Temporal Morphometric MLP param grid
# # -----------------------------
param_grid = {
    "dropout": [0.1,0.2, 0.3, 0.4 ], #
    "temporal_type": ["rnn", "gru","lstm"],
    "temporal_hidden_dim": [64, 128, 256],
    "lr": [5e-5, 5e-4],
    "batch_size": [32, 64],
    "epochs": [1],
    "include_cortex_mlp": [True],
    "cortex_mlp_dropout": [0.4],
    "cortex_mlp_hidden_dim": [64],
    "cortex_mlp_use_residual": [True],
    "cortex_mlp_activation": ["leakyrelu"],
    "cortex_mlp_use_layernorm": [False],
    "cortex_mlp_num_layers": [1],

    "weight_decay": [5e-2],
}

# # -----------------------------
# # Temporal Morphometric GNN param grid
# # -----------------------------
param_grid = {
    "dropout": [0.4 ], #
    "temporal_type": ["rnn", "gru","lstm"],
    "temporal_hidden_dim": [64],
    "lr": [5e-5, 5e-4],
    "batch_size": [32, 64],
    "epochs": [100],

    # "include_gnn": [True],
    # "gnn_hidden_dim": [64],
    # "gnn_num_layers": [1],
    # "gnn_layer": ["gcn"],
    # "gnn_layer_connectivity": [ "skipsum"],
    # "gnn_dropout": [ 0.2 ],
    # "gnn_norm_type": ["layernorm"],
    # "edge_threshold": [1.0],
    # "gnn_use_pre_mlp": [True],
    # "gnn_cnn_input_add_flattened_node_features": [True],
    # "gnn_add_output_skip": [True],
    # "gnn_readout": ["cnn"],

    "weight_decay": [5e-2],
}

# # -----------------------------
# # Temporal param grid
# # -----------------------------
param_grid = {
    "dropout": [0.4, 0.3, 0.2, 0.1], #
    "temporal_type": ["rnn", "gru","lstm"],
    "temporal_hidden_dim": [64, 128, 256],
    "lr": [5e-5, 5e-4],
    "batch_size": [64, 32],
    "epochs": [75],

    "weight_decay": [5e-2],
}

#
# # =========================================================
# Helpers for robust param matching
# =========================================================
def _canon_value(v):
    """
    Normalize values so JSON values vs Python values compare reliably.
    """
    if isinstance(v, (bool, int, str)) or v is None:
        return v
    if isinstance(v, float):
        return round(v, 12)
    if isinstance(v, (list, tuple)):
        return tuple(_canon_value(x) for x in v)
    return str(v)


def _params_signature(params, keys):
    """
    Make a hashable signature using only specified keys.
    Missing keys are represented as a sentinel to avoid accidental matches.
    """
    sentinel = "__MISSING__"
    items = []
    for k in sorted(keys):
        items.append((k, _canon_value(params.get(k, sentinel))))
    return tuple(items)


# =========================================================
# Existing-run scan
# =========================================================
def build_existing_run_index(output_root, tuned_keys):
    """
    Returns:
      - existing_sigs: set of signatures for runs that have hyperparams.json
      - sig_to_run:    dict(signature -> run_dir)
    """
    existing_sigs = set()
    sig_to_run = {}

    output_root = Path(output_root)
    if not output_root.is_dir():
        return existing_sigs, sig_to_run

    for run_dir in output_root.iterdir():
        if not run_dir.is_dir():
            continue

        hp_path = run_dir / "hyperparams.json"
        if not hp_path.exists():
            continue

        try:
            with open(hp_path, "r", encoding="utf-8") as f:
                hp = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read {hp_path}: {e}")
            continue

        sig = _params_signature(hp, tuned_keys)
        existing_sigs.add(sig)
        sig_to_run.setdefault(sig, str(run_dir))

    return existing_sigs, sig_to_run


# =========================================================
# Command building
# =========================================================
def make_cmd(params, run_dir):
    cmd = BASE_CMD.copy()

    cmd.append(f"--dataset_path={DATASET_PATH}")
    cmd.append(f"--cross_val_pkl={CROSS_VAL_PKL_PATH}")
    cmd.append(f"--run_dir={run_dir}")

    cmd.append("--excluded_node_features=std_min_max")
    cmd.append("--node_feature_set=ct_vol_sa_mc_sd")

    for k, v in params.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        elif isinstance(v, (list, tuple)):
            cmd.append(f"--{k}")
            cmd.extend(str(x) for x in v)
        else:
            cmd.append(f"--{k}={v}")

    return cmd


def make_unique_run_dir(output_root):
    """
    Create a unique run directory safely for parallel launching.
    """
    output_root = Path(output_root)
    while True:
        run_id = (
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            f"_{uuid.uuid4().hex[:8]}"
        )
        run_dir = output_root / run_id
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue


# =========================================================
# Process launching / waiting
# =========================================================
def launch_experiment(params, gpu_id=None):
    """
    Launch a single experiment asynchronously via subprocess.Popen.
    Returns a dict describing the launched process.
    """
    run_dir = make_unique_run_dir(OUTPUT_ROOT)
    cmd = make_cmd(params, str(run_dir))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Optional CPU-thread limiting to reduce oversubscription in parallel runs
    # Uncomment if useful:
    # env["OMP_NUM_THREADS"] = "1"
    # env["MKL_NUM_THREADS"] = "1"

    log_path = run_dir / "training_log.txt"
    log_f = open(log_path, "w", encoding="utf-8", buffering=1)

    start_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_f.write(f"=== START at {start_str} ===\n")
    log_f.write(f"RUN_DIR: {run_dir}\n")
    log_f.write(f"GPU_ID: {gpu_id}\n")
    log_f.write("PARAMS:\n")
    log_f.write(json.dumps(params, indent=2, ensure_ascii=False))
    log_f.write("\n\nCMD:\n")
    log_f.write(" ".join(map(str, cmd)))
    log_f.write("\n\n")
    log_f.flush()

    p = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    print(f"[LAUNCHED] pid={p.pid} gpu={gpu_id} run_dir={run_dir}")

    return {
        "params": params,
        "process": p,
        "log_f": log_f,
        "run_dir": run_dir,
        "log_path": log_path,
        "gpu_id": gpu_id,
        "start_time": time.time(),
    }


def finalize_finished_processes(active_procs, finished_records):
    """
    Check currently active processes.
    Close and finalize the ones that finished.
    Returns the filtered active process list.
    """
    still_active = []

    for rec in active_procs:
        p = rec["process"]
        rc = p.poll()

        if rc is None:
            still_active.append(rec)
            continue

        end_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rec["log_f"].write(f"\n=== END rc={rc} at {end_str} ===\n")
        rec["log_f"].flush()
        rec["log_f"].close()

        rec["returncode"] = rc
        finished_records.append(rec)

        status = "OK" if rc == 0 else "FAILED"
        print(f"[{status}] pid={p.pid} rc={rc} run_dir={rec['run_dir']}")

    return still_active


def wait_until_slot_available(active_procs, finished_records, max_parallel):
    """
    Block until number of active processes is below max_parallel.
    """
    while len(active_procs) >= max_parallel:
        time.sleep(1)
        active_procs[:] = finalize_finished_processes(active_procs, finished_records)


def wait_for_all(active_procs, finished_records):
    """
    Wait for all remaining active processes to finish.
    """
    while active_procs:
        time.sleep(1)
        active_procs[:] = finalize_finished_processes(active_procs, finished_records)


# =========================================================
# Main
# =========================================================
def main():
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    tuned_keys = set(param_grid.keys())

    # Existing runs already completed on disk
    existing_sigs, sig_to_run = build_existing_run_index(output_root, tuned_keys)
    print(f"Found {len(existing_sigs)} existing runs with hyperparams.json under OUTPUT_ROOT.")

    # Build full grid
    keys, values = zip(*param_grid.items())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*values)]
    print(f"Total experiments (grid size): {len(combos)}")

    # Remove duplicates within this script invocation too
    seen_pending_sigs = set()
    pending = []

    for params in combos:
        sig = _params_signature(params, tuned_keys)

        if sig in existing_sigs:
            print(f"Skipping existing run for params={params}")
            print(f"  Found at: {sig_to_run.get(sig, '(unknown)')}")
            continue

        if sig in seen_pending_sigs:
            print(f"Skipping duplicate combo within this invocation: {params}")
            continue

        seen_pending_sigs.add(sig)
        pending.append(params)

    print(f"Pending experiments to launch: {len(pending)}")

    if not pending:
        print("Nothing to run.")
        return

    active_procs = []
    finished_records = []

    for i, params in enumerate(pending, start=1):
        wait_until_slot_available(active_procs, finished_records, MAX_PARALLEL)

        gpu_id = None
        if GPU_IDS:
            gpu_id = GPU_IDS[(i - 1) % len(GPU_IDS)]

        rec = launch_experiment(params, gpu_id=gpu_id)
        active_procs.append(rec)

        print(f"[{i}/{len(pending)}] Submitted run -> {rec['run_dir']}")

        if i < len(pending):
            time.sleep(LAUNCH_STAGGER_SECONDS)

    print("\nAll jobs launched. Waiting for remaining processes to finish...")
    wait_for_all(active_procs, finished_records)

    failed = [r for r in finished_records if r.get("returncode", 1) != 0]

    print("\n=========================================================")
    print("Finished.")
    print(f"Total launched: {len(finished_records)}")
    print(f"Successful:     {len(finished_records) - len(failed)}")
    print(f"Failed:         {len(failed)}")
    print("=========================================================")

    if failed:
        print("\nFailed runs:")
        for r in failed:
            print(f"  - rc={r['returncode']} | run_dir={r['run_dir']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()