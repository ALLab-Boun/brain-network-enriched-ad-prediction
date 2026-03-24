import subprocess
import threading
import time
import sys

seeds = [42, 123, 2021, 7, 99]

print("Starting experiments with different seeds in parallel...")

base_command = [
    # "python",
    r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
    "exp4_main_deterministic.py",

    "--dataset", "adni",
    "--dataset_path", "./data/adni/CT_Vol_graphs_complete_features_filtered_negative.pt",
    # TUNING DATA single split
    # "--cross_val_pkl", "/path/to/cross_val_splits_1fold_tuning.pkl",

    # 5 fold CV report data
    "--cross_val_pkl", "/content/data/adni/splits/reporting_cv_splits.pkl",

    # 5 fold CV tuning data
    # "--cross_val_pkl", "/path/to/cross_val_splits_5fold_for_tuning.pkl",

    "--epochs", "30",
    "--lr", "5e-5",
    "--batch_size", "64",
    "--weight_decay", "0.05",
    "--dropout", "0.4",

    # GNN specific
    "--include_gnn",
    "--gnn_dropout", "0.2",
    "--edge_threshold", "1.0",
    "--gnn_num_layers", "2",
    "--gnn_use_pre_mlp",
    "--gnn_cnn_input_add_flattened_node_features",
    "--gnn_add_output_skip",
    "--gnn_layer_connectivity", "skipsum",
    "--gnn_hidden_dim", "128",

    "--include_cortex_mlp",
    "--cortex_mlp_dropout", "0.4",
    "--cortex_mlp_use_residual",
    "--cortex_mlp_hidden_dim", "64",
    "--cortex_mlp_num_layers", "1",

    "--early_stopping",
    "--es_monitor", "es_f1_weighted",
    "--es_mode", "max",
    "--es_patience", "5",
    "--es_min_delta", "0.005",

    "--excluded_node_features", "std_min_max",
    "--node_feature_set", "ct_vol_sa_mc_sd"

    # "--add_adj_row_as_node_feature",
    # "--separate_adj_features_instead_of_concat",

]


def stream_output(seed, process):
    try:
        for line in process.stdout:
            print(f"[seed {seed}] {line}", end="")
    except Exception as e:
        print(f"[seed {seed}] Error while reading output: {e}")


processes = []
threads = []

for i, seed in enumerate(seeds):
    cmd = base_command + ["--seed", str(seed)]
    print(f"Launching seed {seed}...")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    thread = threading.Thread(
        target=stream_output,
        args=(seed, process),
        daemon=True
    )
    thread.start()

    processes.append((seed, process))
    threads.append(thread)

    # 5-second stagger after each launch except the last one
    if i < len(seeds) - 1:
        time.sleep(5)

exit_codes = {}
for seed, process in processes:
    exit_codes[seed] = process.wait()

for thread in threads:
    thread.join()

print("\nAll runs completed.\n")
for seed in seeds:
    print(f"Seed {seed} finished with exit code {exit_codes[seed]}")

failed = [seed for seed, code in exit_codes.items() if code != 0]
if failed:
    print(f"\nThese seeds failed: {failed}")
    sys.exit(1)
else:
    print("\nAll seeds finished successfully.")