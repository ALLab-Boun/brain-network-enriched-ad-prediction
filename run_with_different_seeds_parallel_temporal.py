import subprocess
import threading
import time
import sys

seeds = [7] #[42, 123, 2021, 7, 99]

print("Starting experiments with different seeds in parallel...")

base_command = [
    # "python",
    r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
    "temporal_fusion.py",

    # ADNI
    "--dataset", "adni",
    "--dataset_path", "./data/adni/CT_Vol_graphs_complete_features_filtered_negative.pt",
    "--cross_val_pkl", "./data/adni/splits/reporting_cv_splits.pkl",

    # OASIS 
    # "--dataset", "oasis", 
    # "--dataset_path", "./data/oasis3/CTVOL_all_graphs_relabeled_6m_filtered_negative.pt",
    # "--cross_val_pkl", "./data/oasis3/splits/oasis_cv_1foldval.pkl",

    # Feature settings
    "--excluded_node_features", "std_min_max",
    "--node_feature_set", "ct_vol_sa_mc_sd",
    # "--add_adj_row_as_node_feature", # for adj models
    # "--separate_adj_features_instead_of_concat", # for adj models

    "--epochs", "30",
    "--lr", "5e-5",
    "--batch_size", "32",
    "--weight_decay", "0.05",
    
    # Temporal settings
    "--temporal_type", "rnn",
    "--dropout", "0.2",

    # ----------------------------------------
    # ------ Morphometric Model Configs ------
    # ----------------------------------------
    # Morphometric MLP 
    # LR 5e-5, Tuning best epoch at 15, clf dropout 0.0
    # "--include_cortex_mlp",
    # "--cortex_mlp_dropout", "0.4",
    # "--cortex_mlp_hidden_dim", "64",
    # "--cortex_mlp_use_residual",
    # "--cortex_mlp_activation", "leakyrelu",
    # # "cortex_mlp_use_layernorm": false,
    # "--cortex_mlp_num_layers", "1",
    # # "cortex_mlp_hidden_dims" null,
    # "--cortex_mlp_width_mode", "constant",

    # Morphometric GNN
    # LR 5e-5, Tuning best epoch at 18, clf dropout 0.2
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

    # Morphometric Transformer 
    # LR 5e-4, Tuning best epoch at 8, clf dropout 0.4
    # "--include_transformer",
    # "--cort_transformer_dropout", "0.4",
    # "--cortex_transformer_hidden_dim", "128",
    # "--cortex_transformer_num_layers", "3",
    # "--cortex_transformer_num_heads", "8",
    # "--cortex_transformer_cnn_input_add_flattened_node_features",
    ### "--cortex_transformer_add_output_skip", "false", # No output skip for transformer

    # ----------------------------------------
    # ----------------------------------------
    # ----------------------------------------

    # ****************************************
    
    # ----------------------------------------
    # ------ Adjacency Model Configs ---------
    # ----------------------------------------



    # ----------------------------------------
    # ----------------------------------------
    # ----------------------------------------

    # ****************************************

    # ----------------------------------------
    # ------ Cognitive Model Configs ---------
    # ----------------------------------------


    # ----------------------------------------
    # ----------------------------------------
    # ----------------------------------------

    # ****************************************

    # ----------------------------------------
    # ------ Early Stopping Configs ---------
    # ----------------------------------------
    # "--early_stopping",
    # "--es_monitor", "es_f1_weighted",
    # "--es_mode", "max",
    # "--es_patience", "10",
    # "--es_min_delta", "0.0025",

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