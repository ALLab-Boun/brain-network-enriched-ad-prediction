import subprocess
import threading
import time
import sys

seeds = [7] #[42, 123, 2021, 7, 99]

print("Starting experiments with different seeds in parallel...")

base_command = [
    # "python",
    # r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
    r"C:\Users\efeka\Documents\thesis_colab_match\Scripts\python.exe"
    "exp4_temporal.py",

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

    "--epochs", "40",
    "--lr", "5e-5",
    "--batch_size", "32",
    "--weight_decay", "0.05",
    
    # Temporal settings
    "--temporal_type", "rnn",
    "--dropout", "0.0",

    # ----------------------------------------
    # ------ Morphometric Model Configs ------
    # ----------------------------------------

    # Morphometric MLP 
    # LR 5e-5, Tuning best epoch at 15, clf dropout 0.0
    "--include_cortex_mlp",
    "--cortex_mlp_dropout", "0.4",
    "--cortex_mlp_hidden_dim", "64",
    "--cortex_mlp_use_residual",
    "--cortex_mlp_activation", "leakyrelu",
    # "cortex_mlp_use_layernorm": false,
    "--cortex_mlp_num_layers", "1",
    # "cortex_mlp_hidden_dims" null,
    "--cortex_mlp_width_mode", "constant",

    # Morphometric GNN
    # LR 5e-5, Tuning best epoch at 18, clf dropout 0.2
    # "--include_gnn",
    # "--gnn_dropout", "0.2",
    # "--gnn_hidden_dim", "64",
    # "--edge_threshold", "1.0",
    # "--gnn_num_layers", "1",
    # "--gnn_layer", "gcn",
    # "--gnn_use_pre_mlp",
    # "--gnn_cnn_input_add_flattened_node_features",
    # "--gnn_add_output_skip",
    # "--gnn_layer_connectivity", "skipsum",
    # "--gnn_norm_type", "layernorm",
    # "--gnn_readout", "cnn",

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

    # Adjacency CNN
    "--include_cnn",
    "--adj_cnn_dropout", "0.5",
    "--adj_cnn_conv_channels", "32","128","256",
    "--adj_cnn_kernel_sizes", "3", "3", "3",
    "--adj_cnn_strides", "2", "2", "1", 
    "--adj_cnn_pool_types", "max", "max", "avg",
    "--adj_cnn_pool_kernel_sizes", "4", "4", "3",
    "--adj_cnn_negative_slope", "0.01",
    "--adj_cnn_norm_type", "group",
    "--adj_cnn_group_norm_groups", "8",
    "--adj_cnn_readout", "flatten",


    # Adjacency GNN
    "--include_gnn",
    "--gnn_dropout", "0.2",
    "--gnn_hidden_dim", "64",
    "--edge_threshold", "1.0",
    "--gnn_num_layers", "2",
    "--gnn_layer", "gcn",
    # "--gnn_use_pre_mlp", "false",
    "--gnn_cnn_input_add_flattened_node_features",
    "--gnn_add_output_skip",
    "--gnn_layer_connectivity", "skipsum",
    "--gnn_norm_type", "layernorm",
    "--gnn_readout", "cnn",
    "--gnn_graph_pool", "mean_max",
    "--add_adj_row_as_node_feature",
    "--separate_adj_features_instead_of_concat",


    # Adjacency Transformer 
    "--include_transformer",
    "--cort_transformer_dropout", "0.4",
    "--cortex_transformer_hidden_dim", "256",
    "--cortex_transformer_num_layers", "2",
    "--cortex_transformer_num_heads", "1",
    "--cortex_transformer_cnn_input_add_flattened_node_features",
    "--cortex_transformer_add_output_skip", 
    "--pos_encoding_type", "learnable",
    "--add_adj_row_as_node_feature",
    "--separate_adj_features_instead_of_concat",

    # ----------------------------------------
    # ----------------------------------------
    # ----------------------------------------

    # ****************************************

    # ----------------------------------------
    # ------ Cognitive Model Configs ---------
    # ----------------------------------------
    "--include_cog_mlp",
    "--cog_hidden_dim", "128",
    "--cog_mlp_dropout", "0.0",
    "--cog_mlp_width_mode", "shrink",
    "--cog_mlp_num_layers", "3",
    # "--cog_mlp_use_residual_to_last", "false",


    # ----------------------------------------
    # ----------------------------------------
    # ----------------------------------------

    # ****************************************

    # ----------------------------------------
    # ------ Early Stopping Configs ---------
    # ----------------------------------------
    "--early_stopping",
    "--es_monitor", "es_f1_weighted",
    "--es_mode", "max",
    "--es_patience", "20",
    "--es_min_delta", "0.0025",

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