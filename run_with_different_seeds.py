import subprocess

seeds =  [42] #[42, 123, 2021, 7, 99]

# ---------- ADNI ----------
print("Starting experiments with different seeds...")
base_command = [
    r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
    r"C:\dev\GitHub\MIND\colab_data\exp4_main_deterministic.py",

    "--base_folder", r"C:\dev\GitHub\MIND\colab_data\cv_tuning_val_974_split",
    "--dataset", "adni",

    # TUNING DATA single split
    # "--cross_val_pkl", r"C:\dev\GitHub\MIND\colab_data\cv_tuning_val_974_split\split_by_prog_category_9_7_4_seed93\tuning\cross_val_splits_1fold_tuning.pkl", 

    # 5 fold CV report data
    "--cross_val_pkl",r"C:\dev\GitHub\MIND\colab_data\data\adni\splits\reporting_cv_splits.pkl",
    
    # 5 fold CV tuning data 
    # "--cross_val_pkl", r"C:\dev\GitHub\MIND\colab_data\cv_tuning_val_974_split\split_by_prog_category_9_7_4_seed93\early_stopping\cross_val_splits_5fold_for_tuning.pkl",
    
    
    # ADJ CNN
    # "--include_cnn",
    # "--lr", "0.0001",
    # "--batch_size", "32",
    # "--dropout", "0.7",
    # "--adj_cnn_dropout", "0.7",
    # "--epochs", "23",

    # GNN
    # "--include_gnn",
    # "--lr", "5e-4",
    # "--batch_size", "64",
    # "--dropout", "0.7",
    # "--gnn_dropout","0.3",
    # "--edge_threshold", "0.2",
    # "--two_layer_gcn",
    # "--epochs", "4",
    # "--gnn_hidden_dim", "256",
    # "--weight_decay", "0.05",

    # CORT TRANSFORMER
    # "--epochs", "27",
    # "--lr", "0.0001",
    # "--seed", "7",
    # "--batch_size", "64",
    # "--dropout", "0.5",
    # "--cort_transformer_dropout", "0.5",
    # "--include_transformer",
    # "--fusion", "concat",
    # "--task", "diagnosis",
    # "--pos_encoding_type", "learnable",
    # "--lpe_dim", "8",
    # "--transformer_hidden_dim", "128",
    # "--weight_decay", "1e-4",
    # "--edge_threshold", "0.0",

    # CORT MLP
    # "--lr", "5e-5",
    # "--batch_size", "64",
    # "--dropout", "0.7",
    # "--cort_mlp_dropout", "0.3",
    # "--epochs", "22",
    # "--include_mlp",
    # "--cortex_mlp_hidden_dim", "128",
    # "--weight_decay", "0.00",

    # "--include_cog_mlp",
    # "--cog_mlp_dropout", "0.5",
    # "--cog_hidden_dim", "128",

    # "--add_adj_row_as_node_feature",
    # "--separate_adj_features_instead_of_concat",

    #  TESTING FOR EARLY STOPPING
    # "--include_gnn",
    # "--lr", "5e-4",
    # "--batch_size", "64",
    # "--dropout", "0.3",
    # "--gnn_dropout","0.3",
    # "--edge_threshold", "0.0",
    # "--epochs", "15",
    # "--gnn_hidden_dim", "64",
    # "--weight_decay", "0.05",
    # "--gnn_layer", "gcn",
    # "--gnn_use_pre_mlp",
    # "--gnn_cnn_input_add_flattened_node_features",
    # "--gnn_add_output_skip",
    # "--gnn_layer_connectivity", "skipcat",


    "--include_cog_mlp",
    "--cog_mlp_dropout", "0.5",
    "--cog_hidden_dim", "128",

    "--early_stopping",
    "--es_monitor", "es_f1_weighted",
    "--es_mode", "max",
    "--es_patience", "10",
    "--es_min_delta", "0.005",

    # "--exclude_min_max_node_features",

    # "--node_feature_set", "ct_vol", # base
    # # "--excluded_node_features", "std_min_max"

    # "--add_weighted_degree_as_node_feature",
    "--excluded_node_features", "std_min_max",
    "--node_feature_set", "ct_vol_sa_mc_sd"
]

for seed in seeds:
    print(f"\nRunning seed {seed}...\n")
    cmd = base_command + ["--seed", str(seed)]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in process.stdout:
        print(line, end="")

    process.wait()


# print("Starting experiments with different seeds...")
# print("Using CT_Vol_SA constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main.py",

#     "--base_folder", r"C:\dev\GitHub\MIND\colab_data\cv_tuning_val_974_split",
#     "--dataset", "adni",

#     "--epochs", "27",
#     "--lr", "0.0001",
#     "--seed", "7",
#     "--batch_size", "64",
#     "--dropout", "0.5",
#     "--include_transformer",
#     "--fusion", "concat",
#     "--task", "diagnosis",
#     "--pos_encoding_type", "learnable",
#     "--lpe_dim", "8",
#     "--transformer_hidden_dim", "128",
#     "--edge_threshold", "0.0",

#     "--node_feature_set", "ct_vol_sa"
# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()


# print("Starting experiments with different seeds...")
# print("Using CT_Vol_SA_MC_SD constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main.py",

#     "--base_folder", r"C:\dev\GitHub\MIND\colab_data\cv_tuning_val_974_split",
#     "--dataset", "adni",

#     "--epochs", "27",
#     "--lr", "0.0001",
#     "--seed", "7",
#     "--batch_size", "64",
#     "--dropout", "0.5",
#     "--include_transformer",
#     "--fusion", "concat",
#     "--task", "diagnosis",
#     "--pos_encoding_type", "learnable",
#     "--lpe_dim", "8",
#     "--transformer_hidden_dim", "128",
#     "--edge_threshold", "0.0",

#     "--node_feature_set", "ct_vol_sa_mc_sd"
# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()



########################
### ADJ CNN OASIS ###
#########################
# print("Starting experiments with different seeds...")
# print("Using CT constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main_oasis.py",
#     "--base_folder", r"C:\Users\efeka\Documents\oasis3\MIND_graphs\CT_complete\filtered_vertices\nx_graphs\pyg",
#     "--dataset", "oasis",

#     "--include_cnn",
#     "--lr", "0.0001",
#     "--batch_size", "64",
#     "--dropout", "0.5",
#     "--epochs", "72",
#     "--use_class_weights",


#     "--node_feature_set", "ct"

# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()


# print("Starting experiments with different seeds...")
# print("Using CT_Vol constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main_oasis.py",
#     "--base_folder", r"C:\Users\efeka\Documents\oasis3\MIND_graphs\CT_Vol_complete\filtered_vertices\nx_graphs\pyg",
#     "--dataset", "oasis",

#     "--include_cnn",
#     "--lr", "0.0001",
#     "--batch_size", "64",
#     "--dropout", "0.5",
#     "--epochs", "72",
#     # "--use_class_weights",
#     "--balanced_batches",


#     "--node_feature_set", "ct_vol"
# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()


# print("Starting experiments with different seeds...")
# print("Using CT_Vol_SA constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main_oasis.py",
#     "--base_folder", r"C:\Users\efeka\Documents\oasis3\MIND_graphs\CT_Vol_SA_complete\filtered_vertices\nx_graphs\pyg",
#     "--dataset", "oasis",

#     "--include_cnn",
#     "--lr", "0.0001",
#     "--batch_size", "64",
#     "--dropout", "0.5",
#     "--epochs", "72",
#     "--use_class_weights",



#     "--node_feature_set", "ct_vol_sa"
# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()


# print("Starting experiments with different seeds...")
# print("Using CT_Vol_SA_MC_SD constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main_oasis.py",
#     "--base_folder", r"C:\Users\efeka\Documents\oasis3\MIND_graphs\CT_Vol_SA_MC_SD_complete\filtered_vertices\nx_graphs\pyg",
#     "--dataset", "oasis",

#     "--include_cnn",
#     "--lr", "0.0001",
#     "--batch_size", "64",
#     "--dropout", "0.5",
#     "--epochs", "72",

#     "--use_class_weights",

#     "--node_feature_set", "ct_vol_sa_mc_sd"
# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()



# #########################
# ### GNN OASIS ###
# #########################
# print("Starting experiments with different seeds...")
# print("Using CT constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main_oasis.py",
#     "--base_folder", r"C:\Users\efeka\Documents\oasis3\MIND_graphs\CT_complete\filtered_vertices\nx_graphs\pyg",
#     "--dataset", "oasis",

#     "--include_gnn",
#     "--lr", "5e-4",
#     "--batch_size", "64",
#     "--dropout", "0.5",
#     "--edge_threshold", "0.2",
#     "--two_layer_gcn",
#     "--epochs", "7",
#     "--gnn_hidden_dim", "256",
#     "--use_class_weights",


#     "--node_feature_set", "ct"

# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()


# print("Starting experiments with different seeds...")
# print("Using CT_Vol constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main_oasis.py",
#     "--base_folder", r"C:\Users\efeka\Documents\oasis3\MIND_graphs\CT_Vol_complete\filtered_vertices\nx_graphs\pyg",
#     "--dataset", "oasis",

#     "--include_gnn",
#     "--lr", "5e-4",
#     "--batch_size", "64",
#     "--dropout", "0.5",
#     "--edge_threshold", "0.2",
#     "--two_layer_gcn",
#     "--epochs", "7",
#     "--gnn_hidden_dim", "256",
#     "--balanced_batches",

#     "--node_feature_set", "ct_vol"
# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()


# print("Starting experiments with different seeds...")
# print("Using CT_Vol_SA constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main_oasis.py",
#     "--base_folder", r"C:\Users\efeka\Documents\oasis3\MIND_graphs\CT_Vol_SA_complete\filtered_vertices\nx_graphs\pyg",
#     "--dataset", "oasis",

#     "--include_gnn",
#     "--lr", "5e-4",
#     "--batch_size", "64",
#     "--dropout", "0.5",
#     "--edge_threshold", "0.2",
#     "--two_layer_gcn",
#     "--epochs", "7",
#     "--gnn_hidden_dim", "256",
#     "--use_class_weights",


#     "--node_feature_set", "ct_vol_sa"
# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()


# print("Starting experiments with different seeds...")
# print("Using CT_Vol_SA_MC_SD constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main_oasis.py",
#     "--base_folder", r"C:\Users\efeka\Documents\oasis3\MIND_graphs\CT_Vol_SA_MC_SD_complete\filtered_vertices\nx_graphs\pyg",
#     "--dataset", "oasis",

#     "--include_gnn",
#     "--lr", "5e-4",
#     "--batch_size", "64",
#     "--dropout", "0.5",
#     "--edge_threshold", "0.2",
#     "--two_layer_gcn",
#     "--epochs", "7",
#     "--gnn_hidden_dim", "256",
#     "--use_class_weights",

#     "--node_feature_set", "ct_vol_sa_mc_sd"
# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()


# #########################
# ### CORT MLP OASIS ###
# #########################
# print("Starting experiments with different seeds...")
# print("Using CT constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main_oasis.py",
#     "--base_folder", r"C:\Users\efeka\Documents\oasis3\MIND_graphs\CT_complete\filtered_vertices\nx_graphs\pyg",
#     "--dataset", "oasis",

#     "--lr", "5e-4",
#     "--batch_size", "64",
#     "--dropout", "0.3",
#     "--epochs", "5",
#     "--include_mlp",
#     "--cortex_mlp_hidden_dim", "32",
#     "--use_class_weights",

#     "--node_feature_set", "ct"

# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()


# print("Starting experiments with different seeds...")
# print("Using CT_Vol constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main_oasis.py",
#     "--base_folder", r"C:\Users\efeka\Documents\oasis3\MIND_graphs\CT_Vol_complete\filtered_vertices\nx_graphs\pyg",
#     "--dataset", "oasis",

#     "--lr", "5e-4",
#     "--batch_size", "64",
#     "--dropout", "0.3",
#     "--epochs", "5",
#     "--include_mlp",
#     "--cortex_mlp_hidden_dim", "32",
#     "--balanced_batches",

#     "--node_feature_set", "ct_vol"
# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()


# print("Starting experiments with different seeds...")
# print("Using CT_Vol_SA constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main_oasis.py",
#     "--base_folder", r"C:\Users\efeka\Documents\oasis3\MIND_graphs\CT_Vol_SA_complete\filtered_vertices\nx_graphs\pyg",
#     "--dataset", "oasis",

#     "--lr", "5e-4",
#     "--batch_size", "64",
#     "--dropout", "0.3",
#     "--epochs", "5",
#     "--include_mlp",
#     "--cortex_mlp_hidden_dim", "32",
#     "--use_class_weights",

#     "--node_feature_set", "ct_vol_sa"
# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()


# print("Starting experiments with different seeds...")
# print("Using CT_Vol_SA_MC_SD constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main_oasis.py",
#     "--base_folder", r"C:\Users\efeka\Documents\oasis3\MIND_graphs\CT_Vol_SA_MC_SD_complete\filtered_vertices\nx_graphs\pyg",
#     "--dataset", "oasis",

#     "--lr", "5e-4",
#     "--batch_size", "64",
#     "--dropout", "0.3",
#     "--epochs", "5",
#     "--include_mlp",
#     "--cortex_mlp_hidden_dim", "32",
#     "--use_class_weights",

#     "--node_feature_set", "ct_vol_sa_mc_sd"
# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()


# #-------------------------------------
# #-------------------------------------
# #-------------------------------------
# #########################
# ### TRANSFORMER OASIS ###
# #########################
# print("Starting experiments with different seeds...")
# print("Using CT constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main_oasis.py",
#     "--base_folder", r"C:\Users\efeka\Documents\oasis3\MIND_graphs\CT_complete\filtered_vertices\nx_graphs\pyg",
#     "--dataset", "oasis",
#     "--epochs", "27",
#     "--lr", "0.0001",
#     "--seed", "7",
#     "--batch_size", "64",
#     "--dropout", "0.5",

#     "--include_transformer",
#     "--fusion", "concat",
#     "--task", "diagnosis",

#     "--pos_encoding_type", "learnable",
#     "--lpe_dim", "8",
#     "--transformer_hidden_dim", "128",

#     "--edge_threshold", "0.0",
#     "--use_class_weights",

#     "--node_feature_set", "ct"

# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()


# print("Starting experiments with different seeds...")
# print("Using CT_Vol constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main_oasis.py",
#     "--base_folder", r"C:\Users\efeka\Documents\oasis3\MIND_graphs\CT_Vol_complete\filtered_vertices\nx_graphs\pyg",
#     "--dataset", "oasis",
#     "--epochs", "27",
#     "--lr", "0.0001",
#     "--seed", "7",
#     "--batch_size", "64",
#     "--dropout", "0.5",

#     "--include_transformer",
#     "--fusion", "concat",
#     "--task", "diagnosis",

#     "--pos_encoding_type", "learnable",
#     "--lpe_dim", "8",
#     "--transformer_hidden_dim", "128",

#     "--edge_threshold", "0.0",
#     "--balanced_batches",

#     "--node_feature_set", "ct_vol"
# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()


# print("Starting experiments with different seeds...")
# print("Using CT_Vol_SA constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main_oasis.py",
#     "--base_folder", r"C:\Users\efeka\Documents\oasis3\MIND_graphs\CT_Vol_SA_complete\filtered_vertices\nx_graphs\pyg",
#     "--dataset", "oasis",
#     "--epochs", "27",
#     "--lr", "0.0001",
#     "--seed", "7",
#     "--batch_size", "64",
#     "--dropout", "0.5",

#     "--include_transformer",
#     "--fusion", "concat",
#     "--task", "diagnosis",

#     "--pos_encoding_type", "learnable",
#     "--lpe_dim", "8",
#     "--transformer_hidden_dim", "128",

#     "--edge_threshold", "0.0",
#     "--use_class_weights",

#     "--node_feature_set", "ct_vol_sa"
# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()


# print("Starting experiments with different seeds...")
# print("Using CT_Vol_SA_MC_SD constructed MIND graphs.")

# base_command = [
#     r"C:\dev\GitHub\MIND\mind_env\Scripts\python.exe",
#     r"C:\dev\GitHub\MIND\colab_data\exp4_main_oasis.py",
#     "--base_folder", r"C:\Users\efeka\Documents\oasis3\MIND_graphs\CT_Vol_SA_MC_SD_complete\filtered_vertices\nx_graphs\pyg",
#     "--dataset", "oasis",
#     "--epochs", "27",
#     "--lr", "0.0001",
#     "--seed", "7",
#     "--batch_size", "64",
#     "--dropout", "0.5",

#     "--include_transformer",
#     "--fusion", "concat",
#     "--task", "diagnosis",

#     "--pos_encoding_type", "learnable",
#     "--lpe_dim", "8",
#     "--transformer_hidden_dim", "128",

#     "--edge_threshold", "0.0",
#     "--use_class_weights",

#     "--node_feature_set", "ct_vol_sa_mc_sd"
# ]

# for seed in seeds:
#     print(f"\nRunning seed {seed}...\n")
#     cmd = base_command + ["--seed", str(seed)]
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()
#-------------------------------------
#-------------------------------------
#-------------------------------------