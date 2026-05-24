# General imports
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse, random, datetime, json, math, copy
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# ML imports
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

# Local imports
from exp4_model import FusionModel

import utils.observe as observe
import utils.general as general
import utils.preprocessing as preprocessing
import utils.plotting as plotting

# seeding function
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_model_kwargs_from_args(args, cog_in_dim):
    cortex_gnn_kwargs = {
        "dropout": args.cortex_gnn_dropout,
        "hidden_dim": args.cortex_gnn_hidden_dim,
        "use_pre_mlp": args.cortex_gnn_use_pre_mlp,
        "cnn_input_add_flattened_node_features": args.cortex_gnn_cnn_input_add_flattened_node_features,
        "add_output_skip": args.cortex_gnn_add_output_skip,
        "layer_connectivity": args.cortex_gnn_layer_connectivity,
        "norm_type": args.cortex_gnn_norm_type,
        "num_layers": args.cortex_gnn_num_layers,
        "layer": args.cortex_gnn_layer,
        "readout": args.cortex_gnn_readout,
        "graph_pool": args.cortex_gnn_graph_pool,
    }

    adjacency_gnn_kwargs = {
        "dropout": args.adjacency_gnn_dropout,
        "hidden_dim": args.adjacency_gnn_hidden_dim,
        "use_pre_mlp": args.adjacency_gnn_use_pre_mlp,
        "cnn_input_add_flattened_node_features": args.adjacency_gnn_cnn_input_add_flattened_node_features,
        "add_output_skip": args.adjacency_gnn_add_output_skip,
        "layer_connectivity": args.adjacency_gnn_layer_connectivity,
        "norm_type": args.adjacency_gnn_norm_type,
        "num_layers": args.adjacency_gnn_num_layers,
        "layer": args.adjacency_gnn_layer,
        "readout": args.adjacency_gnn_readout,
        "graph_pool": args.adjacency_gnn_graph_pool,
    }

    cortex_mlp_kwargs = {
        "hidden_dim": args.cortex_mlp_hidden_dim,
        "use_residual": args.cortex_mlp_use_residual,
        "activation": args.cortex_mlp_activation,
        "use_layernorm": args.cortex_mlp_use_layernorm,
        "num_layers": args.cortex_mlp_num_layers,
        "hidden_dims": args.cortex_mlp_hidden_dims,
        "width_mode": args.cortex_mlp_width_mode,
        "dropout": args.cortex_mlp_dropout,
    }

    cog_mlp_kwargs = {
        "hidden_dim": args.cog_hidden_dim,
        "num_layers": args.cog_mlp_num_layers,
        "width_mode": args.cog_mlp_width_mode,
        "use_residual_to_last": args.cog_mlp_use_residual_to_last,
        "dropout": args.cog_mlp_dropout,
        "cog_in_dim": cog_in_dim,
    }

    adjacency_cnn_kwargs = {
        "dropout": args.adjacency_cnn_dropout,
        "conv_channels": args.adjacency_cnn_conv_channels,
        "kernel_sizes": args.adjacency_cnn_kernel_sizes,
        "strides": args.adjacency_cnn_strides,
        "pool_types": args.adjacency_cnn_pool_types,
        "pool_kernel_sizes": args.adjacency_cnn_pool_kernel_sizes,
        "negative_slope": args.adjacency_cnn_negative_slope,
        "norm_type": args.adjacency_cnn_norm_type,
        "group_norm_groups": args.adjacency_cnn_group_norm_groups,
        "readout": args.adjacency_cnn_readout,
    }


    cortex_transformer_kwargs = {
    "dropout": args.cortex_transformer_dropout,
    "pos_encoding_type": args.pos_encoding_type,
    "lpe_dim": args.lpe_dim,
    "hidden_dim": args.cortex_transformer_hidden_dim,
    "num_layers": args.cortex_transformer_num_layers,
    "num_heads": args.cortex_transformer_num_heads,
    "cnn_input_add_flattened_node_features": args.cortex_transformer_cnn_input_add_flattened_node_features,
    "add_output_skip": args.cortex_transformer_add_output_skip,
    }

    adjacency_transformer_kwargs = {
        "dropout": args.adjacency_transformer_dropout,
        "pos_encoding_type": args.pos_encoding_type,
        "lpe_dim": args.lpe_dim,
        "hidden_dim": args.adjacency_transformer_hidden_dim,
        "num_layers": args.adjacency_transformer_num_layers,
        "num_heads": args.adjacency_transformer_num_heads,
        "cnn_input_add_flattened_node_features": args.adjacency_transformer_cnn_input_add_flattened_node_features,
        "add_output_skip": args.adjacency_transformer_add_output_skip,
    }

    return {
        "cortex_gnn_kwargs": cortex_gnn_kwargs,
        "adjacency_gnn_kwargs": adjacency_gnn_kwargs,
        "cortex_mlp_kwargs": cortex_mlp_kwargs,
        "cog_mlp_kwargs": cog_mlp_kwargs,
        "adjacency_cnn_kwargs": adjacency_cnn_kwargs,
        "cortex_transformer_kwargs": cortex_transformer_kwargs,
        "adjacency_transformer_kwargs": adjacency_transformer_kwargs,
    }


# Main
def main(args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATASET_PATH = args.dataset_path
    CROSS_VAL_PKL_PATH = args.cross_val_pkl
    FEATURE_SLICES = preprocessing.get_feature_slices(args.excluded_node_features)
    
    use_es = args.early_stopping  

    # Load and preprocess data
    # if dataset path is a pt file, use load_dataset_from_single_pt
    data_list = general.load_dataset_from_single_pt(DATASET_PATH, convert_labels=False if args.dataset == "oasis" else True) if DATASET_PATH.endswith(".pt") else None
    # if data_list is None:
    #     print("Loading dataset ...")
    #     data_list = general.load_dataset(DATASET_PATH)
    #     print(f"Loaded {len(data_list)} graphs.")

    # Sanity check
    num_nodes = data_list[0].x.shape[0]
    print(f"Each graph has {num_nodes} nodes and {data_list[0].x.shape[1]} node features.")
    #print the values of the first node features for the first graph to check they look reasonable
    print("First node features of the first graph:", data_list[0].x[0])


    # Early stopping data filenames
    # For ADNI, we use a fixed set of early stopping subjects/visits based on the combined tuning splits.
    # For OASIS, 1 fold of the 5-fold CV is used as the early stopping fold.
    early_stopping_data_list_names = None
    if use_es and args.dataset == "adni":
        with open("./data/adni/splits/combined_tuning_filenames.json", "r") as f:
            early_stopping_data_list_names = json.load(f)

    # Load CV splits
    splits = general.read_cross_val(CROSS_VAL_PKL_PATH)
    print(f"Loaded {len(splits)} cross-validation splits.")

    conv_visit_map = {}
    if args.dataset in ["adni", "oasis"]:
        if args.dataset == "adni":
            conv_df = pd.read_excel("./metadata_tables/adni_labels_internal_dataset_plus_last_visit.xlsx")
            ptid_col = "PTID"
            viscode_col = "VISCODE"
        else:
            conv_df = pd.read_excel("./metadata_tables/oasis_dataset_labels.xlsx")
            ptid_col = "OASISID"
            viscode_col = "scan_day"
        
        conv_df[ptid_col] = conv_df[ptid_col].astype(str).str.strip()
        conv_df[viscode_col] = conv_df[viscode_col].astype(str).str.strip()

        if args.task == "diagnosis":
            conv_df["IS_CONV_VISIT"] = conv_df["CURRENT_IS_CONV_VISIT"]  
        elif args.task == "next_diagnosis":
            conv_df["IS_CONV_VISIT"] = conv_df["NEXT_IS_CONV_VISIT"]
        elif args.task == "long_term_conversion":
            # this task does not use visit-wise conversion labels, but we want to keep the column for consistency 
            conv_df["IS_CONV_VISIT"] = -1
        # fill nans with -1 to indicate missing conversion label for that visit
        conv_df["IS_CONV_VISIT"] = conv_df["IS_CONV_VISIT"].fillna(-1).astype(int)
        
        conv_visit_map = {
            (getattr(row, ptid_col), getattr(row, viscode_col)): int(row.IS_CONV_VISIT)
            for row in conv_df.itertuples(index=False)
        }
        next_label_map = {
            (getattr(row, ptid_col), getattr(row, viscode_col)): int(row.NEXT_LABEL) if not pd.isna(row.NEXT_LABEL) else -99
            for row in conv_df.itertuples(index=False)
        }

        print(
            "Loaded IS_CONV_VISIT labels:",
            sum(conv_visit_map.values()),
            "conversion visits out of",
            len(conv_visit_map),
            "rows"
        )
        for data in data_list:
            if args.dataset == "adni":
                ptid = str(data.ptid).strip()
                viscode = str(data.viscode).strip()
            else:
                ptid = str(data.oasis_id).strip()
                viscode = str(data.scan_day).strip()

            flag = conv_visit_map.get((ptid, viscode), -1)
            # Store as tensor so PyG Batch can collate it cleanly
            data.is_conv_visit = torch.tensor(flag, dtype=torch.long)

        # next_diagnosis and long_term_conversion tasks require changing the labels in data.y
        if args.task == "next_diagnosis":
            # change data.y to be the NEXT_LABEL from conv_df
            for data in data_list:
                if args.dataset == "adni":
                    ptid = str(data.ptid).strip()
                    viscode = str(data.viscode).strip()
                else:
                    ptid = str(data.oasis_id).strip()
                    viscode = str(data.scan_day).strip()

                next_label = next_label_map.get((ptid, viscode), -99)
                if args.dataset == "adni": # labels should be shifted from 1/2 to 0/1 for MCI/AD in ADNI, but OASIS is already 0/1 for non-dementia/dementia
                    next_label = next_label - 1 if next_label != -99 else -99  # convert from 1/2 (mci/ad) to 0/1
                data.y = torch.tensor(next_label, dtype=torch.long)
            
            # drop the data objects where we don't have a next diagnosis label 
            # data.y is -99 in this case
            data_list = [data for data in data_list if data.y != -99]
        elif args.task == "long_term_conversion":
            if args.dataset == "adni":
                long_term_conv_table = pd.read_excel("./metadata_tables/adni_progression_table.xlsx")
                long_term_label_map = {
                    (str(row['ptid']).strip(), str(row['viscode']).strip()): row['Progression 24m']
                    for _, row in long_term_conv_table.iterrows()
                }
            elif args.dataset == "oasis":
                long_term_label_map = {}
                pass # TODO
            
            # change data.y to be the "Progression 24m" from long_term_conv_table
            for data in data_list:
                if args.dataset == "adni":
                    ptid = str(data.ptid).strip()
                    viscode = str(data.viscode).strip()
                else:
                    ptid = str(data.oasis_id).strip()
                    viscode = str(data.scan_day).strip()

                prog_label = long_term_label_map.get((ptid, viscode), -99)
                if not pd.isna(prog_label) and prog_label != -99:
                    data.y = torch.tensor(int(prog_label), dtype=torch.long)
                else:
                    data.y = torch.tensor(-99, dtype=torch.long)
            
            # We limit to mci visits for this task: smci/pmci classification
            # AD (y = 2) and CN (y = -1) visits are not relevant for this task, and we also drop any visits where the long-term progression label is missing (y = -99)
            data_list = [data for data in data_list if data.y != -99 and data.y != 2 and data.y != -1]  # keep if y=1 (pmci), drop if y=-99 (missing) or y=0 (cn)

    # PREPROCCESSING STEPS:
    data_list, cog_in_dim, vol_sum_index = preprocessing.preprocess_global_data_list(
        data_list=data_list,
        dataset_path=DATASET_PATH,
        args=args,
        feature_slices=FEATURE_SLICES
    )

    # Build filename --> data map (cross-val splits are defined in terms of filenames, so we need this to quickly get the data objects for each split)
    if args.dataset == "adni":
        filename_to_data = {data.ptid + "_" + data.viscode + ".pt": data for data in data_list}
    elif args.dataset == "oasis": 
        filename_to_data = {data.oasis_id + "_" + data.scan_day + ".pt": data for data in data_list}
    
    results = pd.DataFrame(columns=[
    "FOLD",
    "ACC",
    "BALANCED_ACC",
    "F1_macro",
    "F1_weighted",
    "PRECISION_CLASS_0",
    "RECALL_CLASS_0",
    "PRECISION_CLASS_1",
    "RECALL_CLASS_1",
    "AUC",
    "AUPRC",
    "CONVERSION_RECALL"
    ])
    all_true, all_pred = [], []
    all_train_losses, all_test_losses, all_epoch_metrics = [], [], []

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_dir is not None:
        results_dir = args.run_dir + f"/{now}_seed{args.seed}"
    else:
        results_dir = f"./drive/MyDrive/thesis_gnn_results/mind_graph_exps/{now+str(args.seed)}_fusion_model"
    os.makedirs(results_dir, exist_ok=True)

    # encoder_stats_path = os.path.join(results_dir, "encoder_representation_stats.txt")

    all_prediction_records = []
    observe.print_cv_class_distributions(
        splits=splits,
        filename_to_data=filename_to_data,
        out_path=os.path.join(results_dir, "class_distributions.txt"),
        label_attr="y"
    )

    # early stopping config
    if use_es:
        monitor = args.es_monitor
        mode = args.es_mode
        patience = args.es_patience
        min_delta = args.es_min_delta
    
    # Cross-validation
    for fold, split in enumerate(splits):
        print(f"\n=== Fold {fold+1}/{len(splits)} ===")

        # if val_files exist as a key in split, use them
        # val files exist due to the previous convention, they are not actually used for validation
        # The actual validation for early stopping is done on the separate early_stopping_data which is 
        # loaded from EARLY_STOPPING_DATA_PATH and processed together with train and test data
        if args.dataset == "adni":
            if "val_files" in split:
                train_files = split["train_files"] + split["val_files"]
            else:
                train_files = split["train_files"]
            test_files = split["test_files"]
            train_data = [copy.deepcopy(filename_to_data[f]) for f in train_files if f in filename_to_data]
            test_data  = [copy.deepcopy(filename_to_data[f]) for f in test_files if f in filename_to_data]
            early_stopping_data = None
            if use_es:
                early_stopping_data = [copy.deepcopy(filename_to_data[f]) for f in early_stopping_data_list_names if f in filename_to_data]
                print(f"Train size: {len(train_data)}, Test size: {len(test_data)}, Early stopping size: {len(early_stopping_data)}")
            else:
                print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        elif args.dataset == "oasis":
            train_files = split["train_files"]
            test_files = split["test_files"]
            train_data = [copy.deepcopy(filename_to_data[f]) for f in train_files if f in filename_to_data]
            test_data  = [copy.deepcopy(filename_to_data[f]) for f in test_files if f in filename_to_data]
            print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
            early_stopping_data = None  
            if use_es:
                # for OASIS, we can use the val_files from the split as early stopping data since we don't have a separate tuning set defined
                early_stopping_files = split["val_files"] 
                early_stopping_data = [copy.deepcopy(filename_to_data[f]) for f in early_stopping_files if f in filename_to_data]
                print(f"Early stopping size: {len(early_stopping_data)}")

        # Check that there are no subject overlaps between train, test, and early stopping sets
        train_subjects = set(f.rsplit('_',1)[0] for f in train_files if f in filename_to_data)
        test_subjects = set(f.rsplit('_',1)[0] for f in test_files if f in filename_to_data)
        assert len(train_subjects.intersection(test_subjects)) == 0, "Overlap between train and test sets!"
        
        if use_es:
            es_files = early_stopping_files if args.dataset == "oasis" else early_stopping_data_list_names
            es_subjects = set(f.rsplit('_',1)[0] for f in es_files if f in filename_to_data)
            assert len(train_subjects.intersection(es_subjects)) == 0, "Overlap between train and early stopping sets!"
            assert len(test_subjects.intersection(es_subjects)) == 0, "Overlap between test and early stopping sets!"

        # set seed for determinism
        fold_seed = args.seed + fold
        seed_all(fold_seed)        


        if args.fusion == "concat":
            print("Using concatenation-based fusion model.")

            branch_kwargs = build_model_kwargs_from_args(args, cog_in_dim)

            model = FusionModel(
                num_nodes=num_nodes,
                node_in_dim=train_data[0].x.shape[1],
                num_classes=2,
                dropout=args.dropout,

                
                include_cortex_gnn=args.include_cortex_gnn,
                include_adjacency_gnn=args.include_adjacency_gnn,
                include_adjacency_cnn=args.include_adjacency_cnn,
                include_cortex_mlp=args.include_cortex_mlp,
                include_cog_mlp=args.include_cog_mlp,
                include_cortex_transformer=args.include_cortex_transformer,
                include_adjacency_transformer=args.include_adjacency_transformer,
                include_linear_x_logits=args.include_linear_x_logits,
                separate_adj_features_instead_of_concat=args.separate_adj_features_instead_of_concat,

                **branch_kwargs,
            ).to(device)
        
        # preprocess cognitive features
        train_data, cog_scaler, cog_mean = preprocessing.preprocess_cognitive_features_train(train_data)
        test_data = preprocessing.preprocess_cognitive_features_test(test_data, cog_scaler, cog_mean)
        if use_es:
            early_stopping_data = preprocessing.preprocess_cognitive_features_test(early_stopping_data, cog_scaler, cog_mean)

        # ICV normalization: fit on training data, apply to both train and test
        if vol_sum_index is not None:
            print("Performing ICV normalization on 'vol' features")
            icv_params = preprocessing.fit_icv_normalizer(train_data, feature_indices=[vol_sum_index], icv_attr="ICV")
            train_data = preprocessing.apply_icv_normalizer(train_data, icv_params)
            test_data  = preprocessing.apply_icv_normalizer(test_data,  icv_params)
            if use_es:
                early_stopping_data = preprocessing.apply_icv_normalizer(early_stopping_data, icv_params)

        # preprocess branch features on training, get the scalers
        train_data, mri_node_scalers = preprocessing.preprocess_mri_node_features(train_data)
        test_data = preprocessing.apply_mri_node_scalers(test_data, mri_node_scalers)
        if use_es:
            early_stopping_data = preprocessing.apply_mri_node_scalers(early_stopping_data, mri_node_scalers)

        # Adjacency scaling (avoided since mind degrees are already 0-1)
        # if args.include_adjacency_cnn or args.include_adjacency_gnn or args.include_adjacency_transformer:
        #     train_data, adjacency_scaler = preprocessing.preprocess_adjacency_matrix_features_train(train_data)
        #     test_data = preprocessing.preprocess_adjacency_matrix_features_test(test_data, adjacency_scaler)

        #     if use_es:
        #         early_stopping_data = preprocessing.preprocess_adjacency_matrix_features_test(
        #             early_stopping_data,
        #             adjacency_scaler
        #         )

        print("Shape of node features:", train_data[0].x.shape)

        # seed for loaders
        g = torch.Generator()
        g.manual_seed(fold_seed)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=0, drop_last=True)
        test_loader  = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if use_es:
            early_stopping_loader = DataLoader(early_stopping_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        observation_train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=0) 

        # Class weights
        train_y = torch.tensor([int(d.y.item()) for d in train_data])
        cw = compute_class_weight("balanced", classes=np.unique(train_y), y=train_y.numpy())
        unique, counts = np.unique(train_y.numpy(), return_counts=True)
        class_counts = dict(zip(unique, counts))
        print("Class distribution in training set:", class_counts)
        test_y = torch.tensor([int(d.y.item()) for d in test_data])
        unique, counts = np.unique(test_y.numpy(), return_counts=True)
        class_counts = dict(zip(unique, counts))
        print("Class distribution in test set:", class_counts)
        
        if args.use_class_weights:
            print("Using class weights in loss function.")
            print("Class weights:", cw)
            criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=device))
        else:
            criterion = torch.nn.CrossEntropyLoss()


        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, decoupled_weight_decay=True)
        print("Using optimizer:", optimizer.__repr__(), "with weight_decay:", args.weight_decay)

        best_score = None
        improved_fn = None
        best_epoch = 0
        bad_epochs = 0
        best_state = None

        if use_es:
            if mode == "min":
                best_score = math.inf
                improved_fn = lambda current, best: (best - current) > min_delta
            else:
                best_score = -math.inf
                improved_fn = lambda current, best: (current - best) > min_delta
        
        # best_ckpt_path = os.path.join(results_dir, f"fold{fold+1}_best.pt")
        # Training
        fold_train_losses, fold_test_losses, epoch_metrics = [], [], []

        # Epoch 0 (initialization) logging 
        # ES placeholders 
        es_loss = es_acc = es_f1_weighted = es_f1_macro = es_auc = None
        es_precision = [None, None]
        es_recall = [None, None]
        es_conv_recall = None
        init_current = None
        # Evaluate before any training step
        init_test_loss, init_test_acc, init_f1_weighted, init_f1_macro, init_precision, init_recall, init_auc, init_conv_recall = \
            general.evaluate(model, test_loader, device, criterion=criterion)

        init_tr_loss, init_tr_acc, init_tr_f1_weighted, init_tr_f1_macro, init_tr_precision, init_tr_recall, init_tr_auc, init_tr_conv_recall = \
            general.evaluate(model, observation_train_loader, device, criterion=criterion)

        if use_es:
            es_loss, es_acc, es_f1_weighted, es_f1_macro, es_precision, es_recall, es_auc, es_conv_recall = \
                general.evaluate(model, early_stopping_loader, device, criterion=criterion)

            init_es_metrics = {
                "es_loss": es_loss,
                "es_acc": es_acc,
                "es_f1_weighted": es_f1_weighted,
                "es_f1_macro": es_f1_macro,
                "es_auc": es_auc,
            }
            init_current = init_es_metrics[monitor]

            # allow epoch0 to be candidate best
            if improved_fn(init_current, best_score):
                best_score = init_current
                best_epoch = 0
                bad_epochs = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs = 0

        # If you want loss curves to start at epoch 0:
        fold_train_losses, fold_test_losses, epoch_metrics = [], [], []
        fold_train_losses.append(init_tr_loss)
        fold_test_losses.append(init_test_loss)

        # Save epoch 0 metrics row
        epoch_metrics.append({
            "epoch": 0,
            "train_loss": init_tr_loss,
            "train_acc": init_tr_acc,
            "train_f1_weighted": init_tr_f1_weighted,
            "train_f1_macro": init_tr_f1_macro,
            "train_class0_precision": init_tr_precision[0],
            "train_class1_precision": init_tr_precision[1],
            "train_class0_recall": init_tr_recall[0],
            "train_class1_recall": init_tr_recall[1],
            "train_auc": init_tr_auc,
            "train_conv_recall": init_tr_conv_recall,

            "test_loss": init_test_loss,
            "test_acc": init_test_acc,
            "test_f1_weighted": init_f1_weighted,
            "test_f1_macro": init_f1_macro,
            "test_class0_precision": init_precision[0],
            "test_class1_precision": init_precision[1],
            "test_class0_recall": init_recall[0],
            "test_class1_recall": init_recall[1],
            "test_auc": init_auc,
            "test_conv_recall": init_conv_recall,

            "es_loss": es_loss,
            "es_acc": es_acc,
            "es_f1_weighted": es_f1_weighted,
            "es_f1_macro": es_f1_macro,
            "es_class0_precision": es_precision[0],
            "es_class1_precision": es_precision[1],
            "es_class0_recall": es_recall[0],
            "es_class1_recall": es_recall[1],
            "es_auc": es_auc,
            "es_conv_recall": es_conv_recall,
        })

        msg = (
            f"Epoch 000 | "
            f"TrLoss {init_tr_loss:.4f} | TrAcc {init_tr_acc:.3f} | "
            f"TestLoss {init_test_loss:.4f} | TestAcc {init_test_acc:.3f}"
        )
        if use_es:
            msg += f" | ES {monitor}={init_current:.4f} (best={best_score:.4f} @ {best_epoch})"
        print(msg)


        for epoch in range(1, args.epochs + 1):
            tr_loss = general.train_one_epoch(model, train_loader, optimizer, device, criterion=criterion)

            test_loss, test_acc, f1_weighted, f1_macro, precision, recall, auc, conv_recall= general.evaluate(model, test_loader, device,criterion=criterion)
            tr_after_epoch_loss, tr_acc, tr_f1_weighted, tr_f1_macro, tr_precision, tr_recall, tr_auc, tr_conv_recall= general.evaluate(model, observation_train_loader, device,criterion=criterion)

            # early stopping evaluation
            if use_es:
                es_loss, es_acc, es_f1_weighted, es_f1_macro, es_precision, es_recall, es_auc, es_conv_recall = general.evaluate(model, early_stopping_loader, device, criterion=criterion)
                es_metrics = {"es_loss": es_loss, "es_acc": es_acc, "es_f1_weighted": es_f1_weighted, "es_f1_macro": es_f1_macro, "es_auc": es_auc,}
                current = es_metrics[args.es_monitor]
            else:
                current = None

            fold_train_losses.append(tr_after_epoch_loss)
            fold_test_losses.append(test_loss)

            if use_es and improved_fn(current, best_score):
                best_score = current
                best_epoch = epoch
                bad_epochs = 0
                best_state = copy.deepcopy(model.state_dict())
            elif use_es:
                bad_epochs += 1

            if epoch % 1 == 0 or epoch == args.epochs:
                msg = (
                    f"Epoch {epoch:03d} | Train {tr_loss:.4f} | "
                    f"Tr_Acc {tr_acc:.3f} | Tr_F1w {tr_f1_weighted:.3f} | "
                    f"Test {test_loss:.4f} | Acc {test_acc:.3f} | F1w {f1_weighted:.3f}"
                )
                if use_es:
                    msg += f" | ES {monitor}={current:.4f} (best={best_score:.4f} @ {best_epoch})"
                    msg += f" | bad_epochs={bad_epochs}/{patience}"
                print(msg)

            # Save epoch metrics for potential later analysis
            epoch_metrics.append({
                "epoch": epoch,
                "train_loss": tr_after_epoch_loss,
                "train_acc": tr_acc,
                "train_f1_weighted": tr_f1_weighted,
                "train_f1_macro": tr_f1_macro,
                "train_class0_precision": tr_precision[0] ,
                "train_class1_precision": tr_precision[1] ,
                "train_class0_recall": tr_recall[0] ,
                "train_class1_recall": tr_recall[1],
                "train_auc": tr_auc,
                "train_conv_recall": tr_conv_recall,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_f1_weighted": f1_weighted,
                "test_f1_macro": f1_macro,
                "test_class0_precision": precision[0],
                "test_class1_precision": precision[1],
                "test_class0_recall": recall[0],
                "test_class1_recall": recall[1],
                "test_auc": auc,
                "test_conv_recall": conv_recall,

                # early stopping metrics
                "es_loss": es_loss,
                "es_acc": es_acc,
                "es_f1_weighted": es_f1_weighted,
                "es_f1_macro": es_f1_macro,
                "es_class0_precision": es_precision[0],
                "es_class1_precision": es_precision[1],
                "es_class0_recall": es_recall[0],
                "es_class1_recall": es_recall[1],
                "es_auc": es_auc,
                "es_conv_recall": es_conv_recall,
            })

            if use_es and bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best {monitor}={best_score:.4f} at epoch {best_epoch}).")
                break
        
        if use_es and best_state is not None:
            model.load_state_dict(best_state)
        else:
            pass
            
        # Save the model weights for this fold
        model_save_path = os.path.join(results_dir, f"fold{fold+1}_model_weights.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved fold {fold+1} model weights to {model_save_path}")

        all_train_losses.append(fold_train_losses)
        all_test_losses.append(fold_test_losses)
        all_epoch_metrics.append(epoch_metrics)

        # Final evaluation per fold
        test_loss, test_acc, f1_weighted, f1_macro, precision, recall, auc, conv_recall= general.evaluate(model, test_loader, device,criterion=criterion)
        print(f"Final Test Metrics for Fold {fold+1}: Loss {test_loss:.4f} | Acc {test_acc:.3f} | F1w {f1_weighted:.3f} | F1m {f1_macro:.3f} | Precision {precision} | Recall {recall} | AUC {auc:.3f} | Conv_Recall {conv_recall:.3f}")
        y_true, y_pred, y_prob_class_1 = [], [], []
        conv_true_count = 0
        conv_pred_positive_count = 0
        model.eval()
        for data in test_loader:
            data = data.to(device)
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            if hasattr(data, "is_conv_visit"):
                conv_flags = data.is_conv_visit

                if not torch.is_tensor(conv_flags):
                    conv_flags = torch.tensor(conv_flags, device=device)

                conv_flags = conv_flags.to(device).long()

                conv_mask = conv_flags == 1

                conv_true_count += int(conv_mask.sum().item())
                conv_pred_positive_count += int((preds[conv_mask] == 1).sum().item())
            else:
                pass # Already handled by other prints or irrelevant for dataset

            for i in range(data.num_graphs):
                if args.dataset == "adni":
                    ptid = data.ptid[i] if isinstance(data.ptid, list) else data.ptid
                    viscode = data.viscode[i] if isinstance(data.viscode, list) else data.viscode
                    label = int(data.y[i].cpu().item())
                    status = data.status[i] # MCI, MCI to Dementia, Dementia etc.
                    is_conv_visit = int(data.is_conv_visit[i].cpu().item())
                elif args.dataset == "oasis":
                    ptid = data.oasis_id[i] if isinstance(data.oasis_id, list) else data.oasis_id
                    viscode = data.scan_day[i] if isinstance(data.scan_day, list) else data.scan_day
                    label = int(data.y[i].cpu().item())
                    status = f"CDRTOT_{data.CDRTOT[i].cpu().item()}"  # Consider CDRTOT as the status for OASIS
                    is_conv_visit = int(data.is_conv_visit[i].cpu().item())
                
                prediction = int(preds[i].cpu().item())
                prob_mci = float(probs[i, 0].cpu().item())
                prob_ad  = float(probs[i, 1].cpu().item())

                record = {
                    "fold": fold + 1,
                    "ptid": ptid,
                    "viscode": viscode,
                    "label": label,
                    "status": status,
                    "prediction": prediction,
                    "prob_mci": prob_mci,
                    "prob_ad":  prob_ad,
                    "is_conv_visit": is_conv_visit,
                }
                all_prediction_records.append(record)
            
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob_class_1.extend(probs[:, 1].detach().cpu().numpy())


        all_true.extend(y_true)
        all_pred.extend(y_pred)

        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0
        )
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        try:
            final_auc = roc_auc_score(y_true, y_prob_class_1)
        except ValueError:
            final_auc = float("nan")

        try:
            final_auprc = average_precision_score(y_true, y_prob_class_1)
        except ValueError:
            final_auprc = float("nan")

        final_conv_recall = float(conv_pred_positive_count) / conv_true_count if conv_true_count > 0 else float("nan")
        results.loc[fold] = [
            fold + 1,
            test_acc,
            balanced_acc,
            f1_score(y_true, y_pred, average="macro"), 
            f1_score(y_true, y_pred, average="weighted"), 
            precision[0],
            recall[0],
            precision[1],
            recall[1],
            final_auc,
            final_auprc,
            final_conv_recall
        ]

    # Save logs
    prediction_df = pd.DataFrame(all_prediction_records)
    prediction_df.to_excel(os.path.join(results_dir, "all_predictions.xlsx"), index=False)

    os.makedirs(os.path.join(results_dir, "training_logs_plots"), exist_ok=True)
    # 
    if len(all_epoch_metrics) ==1:
        epoch_metrics_df = pd.DataFrame(all_epoch_metrics[0])
        epoch_metrics_df.to_csv(os.path.join(results_dir, "training_logs_plots", "epoch_metrics.csv"), index=False)
    else:
        per_fold_epoch_dfs = []

        for i, fold_metrics in enumerate(all_epoch_metrics):
            fold_metrics_df = pd.DataFrame(fold_metrics)
            fold_metrics_df.to_csv(
                os.path.join(results_dir, "training_logs_plots", f"fold{i+1}_epoch_metrics.csv"),
                index=False
            )

            fold_metrics_df = fold_metrics_df.copy()
            fold_metrics_df["fold"] = i + 1
            per_fold_epoch_dfs.append(fold_metrics_df)

        # Only compute average/std per epoch when early stopping is NOT used
        if not use_es:
            combined_epoch_df = pd.concat(per_fold_epoch_dfs, ignore_index=True)

            # numeric metric columns except identifiers
            metric_cols = [
                c for c in combined_epoch_df.columns
                if c not in ["epoch", "fold"]
                and pd.api.types.is_numeric_dtype(combined_epoch_df[c])
            ]

            avg_epoch_metrics = (
                combined_epoch_df
                .groupby("epoch")[metric_cols]
                .agg(["mean", "std"])
                .reset_index()
            )

            # flatten MultiIndex columns:
            # ('train_loss', 'mean') -> 'train_loss_mean'
            # ('train_loss', 'std')  -> 'train_loss_std'
            avg_epoch_metrics.columns = [
                "epoch" if col == ("epoch", "") else f"{col[0]}_{col[1]}"
                for col in avg_epoch_metrics.columns
            ]

            avg_epoch_metrics.to_csv(
                os.path.join(results_dir, "training_logs_plots", "average_epoch_metrics.csv"),
                index=False
            )

    results.to_csv(os.path.join(results_dir, "fusion_results.csv"), index=False)
    means = results.drop(columns=["FOLD"]).mean()
    stds = results.drop(columns=["FOLD"]).std()
    summary = pd.DataFrame({"Mean": means, "Std": stds})
    summary.to_csv(os.path.join(results_dir, "fusion_mean_std_results.csv"))

    # Plot losses
    # Decide how to save based on number of folds
    if len(all_epoch_metrics) == 1:
        # Single fold -> one figure
        plotting.plot_fold_curves(
            all_epoch_metrics[0],
            out_path=os.path.join(results_dir,"training_logs_plots", "fold1_curves.png"),
            fold_idx=1
        )
    else:
        # Multi-fold -> one per fold + optionally a combined overview (saved as separate images)
        for i, fold_metrics in enumerate(all_epoch_metrics):
            plotting.plot_fold_curves(
                fold_metrics,
                out_path=os.path.join(results_dir, "training_logs_plots", f"fold{i+1}_curves.png"),
                fold_idx=i + 1
            )


    val_key = "es_loss" if use_es else None
    plotting.plot_all_folds_losses(
        all_epoch_metrics,
        out_path=os.path.join(results_dir, "loss_curves.png"),
        validation_loss_key=val_key
    )

    args_dict = vars(args)
    config_path = os.path.join(results_dir, "hyperparams.json")
    with open(config_path, "w") as f:
        json.dump(args_dict, f, indent=4)

    # save the model summary
    observe.save_model_summary(model, os.path.join(results_dir, "model_summary.txt"))

    # save the dataset and crossval paths used
    with open(os.path.join(results_dir, "data_paths.txt"), "w") as f:
        f.write(f"DATASET_PATH: {DATASET_PATH}\n")
        f.write(f"CROSS_VAL_PKL_PATH: {CROSS_VAL_PKL_PATH}\n")
        f.write(f"Node feature set: {args.node_feature_set}\n")

    # save the loss curves data
    loss_curves_data = {
        "train_losses": all_train_losses,
        "test_losses": all_test_losses
    }
    with open(os.path.join(results_dir, "loss_curves_data.json"), "w") as f:
        json.dump(loss_curves_data, f, indent=4)

    print(f"Results saved to {results_dir}")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion model (GNN + 2D CNN + MLP) with cross-validation")
    parser.add_argument("--base_folder", type=str, default=".")
    parser.add_argument("--dataset_path", type=str, default=r"C:\Users\efeka\Documents\MIND_graphs\ADNI\MIND_graphs_CT_Vol\CT_Vol_graphs_complete_features_filtered_negative\pyg\CT_Vol_graphs_complete_features_filtered_negative.pt")
    parser.add_argument("--cross_val_pkl", type=str, default=r"C:\dev\GitHub\MIND\colab_data\cv_tuning_val_974_split\split_by_prog_category_9_7_4_seed93\cv\cross_val_splits_5fold_10perc_early_stop.pkl")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, choices=["adni", "oasis"], default="adni")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default = 0.5) # classifier head dropout
    parser.add_argument("--fusion", type=str, choices=["attention", "concat"], default="concat")
    parser.add_argument("--task", type=str, choices=["diagnosis", "next_diagnosis", "long_term_conversion"], default="diagnosis")
    parser.add_argument("--lt_conversion_window", type=int, default=24, help="Time window in months for defining long-term conversion (only relevant if task is 'long_term_conversion')")

    # Branch inclusion args
    parser.add_argument("--include_cortex_mlp", action="store_true")
    parser.add_argument("--include_cortex_gnn", action="store_true")
    parser.add_argument("--include_cortex_transformer", action="store_true")
    parser.add_argument("--include_adjacency_cnn", action="store_true")
    parser.add_argument("--include_adjacency_gnn", action="store_true")
    parser.add_argument("--include_adjacency_transformer", action="store_true")
    parser.add_argument("--include_cog_mlp", action="store_true")
    parser.add_argument("--include_linear_x_logits", action="store_true")

    parser.add_argument("--edge_threshold", type=float, default=1.0)
    parser.add_argument("--add_adj_row_as_node_feature", action="store_true")
    parser.add_argument("--separate_adj_features_instead_of_concat", action="store_true")
    parser.add_argument("--add_weighted_degree_as_node_feature", action="store_true")


    # Cortex MLP
    parser.add_argument("--cortex_mlp_dropout", type=float, default=0.5)
    parser.add_argument("--cortex_mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--cortex_mlp_use_residual", action="store_true")
    parser.add_argument("--cortex_mlp_activation", type=str, choices=["relu", "gelu", "elu", "leakyrelu"], default="leakyrelu")
    parser.add_argument("--cortex_mlp_use_layernorm", action="store_true")
    parser.add_argument("--cortex_mlp_num_layers", type=int, default=1)
    parser.add_argument("--cortex_mlp_hidden_dims", type=int, nargs="+", default=None)
    parser.add_argument("--cortex_mlp_width_mode", type=str, default="constant" )

    # Cortex GNN
    parser.add_argument("--cortex_gnn_dropout", type=float, default=0.2)
    parser.add_argument("--cortex_gnn_hidden_dim", type=int, default=64)
    parser.add_argument("--cortex_gnn_num_layers", type=int, default=1)
    parser.add_argument("--cortex_gnn_layer", type=str, default="gcn")
    parser.add_argument("--cortex_gnn_readout", type=str, default="cnn")
    parser.add_argument("--cortex_gnn_graph_pool", type=str, default="mean_max")
    parser.add_argument("--cortex_gnn_norm_type", type=str, default="layernorm")
    parser.add_argument("--cortex_gnn_use_pre_mlp", action="store_true")
    parser.add_argument("--cortex_gnn_cnn_input_add_flattened_node_features", action="store_true")
    parser.add_argument("--cortex_gnn_add_output_skip", action="store_true")
    parser.add_argument("--cortex_gnn_layer_connectivity", type=str, default="stack")

    # Cortex Transformer
    parser.add_argument("--cortex_transformer_dropout", type=float, default=0.5)
    parser.add_argument("--cortex_transformer_hidden_dim", type=int, default=128)
    parser.add_argument("--cortex_transformer_num_layers", type=int, default=2)
    parser.add_argument("--cortex_transformer_num_heads", type=int, default=4)
    parser.add_argument("--cortex_transformer_cnn_input_add_flattened_node_features", action="store_true")
    parser.add_argument("--cortex_transformer_add_output_skip", action="store_true")

    # Adjacency Transformer
    parser.add_argument("--adjacency_transformer_dropout", type=float, default=0.5)
    parser.add_argument("--adjacency_transformer_hidden_dim", type=int, default=128)
    parser.add_argument("--adjacency_transformer_num_layers", type=int, default=2)
    parser.add_argument("--adjacency_transformer_num_heads", type=int, default=4)
    parser.add_argument("--adjacency_transformer_cnn_input_add_flattened_node_features", action="store_true")
    parser.add_argument("--adjacency_transformer_add_output_skip", action="store_true")


    # Adjacency CNN
    parser.add_argument("--adjacency_cnn_dropout", type=float, default=0.5)
    parser.add_argument("--adjacency_cnn_conv_channels", type=int, nargs="+", default=[32, 256, 2048])
    parser.add_argument("--adjacency_cnn_kernel_sizes", type=int, nargs="+", default=[7, 5, 3])
    parser.add_argument("--adjacency_cnn_strides", type=int, nargs="+", default=[2, 2, 1])
    parser.add_argument("--adjacency_cnn_pool_types", type=str, nargs="+", default=["max", "max", "avg"])
    parser.add_argument("--adjacency_cnn_pool_kernel_sizes", type=int, nargs="+", default=[4, 4, 4])
    parser.add_argument("--adjacency_cnn_negative_slope", type=float, default=0.01)
    parser.add_argument("--adjacency_cnn_norm_type", type=str, default=None)
    parser.add_argument("--adjacency_cnn_group_norm_groups", type=int, default=8)
    parser.add_argument("--adjacency_cnn_readout", type=str, choices=["flatten", "gap", "gmp", "gap_gmp"], default="flatten")

    # Adjacency GNN
    parser.add_argument("--adjacency_gnn_dropout", type=float, default=0.3)
    parser.add_argument("--adjacency_gnn_hidden_dim", type=int, default=64)
    parser.add_argument("--adjacency_gnn_num_layers", type=int, default=2)
    parser.add_argument("--adjacency_gnn_layer", type=str, default="gcn")
    parser.add_argument("--adjacency_gnn_readout", type=str, default="cnn")
    parser.add_argument("--adjacency_gnn_graph_pool", type=str, default="mean")
    parser.add_argument("--adjacency_gnn_norm_type", type=str, default="layernorm")
    parser.add_argument("--adjacency_gnn_use_pre_mlp", action="store_true")
    parser.add_argument("--adjacency_gnn_cnn_input_add_flattened_node_features", action="store_true")
    parser.add_argument("--adjacency_gnn_add_output_skip", action="store_true")
    parser.add_argument("--adjacency_gnn_layer_connectivity", type=str, default="stack")

    # Cognitive MLP
    parser.add_argument("--cog_hidden_dim", type=int, default=128)
    parser.add_argument("--cog_mlp_dropout", type=float, default=0.5)
    parser.add_argument("--cog_mlp_width_mode", type=str, default="constant")
    parser.add_argument("--cog_mlp_num_layers", type=int, default=2)
    parser.add_argument("--cog_mlp_use_residual_to_last", action="store_true")

    # positional encoding (used in transformer branch, and optionally can be added to GNN node features as well if adapted)
    parser.add_argument("--pos_encoding_type", type=str, choices=["none", "sinusoidal", "learnable", "lpe"], default="learnable")
    parser.add_argument("--add_laplacian_pe", action="store_true")
    parser.add_argument("--lpe_dim", type=int, default=8)

    # other model configs and hyperparams
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--balanced_batches", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    # Feature set configs
    parser.add_argument("--node_feature_set", type=str, default="ct_vol_sa_mc_sd")
    parser.add_argument("--excluded_node_features", choices=[None, "min_max", "std_min_max"], default="std_min_max")
    parser.add_argument("--cog_feature_set", type=str, choices=["all", "no_adas"], default="all")

    # Early Stopping
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--es_monitor", type=str, default="es_f1_weighted",
                        choices=["es_loss", "es_f1_weighted", "es_f1_macro", "es_acc", "es_auc"])
    parser.add_argument("--es_patience", type=int, default=10)
    parser.add_argument("--es_min_delta", type=float, default=1e-4)
    parser.add_argument("--es_mode", type=str, default="max", choices=["min", "max"])  # "min" for loss, "max" for F1/AUC/acc

    args = parser.parse_args()
    main(args, seed=args.seed)