# General imports
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse, random, datetime, json, math, copy
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# ML imports
from sklearn.metrics import f1_score, precision_recall_fscore_support
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

# Main
def main(args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE_FOLDER = args.base_folder
    DATASET_PATH = args.dataset_path
    CROSS_VAL_PKL_PATH = args.cross_val_pkl
    use_es = args.early_stopping  


    if args.excluded_node_features == None:
        FEATURE_SLICES = {
        "ct":  slice(0, 4),
        "vol": slice(4, 8),
        "sa":  slice(8, 12),
        "mc":  slice(12, 16),
        "sd":  slice(16, 20),
        }
    elif args.excluded_node_features == "min_max":
        FEATURE_SLICES = {
        "ct":  slice(0, 2),
        "vol": slice(4, 6),
        "sa":  slice(8, 10),
        "mc":  slice(12, 14),
        "sd":  slice(16, 18),
    }
    elif args.excluded_node_features == "std_min_max":
        FEATURE_SLICES = {
        "ct":  slice(0, 1),
        "vol": slice(4, 5),
        "sa":  slice(8, 9),
        "mc":  slice(12, 13),
        "sd":  slice(16, 17),
    }
    else:        
        raise ValueError(f"Unknown option for --excluded_node_features: {args.excluded_node_features}")
        

    
    # Load and preprocess data
    # if dataset path is a pt file, use load_dataset_from_single_pt
    data_list = general.load_dataset_from_single_pt(DATASET_PATH) if DATASET_PATH.endswith(".pt") else None
    if data_list is None:
        print("Loading dataset ...")
        data_list = general.load_dataset(DATASET_PATH)
        print(f"Loaded {len(data_list)} graphs.")
    num_nodes = data_list[0].x.shape[0]
    print(f"Each graph has {num_nodes} nodes and {data_list[0].x.shape[1]} node features.")


    # early stopping data
    # list the filenames in the early stopping data path
    early_stopping_data_list_names = None
    if use_es:
        with open("./data/adni/splits/combined_tuning_filenames.json", "r") as f:
            early_stopping_data_list_names = json.load(f)

    # Load CV splits
    splits = general.read_cross_val(CROSS_VAL_PKL_PATH)
    print(f"Loaded {len(splits)} cross-validation splits.")

    # Filter data_list to only include samples in the CV splits
    # used_filenames = get_used_filenames_from_splits(splits, include_val_in_train=True)
    # data_list = filter_data_list_by_splits(data_list, used_filenames, dataset=args.dataset)
    
    # PREPROCCESSING STEPS:
    data_list, cog_in_dim, vol_sum_index = preprocessing.preprocess_global_data_list(
        data_list=data_list,
        dataset_path=DATASET_PATH,
        args=args,
        feature_slices=FEATURE_SLICES
    )

    # Build filename --> data map
    if args.dataset == "adni":
        filename_to_data = {data.ptid + "_" + data.viscode + ".pt": data for data in data_list}
    elif args.dataset == "oasis": # TODO: FIX !!!
        filename_to_data = {data.ptid + "_" + data.viscode + ".pt": data for data in data_list}


    # Results containers
    results = pd.DataFrame(columns=[
        "FOLD", "ACC", "F1_macro", "F1_weighted",
        "PRECISION_CLASS_0", "RECALL_CLASS_0",
        "PRECISION_CLASS_1", "RECALL_CLASS_1"
    ])
    all_true, all_pred = [], []
    all_train_losses, all_test_losses, all_epoch_metrics = [], [], []


    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_dir is not None:
        results_dir = args.run_dir
    else:
        results_dir = f"./drive/MyDrive/thesis_gnn_results/mind_graph_exps/{now+str(args.seed)}_fusion_model"
    os.makedirs(results_dir, exist_ok=True)

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
        if "val_files" in split:
            train_files = split["train_files"] + split["val_files"]
        else:
            train_files = split["train_files"]
        test_files = split["test_files"]

        train_data = [copy.deepcopy(filename_to_data[f]) for f in train_files if f in filename_to_data]
        test_data  = [copy.deepcopy(filename_to_data[f]) for f in test_files if f in filename_to_data]

        # set seed for determinism
        fold_seed = args.seed + fold
        seed_all(fold_seed)        

        if args.fusion == "concat":
            print("Using concatenation-based fusion model.")
            model = FusionModel(
                num_nodes=num_nodes,
                node_in_dim=train_data[0].x.shape[1],
                num_classes=2,
                include_cnn=args.include_cnn,
                include_mlp=args.include_cortex_mlp,
                include_cog_mlp=args.include_cog_mlp,
                include_transformer=args.include_transformer,
                cortex_mlp_hidden_dim=args.cortex_mlp_hidden_dim,

                # GNN
                include_gnn=args.include_gnn,
                gnn_dropout=args.gnn_dropout,
                gnn_hidden_dim=args.gnn_hidden_dim,
                gnn_use_pre_mlp=args.gnn_use_pre_mlp,
                gnn_cnn_input_add_flattened_node_features=args.gnn_cnn_input_add_flattened_node_features,
                gnn_add_output_skip=args.gnn_add_output_skip,
                gnn_layer_connectivity=args.gnn_layer_connectivity,
                gnn_norm_type=args.gnn_norm_type,
                gnn_num_layers=args.gnn_num_layers,
                gnn_layer=args.gnn_layer,


                # Cortex MLP
                cortex_mlp_use_residual=args.cortex_mlp_use_residual,
                cortex_mlp_activation = args.cortex_mlp_activation,
                cortex_mlp_use_layernorm = args.cortex_mlp_use_layernorm,
                cortex_mlp_num_layers = args.cortex_mlp_num_layers,
                cortex_mlp_hidden_dims=args.cortex_mlp_hidden_dims,
                cortex_mlp_width_mode=args.cortex_mlp_width_mode,
                cortex_mlp_dropout=args.cortex_mlp_dropout,

                # cognitive MLP
                cog_hidden_dim=args.cog_hidden_dim,
                cog_mlp_num_layers = args.cog_mlp_num_layers,
                cog_mlp_width_mode=args.cog_mlp_width_mode,
                cog_mlp_use_residual_to_last=args.cog_mlp_use_residual_to_last,
                cog_mlp_dropout=args.cog_mlp_dropout,
                cog_in_dim=cog_in_dim,


                # other fusion and general hyperparameters
                dropout=args.dropout,
                adj_cnn_dropout=args.adj_cnn_dropout,
                adj_cnn_conv_channels=args.adj_cnn_conv_channels,
                adj_cnn_kernel_sizes=args.adj_cnn_kernel_sizes,
                adj_cnn_strides=args.adj_cnn_strides,
                adj_cnn_pool_types=args.adj_cnn_pool_types,
                adj_cnn_pool_kernel_sizes=args.adj_cnn_pool_kernel_sizes,
                adj_cnn_negative_slope=args.adj_cnn_negative_slope,
                adj_cnn_norm_type=args.adj_cnn_norm_type,
                adj_cnn_group_norm_groups=args.adj_cnn_group_norm_groups,
                adj_cnn_readout = args.adj_cnn_readout,
                cort_transformer_dropout=args.cort_transformer_dropout,
                pos_encoding_type=args.pos_encoding_type,
                lpe_dim=args.lpe_dim,
                transformer_hidden_dim=args.transformer_hidden_dim,
                separate_adj_features_instead_of_concat=args.separate_adj_features_instead_of_concat,
            ).to(device)

        early_stopping_data = None
        if use_es:
            early_stopping_data = [copy.deepcopy(filename_to_data[f]) for f in early_stopping_data_list_names if f in filename_to_data]
            print(f"Train size: {len(train_data)}, Test size: {len(test_data)}, Early stopping size: {len(early_stopping_data)}")
        else:
            print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        
        # preprocess cognitive features
        if args.dataset == "adni":
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
        print("Using optimizer:", optimizer, "with weight_decay:", args.weight_decay)

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

        # ---- Epoch 0 (initialization) logging ----

        # ES placeholders (so you can always append a row without NameError)
        es_loss = es_acc = es_f1_weighted = es_f1_macro = es_auc = None
        es_precision = [None, None]
        es_recall = [None, None]
        es_conv_recall = None
        init_current = None
        # Evaluate BEFORE any training step
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

        # Optional: let early stopping consider epoch 0 as a candidate best
        if use_es and improved_fn(init_current, best_score):
            best_score = init_current
            best_epoch = 0
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        elif use_es:
            # Usually keep bad_epochs=0 at init; don't penalize before training
            bad_epochs = 0

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
            # tr_after_epoch_loss, tr_acc, tr_f1_weighted, tr_f1_macro, tr_precision, tr_recall, tr_auc, tr_conv_recall= evaluate(model, train_loader, device,criterion=criterion)

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
            # if early stopping enabled but never improved, you might still want to load best_ckpt_path if it exists
            pass
        all_train_losses.append(fold_train_losses)
        all_test_losses.append(fold_test_losses)
        all_epoch_metrics.append(epoch_metrics)

        # Final evaluation per fold
        test_loss, test_acc, f1_weighted, f1_macro, precision, recall, auc, conv_recall= general.evaluate(model, test_loader, device,criterion=criterion)
        print(f"Final Test Metrics for Fold {fold+1}: Loss {test_loss:.4f} | Acc {test_acc:.3f} | F1w {f1_weighted:.3f} | F1m {f1_macro:.3f} | Precision {precision} | Recall {recall} | AUC {auc:.3f} | Conv_Recall {conv_recall:.3f}")
        y_true, y_pred, y_pred_raw = [], [], []
        model.eval()
        for data in test_loader:
            data = data.to(device)
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            for i in range(data.num_graphs):
                ptid = data.ptid[i] if isinstance(data.ptid, list) else data.ptid
                viscode = data.viscode[i] if isinstance(data.viscode, list) else data.viscode
                label = int(data.y[i].cpu().item())
                status = data.status[i]

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
                }
                all_prediction_records.append(record)
            
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())


        all_true.extend(y_true)
        all_pred.extend(y_pred)

        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0
        )

        results.loc[fold] = [
            fold + 1,
            test_acc,
            f1_score(y_true, y_pred, average="macro"), 
            f1_score(y_true, y_pred, average="weighted"), 
            precision[0],
            recall[0],
            precision[1],
            recall[1],
        ]
        # if args.fusion == "attention":
        #     # Save attention weights
        #     y_true, y_pred, attn_df = evaluate_with_attention(model, test_loader, device)
            
        #     attn_df.to_csv(os.path.join(results_dir, f"fold{fold+1}_attention_weights.csv"), index=False)

    # Save logs

    prediction_df = pd.DataFrame(all_prediction_records)
    prediction_df.to_excel(os.path.join(results_dir, "validation_predictions.xlsx"), index=False)

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

    # --- Conversion recall per fold ---
    conv_recall_per_fold = (
        prediction_df[prediction_df["status"] == "MCI to Dementia"]
            .groupby("fold")["prediction"]
            .apply(lambda s: (s == 1).mean())
            .rename("Conversion_Recall")
    )
    # --- Count predicted positives / negatives per fold ---
    pred_counts_per_fold = (
        prediction_df[prediction_df["status"] == "MCI to Dementia"]
            .groupby("fold")["prediction"]
            .agg(
                Num_Predicted_Positive=lambda s: (s == 1).sum(),
                Num_Predicted_Negative=lambda s: (s == 0).sum()
            )
    )
    conv_recall_summary = pd.concat(
        [conv_recall_per_fold, pred_counts_per_fold],
        axis=1
    )
    conv_recall_summary.to_csv(
        os.path.join(results_dir, "conversion_recall_per_fold.csv")
    )

    conv_recall_mean = conv_recall_per_fold.mean()
    conv_recall_std  = conv_recall_per_fold.std()
    results.to_csv(os.path.join(results_dir, "fusion_results.csv"), index=False)
    means = results.drop(columns=["FOLD"]).mean()
    stds = results.drop(columns=["FOLD"]).std()
    summary = pd.DataFrame({"Mean": means, "Std": stds})
        # append the rows for conversion recall
    summary.loc["Conversion_Recall"] = [conv_recall_mean, conv_recall_std]
    summary.to_csv(os.path.join(results_dir, "fusion_mean_std_results.csv"))

    # Plot losses
    # Decide how to save based on number of folds
    # if len(all_epoch_metrics) == 1:
    #     # Single fold -> one figure
    #     plotting.plot_fold_curves(
    #         all_epoch_metrics[0],
    #         out_path=os.path.join(results_dir,"training_logs_plots", "fold1_curves.png"),
    #         fold_idx=1
    #     )
    # else:
    #     # Multi-fold -> one per fold + optionally a combined overview (saved as separate images)
    #     for i, fold_metrics in enumerate(all_epoch_metrics):
    #         plotting.plot_fold_curves(
    #             fold_metrics,
    #             out_path=os.path.join(results_dir, "training_logs_plots", f"fold{i+1}_curves.png"),
    #             fold_idx=i + 1
    #         )


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
    print(f"Model summary saved to {os.path.join(results_dir, 'model_summary.txt')}")

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
    # parser.add_argument("--base_folder", type=str, default="./cv_tuning_val_974_split")
    parser.add_argument("--base_folder", type=str, default=r"C:\dev\GitHub\MIND\colab_data\cv_tuning_val_974_split")
    parser.add_argument("--dataset_path", type=str, default=r"C:\Users\efeka\Documents\MIND_graphs\ADNI\MIND_graphs_CT_Vol\CT_Vol_graphs_complete_features_filtered_negative\pyg\CT_Vol_graphs_complete_features_filtered_negative.pt")
    parser.add_argument("--cross_val_pkl", type=str, default=r"C:\dev\GitHub\MIND\colab_data\cv_tuning_val_974_split\split_by_prog_category_9_7_4_seed93\cv\cross_val_splits_5fold_10perc_early_stop.pkl")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, choices=["adni", "oasis"], default="adni")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default = 0.5) # classifier head dropout
    parser.add_argument("--include_cnn", action="store_true")
    parser.add_argument("--include_transformer", action="store_true")
    parser.add_argument("--fusion", type=str, choices=["attention", "concat"], default="concat")
    parser.add_argument("--task", type=str, choices=["diagnosis", "conversion"], default="diagnosis")

    # GNN
    parser.add_argument("--include_gnn", action="store_true")
    parser.add_argument("--gnn_dropout", type=float, default=0.5)
    parser.add_argument("--gnn_hidden_dim", type=int, default=256)
    parser.add_argument("--edge_threshold", type=float, default=1.0)
    parser.add_argument("--gnn_num_layers", type=int, default=2)
    parser.add_argument("--gnn_layer", type=str, choices=["gcn", "sage", "gatv2", "gin"], default="gcn") # "gcn" | "sage" | "gatv2" | "gin"
    parser.add_argument("--add_adj_row_as_node_feature", action="store_true")
    parser.add_argument("--separate_adj_features_instead_of_concat", action="store_true")
    parser.add_argument("--add_weighted_degree_as_node_feature", action="store_true")
    parser.add_argument("--gnn_use_pre_mlp", action="store_true")
    parser.add_argument("--gnn_cnn_input_add_flattened_node_features", action="store_true")
    parser.add_argument("--gnn_add_output_skip", action="store_true")
    parser.add_argument("--gnn_layer_connectivity", type=str, choices=["stack", "skipcat", "skipsum"], default="skipsum")
    parser.add_argument("--gnn_norm_type", type=str, default="layernorm")


    # Cortex MLP
    parser.add_argument("--include_cortex_mlp", action="store_true")
    parser.add_argument("--cortex_mlp_dropout", type=float, default=0.5)
    parser.add_argument("--cortex_mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--cortex_mlp_use_residual", action="store_true")
    parser.add_argument("--cortex_mlp_activation", type=str, choices=["relu", "gelu", "elu", "leakyrelu"], default="leakyrelu")
    parser.add_argument("--cortex_mlp_use_layernorm", action="store_true")
    parser.add_argument("--cortex_mlp_num_layers", type=int, default=3)
    parser.add_argument("--cortex_mlp_hidden_dims", type=int, nargs="+", default=None)
    parser.add_argument("--cortex_mlp_width_mode", type=str, default="constant" )

    # Adjacency CNN
    parser.add_argument("--adj_cnn_dropout", type=float, default=0.5)
    parser.add_argument("--adj_cnn_conv_channels", type=int, nargs="+", default=[32, 256, 2048])
    parser.add_argument("--adj_cnn_kernel_sizes", type=int, nargs="+", default=[7, 5, 3])
    parser.add_argument("--adj_cnn_strides", type=int, nargs="+", default=[2, 2, 1])
    parser.add_argument("--adj_cnn_pool_types", type=str, nargs="+", default=["max", "max", "avg"])
    parser.add_argument("--adj_cnn_pool_kernel_sizes", type=int, nargs="+", default=[4, 4, 4])
    parser.add_argument("--adj_cnn_negative_slope", type=float, default=0.01)
    parser.add_argument("--adj_cnn_norm_type", type=str, default=None)
    parser.add_argument("--adj_cnn_group_norm_groups", type=int, default=8)
    parser.add_argument("--adj_cnn_readout", type=str, choices=["flatten", "gap", "gmp", "gap_gmp"], default="flatten")   

    # Cortex Transformer
    parser.add_argument("--cort_transformer_dropout", type=float, default=0.5)
    parser.add_argument("--transformer_hidden_dim", type=int, default=128)

    # Cognitive MLP
    parser.add_argument("--include_cog_mlp", action="store_true")
    parser.add_argument("--cog_hidden_dim", type=int, default=128)
    parser.add_argument("--cog_mlp_dropout", type=float, default=0.5)
    parser.add_argument("--cog_mlp_width_mode", type=str, default="constant")
    parser.add_argument("--cog_mlp_num_layers", type=int, default=2)
    parser.add_argument("--cog_mlp_use_residual_to_last", action="store_true")


    # positional encoding
    parser.add_argument("--add_laplacian_pe", action="store_true")
    parser.add_argument("--pos_encoding_type", type=str, choices=["none", "sinusoidal", "learnable", "lpe"], default="sinusoidal")
    parser.add_argument("--lpe_dim", type=int, default=8)

    # other model configs and hyperparams
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--balanced_batches", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=1e-2)


    # Feature set configs
    parser.add_argument("--node_feature_set", type=str, default="ct_vol_sa_mc_sd")
    parser.add_argument("--excluded_node_features", choices=[None, "min_max", "std_min_max"], default=None)
    parser.add_argument("--cog_feature_set", type=str, choices=["all", "no_adas"], default="all")

    # Early Stopping
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--es_monitor", type=str, default="es_loss",
                        choices=["es_loss", "es_f1_weighted", "es_f1_macro", "es_acc", "es_auc"])
    parser.add_argument("--es_patience", type=int, default=10)
    parser.add_argument("--es_min_delta", type=float, default=1e-4)
    parser.add_argument("--es_mode", type=str, default="min", choices=["min", "max"])  # "min" for loss, "max" for F1/AUC/acc

    args = parser.parse_args()
    main(args, seed=args.seed)