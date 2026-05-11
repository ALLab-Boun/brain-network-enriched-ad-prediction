import copy
import os
import pickle
import torch
import numpy as np
import pandas as pd
import json
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

from datetime import datetime

import utils.observe as observe
import utils.general as general
import utils.preprocessing as preprocessing
import utils.plotting as plotting

from sklearn.base import clone

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--random_seed", default=42,type=int, help="Random seed for reproducibility")
    args.add_argument("--task", default="classification", type=str, choices=["classification",  "next_diagnosis", "long_term_conversion"])
    args.add_argument("--is_tuning", action="store_true", help="Whether this is a hyperparameter tuning run")
    args.add_argument("--class_weights", action="store_true", help="Whether to use class weights in classifiers")
    args.add_argument("--dataset", default="adni", type=str, choices=["adni", "oasis"], help="Dataset name")
    args.add_argument("--cog_feature_set", type=str, choices=["all", "no_adas"], default="all")
    args.add_argument("--expected_nodes", type=int, default=68, help="Expected number of nodes in each graph (used for validation)")
    args.add_argument("--node_feature_set", type=str, default="ct_vol_sa_mc_sd", help="Which node features to include. Default is all Freesurfer features (cortical thickness, volume, surface area, mean curvature, sulcal depth).")
    args.add_argument("--excluded_node_features", choices=["mean_min_max", "min_max", "std_min_max"], default="std_min_max")
    args.add_argument("--graph_measures_type", type=str, choices=["mean", "full"])
    # run_dir, if not specified, will be created with a timestamp
    args.add_argument("--run_dir", type=str, default=None, help="Output directory for results. If not specified, a new directory will be created with a timestamp.")

    # PCA options
    args.add_argument("--apply_pca", action="store_true", help="Whether to apply PCA to cortical features")
    args.add_argument("--pca_components", type=float, default=0.8, help="Number of PCA components to keep (if --apply_pca is set). If <1, interpreted as variance ratio to preserve. If >=1, interpreted as number of components.")

    # Paths
    args.add_argument("--dataset_path", type=str, default=r"C:\dev\GitHub\graph-based-dementia-prediction\data\adni\CT_Vol_graphs_complete_features_filtered_negative.pt", help="Path to dataset (directory of .pt files or single .pt file)")
    args.add_argument("--cross_val_pkl_path", type=str, default=r"C:\dev\GitHub\graph-based-dementia-prediction\data\adni\splits\reporting_cv_splits.pkl", help="Path to cross-validation splits pickle file")
    
    # Which models to train/evaluate
    args.add_argument("--models", nargs="+",
    default=["Logistic_Regression", "Random_Forest", "SVM_RBF", "XGBoost"],
    choices=["Logistic_Regression", "Random_Forest", "SVM_RBF", "XGBoost"])

    # Logistic Regression hyperparameters
    args.add_argument("--lr_C", type=float, default=0.001)
    args.add_argument("--lr_max_iter", type=int, default=4000)
    args.add_argument("--lr_solver", type=str, default="liblinear")

    # Random Forest hyperparameters
    def int_or_none(x):
        if x.lower() == "none":
            return None
        return int(x)
    args.add_argument("--rf_n_estimators", type=int, default=800)
    args.add_argument("--rf_max_depth", type=int_or_none, default=30)

    # SVM RBF hyperparameters
    args.add_argument("--svm_C", type=float, default=1)
    args.add_argument("--svm_gamma", type=str, default=0.001)

    # XGBoost hyperparameters
    args.add_argument("--xgb_n_estimators", type=int, default=200)
    args.add_argument("--xgb_max_depth", type=int, default=7)
    args.add_argument("--xgb_learning_rate", type=float, default=0.03)
    args.add_argument("--xgb_subsample", type=float, default=0.8)
    args.add_argument("--xgb_colsample_bytree", type=float, default=1.0)
    args.add_argument("--xgb_reg_lambda", type=float, default=0)

    # Specify whether to include each feature type
    args.add_argument("--include_x", action="store_true", help="Whether to include Freesurfer cortical features (obtained from graph node data)")
    args.add_argument("--include_cog", action="store_true", help="Whether to include cognitive features")
    args.add_argument("--include_mri", action="store_true", help="Whether to include overall MRI features")
    args.add_argument("--include_ucsffsx", action="store_true", help="Whether to include UCSFFSX features")
    args.add_argument("--include_graph_measures", action="store_true", help="Whether to include graph measures (degree, clustering coefficient, betweenness centrality, eigenvector centrality)")
    args.add_argument("--include_adjacency", action="store_true",
                  help="Whether to include adjacency features using upper-triangle values")

    args.add_argument("--add_adj_row_as_node_feature", action="store_true")
    args.add_argument("--separate_adj_features_instead_of_concat", action="store_true")
    args.add_argument("--add_weighted_degree_as_node_feature", action="store_true")

    parsed_args = args.parse_args()

    # Output directory if timestamp is needed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    FEATURE_SLICES = preprocessing.get_feature_slices(parsed_args.excluded_node_features)
    DATASET_PATH = parsed_args.dataset_path
    CROSS_VAL_PKL_PATH = parsed_args.cross_val_pkl_path
    APPLY_PCA = parsed_args.apply_pca
    NUM_COMPONENTS = parsed_args.pca_components
    EXPECTED_NODES = parsed_args.expected_nodes

    # Specify whether to include each feature type
    include_x = True if not (
        parsed_args.include_x or parsed_args.include_cog or parsed_args.include_mri
        or parsed_args.include_ucsffsx or parsed_args.include_graph_measures
        or parsed_args.include_adjacency
    ) else parsed_args.include_x
    include_cog = parsed_args.include_cog # Cognitive features
    include_mri = parsed_args.include_mri # Overall MRI features
    include_ucsffsx = parsed_args.include_ucsffsx # UCSFFSX features
    include_graph_measures = parsed_args.include_graph_measures # Graph measures (degree, clustering coefficient, betweenness centrality, eigenvector centrality)
    include_adjacency = parsed_args.include_adjacency

    print("ADjacency features will be included:", include_adjacency )
    if parsed_args.run_dir is not None:
        RESULTS_DIR = parsed_args.run_dir
    else:
        RESULTS_DIR = f"./results_baseline_models/training_fs_{include_x}_cog_{include_cog}_mri_{include_mri}_ucsffsx_{include_ucsffsx}_graph_measures_{include_graph_measures}_{timestamp}_cw"
        os.makedirs(RESULTS_DIR, exist_ok=True)

    # Write args to a json file in the results directory
    args_dict = vars(parsed_args)
    with open(os.path.join(RESULTS_DIR, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

    # for summary csvs, create a results directory
    summary_dfs_dir = os.path.join(RESULTS_DIR, f"summary_csvs_{timestamp}")
    os.makedirs(summary_dfs_dir, exist_ok=True)


    # Load and preprocess data
    # if dataset path is a pt file, use load_dataset_from_single_pt
    if parsed_args.dataset == "oasis":
        print("Using OASIS dataset, loading from single .pt file...")
        data_list = general.load_dataset_from_single_pt(DATASET_PATH, convert_labels=False) if DATASET_PATH.endswith(".pt") else None
    elif parsed_args.dataset == "adni":
        print("Using ADNI dataset, loading from directory of .pt files...")
        data_list = general.load_dataset_from_single_pt(DATASET_PATH, convert_labels=True) if DATASET_PATH.endswith(".pt") else None



        

    # PREPROCCESSING STEPS:
    data_list, cog_in_dim, vol_sum_index = preprocessing.preprocess_global_data_list_for_baseline(
        data_list=data_list,
        args=parsed_args,
        feature_slices=FEATURE_SLICES
    )

    # if parsed_args.dataset == "oasis":
    #     for data in data_list:
    #         if not hasattr(data, "CDRTOT"):
    #             raise AttributeError("Graph is missing attribute `CDRTOT` needed for conversion task.")

    #         # ensure torch.long tensor
    #         y_val = 0 if data.CDRTOT == 0 else 1  
    #         data.y = torch.tensor([y_val], dtype=torch.long)

    print(f"Loaded {len(data_list)} graphs")

    conv_visit_map = {}

    if parsed_args.dataset == "adni":
        conv_df = pd.read_excel("metadata_tables/adni_labels_internal_dataset_plus_last_visit.xlsx")

        conv_df["PTID"] = conv_df["PTID"].astype(str).str.strip()
        conv_df["VISCODE"] = conv_df["VISCODE"].astype(str).str.strip()

        if parsed_args.task == "next_diagnosis":
            conv_df["IS_CONV_VISIT"] = conv_df["NEXT_IS_CONV_VISIT"]
        elif parsed_args.task == "classification":
            conv_df["IS_CONV_VISIT"] = conv_df["CURRENT_IS_CONV_VISIT"]
        elif parsed_args.task == "long_term_conversion":
            # This task uses subject/visit-level long-term progression labels,
            # not visit-wise conversion labels.
            conv_df["IS_CONV_VISIT"] = -1
        conv_df["IS_CONV_VISIT"] = conv_df["IS_CONV_VISIT"].fillna(-1).astype(int)

        conv_visit_map = {
            (str(row.PTID).strip(), str(row.VISCODE).strip()): int(row.IS_CONV_VISIT)
            for row in conv_df.itertuples(index=False)
        }
        next_label_map = {
            (str(row.PTID).strip(), str(row.VISCODE).strip()):
                int(row.NEXT_LABEL) if not pd.isna(row.NEXT_LABEL) else -99
            for row in conv_df.itertuples(index=False)
        }

        print(
            "Loaded IS_CONV_VISIT labels:",
            sum(v == 1 for v in conv_visit_map.values()),
            "conversion visits out of",
            len(conv_visit_map),
            "rows"
        )

        for data in data_list:
            ptid = str(data.ptid).strip()
            viscode = str(data.viscode).strip()
            flag = conv_visit_map.get((ptid, viscode), -1)
            data.is_conv_visit = torch.tensor(flag, dtype=torch.long)

    elif parsed_args.dataset == "oasis":
        # Optional: if you do not have conversion-visit labels for OASIS,
        # keep the field but mark it unavailable.
        for data in data_list:
            data.is_conv_visit = torch.tensor(-1, dtype=torch.long)
    
    if parsed_args.task == "next_diagnosis":
        print("Using next_diagnosis task: switching labels to NEXT_LABEL")

        for data in data_list:
            ptid = str(data.ptid).strip()
            viscode = str(data.viscode).strip()

            next_label = next_label_map.get((ptid, viscode), -99)

            # ADNI NEXT_LABEL is likely 1/2 for MCI/AD, convert to 0/1
            next_label = next_label - 1 if next_label != -99 else -99

            data.y = torch.tensor([next_label], dtype=torch.long)

        before_n = len(data_list)
        data_list = [
            data for data in data_list
            if int(data.y.item()) != -99
        ]
        after_n = len(data_list)

        print(f"Dropped {before_n - after_n} samples without NEXT_LABEL. Remaining: {after_n}")
    elif parsed_args.task == "long_term_conversion":
        print("Using long_term_conversion task: switching labels to Progression 24m")

        if parsed_args.dataset == "adni":
            long_term_conv_table = pd.read_excel(
                "./metadata_tables/adni_progression_table.xlsx"
            )

            long_term_label_map = {
                (str(row["ptid"]).strip(), str(row["viscode"]).strip()): row["Progression 24m"]
                for _, row in long_term_conv_table.iterrows()
            }

        elif parsed_args.dataset == "oasis":
            raise NotImplementedError(
                "long_term_conversion is not implemented for OASIS yet."
            )

        for data in data_list:
            ptid = str(data.ptid).strip()
            viscode = str(data.viscode).strip()

            prog_label = long_term_label_map.get((ptid, viscode), -99)

            if not pd.isna(prog_label) and prog_label != -99:
                data.y = torch.tensor([int(prog_label)], dtype=torch.long)
            else:
                data.y = torch.tensor([-99], dtype=torch.long)

        before_n = len(data_list)

        data_list = [
            data for data in data_list
            if int(data.y.item()) not in [-99, -1, 2]
        ]

        after_n = len(data_list)

        print(
            f"Dropped {before_n - after_n} samples without valid long-term conversion label. "
            f"Remaining: {after_n}"
        )

    labels = [int(data.y.item()) for data in data_list]
    print("Task:", parsed_args.task)
    print("Unique labels after task remapping:", sorted(set(labels)))
    print("Label counts:", pd.Series(labels).value_counts().sort_index().to_dict())

    # assert set(labels).issubset({0, 1}), f"Unexpected labels found: {set(labels)}"

    # Map filename to graph
    if parsed_args.dataset == "adni":
        filename_to_data = {data.ptid + "_" + data.viscode + ".pt": data for data in data_list}
    elif parsed_args.dataset == "oasis":
        filename_to_data = {data.oasis_id + "_" + data.scan_day + ".pt": data for data in data_list}

    if include_graph_measures:
        if parsed_args.graph_measures_type == "mean":
            if parsed_args.dataset == "adni":
                graph_measures_path = r"C:\Users\efeka\Documents\thesis_results\thesis_results\2026_04_11\adni_graph_measures\ctvol_graph_measures_1.0_density_mean.xlsx"
            elif parsed_args.dataset == "oasis":
                graph_measures_path = r"C:\Users\efeka\Documents\thesis_results\thesis_results\2026_04_11\oasis3_graph_measures\ctvol_graph_measures_1.0_density_mean.xlsx"
            print(f"Loading graph measures from {graph_measures_path}")
            df_graph = pd.read_excel(graph_measures_path)
            
            # The excels have .csv filenames, convert them to .pt to match our dictionary
            df_graph["filename"] = df_graph["filename"].str.replace(".csv", ".pt", regex=False)
            
            # Drop duplicates since we need a unique index for the dictionary
            df_graph = df_graph.drop_duplicates(subset=["filename"])

            graph_measures_dict = df_graph.set_index("filename")[
                ["mean_strength", "mean_betweenness", "mean_eigenvector_centrality", "mean_clustering_coefficient"]
            ].to_dict(orient="index")
            
            for fname, data in filename_to_data.items():
                if fname in graph_measures_dict:
                    measures = graph_measures_dict[fname]
                    data.x_graph_measures = torch.tensor([
                        measures["mean_strength"], 
                        measures["mean_betweenness"], 
                        measures["mean_eigenvector_centrality"], 
                        measures["mean_clustering_coefficient"]
                    ], dtype=torch.float)
                else:
                    print(f"Warning: Graph measures not found for {fname}")
                    data.x_graph_measures = torch.zeros(4, dtype=torch.float)
        elif parsed_args.graph_measures_type == "full":
            if parsed_args.dataset == "adni":
                graph_measures_path = r"C:\Users\efeka\Documents\thesis_results\thesis_results\2026_04_11\adni_graph_measures\ctvol_graph_measures_1.0_density.csv"
            elif parsed_args.dataset == "oasis":
                graph_measures_path = r"C:\Users\efeka\Documents\thesis_results\thesis_results\2026_04_11\oasis3_graph_measures\ctvol_graph_measures_1.0_density.csv"
            print(f"Loading graph measures from {graph_measures_path}")
            df_graph = pd.read_csv(graph_measures_path)
            
            # The excels have .csv filenames, convert them to .pt to match our dictionary
            df_graph["filename"] = df_graph["filename"].str.replace(".csv", ".pt", regex=False)
            if parsed_args.dataset == "oasis":
                # delete "MR_d" substring from filenames to match our dictionary
                df_graph["filename"] = df_graph["filename"].str.replace("MR_d", "", regex=False)
            
            # Drop duplicates since we need a unique index for the dictionary
            df_graph = df_graph.drop_duplicates(subset=["filename"])

            measure_cols = [col for col in df_graph.columns if col.endswith(("strength", "betweenness", "eigenvector_centrality", "clustering_coefficient"))]
            print(f"Using graph measure columns: {measure_cols}")
            graph_measures_dict = df_graph.set_index("filename")[measure_cols].to_dict(orient="index")
            
            for fname, data in filename_to_data.items():
                if fname in graph_measures_dict:
                    measures = graph_measures_dict[fname]
                    data.x_graph_measures = torch.tensor([measures[col] for col in measure_cols], dtype=torch.float)
                else:
                    print(f"Warning: Graph measures not found for {fname}")
                    data.x_graph_measures = torch.zeros(len(measure_cols), dtype=torch.float)

    # # Load conversion subjects
    # if os.path.exists(CONVERSION_SUBJECTS_PATH):
    #     conversion_split = read_cross_val(CONVERSION_SUBJECTS_PATH)
    #     conversion_subjects = conversion_split[0]['test_subjects']
    # Load splits
    splits = general.read_cross_val(CROSS_VAL_PKL_PATH)

    # Classifiers
    lr_class_weight = "balanced" if parsed_args.class_weights else None
    rf_class_weight = "balanced" if parsed_args.class_weights else None
    svm_class_weight = "balanced" if parsed_args.class_weights else None


    svm_gamma = parsed_args.svm_gamma
    try:
        svm_gamma = float(svm_gamma)
    except ValueError:
        pass

    models = {
        "Logistic_Regression": LogisticRegression(
            random_state=parsed_args.random_seed,
            max_iter=parsed_args.lr_max_iter,
            C=parsed_args.lr_C,
            solver=parsed_args.lr_solver,
            class_weight=lr_class_weight,
        ),
        "Random_Forest": RandomForestClassifier(
            random_state=parsed_args.random_seed,
            n_estimators=parsed_args.rf_n_estimators,
            max_depth=parsed_args.rf_max_depth,
            class_weight=rf_class_weight,
            n_jobs=-1,
        ),
        "SVM_RBF": SVC(
            random_state=parsed_args.random_seed,
            kernel="rbf",
            C=parsed_args.svm_C,
            gamma=svm_gamma,
            class_weight=svm_class_weight,
            probability=True,
        ),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=parsed_args.xgb_n_estimators,
            max_depth=parsed_args.xgb_max_depth,
            learning_rate=parsed_args.xgb_learning_rate,
            subsample=parsed_args.xgb_subsample,
            colsample_bytree=parsed_args.xgb_colsample_bytree,
            reg_lambda=parsed_args.xgb_reg_lambda,  
            random_state=parsed_args.random_seed,
            n_jobs=-1,
        ),
    }

    selected_models = parsed_args.models
    models = {name: models[name] for name in selected_models}

    excel_path = os.path.join(RESULTS_DIR, "all_models_mean_std_results.xlsx")
    excel_writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')

    # Evaluation
    for model_name, base_clf in models.items():
        print(f"\n=== Model: {model_name} ===")

        fold_metrics = []
        all_prediction_records = []
        all_preds = []
        all_labels = []

        for fold, split in enumerate(splits):
            clf = clone(base_clf)
            if "val_files" in split:
                # train_filenames = split["train_files"] + split["val_files"] # no validation, add it to train
                if parsed_args.dataset == "oasis":
                    train_filenames = split["train_files"]
                elif parsed_args.dataset == "adni":
                    train_filenames = split["train_files"] + split["val_files"] 
            else:
                train_filenames = split["train_files"]
            # train_files = split['train_files'] #+ split['val_files'] # Split with conversion subjects
            test_filenames = split['test_files']

            # Find the corresponding data objects and copy them
            train_data = [copy.deepcopy(filename_to_data[fname]) 
                        for fname in train_filenames if fname in filename_to_data]

            test_data = [copy.deepcopy(filename_to_data[fname]) 
                        for fname in test_filenames if fname in filename_to_data]

            print(  f"Fold {fold + 1} - Train files: {len(train_data)}, Test files: {len(test_data)}")

            # preprocess branch features on training, get the scalers
            if include_x:
                icv_params = preprocessing.fit_icv_normalizer(train_data, feature_indices=[vol_sum_index], icv_attr="ICV")
                train_data = preprocessing.apply_icv_normalizer(train_data, icv_params)
                test_data  = preprocessing.apply_icv_normalizer(test_data,  icv_params)

                train_data, mri_node_scalers = preprocessing.preprocess_mri_node_features(train_data)
                test_data = preprocessing.apply_mri_node_scalers(test_data, mri_node_scalers)

            if include_cog:
                train_data, cog_scaler, cog_mean = preprocessing.preprocess_cognitive_features_train(train_data)
                test_data = preprocessing.preprocess_cognitive_features_test(test_data, cog_scaler, cog_mean)

            if include_mri:
                train_data, mri_scaler, mri_mean = preprocessing.preprocess_mri_features_train(train_data)
                test_data = preprocessing.preprocess_mri_features_test(test_data, mri_scaler, mri_mean)

            if include_ucsffsx:
                train_data, ucsffsx_scaler, ucsffsx_mean = preprocessing.preprocess_ucsffsx_features_train(train_data)
                test_data = preprocessing.preprocess_ucsffsx_features_test(test_data, ucsffsx_scaler, ucsffsx_mean)

            if include_graph_measures:
                train_data, graph_measures_scaler, graph_measures_mean = preprocessing.preprocess_graph_measures_train(train_data)
                test_data = preprocessing.preprocess_graph_measures_test(test_data, graph_measures_scaler, graph_measures_mean)

            if include_adjacency:
                train_data, adjacency_scaler, adjacency_mean = preprocessing.preprocess_adjacency_features_train(train_data)
                test_data = preprocessing.preprocess_adjacency_features_test(test_data, adjacency_scaler, adjacency_mean)

            if APPLY_PCA and include_x:
                train_data, pca_cortex = preprocessing.fit_pca_on_train(train_data, n_components=NUM_COMPONENTS)
                test_data = preprocessing.apply_pca_to_graphs(test_data, pca_cortex)

            X_train, y_train, feature_names = preprocessing.get_flattened_features(train_data, include_x=include_x,
                                                        include_cog= include_cog, include_mri=include_mri, include_ucsffsx=include_ucsffsx, include_graph_measures=include_graph_measures,
                                                        include_adjacency=include_adjacency, expected_nodes=EXPECTED_NODES)
            X_test, y_test, feature_names = preprocessing.get_flattened_features(test_data, include_x=include_x,
                                                        include_cog= include_cog, include_mri=include_mri, include_ucsffsx=include_ucsffsx, include_graph_measures=include_graph_measures,
                                                        include_adjacency=include_adjacency, expected_nodes= EXPECTED_NODES)
            print(X_train.shape, y_train.shape)

            if model_name == "XGBoost" and parsed_args.class_weights:
                # Count the number of samples per class
                n_neg = np.sum(y_train == 0)
                n_pos = np.sum(y_train == 1)
                if n_pos == 0:
                    scale_pos_weight = 1.0  # avoid division by zero
                else:
                    scale_pos_weight = n_neg / n_pos

                print(f"Fold {fold + 1}: n_neg={n_neg}, n_pos={n_pos}, scale_pos_weight={scale_pos_weight:.3f}")

                # Reinitialize XGBoost with this fold's class weight
                clf = XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=parsed_args.xgb_n_estimators,
                    max_depth=parsed_args.xgb_max_depth,
                    learning_rate=parsed_args.xgb_learning_rate,
                    subsample=parsed_args.xgb_subsample,
                    colsample_bytree=parsed_args.xgb_colsample_bytree,
                    random_state=parsed_args.random_seed,
                    n_jobs=-1,
                    scale_pos_weight=scale_pos_weight,
                )


            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Get probability for class 1, if available
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test)
                y_prob_class_1 = y_prob[:, 1]
            else:
                y_prob = None
                y_prob_class_1 = None

            acc = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)

            f1_macro = f1_score(y_test, y_pred, average="macro")
            f1_weighted = f1_score(y_test, y_pred, average="weighted")

            precision, recall, _, _ = precision_recall_fscore_support(
                y_test,
                y_pred,
                labels=[0, 1],
                zero_division=0
            )

            try:
                final_auc = roc_auc_score(y_test, y_prob_class_1) if y_prob_class_1 is not None else float("nan")
            except ValueError:
                final_auc = float("nan")

            try:
                final_auprc = average_precision_score(y_test, y_prob_class_1) if y_prob_class_1 is not None else float("nan")
            except ValueError:
                final_auprc = float("nan")

            # among true conversion visits, how many were predicted as positive class 1?
            conv_true_count = 0
            conv_pred_positive_count = 0

            for i, data_obj in enumerate(test_data):
                if hasattr(data_obj, "is_conv_visit"):
                    conv_flag = data_obj.is_conv_visit
                    if torch.is_tensor(conv_flag):
                        conv_flag = int(conv_flag.item())
                    else:
                        conv_flag = int(conv_flag)

                    if conv_flag == 1:
                        conv_true_count += 1
                        if int(y_pred[i]) == 1:
                            conv_pred_positive_count += 1

            final_conv_recall = (
                float(conv_pred_positive_count) / conv_true_count
                if conv_true_count > 0
                else float("nan")
            )

            
            if model_name == "Logistic_Regression":
                # feature importance for Logistic Regression
                importances = clf.coef_[0]
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                })
                importance_df['abs_importance'] = importance_df['importance'].abs()
                importance_df = importance_df.sort_values(by='abs_importance', ascending=False)
                importance_df.to_csv(os.path.join(RESULTS_DIR, f"{model_name}_feature_importance_fold_{fold + 1}.csv"), index=False)
         
            fold_metrics.append({
                "FOLD": fold + 1,
                "ACC": acc,
                "BALANCED_ACC": balanced_acc,
                "F1_macro": f1_macro,
                "F1_weighted": f1_weighted,
                "PRECISION_CLASS_0": precision[0],
                "RECALL_CLASS_0": recall[0],
                "PRECISION_CLASS_1": precision[1],
                "RECALL_CLASS_1": recall[1],
                "AUC": final_auc,
                "AUPRC": final_auprc,
                "CONVERSION_RECALL": final_conv_recall,
            })

            for i, data_obj in enumerate(test_data):
                ptid = getattr(data_obj, "ptid", getattr(data_obj, "oasis_id", None))
                viscode = getattr(data_obj, "viscode", getattr(data_obj, "scan_day", None))
                label = int(y_test[i])
                prediction = int(y_pred[i])

                if y_prob is not None:
                    prob_class0 = float(y_prob[i][0])
                    prob_class1 = float(y_prob[i][1])
                else:
                    prob_class0 = None
                    prob_class1 = None

                status = getattr(data_obj, "status", None)

                is_conv_visit = getattr(data_obj, "is_conv_visit", -1)
                if torch.is_tensor(is_conv_visit):
                    is_conv_visit = int(is_conv_visit.item())
                else:
                    is_conv_visit = int(is_conv_visit)

                record = {
                    "fold": fold + 1,
                    "ptid": ptid,
                    "viscode": viscode,
                    "task": parsed_args.task,
                    "label": label,
                    "status": status,
                    "prediction": prediction,
                    "prob_mci": prob_class0,
                    "prob_ad": prob_class1,
                    "is_conv_visit": is_conv_visit,
                    "model": model_name,
                }

                all_prediction_records.append(record)

            all_preds.extend(y_pred)
            all_labels.extend(y_test)

        # Save per-fold results
        df_folds = pd.DataFrame(fold_metrics)
        df_folds.to_csv(os.path.join(RESULTS_DIR, f"{model_name}_fold_results.csv"), index=False)
        print(f"Saved fold results to {model_name}_fold_results.csv")

        prediction_df = pd.DataFrame(all_prediction_records)
        pred_path = os.path.join(RESULTS_DIR, f"{model_name}_validation_predictions.xlsx")
        prediction_df.to_excel(pred_path, index=False)
        print(f"Saved validation predictions to {pred_path}")

        # Calculate mean/std
        means = df_folds.drop(columns=["FOLD"]).mean()
        stds = df_folds.drop(columns=["FOLD"]).std()
        summary_df = pd.DataFrame({
            "Mean": means,
            "Std": stds,
            "Mean +- SD": means.map("{:.3f}".format) + " +- " + stds.map("{:.3f}".format)
        })

        summary_df.to_csv(os.path.join(summary_dfs_dir, f"{model_name}_summary.csv"), index=True)
        summary_df.to_excel(excel_writer, sheet_name=model_name[:31])  # Sheet names must be ≤31 characters

        models_txt_path = os.path.join(RESULTS_DIR, "models.txt")
        with open(models_txt_path, "w", encoding="utf-8") as f:
            f.write(str(models))

        print(f"Saved models dictionary to {models_txt_path}")

        # save include_x, include_cog, include_mri to a text file
    with open(os.path.join(RESULTS_DIR, f"features_used.txt"), "w") as f:
        f.write(f"include_x: {include_x}\n")
        f.write(f"include_cog: {include_cog}\n")
        f.write(f"include_mri: {include_mri}\n")
        f.write(f"include_ucsffsx: {include_ucsffsx}\n")
        f.write(f"Dataset path: {DATASET_PATH}\n")
        f.write(f"NUM_COMPONENTS (if PCA applied): {NUM_COMPONENTS}\n")
        f.write(f"APPLY_PCA: {APPLY_PCA}\n")
        f.write(f"Random Seed: {parsed_args.random_seed}\n")
        f.write(f"Task: {parsed_args.task}\n")
        f.write(f"Dataset: {parsed_args.dataset}\n")
        f.write(f"Class Weights: {parsed_args.class_weights}\n")
        f.write(f"Is Tuning: {parsed_args.is_tuning}\n")
        f.write(f"Cross-validation splits path: {CROSS_VAL_PKL_PATH}\n")

    excel_writer.close()
    print(f"\nAll mean/std results written to {excel_path}")
