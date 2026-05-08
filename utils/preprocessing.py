from .general import attach_weighted_adj_matrices, add_laplacian_pe, remove_low_weight_edges_pyg, load_dataset_from_single_pt, load_dataset
import torch

def preprocess_global_data_list(
    data_list,
    dataset_path: str,
    args,
    feature_slices: dict,
):
    """
    Preprocessing that is safe to do ONCE on the whole dataset (no train/test leakage).
    Returns: (data_list, cog_in_dim)
    """

    sample = data_list[0]
    if args.dataset == "adni":
        print(sample.ptid, sample.viscode, "has weighted_adj_matrix with shape", sample.weighted_adj_matrix.shape)
    elif args.dataset == "oasis":
        print(sample.oasis_id, sample.scan_day, "has weighted_adj_matrix with shape", sample.weighted_adj_matrix.shape)

    # --- Optional Laplacian PE (global; doesn't use labels) ---
    if args.add_laplacian_pe:
        print("Adding Laplacian positional encodings...")
        print("Dimension of node features before adding Laplacian PE:", data_list[0].x.shape)
        data_list = add_laplacian_pe(data_list, pe_dim=args.lpe_dim)
        print("Dimension of node features after adding Laplacian PE:", data_list[0].x.shape)
        print("Laplacian PE added as attribute 'laplacian_pe' with shape:", data_list[0].laplacian_pe.shape)

    # --- Conversion task label switch (pmci -> y) ---
    if args.task == "conversion":
        print("Using conversion task: switching labels to data.pmci")
        for d in data_list:
            if not hasattr(d, "pmci"):
                raise AttributeError("Graph is missing attribute `pmci` needed for conversion task.")
            d.y = torch.tensor([int(d.pmci)], dtype=torch.long)

    # --- Optional edge pruning (global; no leakage) ---
    if args.edge_threshold < 1.0:
        print(data_list[0].edge_index)
        print(f"Keeping top {args.edge_threshold*100:.1f}% of edges by weight, removing the rest...")
        data_list = [remove_low_weight_edges_pyg(d, args.edge_threshold) for d in data_list]
        # print the number of edges after pruning for the first graph as a sanity check
        print(f"After pruning, graph 0 has {data_list[0].edge_index.size(1)} edges.")
        # print some edge weights after pruning for the first graph as a sanity check
        # edge_attr has edge weights
        if hasattr(data_list[1], "edge_attr") and data_list[0].edge_attr is not None:
            print("Sample edge weights after pruning (graph 0):", data_list[0].edge_attr[:])
            print(data_list[1].edge_index)

    # --- Node feature selection (global; no leakage) ---
    selected = args.node_feature_set.lower()
    if selected == "all":
        selected_features = list(feature_slices.keys())
    else:
        selected_features = selected.split("_")

    indices = []
    for feat in selected_features:
        if feat == "":
            print("No node features selected (empty set).")
            continue
        if feat not in feature_slices:
            raise ValueError(f"Unknown feature: {feat}")
        s = feature_slices[feat]
        
        # retrieve the volume sum index for later icv normalization
        if feat == "vol":
            vol_sum_index = len(indices) 
            print(f"Volume sum index (for later ICV normalization): {vol_sum_index}")
        indices.extend(range(s.start, s.stop))
    indices = torch.tensor(indices, dtype=torch.long)

    print(f"Using node feature set: {args.node_feature_set} -> columns {indices.tolist()}")
    for d in data_list:
        d.x = d.x[:, indices]


    add_adj = bool(getattr(args, "add_adj_row_as_node_feature", False))
    separate_adj_features = bool(getattr(args, "separate_adj_features_instead_of_concat", False))
    if args.include_adjacency_gnn:
        print("Creating adjacency row node features ...")

        for d in data_list:
            if not hasattr(d, "weighted_adj_matrix") or d.weighted_adj_matrix is None:
                raise AttributeError("Cannot create adjacency row features: missing `weighted_adj_matrix`.")

            A = d.weighted_adj_matrix

            if A.dim() == 3 and A.size(0) == 1:
                A = A.squeeze(0)

            if A.dim() != 2:
                raise ValueError(f"Expected weighted_adj_matrix to be [N,N], got {tuple(A.shape)}")

            N = d.x.size(0)
            if A.size(0) != N or A.size(1) != N:
                raise ValueError(f"Adjacency shape {tuple(A.shape)} does not match num nodes {N}")

            A = A.to(device=d.x.device, dtype=d.x.dtype)

            d.x_adj_row = A

    if add_adj:
        print("Appending adjacency row (dense) as node features...")
        for d in data_list:
            if not hasattr(d, "weighted_adj_matrix") or d.weighted_adj_matrix is None:
                raise AttributeError("Cannot append adjacency row: missing `weighted_adj_matrix`.")

            A = d.weighted_adj_matrix
            if A.dim() == 3 and A.size(0) == 1:
                A = A.squeeze(0)  # [N, N]
            if A.dim() != 2:
                raise ValueError(f"Expected weighted_adj_matrix to be [N,N], got {tuple(A.shape)}")

            # sanity check
            N = d.x.size(0)
            if A.size(0) != N or A.size(1) != N:
                raise ValueError(f"Adjacency shape {tuple(A.shape)} does not match num nodes {N}")

            if separate_adj_features:
                d.x_adj_row = A.to(device=d.x.device, dtype=d.x.dtype) # store separately for model to handle as needed
            else:
                d.x = torch.cat([d.x, A.to(d.x.device).to(d.x.dtype)], dim=1)

    add_deg = bool(getattr(args, "add_weighted_degree_as_node_feature", False))
    if add_deg:
        print("Appending weighted node degree as a node feature...")
        for d in data_list:
            if not hasattr(d, "weighted_adj_matrix") or d.weighted_adj_matrix is None:
                raise AttributeError("Cannot append degree: missing `weighted_adj_matrix`.")

            A = d.weighted_adj_matrix
            if A.dim() == 3 and A.size(0) == 1:
                A = A.squeeze(0)  # [N, N]
            if A.dim() != 2:
                raise ValueError(f"Expected weighted_adj_matrix to be [N,N], got {tuple(A.shape)}")

            # sanity check
            N = d.x.size(0)
            if A.size(0) != N or A.size(1) != N:
                raise ValueError(f"Adjacency shape {tuple(A.shape)} does not match num nodes {N}")

            # move/cast to match x
            A = A.to(device=d.x.device, dtype=d.x.dtype)

            # weighted degree: row-sum -> [N]
            deg = A.sum(dim=1, keepdim=True)  # [N, 1]

            # append as one extra node feature
            d.x = torch.cat([d.x, deg], dim=1)

        print("New node feature dim:", data_list[0].x.shape)

    # --- Cognitive feature selection (global) ---
    if args.cog_feature_set == "no_adas":
        print("Excluding ADAS cognitive features.")
        for d in data_list:
            if hasattr(d, "x_cog") and d.x_cog is not None:
                d.x_cog = d.x_cog[2:]  # your assumption: first 2 are ADAS

    # --- Determine cog input dim (after selection) ---
    cog_in_dim = None
    if hasattr(data_list[0], "x_cog") and data_list[0].x_cog is not None:
        cog_in_dim = int(data_list[0].x_cog.shape[0])

    return data_list, cog_in_dim, vol_sum_index if 'vol' in selected_features else None



def preprocess_global_data_list_for_baseline(
    data_list,
    args,
    feature_slices: dict,
):
    """
    Preprocessing that is safe to do ONCE on the whole dataset (no train/test leakage).
    Returns: (data_list, cog_in_dim)
    """


    # --- Node feature selection (global; no leakage) ---
    selected = args.node_feature_set.lower()
    if selected == "all":
        selected_features = list(feature_slices.keys())
    else:
        selected_features = selected.split("_")

    indices = []
    for feat in selected_features:
        if feat == "":
            print("No node features selected (empty set).")
            continue
        if feat not in feature_slices:
            raise ValueError(f"Unknown feature: {feat}")
        s = feature_slices[feat]
        
        # retrieve the volume sum index for later icv normalization
        if feat == "vol":
            vol_sum_index = len(indices) 
            print(f"Volume sum index (for later ICV normalization): {vol_sum_index}")
        indices.extend(range(s.start, s.stop))
    indices = torch.tensor(indices, dtype=torch.long)

    print(f"Using node feature set: {args.node_feature_set} -> columns {indices.tolist()}")
    for d in data_list:
        d.x = d.x[:, indices]


    add_adj = bool(getattr(args, "add_adj_row_as_node_feature", False))
    if add_adj:
        print("Appending adjacency row (dense) as node features...")
        for d in data_list:
            if not hasattr(d, "weighted_adj_matrix") or d.weighted_adj_matrix is None:
                raise AttributeError("Cannot append adjacency row: missing `weighted_adj_matrix`.")

            A = d.weighted_adj_matrix
            if A.dim() == 3 and A.size(0) == 1:
                A = A.squeeze(0)  # [N, N]
            if A.dim() != 2:
                raise ValueError(f"Expected weighted_adj_matrix to be [N,N], got {tuple(A.shape)}")

            # sanity check
            N = d.x.size(0)
            if A.size(0) != N or A.size(1) != N:
                raise ValueError(f"Adjacency shape {tuple(A.shape)} does not match num nodes {N}")

            d.x = torch.cat([d.x, A.to(d.x.device).to(d.x.dtype)], dim=1)

        print("New node feature dim:", data_list[0].x.shape)
    
    add_deg = bool(getattr(args, "add_weighted_degree_as_node_feature", False))
    if add_deg:
        print("Appending weighted node degree as a node feature...")
        for d in data_list:
            if not hasattr(d, "weighted_adj_matrix") or d.weighted_adj_matrix is None:
                raise AttributeError("Cannot append degree: missing `weighted_adj_matrix`.")

            A = d.weighted_adj_matrix
            if A.dim() == 3 and A.size(0) == 1:
                A = A.squeeze(0)  # [N, N]
            if A.dim() != 2:
                raise ValueError(f"Expected weighted_adj_matrix to be [N,N], got {tuple(A.shape)}")

            # sanity check
            N = d.x.size(0)
            if A.size(0) != N or A.size(1) != N:
                raise ValueError(f"Adjacency shape {tuple(A.shape)} does not match num nodes {N}")

            # move/cast to match x
            A = A.to(device=d.x.device, dtype=d.x.dtype)

            # weighted degree: row-sum -> [N]
            deg = A.sum(dim=1, keepdim=True)  # [N, 1]

            # append as one extra node feature
            d.x = torch.cat([d.x, deg], dim=1)

        print("New node feature dim:", data_list[0].x.shape)

    # --- Cognitive feature selection (global) ---
    if args.cog_feature_set == "no_adas":
        print("Excluding ADAS cognitive features.")
        for d in data_list:
            if hasattr(d, "x_cog") and d.x_cog is not None:
                d.x_cog = d.x_cog[2:]  # your assumption: first 2 are ADAS

    # --- Determine cog input dim (after selection) ---
    cog_in_dim = None
    if hasattr(data_list[0], "x_cog") and data_list[0].x_cog is not None:
        cog_in_dim = int(data_list[0].x_cog.shape[0])

    return data_list, cog_in_dim, vol_sum_index if 'vol' in selected_features else None


import torch
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_mri_node_features(data_list_train):
    """
    Standardize node features per node across training graphs only.
    
    Args:
        data_list_train (list[torch_geometric.data.Data]): training graphs
    
    Returns:
        data_list_train (list[torch_geometric.data.Data]): scaled training graphs
        scalers (list[StandardScaler]): one fitted scaler per node
    """
    num_nodes = data_list_train[0].x.size(0)
    num_features = data_list_train[0].x.size(1)

    # Collect feature vectors of each node across all training graphs
    node_feature_matrix = [[] for _ in range(num_nodes)]
    for data in data_list_train:
        for node_idx in range(num_nodes):
            node_feature_matrix[node_idx].append(data.x[node_idx].numpy())

    # Fit a scaler per node
    scalers = []
    for node_idx in range(num_nodes):
        # X = torch.tensor(node_feature_matrix[node_idx])  # [num_graphs, num_features]
        arr = np.array(node_feature_matrix[node_idx])   # fast numpy array construction
        X = torch.from_numpy(arr).float()               # efficient tensor conversion

        scaler = StandardScaler()
        scaler.fit(X)   # fit only on training data
        scalers.append(scaler)

    # Apply scalers back to training graphs
    for data in data_list_train:
        x_scaled = []
        for node_idx in range(num_nodes):
            scaled = scalers[node_idx].transform(
                data.x[node_idx].unsqueeze(0).numpy()
            )
            x_scaled.append(torch.tensor(scaled.squeeze(), dtype=torch.float32))
        data.x = torch.stack(x_scaled)

    return data_list_train, scalers

def apply_mri_node_scalers(data_list_test, scalers):
    """
    Apply pre-fitted per-node scalers to test/validation graphs.

    Args:
        data_list_test (list[torch_geometric.data.Data]): graphs to transform
        scalers (list[StandardScaler]): fitted per-node scalers from training

    Returns:
        data_list_test (list[torch_geometric.data.Data]): scaled test graphs
    """
    expected_num_nodes = len(scalers)
    actual_num_nodes = data_list_test[0].x.size(0)

    if actual_num_nodes != expected_num_nodes:
        raise ValueError(
            f"Node count mismatch: scalers expect {expected_num_nodes} nodes, "
            f"but test graphs have {actual_num_nodes} nodes."
        )

    for data in data_list_test:
        if data.x.size(0) != expected_num_nodes:
            raise ValueError(
                f"Graph with {data.x.size(0)} nodes found, "
                f"but expected {expected_num_nodes} nodes."
            )

        x_scaled = []
        for node_idx in range(expected_num_nodes):
            scaled = scalers[node_idx].transform(
                data.x[node_idx].unsqueeze(0).numpy()
            )
            x_scaled.append(torch.tensor(scaled.squeeze(), dtype=torch.float32))
        data.x = torch.stack(x_scaled)

    return data_list_test

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# TRAIN: fit imputation means + scaler on TRAIN, apply to TRAIN
# ------------------------------------------------------------
def preprocess_cognitive_features_train(data_list_train):
    """
    • imputes NaNs in data.x_cog using TRAIN per-feature means
    • standard-scales with StandardScaler fitted on TRAIN (after imputation)
    • guarantees each graph stores x_cog with shape (1, F)
    
    Returns:
        data_list_train: transformed training graphs (in-place modified)
        scaler: fitted StandardScaler
        feat_mean: ndarray of shape (F,) for re-use on test/val
    """
    if len(data_list_train) == 0:
        raise ValueError("data_list_train is empty.")

    # ---------- 1) stack all cognitive vectors (N, F) ----------
    F = data_list_train[0].x_cog.view(1, -1).size(1)
    cog_rows = [data.x_cog.view(1, -1) for data in data_list_train]  # each (1, F)
    cog_mat  = torch.cat(cog_rows, dim=0)                            # (N, F)

    # ---------- 2) impute NaNs with TRAIN per-feature mean -------
    cog_np   = cog_mat.numpy()
    feat_mean = np.nanmean(cog_np, axis=0)                           # (F,)
    nan_idx   = np.where(np.isnan(cog_np))
    print("Number of NaNs in TRAIN cognitive features: ", len(nan_idx[0]))
    if len(nan_idx[0]) > 0:
        cog_np[nan_idx] = feat_mean[nan_idx[1]]
    cog_imputed = torch.tensor(cog_np, dtype=torch.float32)          # (N, F)

    # ---------- 3) write back imputed to each Data (shape: (1, F)) -
    for i, data in enumerate(data_list_train):
        data.x_cog = cog_imputed[i].unsqueeze(0)

    # ---------- 4) fit StandardScaler on TRAIN (imputed) ----------
    print("Scaling TRAIN cognitive features …")
    scaler = StandardScaler().fit(cog_imputed.numpy())

    # ---------- 5) transform TRAIN and write back -----------------
    for data in data_list_train:
        data.x_cog = torch.tensor(
            scaler.transform(data.x_cog.numpy()),
            dtype=torch.float32
        )  # (1, F)

    # Check whether cognitive features are now NaN-free and scaled (mean ~0, std ~1)
    # for data in data_list_train:
    #     if torch.isnan(data.x_cog).any():
    #         raise ValueError("NaN values found in cognitive features after preprocessing.")
    #     mean = data.x_cog.mean().item()
    #     std = data.x_cog.std().item()
    #     print(f"Post-scaling cognitive features: mean={mean:.4f}, std={std:.4f}")

    return data_list_train, scaler, feat_mean


# ------------------------------------------------------------
# TEST/VAL: reuse TRAIN feat_mean + scaler; no refit
# ------------------------------------------------------------
def preprocess_cognitive_features_test(data_list_test, scaler, feat_mean):
    """
    Apply TRAIN-time imputation mean and TRAIN-fitted scaler to TEST/VAL graphs.
    • No refitting; strictly uses provided feat_mean and scaler.
    • Guarantees each graph stores x_cog with shape (1, F)

    Args:
        data_list_test: list of graphs to transform (modified in-place)
        scaler: StandardScaler fitted on TRAIN
        feat_mean: ndarray (F,) computed on TRAIN for NaN imputation

    Returns:
        data_list_test: transformed test/val graphs
    """
    if len(data_list_test) == 0:
        raise ValueError("data_list_test is empty.")
    if feat_mean.ndim != 1:
        raise ValueError("feat_mean must be a 1D array of shape (F,).")

    F_expected = feat_mean.shape[0]
    F_test = data_list_test[0].x_cog.view(1, -1).size(1)
    if F_test != F_expected:
        raise ValueError(
            f"Feature dimension mismatch: TRAIN has F={F_expected} but TEST has F={F_test}."
        )

    # ---------- 1) stack (N, F) ----------------------------------
    cog_rows = [data.x_cog.view(1, -1) for data in data_list_test]
    cog_mat  = torch.cat(cog_rows, dim=0)                            # (N, F)

    # ---------- 2) impute NaNs using TRAIN feat_mean --------------
    cog_np  = cog_mat.numpy()
    nan_idx = np.where(np.isnan(cog_np))
    print("Number of NaNs in TEST cognitive features: ", len(nan_idx[0]))
    if len(nan_idx[0]) > 0:
        cog_np[nan_idx] = feat_mean[nan_idx[1]]
    cog_imputed = torch.tensor(cog_np, dtype=torch.float32)          # (N, F)

    # ---------- 3) transform with TRAIN-fitted scaler -------------
    print("Scaling TEST cognitive features …")
    cog_scaled = scaler.transform(cog_imputed.numpy())               # (N, F)

    # ---------- 4) write back to each Data ------------------------
    cog_scaled_t = torch.tensor(cog_scaled, dtype=torch.float32)
    for i, data in enumerate(data_list_test):
        data.x_cog = cog_scaled_t[i].unsqueeze(0)                    # (1, F)

    return data_list_test
# ---------------------------------------------------------------------------

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# TRAIN: fit imputation means + scaler on TRAIN, apply to TRAIN
# ------------------------------------------------------------
def preprocess_mri_features_train(data_list_train):
    """
    • imputes NaNs in data.x_mri using TRAIN per-feature means
    • standard-scales with StandardScaler fitted on TRAIN (after imputation)
    • guarantees each graph stores x_mri with shape (1, F)
    
    Returns:
        data_list_train: transformed training graphs (in-place modified)
        scaler: fitted StandardScaler
        feat_mean: ndarray of shape (F,) for re-use on test/val
    """
    if len(data_list_train) == 0:
        raise ValueError("data_list_train is empty.")

    # ---------- 1) stack all MRI feature matrices (N, F) ----------
    F = data_list_train[0].x_mri.view(1, -1).size(1)
    mri_rows = [data.x_mri.view(1, -1) for data in data_list_train]  # (1, F)
    mri_mat  = torch.cat(mri_rows, dim=0)                            # (N, F)

    # ---------- 2) impute NaNs with TRAIN per-feature mean --------
    mri_np   = mri_mat.numpy()
    feat_mean = np.nanmean(mri_np, axis=0)                           # (F,)
    nan_idx   = np.where(np.isnan(mri_np))
    # print("nan_idx", nan_idx)
    print("Number of NaNs in TRAIN MRI features: ", len(nan_idx[0]))
    if len(nan_idx[0]) > 0:
        mri_np[nan_idx] = feat_mean[nan_idx[1]]
    mri_imputed = torch.tensor(mri_np, dtype=torch.float32)          # (N, F)

    # ---------- 3) write back imputed -----------------------------
    for i, data in enumerate(data_list_train):
        data.x_mri = mri_imputed[i].unsqueeze(0)                     # (1, F)

    # ---------- 4) fit StandardScaler on TRAIN --------------------
    print("Scaling TRAIN MRI features …")
    scaler = StandardScaler().fit(mri_imputed.numpy())

    # ---------- 5) transform TRAIN and write back -----------------
    for data in data_list_train:
        data.x_mri = torch.tensor(
            scaler.transform(data.x_mri.numpy()),
            dtype=torch.float32
        )  # (1, F)

    return data_list_train, scaler, feat_mean


# ------------------------------------------------------------
# TEST/VAL: reuse TRAIN feat_mean + scaler; no refit
# ------------------------------------------------------------
def preprocess_mri_features_test(data_list_test, scaler, feat_mean):
    """
    Apply TRAIN-time imputation mean and TRAIN-fitted scaler to TEST/VAL graphs.
    • No refitting; strictly uses provided feat_mean and scaler.
    • Guarantees each graph stores x_mri with shape (1, F)

    Args:
        data_list_test: list of graphs to transform (modified in-place)
        scaler: StandardScaler fitted on TRAIN
        feat_mean: ndarray (F,) computed on TRAIN for NaN imputation

    Returns:
        data_list_test: transformed test/val graphs
    """
    if len(data_list_test) == 0:
        raise ValueError("data_list_test is empty.")
    if feat_mean.ndim != 1:
        raise ValueError("feat_mean must be a 1D array of shape (F,).")

    F_expected = feat_mean.shape[0]
    F_test = data_list_test[0].x_mri.view(1, -1).size(1)
    if F_test != F_expected:
        raise ValueError(
            f"Feature dimension mismatch: TRAIN has F={F_expected} but TEST has F={F_test}."
        )

    # ---------- 1) stack (N, F) ----------------------------------
    mri_rows = [data.x_mri.view(1, -1) for data in data_list_test]
    mri_mat  = torch.cat(mri_rows, dim=0)                            # (N, F)

    # ---------- 2) impute NaNs using TRAIN feat_mean --------------
    mri_np  = mri_mat.numpy()
    nan_idx = np.where(np.isnan(mri_np))
    print("Number of NaNs in TEST MRI features: ", len(nan_idx[0]))
    if len(nan_idx[0]) > 0:
        mri_np[nan_idx] = feat_mean[nan_idx[1]]
    mri_imputed = torch.tensor(mri_np, dtype=torch.float32)          # (N, F)

    # ---------- 3) transform with TRAIN-fitted scaler -------------
    print("Scaling TEST MRI features …")
    mri_scaled = scaler.transform(mri_imputed.numpy())               # (N, F)

    # ---------- 4) write back to each Data ------------------------
    mri_scaled_t = torch.tensor(mri_scaled, dtype=torch.float32)
    for i, data in enumerate(data_list_test):
        data.x_mri = mri_scaled_t[i].unsqueeze(0)                    # (1, F)

    return data_list_test

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# TRAIN: fit imputation means + scaler on TRAIN, apply to TRAIN
# ------------------------------------------------------------
def preprocess_ucsffsx_features_train(data_list_train):
    """
    • imputes NaNs in data.x_ucsffsx using TRAIN per-feature means
    • standard-scales with StandardScaler fitted on TRAIN (after imputation)
    
    Returns:
        data_list_train : transformed training graphs (modified in-place)
        scaler          : fitted StandardScaler
        feat_mean       : ndarray (F,) for re-use on test/val
    """
    if len(data_list_train) == 0:
        raise ValueError("data_list_train is empty.")

    # 1) stack (N, F)
    F = data_list_train[0].x_ucsffsx.view(1, -1).size(1)
    rows = [data.x_ucsffsx.view(1, -1) for data in data_list_train]   # (1, F)
    mat  = torch.cat(rows, dim=0)                                     # (N, F)

    # 2) impute NaNs with TRAIN per-feature mean
    np_mat   = mat.detach().cpu().numpy()
    feat_mean = np.nanmean(np_mat, axis=0)                             # (F,)
    nan_idx   = np.where(np.isnan(np_mat))
    print("Number of NaNs in TRAIN UCSFFSX features:", len(nan_idx[0]))
    if len(nan_idx[0]) > 0:
        np_mat[nan_idx] = feat_mean[nan_idx[1]]
    imputed = torch.tensor(np_mat, dtype=torch.float32)                # (N, F)

    # 3) write back imputed (shape: (1, F))
    for i, data in enumerate(data_list_train):
        data.x_ucsffsx = imputed[i].unsqueeze(0)

    # 4) fit scaler on TRAIN (imputed)
    print("Scaling TRAIN UCSFFSX features …")
    scaler = StandardScaler().fit(imputed.detach().cpu().numpy())

    # 5) transform TRAIN and write back
    for data in data_list_train:
        data.x_ucsffsx = torch.tensor(
            scaler.transform(data.x_ucsffsx.detach().cpu().numpy()),
            dtype=torch.float32
        )  # (1, F)

    return data_list_train, scaler, feat_mean


# ------------------------------------------------------------
# TEST/VAL: reuse TRAIN feat_mean + scaler; no refit
# ------------------------------------------------------------
def preprocess_ucsffsx_features_test(data_list_test, scaler, feat_mean):
    """
    Apply TRAIN-time imputation mean and TRAIN-fitted scaler to TEST/VAL graphs.
    • No refitting; strictly uses provided feat_mean and scaler.
    
    Args:
        data_list_test : list of graphs to transform (modified in-place)
        scaler         : StandardScaler fitted on TRAIN
        feat_mean      : ndarray (F,) computed on TRAIN for NaN imputation
    
    Returns:
        data_list_test : transformed test/val graphs
    """
    if len(data_list_test) == 0:
        raise ValueError("data_list_test is empty.")
    if feat_mean.ndim != 1:
        raise ValueError("feat_mean must be a 1D array of shape (F,).")

    F_expected = feat_mean.shape[0]
    F_test = data_list_test[0].x_ucsffsx.view(1, -1).size(1)
    if F_test != F_expected:
        raise ValueError(
            f"Feature dimension mismatch: TRAIN has F={F_expected} but TEST has F={F_test}."
        )

    # 1) stack (N, F)
    rows = [data.x_ucsffsx.view(1, -1) for data in data_list_test]
    mat  = torch.cat(rows, dim=0)                                      # (N, F)

    # 2) impute NaNs using TRAIN feat_mean
    np_mat  = mat.detach().cpu().numpy()
    nan_idx = np.where(np.isnan(np_mat))
    print("Number of NaNs in TEST UCSFFSX features:", len(nan_idx[0]))
    if len(nan_idx[0]) > 0:
        np_mat[nan_idx] = feat_mean[nan_idx[1]]
    imputed = torch.tensor(np_mat, dtype=torch.float32)                # (N, F)

    # 3) transform with TRAIN-fitted scaler
    print("Scaling TEST UCSFFSX features …")
    scaled = scaler.transform(imputed.detach().cpu().numpy())          # (N, F)

    # 4) write back
    scaled_t = torch.tensor(scaled, dtype=torch.float32)
    for i, data in enumerate(data_list_test):
        data.x_ucsffsx = scaled_t[i].unsqueeze(0)                      # (1, F)

    return data_list_test


import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------
def preprocess_gae_embeddings_train(data_list_train):
    """
    • imputes NaNs in data.gae_embedding using TRAIN means
    • fits StandardScaler on TRAIN (after imputation)
    • transforms training embeddings
    • guarantees each gae_embedding has shape (1, latent_dim)

    Returns:
        data_list_train : transformed training graphs
        scaler          : fitted StandardScaler
        feat_mean       : ndarray (latent_dim,)
    """
    if len(data_list_train) == 0:
        raise ValueError("data_list_train is empty.")

    # 1) stack embeddings
    emb_rows = []
    for data in data_list_train:
        if not hasattr(data, "gae_embedding"):
            raise AttributeError("Data object missing gae_embedding attribute")
        emb_rows.append(data.gae_embedding.view(1, -1))
    emb_mat = torch.cat(emb_rows, dim=0)                     # (N, latent_dim)

    # 2) impute NaNs with TRAIN means
    emb_np   = emb_mat.detach().cpu().numpy()
    feat_mean = np.nanmean(emb_np, axis=0)                   # (latent_dim,)
    nan_idx   = np.where(np.isnan(emb_np))
    print("Number of NaNs in TRAIN GAE embeddings:", len(nan_idx[0]))
    if len(nan_idx[0]) > 0:
        emb_np[nan_idx] = feat_mean[nan_idx[1]]
    emb_imputed = torch.tensor(emb_np, dtype=torch.float32)  # (N, latent_dim)

    # 3) fit scaler
    print("Scaling TRAIN GAE embeddings …")
    scaler = StandardScaler().fit(emb_imputed.numpy())

    # 4) transform and write back
    emb_scaled = scaler.transform(emb_imputed.numpy())
    for i, data in enumerate(data_list_train):
        data.gae_embedding = torch.tensor(
            emb_scaled[i], dtype=torch.float32
        ).unsqueeze(0)                                       # (1, latent_dim)

    return data_list_train, scaler, feat_mean


# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------
def preprocess_gae_embeddings_test(data_list_test, scaler, feat_mean):
    """
    Apply TRAIN-time imputation mean + scaler to TEST/VAL GAE embeddings.
    • No refitting; strictly uses TRAIN stats.
    • guarantees each gae_embedding has shape (1, latent_dim)

    Args:
        data_list_test : list of Data objects with gae_embedding
        scaler         : StandardScaler fitted on TRAIN
        feat_mean      : ndarray (latent_dim,) for NaN imputation

    Returns:
        data_list_test : transformed test/val graphs
    """
    if len(data_list_test) == 0:
        raise ValueError("data_list_test is empty.")
    if feat_mean.ndim != 1:
        raise ValueError("feat_mean must be a 1D array (latent_dim,).")

    # 1) stack embeddings
    emb_rows = []
    for data in data_list_test:
        if not hasattr(data, "gae_embedding"):
            raise AttributeError("Data object missing gae_embedding attribute")
        emb_rows.append(data.gae_embedding.view(1, -1))
    emb_mat = torch.cat(emb_rows, dim=0)                     # (N, latent_dim)

    # 2) impute NaNs with TRAIN means
    emb_np   = emb_mat.detach().cpu().numpy()
    nan_idx  = np.where(np.isnan(emb_np))
    print("Number of NaNs in TEST GAE embeddings:", len(nan_idx[0]))
    if len(nan_idx[0]) > 0:
        emb_np[nan_idx] = feat_mean[nan_idx[1]]
    emb_imputed = torch.tensor(emb_np, dtype=torch.float32)  # (N, latent_dim)

    # 3) transform with TRAIN scaler
    print("Scaling TEST GAE embeddings …")
    emb_scaled = scaler.transform(emb_imputed.numpy())

    # 4) write back
    for i, data in enumerate(data_list_test):
        data.gae_embedding = torch.tensor(
            emb_scaled[i], dtype=torch.float32
        ).unsqueeze(0)                                       # (1, latent_dim)

    return data_list_test

def add_similarity_rows_to_x(data_list, discard_fs_features=False, make_symmetric=False):
    """
    For each PyG Data in data_list:
      - Build a dense adjacency-weight matrix A from edge_index/edge_attr
      - Optionally symmetrize A (useful if your graph is undirected but stored as one direction)
      - Concatenate A to existing node features: data.x = [x | A]
    
    After this, if data.x was (N, F), it becomes (N, F + N).
    
    Args:
        data_list (list[torch_geometric.data.Data]): list of Data objects
        make_symmetric (bool): if True, use A = 0.5 * (A + A^T)
    Returns:
        data_list
    """
    for data in data_list:
        # --- sanity checks
        if data.x is None:
            raise ValueError("data.x is None; expected node features to exist.")
        if data.edge_index is None or data.edge_attr is None:
            raise ValueError("edge_index/edge_attr must be present.")
        
        num_nodes = data.num_nodes
        N, F = data.x.size()
        if N != num_nodes:
            raise ValueError(f"data.x has {N} nodes but data.num_nodes={num_nodes}.")

        # --- build dense adjacency-weight matrix A (N x N)
        # keep dtype/device consistent with data.x
        A = torch.zeros((num_nodes, num_nodes), dtype=data.x.dtype, device=data.x.device)

        src, dst = data.edge_index  # shape [E], [E]
        # edge_attr expected shape [E, 1] or [E]; squeeze to [E]
        w = data.edge_attr
        if w.dim() == 2 and w.size(1) == 1:
            w = w.squeeze(1)
        elif w.dim() != 1:
            raise ValueError(f"edge_attr must be shape [E] or [E,1], got {tuple(data.edge_attr.shape)}")

        # place weights into A at (src, dst)
        # if duplicate edges exist, the last assignment wins; to accumulate, use scatter_add instead
        A[src, dst] = w.to(A.dtype)

        # optional symmetrization for undirected similarities
        if make_symmetric:
            A = 0.5 * (A + A.t())

        # --- concatenate to features: [x | A]
        if not discard_fs_features:
            data.x = torch.cat([data.x, A], dim=1)  # (N, F + N)
        else:
            data.x = A
    return data_list



# Utility to flatten node features
import pickle
import pandas as pd
def get_flattened_features(graphs, expected_nodes, include_x = True, include_cog=True, include_mri=True, include_ucsffsx=True, include_gae_embeddings=False, return_visit_details=False):
    X, y = [], []

    # if include_ucsffsx:
    #     with open("./mind_adni1/sparse_pca_model.pkl", "rb") as f:
    #         spca_loaded = pickle.load(f)
        
        
    visit_details = []  # To store visit details if required

    for data in graphs:
        # Flatten x and x_cog separately
        x_flat = data.x.view(-1).numpy()
        if include_cog:
            x_cog_flat = data.x_cog.view(-1).numpy()
        if include_mri:
            x_mri_flat = data.x_mri.view(-1).numpy() 
        if include_ucsffsx:
            x_ucsffsx_flat = data.x_ucsffsx.view(-1).numpy() 
            # Ensure it's 2D: shape (1, n_features)
            x_input = x_ucsffsx_flat.reshape(1, -1)
            # Apply SPCA transformation
            # x_ucsffsx_flat = spca_loaded.transform(x_input)
        if include_gae_embeddings and hasattr(data, 'gae_embedding'):
            gae_flat = data.gae_embedding.view(-1).numpy() 
            # print("GAE embedding shape:", gae_flat.shape)
        features = []

        if include_x:
            features.append(x_flat)
        if include_cog:
            features.append(x_cog_flat)
        if include_mri:
            features.append(x_mri_flat)
        if include_ucsffsx:
            features.append(x_ucsffsx_flat)
        if include_gae_embeddings :
            features.append(gae_flat)



        if return_visit_details:
            ptid = data.ptid
            viscode = data.viscode
            label = int(data.y.cpu().item())
            status = data.status # MCI to AD etc.
            visit_details.append({
                "ptid": ptid,
                "viscode": viscode,
                "label": label,
                "status": status
            })

        feature_names = []
        if include_x:
            num_x = data.x.view(-1).shape[0]
            feature_names += [f"x_pc{i}" for i in range(num_x)]
        if include_cog:
            num_cog = data.x_cog.view(-1).shape[0]
            feature_names += [f"cog_{i}" for i in range(num_cog)]
        if include_mri:
            num_mri = data.x_mri.view(-1).shape[0]
            feature_names += [f"mri_{i}" for i in range(num_mri)]
        if include_ucsffsx:
            num_ucsffsx = data.x_ucsffsx.view(-1).shape[0]
            feature_names += [f"ucsffsx_{i}" for i in range(num_ucsffsx)]
        if include_gae_embeddings and gae_flat is not None:
            num_gae = data.gae_embedding.view(-1).shape[0]
            feature_names += [f"gae_emb_{i}" for i in range(num_gae)]

        if len(features) == 0  :
            raise ValueError("At least one feature set must be included.")

        # Combine all selected features
        combined = np.concatenate(features, axis=0)

        
        # combined = x_cog_flat
        X.append(combined)
        y.append(data.y.item())
        
    if return_visit_details:
        return np.array(X), np.array(y), feature_names, visit_details
    else:
        return np.array(X), np.array(y), feature_names

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch_geometric.data import Data


def fit_pca_on_train(train_data, n_components=0.80, verbose=False):
    """
    Fits PCA on the flattened node features from training graphs
    and updates train_data.x with PCA-reduced features.

    Args:
        train_data (list[Data]): Training graphs (PyG Data objects).
        n_components (int or float): If int, number of PCA components to keep.
                                     If float (0-1), target explained variance.
        verbose (bool): Print variance info.

    Returns:
        tuple: (updated_train_data, fitted_pca)
    """
    # Collect flattened features for all training graphs
    all_features = []
    for data in train_data:
        x_flat = data.x.view(-1).numpy()
        all_features.append(x_flat)
    all_features = np.stack(all_features)  # [num_graphs, num_nodes*num_features]

    # Fit PCA
    pca = PCA(n_components=n_components)
    all_features_pca = pca.fit_transform(all_features)

    if verbose:
        print(f"PCA n_components selected: {pca.n_components_}")
        print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
        print(f"Cumulative explained variance: {pca.explained_variance_ratio_.sum():.2f}")

    # Update train graphs with reduced features
    for idx, data in enumerate(train_data):
        reduced = torch.tensor(all_features_pca[idx], dtype=torch.float32)
        data.x = reduced.unsqueeze(0)  # [1, num_components]

    return train_data, pca


def apply_pca_to_graphs(data_list, pca):
    """
    Applies an already fitted PCA to a new list of graphs (e.g., test set).

    Args:
        data_list (list[Data]): List of PyG Data objects.
        pca (PCA): Fitted sklearn PCA object.

    Returns:
        list: Updated data_list with PCA-reduced features.
    """
    all_features = []
    for data in data_list:
        x_flat = data.x.view(-1).numpy()
        all_features.append(x_flat)
    all_features = np.stack(all_features)

    # Transform using the fitted PCA
    all_features_pca = pca.transform(all_features)

    # Update each graph
    for idx, data in enumerate(data_list):
        reduced = torch.tensor(all_features_pca[idx], dtype=torch.float32)
        data.x = reduced.unsqueeze(0)

    return data_list



import numpy as np
import torch

def fit_icv_normalizer(train_data, feature_indices, icv_attr="ICV"):
    """
    Fits OASIS-style ICV normalization parameters on training fold only.

    For each node and each selected feature:
        ROI = a + B*ICV
    Returns mean_icv (scalar) and B of shape (num_nodes, len(feature_indices)).
    """
    # Collect ICV (N,)
    icv = []
    for d in train_data:
        if not hasattr(d, icv_attr):
            raise AttributeError(f"Missing {icv_attr} on a graph.")
        val = float(getattr(d, icv_attr))
        icv.append(val)
    icv = np.asarray(icv, dtype=np.float64)  # (N,)
    mean_icv = float(icv.mean())

    # Stack features: (N, num_nodes, num_features)
    X = []
    for d in train_data:
        x = d.x.detach().cpu().numpy()
        X.append(x)
    X = np.stack(X, axis=0)  # (N, num_nodes, F)

    num_nodes = X.shape[1]
    feats = np.array(feature_indices, dtype=int)
    Y = X[:, :, feats]  # (N, num_nodes, K)

    # Compute slope B (unstandardized) for each node+feature:
    # B = Cov(ICV, ROI) / Var(ICV)
    icv_centered = icv - mean_icv                        # (N,)
    var_icv = np.mean(icv_centered ** 2)                 # scalar
    if var_icv == 0:
        raise ValueError("ICV has zero variance in training fold; cannot fit regression slope.")

    # Cov over subjects: mean( (icv-mean) * (roi-mean_roi) )
    Y_mean = Y.mean(axis=0, keepdims=True)               # (1, num_nodes, K)
    Y_centered = Y - Y_mean                               # (N, num_nodes, K)

    cov = np.mean(icv_centered[:, None, None] * Y_centered, axis=0)  # (num_nodes, K)
    B = cov / var_icv                                     # (num_nodes, K)

    params = {
        "mean_icv": mean_icv,
        "B": B.astype(np.float32),
        "feature_indices": feats.tolist(),
        "icv_attr": icv_attr,
    }
    return params


def apply_icv_normalizer(data_list, params):
    """
    Applies: normalized = raw - (B * (ICV - meanICV))
    where B is per node per selected feature.
    """
    mean_icv = float(params["mean_icv"])
    B = params["B"]  # (num_nodes, K)
    feats = np.array(params["feature_indices"], dtype=int)
    icv_attr = params.get("icv_attr", "ICV")

    for d in data_list:
        icv = float(getattr(d, icv_attr))
        delta = (icv - mean_icv)  # scalar

        x = d.x.detach().cpu().numpy()  # (num_nodes, F)
        # subtract B * delta for each selected feature
        x[:, feats] = x[:, feats] - (B * delta)
        d.x = torch.tensor(x, dtype=d.x.dtype)
    return data_list

def get_feature_slices(excluded_node_features):
    if excluded_node_features is None:
        return {
            "ct":  slice(0, 4),
            "vol": slice(4, 8),
            "sa":  slice(8, 12),
            "mc":  slice(12, 16),
            "sd":  slice(16, 20),
        }
    elif excluded_node_features == "min_max":
        return {
            "ct":  slice(0, 2),
            "vol": slice(4, 6),
            "sa":  slice(8, 10),
            "mc":  slice(12, 14),
            "sd":  slice(16, 18),
        }
    elif excluded_node_features == "std_min_max":
        return {
            "ct":  slice(0, 1),
            "vol": slice(4, 5),
            "sa":  slice(8, 9),
            "mc":  slice(12, 13),
            "sd":  slice(16, 17),
        }
    else:        
        raise ValueError(f"Unknown option for --excluded_node_features: {excluded_node_features}")