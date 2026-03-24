# Retrieved from exp4 utils
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from torch_geometric.utils import add_self_loops, remove_self_loops
from copy import deepcopy
import pickle
import math
# def remove_low_weight_edges_pyg(data, threshold=0.05):
#     """
#     Remove edges from a PyG Data object whose edge_attr (assumed weight) is below threshold.
#     Assumes undirected graph: if (u,v) is removed, (v,u) is also removed.
#     """
#     data = deepcopy(data)  # Clone the original to avoid in-place changes

#     edge_index = data.edge_index
#     edge_attr = data.edge_attr

#     if edge_attr is None:
#         raise ValueError("Edge attributes (edge_attr) are required for thresholding based on weight.")

#     # Identify edges to keep
#     keep_mask = edge_attr.view(-1) >= threshold

#     # Apply the mask
#     new_edge_index = edge_index[:, keep_mask]
#     new_edge_attr = edge_attr[keep_mask]

#     data.edge_index = new_edge_index
#     data.edge_attr = new_edge_attr

#     return data


def remove_low_weight_edges_pyg(data, threshold=0.8):
    """
    Keep the top percentage of UNIQUE undirected edges in a PyG Data object.

    Parameters
    ----------
    data : torch_geometric.data.Data
        PyG graph with:
          - edge_index of shape [2, E]
          - edge_attr of shape [E] or [E, 1] (edge weights)
        Assumes an undirected graph, typically stored with both (u,v) and (v,u).

    threshold : float
        Fraction of unique undirected edges to keep.
        Examples:
          - 0.0 -> keep no edges
          - 0.8 -> keep top 80% of unique undirected edges by weight
          - 1.0 -> keep all edges
        Must satisfy 0 <= threshold <= 1.

    Returns
    -------
    data : torch_geometric.data.Data
        New copied Data object with filtered edge_index and edge_attr.

    Notes
    -----
    - Unique undirected edges are identified by sorting endpoints:
          (u, v) and (v, u) -> (min(u,v), max(u,v))
    - If duplicate copies of the same undirected edge exist, this function keeps
      the maximum weight among them for ranking.
    - After selection, both directions are written back:
          (u,v) and (v,u)
      with the same weight.
    """

    if not (0 <= threshold <= 1):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    data = deepcopy(data)

    edge_index = data.edge_index
    edge_attr = data.edge_attr

    if edge_attr is None:
        raise ValueError("edge_attr is required for weight-based filtering.")

    if edge_index is None or edge_index.numel() == 0:
        return data

    weights = edge_attr.view(-1)

    if edge_index.size(1) != weights.size(0):
        raise ValueError("edge_index and edge_attr must have the same number of edges.")

    if threshold == 1.0:
        return data

    # Step 1: collect unique undirected edges
    # key = (min(u,v), max(u,v))
    # value = max weight seen for that undirected edge
    undirected_edge_to_weight = {}

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    wts = weights.tolist()

    for u, v, w in zip(src, dst, wts):
        a, b = (u, v) if u <= v else (v, u)
        key = (a, b)

        if key not in undirected_edge_to_weight:
            undirected_edge_to_weight[key] = w
        else:
            undirected_edge_to_weight[key] = max(undirected_edge_to_weight[key], w)

    unique_edges = list(undirected_edge_to_weight.items())
    num_unique = len(unique_edges)

    if num_unique == 0:
        data.edge_index = edge_index[:, :0]
        data.edge_attr = edge_attr[:0]
        return data

    # threshold == 0 -> remove all edges
    if threshold == 0.0:
        data.edge_index = edge_index[:, :0]
        data.edge_attr = edge_attr[:0]
        return data

    # Step 2: compute how many unique undirected edges to keep
    k = math.ceil(num_unique * threshold)
    k = min(k, num_unique)

    # Step 3: sort unique undirected edges by descending weight and keep top-k
    unique_edges.sort(key=lambda x: x[1], reverse=True)
    kept_unique_edges = unique_edges[:k]

    # Step 4: rebuild edge_index and edge_attr with both directions
    new_edges = []
    new_weights = []

    for (u, v), w in kept_unique_edges:
        if u == v:
            # self-loop: keep only once
            new_edges.append([u, v])
            new_weights.append(w)
        else:
            new_edges.append([u, v])
            new_edges.append([v, u])
            new_weights.append(w)
            new_weights.append(w)

    if len(new_edges) == 0:
        new_edge_index = edge_index[:, :0]
        new_edge_attr = edge_attr[:0]
    else:
        new_edge_index = torch.tensor(
            new_edges, dtype=edge_index.dtype, device=edge_index.device
        ).t().contiguous()

        new_edge_attr = torch.tensor(
            new_weights, dtype=edge_attr.dtype, device=edge_attr.device
        )

        # Preserve original edge_attr shape if it was [E, 1] or [E, ...]
        if edge_attr.dim() > 1:
            new_edge_attr = new_edge_attr.view(-1, *edge_attr.shape[1:])

    data.edge_index = new_edge_index
    data.edge_attr = new_edge_attr

    return data


def read_cross_val(pkl_path):
    """
    Read the cross-validation splits from a pickle file.
    
    Args:
        pkl_path (str): Path to the pickle file containing the splits.
        
    Returns:
        list: A list of dictionaries, each containing 'train_files' and 'test_files' for each split.
    """
    with open(pkl_path, 'rb') as f:
        splits = pickle.load(f)
    return splits
# 1. Dataset utilities
def load_dataset(dataset_path: str, apply_log_transform: bool = False, convert_labels: bool = True):
    """Load all PyG data objects (.pt files) from a directory."""
    data_list = []
    fill_value = 1  # weight for self-loops
    for fname in sorted(os.listdir(dataset_path)):
        if fname.endswith(".pt"):
            data = torch.load(os.path.join(dataset_path, fname), weights_only=False)

            if getattr(data, "edge_index", None) is not None:
                try:
                    # Remove existing self-loops (avoid duplicates)
                    edge_index, edge_attr = remove_self_loops(data.edge_index, data.edge_attr)

                    # Add self-loops for all nodes with weight = 1
                    num_nodes = data.x.size(0)
                    edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=fill_value, num_nodes=num_nodes)

                    # Optional: apply log-transform to edge weights
                    if apply_log_transform and edge_attr is not None:
                        # TODO: Rewrite this without log1p
                        # # Shift by a small epsilon to avoid log(0)
                        # eps = 1e-6
                        # # Apply log1p (log(1+x)) for numerical stability
                        # edge_attr = torch.log1p(torch.clamp(edge_attr, min=0) + eps)
                        pass

                    # Assign back
                    data.edge_index = edge_index
                    data.edge_attr = edge_attr

                except Exception as e:
                    print(f"Warning: could not process self-loops or log-transform for {fname} ({e})")

            # Convert label to 0/1
            if convert_labels:
                data.y = (data.y - 1).long().view(-1)
            data_list.append(data)

    print(f"Loaded {len(data_list)} PyG graphs (self-loops weight={fill_value}, log-transform={apply_log_transform}).")
    return data_list

def load_dataset_from_single_pt(dataset_path: str, apply_log_transform: bool = False):
    """Load a single .pt file containing a list of PyG data objects."""
    data_list = torch.load(dataset_path, weights_only=False)
    fill_value = 1  # weight for self-loops
    data_list_processed = []
    for data in data_list:
        if getattr(data, "edge_index", None) is not None:
            try:
                # Remove existing self-loops (avoid duplicates)
                edge_index, edge_attr = remove_self_loops(data.edge_index, data.edge_attr)

                # Add self-loops for all nodes with weight = 1
                num_nodes = data.x.size(0)
                edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=fill_value, num_nodes=num_nodes)

                data.edge_index = edge_index
                data.edge_attr = edge_attr


                # Optional: apply log-transform to edge weights
                if apply_log_transform and edge_attr is not None:
                    pass
            except:
                print(f"Warning: could not process self-loops or log-transform for a data object.")
        # Convert label to 0/1
        data.y = (data.y - 1).long().view(-1)
        data_list_processed.append(data)
    print(f"Loaded {len(data_list_processed)} PyG graphs from single .pt file (self-loops weight={fill_value}, log-transform={apply_log_transform}).")
    return data_list_processed


# 2. Precompute adjacency matrices
def attach_weighted_adj_matrices(dataset_path: str, num_nodes: int):
    """
    Compute weighted adjacency matrices once and attach as data.weighted_adj_matrix.
    Works for:
      • a single .pt file containing a LIST of PyG Data objects
      • a directory containing many .pt files, each a single PyG Data object

    Saves the updated objects back to disk.
    """

    # CASE 1: Single .pt file containing a LIST of Data objects
    if dataset_path.endswith(".pt") and os.path.isfile(dataset_path):
        print(f"Loading single .pt file: {dataset_path}")
        data_list = torch.load(dataset_path, weights_only=False)

        if not isinstance(data_list, list):
            raise ValueError("Expected a list of PyG Data objects in the .pt file.")

        print(f"Precomputing weighted adjacency matrices for {len(data_list)} graphs...")

        for data in tqdm(data_list):

            adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

            if hasattr(data, "edge_index") and hasattr(data, "edge_attr"):
                edge_index = data.edge_index
                edge_attr = data.edge_attr.view(-1)

                adj[edge_index[0], edge_index[1]] = edge_attr
                adj[edge_index[1], edge_index[0]] = edge_attr

            data.weighted_adj_matrix = adj.unsqueeze(0)  # shape [1, N, N]

            # print the  shapes of components
            print(f"data.edge_index shape: {data.edge_index.shape}, data.edge_attr shape: {data.edge_attr.shape}, data.weighted_adj_matrix shape: {data.weighted_adj_matrix.shape}")
            
        # overwrite the single file
        torch.save(data_list, dataset_path)
        print("Done. Updated the single data_list .pt file.")
        return

    # ------------------------------------------------------
    # CASE 2: A directory containing many single-graph .pt files
    # ------------------------------------------------------
    elif os.path.isdir(dataset_path):
        files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".pt")])
        print(f"Precomputing weighted adjacency matrices for {len(files)} graph files...")

        for fname in tqdm(files):
            fpath = os.path.join(dataset_path, fname)
            data = torch.load(fpath, weights_only=False)

            adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

            if hasattr(data, "edge_index") and hasattr(data, "edge_attr"):
                edge_index = data.edge_index
                edge_attr = data.edge_attr.view(-1)

                adj[edge_index[0], edge_index[1]] = edge_attr
                adj[edge_index[1], edge_index[0]] = edge_attr

            data.weighted_adj_matrix = adj.unsqueeze(0)  # [1, N, N]

            torch.save(data, fpath)

        print("Done. Updated all .pt files in the directory.")
        return

    else:
        raise NotADirectoryError(f"Invalid path: {dataset_path}")


# 3. Training / Evaluation
def train_one_epoch(model, loader, optimizer, device, criterion=None):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        if criterion is None: 
            loss = F.cross_entropy(logits, data.y)
        else:
            loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


# @torch.no_grad()
# def evaluate(model, loader, device):
#     model.eval()
#     y_true, y_pred = [], []
#     total_loss = 0.0

#     for data in loader:
#         data = data.to(device)
#         logits = model(data)
#         loss = F.cross_entropy(logits, data.y)
#         total_loss += loss.item() * data.y.size(0)  # batch-size weighted loss

#         preds = logits.argmax(dim=1)

#         # Extend lists with the batch contents, not append
#         y_true.extend(data.y.cpu().tolist())
#         y_pred.extend(preds.cpu().tolist())

#     # Convert to tensors
#     y_true = torch.tensor(y_true)
#     y_pred = torch.tensor(y_pred)

#     acc = accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred, average="weighted")

#     mean_loss = total_loss / len(loader.dataset)
#     return mean_loss, acc, f1

@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    y_true, y_pred, all_status, y_probs = [], [], [], []
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        logits = model(data)
        probs = F.softmax(logits, dim=1)  # class probabilities
        if criterion is None:
            loss = F.cross_entropy(logits, data.y)
        else:
            loss = criterion(logits, data.y)
        total_loss += loss.item() * data.y.size(0)

        preds = logits.argmax(dim=1)
        y_probs.extend(probs[:, 1].detach().cpu().tolist())  # probability of AD class

        y_true.extend(data.y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        # Keep track of patient status if available
        if hasattr(data, "status"):
            if isinstance(data.status, list):
                all_status.extend(data.status)
            else:
                all_status.extend([data.status] * data.num_graphs)

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

# Compute AUC (only if both classes are present)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = np.nan  # occurs when only one class is present in y_true

    # Compute conversion recall (for "MCI to Dementia" subset)
    if len(all_status) > 0:
        conv_mask = np.array([s == "MCI to Dementia" for s in all_status])
        if conv_mask.sum() > 0:
            conv_recall = (y_pred[conv_mask] == 1).mean()  # prediction=1 means AD
        else:
            conv_recall = np.nan
    else:
        conv_recall = np.nan

    mean_loss = total_loss / len(loader.dataset)

    return mean_loss, acc, f1_weighted, f1_macro, precision, recall, auc, conv_recall



@torch.no_grad()
def evaluate_with_attention(model, loader, device, branch_names=("GNN", "CNN", "MLP")):
    model.eval()
    y_true, y_pred = [], []
    attn_records = []  # list of dicts {sample_id, branch_name, weight}

    for data in tqdm(loader, desc="Eval", leave=False):
        data = data.to(device)
        logits, attn_w = model(data, return_attn=True)   # logits: [B, C], attn_w: [B, num_branches]
        preds = logits.argmax(dim=1)                     # [B]
        batch_size = preds.size(0)

        # ✅ Append all predictions and labels (no .item() on batched tensors)
        y_true.extend(data.y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        # ✅ Handle attention weights per sample in the batch
        attn_weights = attn_w.detach().cpu().tolist()  # [[w1,w2,w3], ...]

        # Handle ptid and viscode (could be lists, tensors, or single values)
        ptid_attr = getattr(data, "ptid", ["unknown"] * batch_size)
        viscode_attr = getattr(data, "viscode", ["unknown"] * batch_size)

        # Normalize to list of strings
        if not isinstance(ptid_attr, (list, tuple)):
            ptid_list = [ptid_attr] * batch_size
        else:
            ptid_list = ptid_attr

        if not isinstance(viscode_attr, (list, tuple)):
            viscode_list = [viscode_attr] * batch_size
        else:
            viscode_list = viscode_attr

        # ✅ Loop over batch elements
        for i in range(batch_size):
            ptid = ptid_list[i]
            viscode = viscode_list[i]
            if torch.is_tensor(ptid):
                ptid = ptid.item() if ptid.numel() == 1 else str(ptid.tolist())
            if torch.is_tensor(viscode):
                viscode = viscode.item() if viscode.numel() == 1 else str(viscode.tolist())

            sample_id = f"{ptid}_{viscode}"
            rec = {"sample_id": sample_id}

            for bname, w in zip(branch_names, attn_weights[i]):
                rec[bname] = w

            attn_records.append(rec)

    df_attn = pd.DataFrame(attn_records)
    return y_true, y_pred, df_attn



from torch_geometric.transforms import AddLaplacianEigenvectorPE

def add_laplacian_pe(data_list, pe_dim=8):
    transform = AddLaplacianEigenvectorPE(
        k=pe_dim,
        is_undirected=True,
        attr_name="laplacian_pe",   # new attribute stored inside data
        normalization='sym'
    )

    new_list = []
    for data in data_list:

        # Convert [1, N, N] → [N, N]
        if hasattr(data, "weighted_adj_matrix"):
            A = data.weighted_adj_matrix
            if A.dim() == 3 and A.size(0) == 1:
                A = A.squeeze(0)     # FIX: remove batch dim

            # Should now be [N, N]
            assert A.dim() == 2, f"Expected 2D adjacency, got {A.shape}"

            # Convert to PyG edge_index / edge_weight
            edge_index = A.nonzero(as_tuple=False).t().contiguous()
            edge_weight = A[edge_index[0], edge_index[1]].float()

            data.edge_index = edge_index
            data.edge_weight = edge_weight

        # Compute Laplacian PE
        data = transform(data)
        new_list.append(data)

    return new_list



def compute_lpe_for_graph(data, k=8):
    """
    data must have a correct `edge_index` and optionally `edge_weight`.
    Returns data with data.laplacian_pe of size [N, k].
    """
    transform = AddLaplacianEigenvectorPE(
        k=k,
        attr_name="laplacian_pe",   # new attribute stored inside data
        is_undirected=True,
        normalization='sym'
    )
    A = data.weighted_adj_matrix.squeeze(0)  # [N, N]
    edge_index = A.nonzero(as_tuple=False).t().contiguous()
    edge_weight = A[edge_index[0], edge_index[1]].float()
    data.edge_index = edge_index
    data.edge_weight = edge_weight

    return transform(data)


def get_used_filenames_from_splits(splits, include_val_in_train=True):
    used = set()
    for sp in splits:
        used.update(sp.get("train_files", []))
        used.update(sp.get("test_files", []))
        if include_val_in_train:
            used.update(sp.get("val_files", []))
    return used

def filter_data_list_by_splits(data_list, used_filenames, dataset="adni"):
    kept = []
    missing_key = 0

    for d in data_list:
        # Build the key exactly like your split filenames
        # (you currently use ptid + "_" + viscode + ".pt")
        key = f"{d.ptid}_{d.viscode}.pt"

        if key in used_filenames:
            kept.append(d)
        else:
            missing_key += 1

    print(f"Filtered data_list: kept {len(kept)}/{len(data_list)} graphs (dropped {missing_key}).")
    return kept


from torch_geometric.utils import sort_edge_index

def sort_edges_for_lstm_aggr(data):
    # Sort by destination nodes (edge_index[1]) so LSTM aggregation works
    if getattr(data, "edge_index", None) is None:
        return data

    if getattr(data, "edge_attr", None) is not None:
        data.edge_index, data.edge_attr = sort_edge_index(
            data.edge_index, data.edge_attr, sort_by_row=False
        )
    else:
        data.edge_index = sort_edge_index(data.edge_index, sort_by_row=False)

    return data