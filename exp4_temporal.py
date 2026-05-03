# General imports
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse, random, datetime, json, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML imports
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

# Local imports
from exp4_model import FusionModel  

import utils.observe as observe
import utils.general as general
import utils.preprocessing as preprocessing
import utils.plotting as plotting

from collections import defaultdict
import re


# -------------------------------------------------------------------
# SEEDING
# -------------------------------------------------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------
# VISIT SORTING
# -------------------------------------------------------------------
def viscode_to_month(viscode):
    """
    Convert ADNI VISCODE strings to a sortable numeric time.

    Examples:
        bl -> 0
        sc -> 0
        m06 -> 6
        m12 -> 12
        m24 -> 24

    Unknown / unparsable visits are pushed to the end.
    """
    if viscode is None:
        return float("inf")

    v = str(viscode).strip().lower()

    if v in {"bl", "sc"}:
        return 0

    m = re.fullmatch(r"m(\d+)", v)
    if m:
        return int(m.group(1))

    return float("inf")


def build_subject_to_visits(data_list, subject_attr="ptid", visit_attr="viscode"):
    """
    Group processed visit-level Data objects by subject and sort visits by time.
    """
    subject_to_visits = defaultdict(list)

    for data in data_list:
        subject_id = getattr(data, subject_attr)
        subject_to_visits[subject_id].append(data)

    for subject_id in subject_to_visits:
        subject_to_visits[subject_id] = sorted(
            subject_to_visits[subject_id],
            key=lambda d: viscode_to_month(getattr(d, visit_attr, None))
        )

    return dict(subject_to_visits)




def make_many_to_many_pyg_visit_samples(subject_to_visits):
    """
    Build one full PyG-Data sequence per subject.

    Each sample contains:
      - data_seq: list of PyG Data objects
      - y_seq: Tensor [T]
      - ptid
      - viscodes
      - seq_len
      - status_seq
    """
    samples = []

    for ptid, visits in subject_to_visits.items():
        if len(visits) == 0:
            continue

        valid_visits = []
        y_list = []

        for v in visits:
            y_val = int(v.y.item()) if torch.is_tensor(v.y) else int(v.y)

            if y_val not in [0, 1]:
                continue

            valid_visits.append(v)
            y_list.append(y_val)

        if len(valid_visits) == 0:
            continue

        sample = {
            "data_seq": valid_visits,
            "y_seq": torch.tensor(y_list, dtype=torch.long),
            "ptid": ptid,
            "viscodes": [v.viscode for v in valid_visits],
            "seq_len": len(valid_visits),
            "status_seq": [getattr(v, "status", None) for v in valid_visits],
        }

        samples.append(sample)

    return samples

# -------------------------------------------------------------------
# DATASET + COLLATE
# -------------------------------------------------------------------
class TemporalDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

from torch.nn.utils.rnn import pad_sequence


def temporal_collate_fn_pyg_many_to_many(batch, pad_label=-100):
    """
    Keeps data_seq as a list of list[PyG Data].

    Returns:
      data_seqs:  list of length B, each item is list[Data] of length T_i
      y_padded:   [B, T_max]
      lengths:    [B]
      mask:       [B, T_max]
    """
    data_seqs = [item["data_seq"] for item in batch]
    y_seqs = [item["y_seq"] for item in batch]

    lengths = torch.tensor([item["seq_len"] for item in batch], dtype=torch.long)

    y_padded = pad_sequence(
        y_seqs,
        batch_first=True,
        padding_value=pad_label
    )  # [B, T_max]

    T_max = y_padded.size(1)
    mask = torch.arange(T_max).unsqueeze(0) < lengths.unsqueeze(1)

    ptids = [item["ptid"] for item in batch]
    viscodes = [item["viscodes"] for item in batch]
    status_seq = [item["status_seq"] for item in batch]

    return {
        "data_seqs": data_seqs,
        "y_padded": y_padded,
        "lengths": lengths,
        "mask": mask,
        "ptids": ptids,
        "viscodes": viscodes,
        "status_seq": status_seq,
    }



import torch
import torch.nn as nn
from torch_geometric.data import Batch


class FusionRecurrentManyToMany(nn.Module):
    def __init__(
        self,
        fusion_encoder,
        temporal_type="lstm",   # "rnn", "gru", "lstm", or "transformer"
        hidden_dim=256,
        num_layers=1,
        num_classes=2,
        recurrent_dropout=0.0,
        classifier_dropout=0.2,
        freeze_fusion_encoder=False,
        rnn_nonlinearity="tanh",  # only used when temporal_type="rnn"

        # Transformer-specific options
        transformer_nhead=4,
        transformer_dim_feedforward=None,
        transformer_activation="gelu",
        transformer_causal_mask=False,
        use_positional_embedding=True,
        max_seq_len=32,
    ):
        super().__init__()

        self.fusion_encoder = fusion_encoder
        self.freeze_fusion_encoder = freeze_fusion_encoder
        self.temporal_type = temporal_type.lower()

        self.transformer_causal_mask = transformer_causal_mask
        self.use_positional_embedding = use_positional_embedding
        self.max_seq_len = max_seq_len

        if not hasattr(fusion_encoder, "concat_dim"):
            raise ValueError(
                "FusionModel must have self.concat_dim. "
                "Add self.concat_dim = concat_dim inside FusionModel.__init__."
            )

        self.encoder_out_dim = fusion_encoder.concat_dim

        # Dropout applied directly to encoded visit embeddings
        self.pre_recurrent = nn.Dropout(recurrent_dropout)

        internal_recurrent_dropout = recurrent_dropout if num_layers > 1 else 0.0

        if self.temporal_type == "lstm":
            self.temporal = nn.LSTM(
                input_size=self.encoder_out_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=internal_recurrent_dropout,
                bidirectional=False,
            )

            self.temporal_out_dim = hidden_dim

        elif self.temporal_type == "gru":
            self.temporal = nn.GRU(
                input_size=self.encoder_out_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=internal_recurrent_dropout,
                bidirectional=False,
            )

            self.temporal_out_dim = hidden_dim

        elif self.temporal_type == "rnn":
            self.temporal = nn.RNN(
                input_size=self.encoder_out_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=internal_recurrent_dropout,
                bidirectional=False,
                nonlinearity=rnn_nonlinearity,
            )

            self.temporal_out_dim = hidden_dim

        elif self.temporal_type == "transformer":
            if self.encoder_out_dim % transformer_nhead != 0:
                raise ValueError(
                    f"encoder_out_dim={self.encoder_out_dim} must be divisible by "
                    f"transformer_nhead={transformer_nhead}."
                )

            if transformer_dim_feedforward is None:
                transformer_dim_feedforward = 4 * self.encoder_out_dim

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.encoder_out_dim,
                nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward,
                dropout=recurrent_dropout,
                activation=transformer_activation,
                batch_first=True,
                norm_first=True,
            )

            self.temporal = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
            )

            if use_positional_embedding:
                self.positional_embedding = nn.Embedding(
                    max_seq_len,
                    self.encoder_out_dim,
                )
            else:
                self.positional_embedding = None

            self.temporal_out_dim = self.encoder_out_dim

        else:
            raise ValueError(
                f"Unknown temporal_type={temporal_type}. "
                "Expected one of: 'rnn', 'gru', 'lstm', 'transformer'."
            )

        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.temporal_out_dim, num_classes)

        if freeze_fusion_encoder:
            for p in self.fusion_encoder.parameters():
                p.requires_grad = False

    def _make_transformer_attention_mask(self, T, device):
        """
        Returns attention mask of shape [T, T].

        In PyTorch TransformerEncoder:
            mask[q, k] = True means query position q CANNOT attend to key position k.

        Causal next_to_previous mask:
            Later visits can attend to previous visits.
            Earlier visits cannot attend to later visits.

        Allowed:
            t=0 -> 0
            t=1 -> 0,1
            t=2 -> 0,1,2
            t=3 -> 0,1,2,3

        Mask:
            upper triangular part above diagonal is True.
        """

        return torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device),
            diagonal=1
        )

    def forward(self, data_seqs, lengths):
        """
        Parameters
        ----------
        data_seqs:
            list of length B.
            Each item is a list of PyG Data objects for one subject.

        lengths:
            Tensor [B], number of valid visits for each subject.

        Returns
        -------
        logits:
            Tensor [B, T_max, num_classes]
        """
        device = next(self.parameters()).device

        lengths = lengths.to(device)
        B = len(data_seqs)
        T_max = int(lengths.max().item())

        # ---------------------------------------------------------
        # Flatten all valid visits across the batch
        # ---------------------------------------------------------
        flat_visits = []
        positions = []

        for i, seq in enumerate(data_seqs):
            for t, data in enumerate(seq):
                flat_visits.append(data)
                positions.append((i, t))

        if len(flat_visits) == 0:
            raise ValueError("Received an empty temporal batch.")

        pyg_batch = Batch.from_data_list(flat_visits).to(device)

        # ---------------------------------------------------------
        # Encode each visit using your existing FusionModel
        # ---------------------------------------------------------
        if self.freeze_fusion_encoder:
            with torch.no_grad():
                flat_embeddings = self.fusion_encoder.encode(pyg_batch)
        else:
            flat_embeddings = self.fusion_encoder.encode(pyg_batch)

        # flat_embeddings: [num_valid_visits, fusion_dim]
        fusion_dim = flat_embeddings.size(-1)

        # ---------------------------------------------------------
        # Restore padded temporal tensor: [B, T_max, fusion_dim]
        # ---------------------------------------------------------
        x_encoded = torch.zeros(
            B,
            T_max,
            fusion_dim,
            device=device,
            dtype=flat_embeddings.dtype,
        )

        for idx, (i, t) in enumerate(positions):
            x_encoded[i, t] = flat_embeddings[idx]

        # ---------------------------------------------------------
        # Dropout before temporal module
        # ---------------------------------------------------------
        x_encoded = self.pre_recurrent(x_encoded)

        # ---------------------------------------------------------
        # Temporal module over visit embeddings
        # ---------------------------------------------------------
        if self.temporal_type == "transformer":
            if self.use_positional_embedding:
                if T_max > self.max_seq_len:
                    raise ValueError(
                        f"T_max={T_max} is larger than max_seq_len={self.max_seq_len}. "
                        "Increase max_seq_len."
                    )

                pos = torch.arange(T_max, device=device)
                pos_emb = self.positional_embedding(pos)  # [T_max, fusion_dim]
                x_encoded = x_encoded + pos_emb.unsqueeze(0)

            # key_padding_mask:
            # True means this position is padding and should be ignored.
            time_ids = torch.arange(T_max, device=device).unsqueeze(0)  # [1, T_max]
            key_padding_mask = time_ids >= lengths.unsqueeze(1)         # [B, T_max]

            if self.transformer_causal_mask:
                attn_mask = self._make_transformer_attention_mask(
                    T=T_max,
                    device=device,
                )
            else:
                attn_mask = None

            out_padded = self.temporal(
                x_encoded,
                mask=attn_mask,
                src_key_padding_mask=key_padding_mask,
            )

        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                x_encoded,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            packed_out, _ = self.temporal(packed)

            out_padded, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=T_max,
            )

        # ---------------------------------------------------------
        # Classification
        # ---------------------------------------------------------
        out_padded = self.classifier_dropout(out_padded)

        logits = self.classifier(out_padded)  # [B, T_max, num_classes]

        return logits
# -------------------------------------------------------------------
# TRAIN / EVAL / PREDICT
# -------------------------------------------------------------------
def train_one_epoch_temporal_many_to_many(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        data_seqs = batch["data_seqs"]
        y_padded = batch["y_padded"].to(device)
        lengths = batch["lengths"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        logits = model(data_seqs, lengths)
        B, T, C = logits.shape

        loss = criterion(
            logits.reshape(B * T, C),
            y_padded.reshape(B * T)
        )
        loss.backward()
        optimizer.step()

        valid_tokens = mask.sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    return avg_loss


@torch.no_grad()
def evaluate_temporal_many_to_many(model, loader, device, criterion=None):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    all_y_true = []
    all_y_pred = []
    all_prob_pos = []

    for batch in loader:
        data_seqs = batch["data_seqs"]
        y_padded = batch["y_padded"].to(device)
        lengths = batch["lengths"].to(device)
        mask = batch["mask"].to(device)

        logits = model(data_seqs, lengths)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        B, T, C = logits.shape

        if criterion is not None:
            loss = criterion(
                logits.reshape(B * T, C),
                y_padded.reshape(B * T)
            )
            valid_tokens = mask.sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

        valid_y = y_padded[mask]
        valid_preds = preds[mask]
        valid_prob_pos = probs[..., 1][mask]

        all_y_true.extend(valid_y.cpu().numpy().tolist())
        all_y_pred.extend(valid_preds.cpu().numpy().tolist())
        all_prob_pos.extend(valid_prob_pos.cpu().numpy().tolist())

    if len(all_y_true) == 0:
        avg_loss = 0.0
        acc = 0.0
        f1_weighted = 0.0
        f1_macro = 0.0
        precision = [0.0, 0.0]
        recall = [0.0, 0.0]
        auc = float("nan")
        conv_recall = float("nan")
        return avg_loss, acc, f1_weighted, f1_macro, precision, recall, auc, conv_recall

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_prob_pos = np.array(all_prob_pos)

    avg_loss = total_loss / total_tokens if (criterion is not None and total_tokens > 0) else 0.0
    acc = float((all_y_true == all_y_pred).mean())
    f1_weighted = f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)

    precision, recall, _, _ = precision_recall_fscore_support(
        all_y_true,
        all_y_pred,
        labels=[0, 1],
        zero_division=0
    )

    if len(np.unique(all_y_true)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(all_y_true, all_prob_pos)

    conv_recall = float("nan")
    return avg_loss, acc, f1_weighted, f1_macro, precision, recall, auc, conv_recall

@torch.no_grad()
def predict_temporal_many_to_many(model, loader, device):
    model.eval()

    records = []

    for batch in loader:
        data_seqs = batch["data_seqs"]
        y_padded = batch["y_padded"].to(device)
        lengths = batch["lengths"].to(device)

        logits = model(data_seqs, lengths)   # [B, T, C]
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        B = len(data_seqs)

        for i in range(B):
            seq_len = int(batch["lengths"][i].item())

            for t in range(seq_len):
                records.append({
                    "ptid": batch["ptids"][i],
                    "viscode": batch["viscodes"][i][t],
                    "time_index": t,
                    "seq_len": seq_len,
                    "label": int(y_padded[i, t].cpu().item()),
                    "prediction": int(preds[i, t].cpu().item()),
                    "prob_class_0": float(probs[i, t, 0].cpu().item()),
                    "prob_class_1": float(probs[i, t, 1].cpu().item()),
                    "status": batch["status_seq"][i][t],
                })

    return records

def load_pretrained_encoder(
    fusion_encoder,
    checkpoint_path,
    device,
    key_prefix_to_strip=None,
    strict=False,
):
    """
    Load pretrained weights into the FusionModel encoder.

    Supports checkpoints saved as:
      1) raw state_dict
      2) {"state_dict": state_dict}
      3) {"model_state_dict": state_dict}
      4) full temporal model state_dict with keys like "fusion_encoder.xxx"

    Parameters
    ----------
    fusion_encoder : nn.Module
        The FusionModel instance.
    checkpoint_path : str
        Path to .pt/.pth checkpoint.
    device : torch.device
        Device used for map_location.
    key_prefix_to_strip : str or None
        Prefix to remove from checkpoint keys before loading.
        Example: "fusion_encoder."
    strict : bool
        Whether to require exact key match.
    """

    if checkpoint_path is None:
        return

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {checkpoint_path}")

    print(f"Loading pretrained encoder from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(
            "Unsupported checkpoint format. Expected a state_dict or a dict containing "
            "'state_dict' or 'model_state_dict'."
        )

    cleaned_state_dict = {}

    for k, v in state_dict.items():
        new_k = k

        # Common case when saved with DataParallel
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]

        # Optional user-provided prefix stripping
        if key_prefix_to_strip is not None and new_k.startswith(key_prefix_to_strip):
            new_k = new_k[len(key_prefix_to_strip):]

        cleaned_state_dict[new_k] = v

    # If checkpoint is from the full temporal model and user did not specify prefix,
    # automatically keep only fusion_encoder.* keys.
    if key_prefix_to_strip is None:
        fusion_only = {}

        for k, v in cleaned_state_dict.items():
            if k.startswith("fusion_encoder."):
                fusion_only[k[len("fusion_encoder."):]] = v

        if len(fusion_only) > 0:
            print("Detected full temporal model checkpoint. Loading only fusion_encoder.* weights.")
            cleaned_state_dict = fusion_only

    incompatible = fusion_encoder.load_state_dict(cleaned_state_dict, strict=strict)

    print("Pretrained encoder loading complete.")
    print("Missing keys:", incompatible.missing_keys)
    print("Unexpected keys:", incompatible.unexpected_keys)
# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main(args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATASET_PATH = args.dataset_path
    CROSS_VAL_PKL_PATH = args.cross_val_pkl
    use_es = args.early_stopping

    if args.excluded_node_features is None:
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
    
    if not (
    args.include_gnn
    or args.include_cnn
    or args.include_cortex_mlp
    or args.include_transformer
    or args.include_cog_mlp
    ):
        raise ValueError(
            "At least one encoder branch must be enabled: "
            "--include_gnn, --include_cnn, --include_cortex_mlp, "
            "--include_transformer, or --include_cog_mlp"
        )

    # Load and preprocess data
    data_list = general.load_dataset_from_single_pt(
        DATASET_PATH,
        convert_labels=False if args.dataset == "oasis" else True
    ) if DATASET_PATH.endswith(".pt") else None

    num_nodes = data_list[0].x.shape[0]
    print(f"Each graph has {num_nodes} nodes and {data_list[0].x.shape[1]} node features.")

    # early stopping data names
    early_stopping_data_list_names = None
    if use_es and args.dataset == "adni":
        with open("./data/adni/splits/combined_tuning_filenames.json", "r") as f:
            early_stopping_data_list_names = json.load(f)

    # Load CV splits
    splits = general.read_cross_val(CROSS_VAL_PKL_PATH)
    print(f"Loaded {len(splits)} cross-validation splits.")

    # global preprocessing
    data_list, cog_in_dim, vol_sum_index = preprocessing.preprocess_global_data_list(
        data_list=data_list,
        dataset_path=DATASET_PATH,
        args=args,
        feature_slices=FEATURE_SLICES
    )

    # filename --> data
    if args.dataset == "adni":
        filename_to_data = {data.ptid + "_" + data.viscode + ".pt": data for data in data_list}
    elif args.dataset == "oasis":
        filename_to_data = {data.oasis_id + "_" + data.scan_day + ".pt": data for data in data_list}
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # results containers
    results = pd.DataFrame(columns=[
        "FOLD", "ACC", "F1_macro", "F1_weighted",
        "PRECISION_CLASS_0", "RECALL_CLASS_0",
        "PRECISION_CLASS_1", "RECALL_CLASS_1"
    ])

    all_true, all_pred = [], []
    all_train_losses, all_test_losses, all_epoch_metrics = [], [], []

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_dir is not None:
        results_dir = args.run_dir + f"/{now}_seed{args.seed}"
    else:
        results_dir = f"./drive/MyDrive/thesis_gnn_results/mind_graph_exps/{now+str(args.seed)}_temporal_many_to_many"

    os.makedirs(results_dir, exist_ok=True)

    all_prediction_records = []

    observe.print_cv_class_distributions(
        splits=splits,
        filename_to_data=filename_to_data,
        out_path=os.path.join(results_dir, "class_distributions.txt"),
        label_attr="y"
    )

    if use_es:
        monitor = args.es_monitor
        mode = args.es_mode
        patience = args.es_patience
        min_delta = args.es_min_delta

    # Cross-validation
    for fold, split in enumerate(splits):
        print(f"\n=== Fold {fold+1}/{len(splits)} ===")

        if args.dataset == "adni":
            if "val_files" in split:
                train_files = split["train_files"] + split["val_files"]
                train_subjects = split["train_subjects"] + split["val_subjects"]
            else:
                train_files = split["train_files"]
                train_subjects = split["train_subjects"]

            test_files = split["test_files"]
            test_subjects = split["test_subjects"]

            train_data = [copy.deepcopy(filename_to_data[f]) for f in train_files if f in filename_to_data]
            test_data = [copy.deepcopy(filename_to_data[f]) for f in test_files if f in filename_to_data]

            train_ptids = {d.ptid for d in train_data}
            test_ptids = {d.ptid for d in test_data}
            overlap = train_ptids & test_ptids
            
            print(f"Number of train subjects: {len(train_ptids)}, Number of test subjects: {len(test_ptids)}, Overlap: {len(overlap)}")
            if len(overlap) > 0:
                raise ValueError(f"Subject leakage between train and test: {list(overlap)[:10]}")

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
            test_data = [copy.deepcopy(filename_to_data[f]) for f in test_files if f in filename_to_data]

            print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

            early_stopping_data = None
            if use_es:
                early_stopping_files = split["val_files"]
                early_stopping_data = [copy.deepcopy(filename_to_data[f]) for f in early_stopping_files if f in filename_to_data]
                print(f"Early stopping size: {len(early_stopping_data)}")

        # determinism
        fold_seed = args.seed + fold
        seed_all(fold_seed)

        # cognitive preprocessing
        train_data, cog_scaler, cog_mean = preprocessing.preprocess_cognitive_features_train(train_data)
        test_data = preprocessing.preprocess_cognitive_features_test(test_data, cog_scaler, cog_mean)
        if use_es:
            early_stopping_data = preprocessing.preprocess_cognitive_features_test(early_stopping_data, cog_scaler, cog_mean)

        # ICV normalization
        if vol_sum_index is not None:
            print("Performing ICV normalization on 'vol' features")
            icv_params = preprocessing.fit_icv_normalizer(train_data, feature_indices=[vol_sum_index], icv_attr="ICV")
            train_data = preprocessing.apply_icv_normalizer(train_data, icv_params)
            test_data = preprocessing.apply_icv_normalizer(test_data, icv_params)
            if use_es:
                early_stopping_data = preprocessing.apply_icv_normalizer(early_stopping_data, icv_params)

        # MRI node preprocessing
        train_data, mri_node_scalers = preprocessing.preprocess_mri_node_features(train_data)
        test_data = preprocessing.apply_mri_node_scalers(test_data, mri_node_scalers)
        if use_es:
            early_stopping_data = preprocessing.apply_mri_node_scalers(early_stopping_data, mri_node_scalers)

        print("Shape of node features:", train_data[0].x.shape)

        # subject -> sorted visits
        train_subject_to_visits = build_subject_to_visits(train_data)
        test_subject_to_visits = build_subject_to_visits(test_data)

        print(f"Number of train subjects with visits: {len(train_subject_to_visits)}")
        print(f"Number of test subjects with visits: {len(test_subject_to_visits)}")

        example_subject = next(iter(train_subject_to_visits))
        print("Example train subject:", example_subject)
        print("Sorted visits:", [d.viscode for d in train_subject_to_visits[example_subject]])

        train_temporal_samples = make_many_to_many_pyg_visit_samples(
            train_subject_to_visits
        )

        test_temporal_samples = make_many_to_many_pyg_visit_samples(
            test_subject_to_visits
        )


        print(f"Number of train temporal samples: {len(train_temporal_samples)}")
        print(f"Number of test temporal samples: {len(test_temporal_samples)}")

        example = train_temporal_samples[0]
        print("Example sample:")
        print("  number of visits:", len(example["data_seq"]))
        print("  first visit x shape:", tuple(example["data_seq"][0].x.shape))
        print("  y_seq shape:", tuple(example["y_seq"].shape))
        print("  y_seq:", example["y_seq"].tolist())

        train_temporal_dataset = TemporalDataset(train_temporal_samples)
        test_temporal_dataset = TemporalDataset(test_temporal_samples)

        g = torch.Generator()
        g.manual_seed(fold_seed)

        train_loader = DataLoader(
            train_temporal_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=g,
            num_workers=0,
            collate_fn=temporal_collate_fn_pyg_many_to_many,
            drop_last=True, # ???
        )

        observation_train_loader = DataLoader(
            train_temporal_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=temporal_collate_fn_pyg_many_to_many,
        )

        test_loader = DataLoader(
            test_temporal_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=temporal_collate_fn_pyg_many_to_many,
        )

        if use_es and early_stopping_data is not None:
            es_subject_to_visits = build_subject_to_visits(early_stopping_data)

            es_temporal_samples = make_many_to_many_pyg_visit_samples(
                es_subject_to_visits
            )

            es_temporal_dataset = TemporalDataset(es_temporal_samples)

            es_loader = DataLoader(
                es_temporal_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=temporal_collate_fn_pyg_many_to_many,
            )
        else:
            es_loader = None

        # Sanity check one batch from the training loader 
        batch = next(iter(train_loader))

        print("Batch number of subject sequences:", len(batch["data_seqs"]))
        print("Batch y_padded shape:", tuple(batch["y_padded"].shape))
        print("Batch lengths shape:", tuple(batch["lengths"].shape))
        print("Batch mask shape:", tuple(batch["mask"].shape))
        print("First sample length:", int(batch["lengths"][0].item()))
        print("First sample ptid:", batch["ptids"][0])
        print("First sample viscodes:", batch["viscodes"][0])
        print("First sample number of visits:", len(batch["data_seqs"][0]))
        print("First visit x shape:", tuple(batch["data_seqs"][0][0].x.shape))

        fusion_encoder = FusionModel(
            num_nodes=num_nodes,
            node_in_dim=train_data[0].x.shape[1],
            num_classes=2,

            # same flags/configs you already use
            include_gnn=args.include_gnn,
            include_cnn=args.include_cnn,
            include_mlp=args.include_cortex_mlp,
            include_transformer=args.include_transformer,
            include_cog_mlp=args.include_cog_mlp,

            gnn_hidden_dim=args.gnn_hidden_dim,
            gnn_dropout=args.gnn_dropout,
            gnn_use_pre_mlp=args.gnn_use_pre_mlp,
            gnn_cnn_input_add_flattened_node_features=args.gnn_cnn_input_add_flattened_node_features,
            gnn_add_output_skip=args.gnn_add_output_skip,
            gnn_layer_connectivity=args.gnn_layer_connectivity,
            gnn_layer=args.gnn_layer,
            gnn_num_layers=args.gnn_num_layers,
            gnn_norm_type=args.gnn_norm_type,
            gnn_readout=args.gnn_readout,
            gnn_graph_pool=args.gnn_graph_pool,

            cortex_mlp_hidden_dim=args.cortex_mlp_hidden_dim,
            cortex_mlp_use_residual=args.cortex_mlp_use_residual,
            cortex_mlp_activation=args.cortex_mlp_activation,
            cortex_mlp_use_layernorm=args.cortex_mlp_use_layernorm,
            cortex_mlp_num_layers=args.cortex_mlp_num_layers,
            cortex_mlp_hidden_dims=args.cortex_mlp_hidden_dims,
            cortex_mlp_width_mode=args.cortex_mlp_width_mode,
            cortex_mlp_dropout=args.cortex_mlp_dropout,

            cog_mlp_num_layers=args.cog_mlp_num_layers,
            cog_mlp_width_mode=args.cog_mlp_width_mode,
            cog_mlp_use_residual_to_last=args.cog_mlp_use_residual_to_last,
            cog_mlp_dropout=args.cog_mlp_dropout,
            cog_hidden_dim=args.cog_hidden_dim,
            cog_in_dim=cog_in_dim,

            adj_cnn_conv_channels=tuple(args.adj_cnn_conv_channels),
            adj_cnn_kernel_sizes=tuple(args.adj_cnn_kernel_sizes),
            adj_cnn_strides=tuple(args.adj_cnn_strides),
            adj_cnn_dropout=args.adj_cnn_dropout,
            adj_cnn_pool_types=tuple(args.adj_cnn_pool_types),
            adj_cnn_pool_kernel_sizes=tuple(args.adj_cnn_pool_kernel_sizes),
            adj_cnn_negative_slope=args.adj_cnn_negative_slope,
            adj_cnn_norm_type=args.adj_cnn_norm_type,
            adj_cnn_group_norm_groups=args.adj_cnn_group_norm_groups,
            adj_cnn_readout=args.adj_cnn_readout,

            cortex_transformer_num_layers=args.cortex_transformer_num_layers,
            cortex_transformer_hidden_dim=args.cortex_transformer_hidden_dim,
            cort_transformer_dropout=args.cort_transformer_dropout,
            cortex_transformer_num_heads=args.cortex_transformer_num_heads,
            cortex_transformer_cnn_input_add_flattened_node_features=args.cortex_transformer_cnn_input_add_flattened_node_features,
            cortex_transformer_add_output_skip=args.cortex_transformer_add_output_skip,

            pos_encoding_type=args.pos_encoding_type,
            lpe_dim=args.lpe_dim,

            dropout=args.dropout,
            separate_adj_features_instead_of_concat=args.separate_adj_features_instead_of_concat,
        )
        load_pretrained_encoder(
            fusion_encoder=fusion_encoder,
            checkpoint_path=args.pretrained_encoder_path + f"/fold{fold+1}_model_weights.pt",
            device=device,
            key_prefix_to_strip=args.pretrained_encoder_key_prefix,
            strict=args.strict_pretrained_encoder,
        )
        model = FusionRecurrentManyToMany(
            fusion_encoder=fusion_encoder,
            temporal_type=args.temporal_type,
            hidden_dim=args.temporal_hidden_dim,
            num_layers=1,
            num_classes=2,
            recurrent_dropout=args.dropout,
            classifier_dropout=args.dropout,
            rnn_nonlinearity="tanh",
            transformer_causal_mask=True
        ).to(device)
        print(model)

        # class weights from ALL visit labels
        train_y = torch.cat([s["y_seq"] for s in train_temporal_samples], dim=0)
        cw = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_y.numpy()),
            y=train_y.numpy()
        )

        if args.use_class_weights:
            print("Using class weights in loss function.")
            print("Class weights:", cw)
            criterion = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(cw, dtype=torch.float, device=device),
                ignore_index=-100
            )
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            decoupled_weight_decay=True
        )
        print("Using optimizer:", optimizer, "with weight_decay:", args.weight_decay)

        fold_train_losses, fold_test_losses, epoch_metrics = [], [], []

        # optional early stopping state
        best_metric = None
        best_state_dict = None
        epochs_no_improve = 0

        def is_better(curr, best, mode, min_delta):
            if best is None:
                return True
            if mode == "min":
                return curr < (best - min_delta)
            else:
                return curr > (best + min_delta)

        for epoch in range(1, args.epochs + 1):
            tr_loss = train_one_epoch_temporal_many_to_many(
                model, train_loader, optimizer, device, criterion
            )

            test_loss, test_acc, f1_weighted, f1_macro, precision, recall, auc, conv_recall = \
                evaluate_temporal_many_to_many(model, test_loader, device, criterion)

            tr_after_epoch_loss, tr_acc, tr_f1_weighted, tr_f1_macro, tr_precision, tr_recall, tr_auc, tr_conv_recall = \
                evaluate_temporal_many_to_many(model, observation_train_loader, device, criterion)

            es_loss, es_acc, es_f1_weighted, es_f1_macro, es_precision, es_recall, es_auc, es_conv_recall = \
                (float("nan"),) * 8

            if use_es and es_loader is not None:
                es_loss, es_acc, es_f1_weighted, es_f1_macro, es_precision, es_recall, es_auc, es_conv_recall = \
                    evaluate_temporal_many_to_many(model, es_loader, device, criterion)

                monitor_value_map = {
                    "es_loss": es_loss,
                    "es_acc": es_acc,
                    "es_f1_weighted": es_f1_weighted,
                    "es_f1_macro": es_f1_macro,
                    "es_auc": es_auc,
                }
                current_metric = monitor_value_map[monitor]

                if not (isinstance(current_metric, float) and np.isnan(current_metric)):
                    if is_better(current_metric, best_metric, mode, min_delta):
                        best_metric = current_metric
                        best_state_dict = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

            fold_train_losses.append(tr_after_epoch_loss)
            fold_test_losses.append(test_loss)

            if use_es and es_loader is not None:
                print(
                    f"Epoch {epoch}: "
                    f"Train Loss={tr_after_epoch_loss:.4f}, Train Acc={tr_acc:.3f} | "
                    f"ES Loss={es_loss:.4f}, ES Acc={es_acc:.3f} | "
                    f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.3f}"
                )
            else:
                print(
                    f"Epoch {epoch}: "
                    f"Train Loss={tr_after_epoch_loss:.4f}, Train Acc={tr_acc:.3f} | "
                    f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.3f}"
                )

            print(
                f"Train F1 Weighted={tr_f1_weighted:.4f}, Train F1 Macro={tr_f1_macro:.4f}, "
                f"Test F1 Weighted={f1_weighted:.4f}, Test F1 Macro={f1_macro:.4f}"
            )

            epoch_row = {
                "epoch": epoch,
                "train_loss": tr_after_epoch_loss,
                "train_acc": tr_acc,
                "train_f1_weighted": tr_f1_weighted,
                "train_f1_macro": tr_f1_macro,
                "train_class0_precision": tr_precision[0],
                "train_class1_precision": tr_precision[1],
                "train_class0_recall": tr_recall[0],
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
            }

            if use_es and es_loader is not None:
                epoch_row.update({
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

            epoch_metrics.append(epoch_row)

            if use_es and es_loader is not None and epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        # restore best model if early stopping used
        if use_es and es_loader is not None and best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            print("Loaded best model state based on early stopping metric.")

        test_loss, test_acc, f1_weighted, f1_macro, precision, recall, auc, conv_recall = \
            evaluate_temporal_many_to_many(model, test_loader, device, criterion=criterion)

        print(
            f"Final Test Metrics for Fold {fold+1}: "
            f"Loss {test_loss:.4f} | Acc {test_acc:.3f} | "
            f"F1w {f1_weighted:.3f} | F1m {f1_macro:.3f} | "
            f"Precision {precision} | Recall {recall} | AUC {auc:.3f}"
        )

        prediction_records = predict_temporal_many_to_many(model, test_loader, device)

        for rec in prediction_records:
            rec["fold"] = fold + 1
            all_prediction_records.append(rec)

        y_true = [rec["label"] for rec in prediction_records]
        y_pred = [rec["prediction"] for rec in prediction_records]

        all_true.extend(y_true)
        all_pred.extend(y_pred)

        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0
        )

        results.loc[fold] = [
            fold + 1,
            test_acc,
            f1_score(y_true, y_pred, average="macro", zero_division=0),
            f1_score(y_true, y_pred, average="weighted", zero_division=0),
            precision[0], recall[0], precision[1], recall[1],
        ]

        all_train_losses.append(fold_train_losses)
        all_test_losses.append(fold_test_losses)
        all_epoch_metrics.append(epoch_metrics)

    # -------------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------------
    results.to_csv(os.path.join(results_dir, "classification_results.csv"), index=False)

    means = results.drop(columns=["FOLD"]).mean()
    stds = results.drop(columns=["FOLD"]).std()
    summary = pd.DataFrame({"Mean": means, "Std": stds})
    summary.to_csv(os.path.join(results_dir, "classification_mean_std_results.csv"))

    all_prediction_df = pd.DataFrame(all_prediction_records)
    all_prediction_df.to_csv(os.path.join(results_dir, "all_predictions.csv"), index=False)

    os.makedirs(os.path.join(results_dir, "training_logs_plots"), exist_ok=True)

    if len(all_epoch_metrics) == 1:
        epoch_metrics_df = pd.DataFrame(all_epoch_metrics[0])
        epoch_metrics_df.to_csv(
            os.path.join(results_dir, "training_logs_plots", "epoch_metrics.csv"),
            index=False
        )
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

        if not use_es:
            combined_epoch_df = pd.concat(per_fold_epoch_dfs, ignore_index=True)

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

            avg_epoch_metrics.columns = [
                "epoch" if col == ("epoch", "") else f"{col[0]}_{col[1]}"
                for col in avg_epoch_metrics.columns
            ]

            avg_epoch_metrics.to_csv(
                os.path.join(results_dir, "training_logs_plots", "average_epoch_metrics.csv"),
                index=False
            )

    # plots
    if len(all_epoch_metrics) == 1:
        plotting.plot_fold_curves(
            all_epoch_metrics[0],
            out_path=os.path.join(results_dir, "training_logs_plots", "fold1_curves.png"),
            fold_idx=1
        )
    else:
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

    observe.save_model_summary(model, os.path.join(results_dir, "model_summary.txt"))


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal many-to-many LSTM with cross-validation")

    parser.add_argument("--base_folder", type=str, default=".")
    parser.add_argument("--dataset_path", type=str, default=r"C:\Users\efeka\Documents\MIND_graphs\ADNI\MIND_graphs_CT_Vol\CT_Vol_graphs_complete_features_filtered_negative\pyg\CT_Vol_graphs_complete_features_filtered_negative.pt")
    parser.add_argument("--cross_val_pkl", type=str, default=r"C:\dev\GitHub\MIND\colab_data\cv_tuning_val_974_split\split_by_prog_category_9_7_4_seed93\cv\cross_val_splits_5fold_10perc_early_stop.pkl")
    parser.add_argument("--run_dir", type=str, default=None)

    parser.add_argument("--dataset", type=str, choices=["adni", "oasis"], default="adni")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--include_cnn", action="store_true")
    parser.add_argument("--include_transformer", action="store_true")
    parser.add_argument("--include_gnn", action="store_true")
    parser.add_argument("--include_cortex_mlp", action="store_true")
    parser.add_argument("--include_cog_mlp", action="store_true")
    parser.add_argument("--fusion", type=str, choices=["attention", "concat"], default="concat")
    parser.add_argument("--task", type=str, choices=["diagnosis", "conversion"], default="diagnosis")

    # Temporal settings
    parser.add_argument("--temporal_type", type=str, choices=["rnn", "lstm", "gru"], default="rnn")
    parser.add_argument("--dropout", type=float, default=0.5) # dropout for temporal classifier
    parser.add_argument("--temporal_hidden_dim", type=int, default=128)
    # Pretrained encoder loading
    parser.add_argument("--pretrained_encoder_path", type=str, default=None)
    parser.add_argument(
        "--pretrained_encoder_key_prefix",
        type=str,
        default=None,
        help=(
            "Optional prefix to strip from checkpoint keys. "
            "Examples: 'fusion_encoder.', 'module.fusion_encoder.'"
        )
    )
    parser.add_argument(
        "--strict_pretrained_encoder",
        action="store_true",
        help="Use strict=True when loading pretrained encoder weights."
    )

    # GNN
    parser.add_argument("--gnn_dropout", type=float, default=0.5)
    parser.add_argument("--gnn_hidden_dim", type=int, default=256)
    parser.add_argument("--edge_threshold", type=float, default=1.0)
    parser.add_argument("--gnn_num_layers", type=int, default=2)
    parser.add_argument("--gnn_layer", type=str, choices=["gcn", "sage", "gatv2", "gin"], default="gcn")
    parser.add_argument("--add_adj_row_as_node_feature", action="store_true")
    parser.add_argument("--separate_adj_features_instead_of_concat", action="store_true")
    parser.add_argument("--add_weighted_degree_as_node_feature", action="store_true")
    parser.add_argument("--gnn_use_pre_mlp", action="store_true")
    parser.add_argument("--gnn_cnn_input_add_flattened_node_features", action="store_true")
    parser.add_argument("--gnn_add_output_skip", action="store_true")
    parser.add_argument("--gnn_layer_connectivity", type=str, choices=["stack", "skipcat", "skipsum"], default="skipsum")
    parser.add_argument("--gnn_norm_type", type=str, default="layernorm")
    parser.add_argument("--gnn_readout", type=str, choices=["cnn", "pool"], default="cnn")
    parser.add_argument("--gnn_graph_pool", type=str, choices=["mean", "max", "sum", "mean_max"], default="mean_max")

    # Cortex MLP
    parser.add_argument("--cortex_mlp_dropout", type=float, default=0.5)
    parser.add_argument("--cortex_mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--cortex_mlp_use_residual", action="store_true")
    parser.add_argument("--cortex_mlp_activation", type=str, choices=["relu", "gelu", "elu", "leakyrelu"], default="leakyrelu")
    parser.add_argument("--cortex_mlp_use_layernorm", action="store_true")
    parser.add_argument("--cortex_mlp_num_layers", type=int, default=3)
    parser.add_argument("--cortex_mlp_hidden_dims", type=int, nargs="+", default=None)
    parser.add_argument("--cortex_mlp_width_mode", type=str, default="constant")

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
    parser.add_argument("--cortex_transformer_hidden_dim", type=int, default=128)
    parser.add_argument("--cortex_transformer_num_layers", type=int, default=2)
    parser.add_argument("--cortex_transformer_num_heads", type=int, default=4)
    parser.add_argument("--cortex_transformer_cnn_input_add_flattened_node_features", action="store_true")
    parser.add_argument("--cortex_transformer_add_output_skip", action="store_true")

    # Cognitive MLP
    parser.add_argument("--cog_hidden_dim", type=int, default=128)
    parser.add_argument("--cog_mlp_dropout", type=float, default=0.5)
    parser.add_argument("--cog_mlp_width_mode", type=str, default="constant")
    parser.add_argument("--cog_mlp_num_layers", type=int, default=2)
    parser.add_argument("--cog_mlp_use_residual_to_last", action="store_true")

    # positional encoding(used in transformer branch, and optionally can be added to GNN node features as well if adapted)
    parser.add_argument("--add_laplacian_pe", action="store_true")
    parser.add_argument("--pos_encoding_type", type=str, choices=["none", "sinusoidal", "learnable", "lpe"], default="learnable")
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
    parser.add_argument("--es_mode", type=str, default="max", choices=["min", "max"])

    args = parser.parse_args()
    main(args, seed=args.seed)