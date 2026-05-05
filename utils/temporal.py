from collections import defaultdict
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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

