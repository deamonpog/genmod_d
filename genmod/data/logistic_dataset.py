"""PyTorch dataset for logistic map regime classification.

Tokenization: quantize x ∈ [0,1] to 8-bit integer [0,255].
Sequence: [CLS, q(x_0), q(x_1), ..., q(x_{T-1})].
No spatial dimension (space_ids = all zeros).
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


def quantize(x: float, bits: int = 8) -> int:
    """Quantize x ∈ [0,1] to an integer in [0, 2^bits - 1]."""
    max_val = (1 << bits) - 1
    return max(0, min(max_val, int(x * max_val)))


class LogisticMapDataset(Dataset):
    """Sliding-window dataset for logistic map regime classification.

    Each sample is a window of T consecutive quantized values.
    Returns (tokens, time_ids, space_ids, label) matching the ECA interface.
    """

    def __init__(
        self,
        runs_by_label: Dict[int, List[dict]],
        window_T: int = 64,
        cls_token_id: int = 256,
        quantize_bits: int = 8,
    ):
        self.window_T = window_T
        self.cls_token_id = cls_token_id
        self.quantize_bits = quantize_bits

        # Pre-quantize all runs
        self.quantized: List[Tuple[List[int], int]] = []  # (quantized_series, label)
        for label, runs in runs_by_label.items():
            for run in runs:
                series = run["output"]
                q_series = [quantize(x, quantize_bits) for x in series]
                if len(q_series) >= window_T:
                    self.quantized.append((q_series, label))

        # Build window index
        self.index: List[Tuple[int, int]] = []
        for run_idx, (q_series, _) in enumerate(self.quantized):
            for start in range(len(q_series) - window_T + 1):
                self.index.append((run_idx, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        run_idx, start = self.index[idx]
        q_series, label = self.quantized[run_idx]

        window = q_series[start : start + self.window_T]

        # Tokens: [CLS] + window values
        tokens = [self.cls_token_id] + window
        # Time: CLS gets position T, then 0..T-1
        time_ids = [self.window_T] + list(range(self.window_T))
        # Space: no spatial dimension, CLS=1, all others=0
        space_ids = [1] + [0] * self.window_T

        x = torch.tensor(tokens, dtype=torch.long)
        tpos = torch.tensor(time_ids, dtype=torch.long)
        spos = torch.tensor(space_ids, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        return x, tpos, spos, y


def load_logistic_data(data_dir: str, num_classes: int) -> Dict[int, List[dict]]:
    """Load all logistic map data.

    Returns: dict mapping label -> list of run dicts.
    """
    data_path = Path(data_dir)
    all_runs: Dict[int, List[dict]] = {}
    for label in range(num_classes):
        p = data_path / f"r_{label:03d}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Generate data first.")
        all_runs[label] = json.loads(p.read_text())
    return all_runs


def get_logistic_tokenization_params(window_T: int = 64, quantize_bits: int = 8) -> dict:
    """Compute tokenization parameters for the logistic map."""
    vocab_size = (1 << quantize_bits) + 1  # 256 + CLS = 257
    cls_token_id = 1 << quantize_bits      # 256
    seq_len = 1 + window_T                 # CLS + T tokens
    time_size = window_T + 1               # 0..T-1 plus CLS position
    space_size = 2                         # 0 (data) and 1 (CLS)
    return {
        "vocab_size": vocab_size,
        "cls_token_id": cls_token_id,
        "seq_len": seq_len,
        "time_size": time_size,
        "space_size": space_size,
    }
