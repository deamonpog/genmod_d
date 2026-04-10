"""PyTorch datasets for ECA rule classification.

Refactored from train_ca_rule_classifier_transformer.py.
Decoupled from contiguous rule ranges; accepts arbitrary rule sets.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def bits_to_patches(bits: List[int], patch_size: int) -> List[int]:
    """Convert a row of bits to patch tokens."""
    assert len(bits) % patch_size == 0
    out = []
    for i in range(0, len(bits), patch_size):
        v = 0
        for b in bits[i : i + patch_size]:
            v = (v << 1) | int(b)
        out.append(v)
    return out


def patches_to_bits(patches: List[int], patch_size: int) -> List[int]:
    """Convert patch tokens back to bits."""
    bits = []
    for p in patches:
        for shift in reversed(range(patch_size)):
            bits.append((p >> shift) & 1)
    return bits


class CARuleWindowDataset(Dataset):
    """Space-time window dataset for multi-rule classification.

    Each sample is a window of T consecutive states from one run.
    Tokens: [CLS] + (T * S) patch tokens.
    Returns time_ids and space_ids for decomposed positional embeddings.
    Label is rule index in [0..num_rules-1].
    """

    def __init__(
        self,
        runs_by_rule: Dict[int, List[dict]],
        rule_to_label: Dict[int, int],
        lattice_width: int,
        patch_size: int,
        window_T: int,
        cls_token_id: int,
        return_metadata: bool = False,
    ):
        self.rule_to_label = rule_to_label
        self.lattice_width = lattice_width
        self.patch_size = patch_size
        self.window_T = window_T
        self.cls_token_id = cls_token_id
        self.return_metadata = return_metadata

        if lattice_width % patch_size != 0:
            raise ValueError("lattice_width must be divisible by patch_size")
        self.S = lattice_width // patch_size

        # Pre-tokenize all runs
        self.tokenized: List[Tuple[List[List[int]], int, int]] = []  # (rows_patches, label, rule)
        for rule, runs in runs_by_rule.items():
            label = rule_to_label[rule]
            for r in runs:
                rows_bits = r["output"]
                if len(rows_bits) < window_T:
                    continue
                if len(rows_bits[0]) != lattice_width:
                    raise ValueError(
                        f"Width mismatch in rule {rule}: got {len(rows_bits[0])}, expected {lattice_width}"
                    )
                rows_patches = [bits_to_patches(row, patch_size) for row in rows_bits]
                self.tokenized.append((rows_patches, label, rule))

        # Build index of all possible windows (run_idx, start_t)
        self.index: List[Tuple[int, int]] = []
        for run_idx, (rows_patches, _label, _rule) in enumerate(self.tokenized):
            T_total = len(rows_patches)
            for start in range(0, T_total - window_T + 1):
                self.index.append((run_idx, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        run_idx, start = self.index[idx]
        rows_patches, label, rule = self.tokenized[run_idx]

        window = rows_patches[start : start + self.window_T]

        flat_tokens = [self.cls_token_id]
        time_ids = [self.window_T]   # CLS time id
        space_ids = [self.S]          # CLS space id

        for t, row in enumerate(window):
            for s, tok in enumerate(row):
                flat_tokens.append(tok)
                time_ids.append(t)
                space_ids.append(s)

        x = torch.tensor(flat_tokens, dtype=torch.long)
        tpos = torch.tensor(time_ids, dtype=torch.long)
        spos = torch.tensor(space_ids, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)

        if self.return_metadata:
            return x, tpos, spos, y, rule, run_idx
        return x, tpos, spos, y


def load_runs_for_rule(json_path: Path) -> List[dict]:
    """Load all runs for a single rule from JSON."""
    return json.loads(json_path.read_text())


def load_all_rules(
    data_dir: str,
    rules: List[int],
) -> Dict[int, List[dict]]:
    """Load runs for all specified rules.

    Returns: dict mapping rule_number -> list of run dicts.
    """
    data_path = Path(data_dir)
    all_runs: Dict[int, List[dict]] = {}
    for rule in rules:
        p = data_path / f"rule_{rule}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Generate data first.")
        all_runs[rule] = load_runs_for_rule(p)
    return all_runs
