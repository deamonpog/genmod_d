"""PyTorch dataset for Schelling segregation threshold classification.

Tokenization: 2x2 patches on 20x20 grid → 100 patches/snapshot.
Each patch: 4 cells × 3 states = base-3 integer in [0, 80]. Vocab = 82 (81 + CLS).
Multiple snapshots give temporal dimension.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


def patch_to_token(patch: List[List[int]], base: int = 3) -> int:
    """Convert a 2D patch of cell values to a single integer token.

    Each cell is in {0, 1, 2}. Flatten and interpret as a base-3 number.
    """
    flat = []
    for row in patch:
        flat.extend(row)
    token = 0
    for v in flat:
        token = token * base + v
    return token


def grid_to_patch_tokens(grid: List[List[int]], patch_size: int = 2) -> List[int]:
    """Convert a 2D grid to a flat list of patch tokens (row-major order)."""
    size = len(grid)
    tokens = []
    for r in range(0, size, patch_size):
        for c in range(0, size, patch_size):
            patch = []
            for dr in range(patch_size):
                row_vals = []
                for dc in range(patch_size):
                    if r + dr < size and c + dc < size:
                        row_vals.append(grid[r + dr][c + dc])
                    else:
                        row_vals.append(0)
                patch.append(row_vals)
            tokens.append(patch_to_token(patch))
    return tokens


class SchellingDataset(Dataset):
    """Snapshot-window dataset for Schelling threshold classification.

    Each sample uses num_snapshots evenly-spaced snapshots from a run.
    Returns (tokens, time_ids, space_ids, label) matching the common interface.
    """

    def __init__(
        self,
        runs_by_label: Dict[int, List[dict]],
        patch_size: int = 2,
        num_snapshots: int = 5,
        grid_size: int = 20,
    ):
        self.patch_size = patch_size
        self.num_snapshots = num_snapshots
        self.grid_size = grid_size

        patches_per_snapshot = (grid_size // patch_size) ** 2  # 100 for 20x20 with 2x2
        vocab_size_no_cls = (3 ** (patch_size * patch_size))  # 81 for 2x2
        self.cls_token_id = vocab_size_no_cls  # 81
        self.S = patches_per_snapshot
        self.vocab_size = vocab_size_no_cls + 1  # 82

        # Pre-tokenize all runs
        self.tokenized: List[Tuple[List[List[int]], int]] = []  # (snapshot_tokens, label)
        for label, runs in runs_by_label.items():
            for run in runs:
                snapshots = run["snapshots"]
                if len(snapshots) < num_snapshots:
                    # Use all available snapshots, pad with last if needed
                    selected = snapshots + [snapshots[-1]] * (num_snapshots - len(snapshots))
                else:
                    # Evenly space
                    indices = [int(i * (len(snapshots) - 1) / (num_snapshots - 1))
                               for i in range(num_snapshots)]
                    selected = [snapshots[i] for i in indices]

                # Tokenize each selected snapshot
                snap_tokens = []
                for snap in selected:
                    snap_tokens.append(grid_to_patch_tokens(snap, patch_size))

                self.tokenized.append((snap_tokens, label))

        # Each run is one sample (no windowing needed — each run is independent)
        self.index = list(range(len(self.tokenized)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        snap_tokens, label = self.tokenized[idx]

        # Build token sequence: [CLS] + snapshot_0_patches + snapshot_1_patches + ...
        flat_tokens = [self.cls_token_id]
        time_ids = [self.num_snapshots]  # CLS time
        space_ids = [self.S]              # CLS space

        for t, patches in enumerate(snap_tokens):
            for s, tok in enumerate(patches):
                flat_tokens.append(tok)
                time_ids.append(t)
                space_ids.append(s)

        x = torch.tensor(flat_tokens, dtype=torch.long)
        tpos = torch.tensor(time_ids, dtype=torch.long)
        spos = torch.tensor(space_ids, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        return x, tpos, spos, y


def load_schelling_data(data_dir: str, num_classes: int) -> Dict[int, List[dict]]:
    """Load all Schelling data."""
    data_path = Path(data_dir)
    all_runs: Dict[int, List[dict]] = {}
    for label in range(num_classes):
        p = data_path / f"threshold_{label:02d}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Generate data first.")
        all_runs[label] = json.loads(p.read_text())
    return all_runs


def get_schelling_tokenization_params(
    grid_size: int = 20, patch_size: int = 2, num_snapshots: int = 5,
) -> dict:
    """Compute tokenization parameters for Schelling."""
    patches_per_snapshot = (grid_size // patch_size) ** 2  # 100
    vocab_no_cls = 3 ** (patch_size * patch_size)          # 81
    cls_token_id = vocab_no_cls
    vocab_size = vocab_no_cls + 1                           # 82
    seq_len = 1 + num_snapshots * patches_per_snapshot      # 501
    time_size = num_snapshots + 1                           # 6
    space_size = patches_per_snapshot + 1                   # 101
    return {
        "vocab_size": vocab_size,
        "cls_token_id": cls_token_id,
        "seq_len": seq_len,
        "time_size": time_size,
        "space_size": space_size,
    }
