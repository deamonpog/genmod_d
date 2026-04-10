# train_ca_transformer_runs.py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    json_path: str = "GENERATED_DATA/rule_110.json"
    lattice_width: int = 128
    patch_size: int = 8                  # 8-bit patches => vocab=256
    vocab_size: int = 256

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1

    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    epochs: int = 20
    train_frac_runs: float = 0.9         # split by run, not by pair
    seed: int = 42

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Tokenization: bits <-> 8-bit patches
# ----------------------------
def bits_to_patches(bits: List[int], patch_size: int) -> List[int]:
    assert len(bits) % patch_size == 0, "lattice_width must be divisible by patch_size"
    out = []
    for i in range(0, len(bits), patch_size):
        val = 0
        for b in bits[i:i + patch_size]:
            val = (val << 1) | int(b)
        out.append(val)
    return out  # each in [0, 2^patch_size - 1]


def patches_to_bits(patches: List[int], patch_size: int) -> List[int]:
    bits = []
    for p in patches:
        for shift in reversed(range(patch_size)):
            bits.append((p >> shift) & 1)
    return bits


# ----------------------------
# Dataset over RUNS (run-level split)
# Each sample is one transition: state_t -> state_{t+1}
# ----------------------------
class CARunsOneStepDataset(Dataset):
    def __init__(self, runs: List[dict], lattice_width: int, patch_size: int):
        self.patch_size = patch_size
        self.lattice_width = lattice_width

        # Pre-tokenize each run as list of patch-states
        self.run_states: List[List[List[int]]] = []
        for r in runs:
            states_bits = r["output"]  # list of [0/1]*lattice_width
            if len(states_bits) < 2:
                continue
            if len(states_bits[0]) != lattice_width:
                raise ValueError(
                    f"Run has width {len(states_bits[0])}, expected {lattice_width}. "
                    f"Did you generate with size={lattice_width}?"
                )
            states_patches = [bits_to_patches(s, patch_size) for s in states_bits]
            self.run_states.append(states_patches)

        # Build global index of (run_id, t) for each consecutive pair
        self.index: List[Tuple[int, int]] = []
        for run_id, states in enumerate(self.run_states):
            for t in range(len(states) - 1):
                self.index.append((run_id, t))

        self.seq_len = lattice_width // patch_size

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        run_id, t = self.index[idx]
        x = torch.tensor(self.run_states[run_id][t], dtype=torch.long)     # (S,)
        y = torch.tensor(self.run_states[run_id][t + 1], dtype=torch.long) # (S,)
        return x, y


# ----------------------------
# Model: per-position classification over 256 patch tokens
# ----------------------------
class CATransformerOneStep(nn.Module):
    def __init__(self, seq_len: int, vocab_size: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,S) long
        B, S = x.shape
        if S != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {S}")
        h = self.tok_emb(x) + self.pos_emb
        h = self.encoder(h)
        h = self.ln_f(h)
        return self.head(h)  # (B,S,V)

    @torch.no_grad()
    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)  # (B,S)


# ----------------------------
# Metrics: rollout on a held-out run
# ----------------------------
@torch.no_grad()
def rollout_metrics(model: CATransformerOneStep, run_bits: List[List[int]], patch_size: int, device: str, max_steps: int = 200):
    model.eval()
    lattice_width = len(run_bits[0])

    # Start from t=0
    x0_patches = bits_to_patches(run_bits[0], patch_size)
    x = torch.tensor([x0_patches], dtype=torch.long, device=device)  # (1,S)

    steps = min(max_steps, len(run_bits) - 1)
    exact = 0
    ham_sum = 0.0

    for t in range(steps):
        yhat = model.predict_next(x)  # (1,S)
        pred_bits = patches_to_bits(yhat[0].tolist(), patch_size)
        true_bits = run_bits[t + 1]

        if pred_bits == true_bits:
            exact += 1

        diff = sum(int(a != b) for a, b in zip(pred_bits, true_bits))
        ham_sum += diff / lattice_width

        x = yhat

    return {
        "steps": steps,
        "exact_step_rate": exact / max(steps, 1),
        "avg_hamming_rate": ham_sum / max(steps, 1),
    }

# ----------------------------
# Token-level accuracy
# ----------------------------
@torch.no_grad()
def token_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train()
    return correct / max(total, 1)


# ----------------------------
# Train
# ----------------------------
def main(cfg: Config):
    torch.manual_seed(cfg.seed)

    path = Path(cfg.json_path)
    runs = json.loads(path.read_text())

    if cfg.lattice_width % cfg.patch_size != 0:
        raise ValueError("lattice_width must be divisible by patch_size")

    # Split by RUN (prevents leakage)
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(runs), generator=g).tolist()
    n_train = int(cfg.train_frac_runs * len(runs))
    train_runs = [runs[i] for i in perm[:n_train]]
    val_runs = [runs[i] for i in perm[n_train:]]

    train_ds = CARunsOneStepDataset(train_runs, cfg.lattice_width, cfg.patch_size)
    val_ds = CARunsOneStepDataset(val_runs, cfg.lattice_width, cfg.patch_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    seq_len = cfg.lattice_width // cfg.patch_size  # 16 for 128/8
    model = CATransformerOneStep(
        seq_len=seq_len,
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def eval_val_loss():
        model.eval()
        losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(cfg.device), y.to(cfg.device)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                losses.append(loss.item())
        model.train()
        return sum(losses) / max(len(losses), 1)

    # Choose a held-out run for rollout evaluation
    rollout_run = val_runs[0]["output"] if len(val_runs) > 0 else train_runs[-1]["output"]

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        val_loss = eval_val_loss()
        token_acc = token_accuracy(model, val_loader, cfg.device)
        r = rollout_metrics(model, rollout_run, cfg.patch_size, cfg.device, max_steps=200)

        print(
            f"epoch {epoch:02d} | val_loss={val_loss:.4f} | "
            f"token_acc={token_acc*100:.2f}% | "
            f"rollout_exact={r['exact_step_rate']*100:.1f}% | "
            f"rollout_hamming={r['avg_hamming_rate']*100:.2f}%"
        )

    # Quick demo: rollout first 10 steps from the held-out run's t=0
    model.eval()
    x = torch.tensor([bits_to_patches(rollout_run[0], cfg.patch_size)], dtype=torch.long, device=cfg.device)
    print("\nPredicted patch tokens for 10 rollout steps (from held-out run t=0):")
    for t in range(10):
        x = model.predict_next(x)
        print(f"t={t+1:02d}:", x[0].tolist())


if __name__ == "__main__":
    cfg = Config(
        json_path="GENERATED_DATA/rule_110.json",
        lattice_width=32,
        patch_size=8,
    )
    main(cfg)
