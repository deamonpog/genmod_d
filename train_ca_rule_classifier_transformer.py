import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    data_dir: str = "GENERATED_DATA"
    rule_start: int = 100
    rule_end_exclusive: int = 140

    lattice_width: int = 32      # must match generation size
    patch_size: int = 8          # 8-bit patches => vocab 256

    # A) Increase evidence per sample
    window_T: int = 32           # <-- upgraded from 16 to 32

    # model
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1

    # training
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    epochs: int = 15

    train_frac_runs: float = 0.9  # split by runs (per rule)
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # C) calibration / probability usefulness
    label_smoothing: float = 0.05  # try 0.02–0.10


# ----------------------------
# Tokenization: bits -> patch tokens
# ----------------------------
def bits_to_patches(bits: List[int], patch_size: int) -> List[int]:
    assert len(bits) % patch_size == 0
    out = []
    for i in range(0, len(bits), patch_size):
        v = 0
        for b in bits[i:i + patch_size]:
            v = (v << 1) | int(b)
        out.append(v)
    return out  # in [0, 2^patch_size - 1]


# ----------------------------
# Dataset: space-time windows -> rule label (+ time/space ids)
# ----------------------------
class CARuleWindowDataset(Dataset):
    """
    Each sample is a window of T consecutive states from one run.
    Tokens are flattened: [CLS] + (T * S) patch tokens.
    Additionally returns:
      - time_ids per token in [0..T] (T reserved for CLS)
      - space_ids per token in [0..S] (S reserved for CLS)
    Label is rule_id in [0..num_rules-1].
    """
    def __init__(
        self,
        runs_by_rule: Dict[int, List[dict]],
        rule_to_label: Dict[int, int],
        lattice_width: int,
        patch_size: int,
        window_T: int,
        cls_token_id: int,
    ):
        self.rule_to_label = rule_to_label
        self.lattice_width = lattice_width
        self.patch_size = patch_size
        self.window_T = window_T
        self.cls_token_id = cls_token_id

        if lattice_width % patch_size != 0:
            raise ValueError("lattice_width must be divisible by patch_size")
        self.S = lattice_width // patch_size  # patches per row

        # Pre-tokenize all runs as patch-rows: rows_patches[t] = [patch tokens length S]
        self.tokenized: List[Tuple[List[List[int]], int]] = []
        for rule, runs in runs_by_rule.items():
            label = rule_to_label[rule]
            for r in runs:
                rows_bits = r["output"]  # list over time of [0/1]*width
                if len(rows_bits) < window_T:
                    continue
                if len(rows_bits[0]) != lattice_width:
                    raise ValueError(
                        f"Width mismatch in rule {rule}: got {len(rows_bits[0])}, expected {lattice_width}"
                    )
                rows_patches = [bits_to_patches(row, patch_size) for row in rows_bits]
                self.tokenized.append((rows_patches, label))

        # Build index of all possible windows (run_idx, start_t)
        self.index: List[Tuple[int, int]] = []
        for run_idx, (rows_patches, _label) in enumerate(self.tokenized):
            T_total = len(rows_patches)
            for start in range(0, T_total - window_T + 1):
                self.index.append((run_idx, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        run_idx, start = self.index[idx]
        rows_patches, label = self.tokenized[run_idx]

        window = rows_patches[start: start + self.window_T]  # list length T, each length S

        # Flatten tokens + build time/space id arrays aligned to tokens
        flat_tokens = [self.cls_token_id]
        time_ids = [self.window_T]  # CLS time id = T
        space_ids = [self.S]        # CLS space id = S

        for t, row in enumerate(window):
            for s, tok in enumerate(row):
                flat_tokens.append(tok)
                time_ids.append(t)
                space_ids.append(s)

        x = torch.tensor(flat_tokens, dtype=torch.long)   # (1 + T*S,)
        tpos = torch.tensor(time_ids, dtype=torch.long)   # (1 + T*S,)
        spos = torch.tensor(space_ids, dtype=torch.long)  # (1 + T*S,)
        y = torch.tensor(label, dtype=torch.long)
        return x, tpos, spos, y


# ----------------------------
# Transformer classifier with time+space embeddings
# ----------------------------
class TransformerRuleClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_classes: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        time_size: int,   # = window_T + 1 (CLS)
        space_size: int,  # = S + 1 (CLS)
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.time_emb = nn.Embedding(time_size, d_model)
        self.space_emb = nn.Embedding(space_size, d_model)

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
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, tpos: torch.Tensor, spos: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        if L != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {L}")

        h = self.tok_emb(x) + self.time_emb(tpos) + self.space_emb(spos)
        h = self.encoder(h)
        h = self.ln_f(h)
        cls = h[:, 0, :]
        return self.head(cls)  # (B, num_classes)

    @torch.no_grad()
    def probs(self, x: torch.Tensor, tpos: torch.Tensor, spos: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, tpos, spos)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_topk(self, x: torch.Tensor, tpos: torch.Tensor, spos: torch.Tensor, k: int = 5):
        probs = self.probs(x, tpos, spos)
        top_p, top_i = torch.topk(probs, k=min(k, probs.size(-1)), dim=-1)
        return top_i, top_p


# ----------------------------
# Helpers: load + split runs per rule
# ----------------------------
def load_runs_for_rule(json_path: Path) -> List[dict]:
    return json.loads(json_path.read_text())


def split_runs(runs: List[dict], train_frac: float, seed: int) -> Tuple[List[dict], List[dict]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(runs), generator=g).tolist()
    n_train = max(1, int(train_frac * len(runs)))
    train = [runs[i] for i in perm[:n_train]]
    val = [runs[i] for i in perm[n_train:]]
    if len(val) == 0 and len(train) > 1:
        val = [train.pop()]
    return train, val


@torch.no_grad()
def eval_topk(model, loader, device, k_list=(1, 3, 5)):
    model.eval()
    total = 0
    correct = {k: 0 for k in k_list}

    for x, tpos, spos, y in loader:
        x, tpos, spos, y = x.to(device), tpos.to(device), spos.to(device), y.to(device)
        probs = model.probs(x, tpos, spos)
        total += y.size(0)

        for k in k_list:
            topk = torch.topk(probs, k=min(k, probs.size(-1)), dim=-1).indices
            hit = (topk == y.unsqueeze(-1)).any(dim=-1).sum().item()
            correct[k] += hit

    model.train()
    return {k: correct[k] / max(total, 1) for k in k_list}


# ----------------------------
# B) Multi-window inference aggregation (per-run)
# ----------------------------
@torch.no_grad()
def predict_rules_from_run_output(
    model: TransformerRuleClassifier,
    run_output_bits: List[List[int]],
    *,
    patch_size: int,
    window_T: int,
    lattice_width: int,
    cls_token_id: int,
    device: str,
    n_windows: int = 50,
    top_k: int = 5,
    seed: int = 0,
):
    """
    run_output_bits: list of time rows, each row is list of 0/1 of length lattice_width.
    Samples n_windows random windows, averages probabilities, returns top-k candidates.

    Returns: (top_rules_idx, top_probs, avg_probs_vector)
    """
    model.eval()
    if len(run_output_bits) < window_T:
        raise ValueError(f"Run too short: has {len(run_output_bits)} rows, need at least window_T={window_T}")
    if len(run_output_bits[0]) != lattice_width:
        raise ValueError(f"Width mismatch: got {len(run_output_bits[0])}, expected {lattice_width}")
    if lattice_width % patch_size != 0:
        raise ValueError("lattice_width must be divisible by patch_size")

    S = lattice_width // patch_size
    max_start = len(run_output_bits) - window_T
    g = torch.Generator().manual_seed(seed)

    # choose start indices (with replacement if needed)
    if max_start == 0:
        starts = [0] * n_windows
    else:
        starts = torch.randint(low=0, high=max_start + 1, size=(n_windows,), generator=g).tolist()

    # accumulate probs
    avg_probs = torch.zeros((model.num_classes,), dtype=torch.float32, device=device)

    for st in starts:
        window_bits = run_output_bits[st: st + window_T]  # length T

        # build tokens + time/space ids
        tokens = [cls_token_id]
        time_ids = [window_T]
        space_ids = [S]

        for t, row_bits in enumerate(window_bits):
            row_patches = bits_to_patches(row_bits, patch_size)
            for s, tok in enumerate(row_patches):
                tokens.append(tok)
                time_ids.append(t)
                space_ids.append(s)

        x = torch.tensor([tokens], dtype=torch.long, device=device)
        tpos = torch.tensor([time_ids], dtype=torch.long, device=device)
        spos = torch.tensor([space_ids], dtype=torch.long, device=device)

        probs = model.probs(x, tpos, spos)[0]  # (C,)
        avg_probs += probs

    avg_probs /= float(len(starts))

    top_p, top_i = torch.topk(avg_probs, k=min(top_k, avg_probs.numel()))
    return top_i.detach().cpu().tolist(), top_p.detach().cpu().tolist(), avg_probs.detach().cpu()


# ----------------------------
# Main training
# ----------------------------
def main(cfg: Config):
    torch.manual_seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    rules = list(range(cfg.rule_start, cfg.rule_end_exclusive))

    train_runs_by_rule: Dict[int, List[dict]] = {}
    val_runs_by_rule: Dict[int, List[dict]] = {}

    for rule in rules:
        p = data_dir / f"rule_{rule}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Generate it first.")
        runs = load_runs_for_rule(p)
        tr, va = split_runs(runs, cfg.train_frac_runs, seed=cfg.seed + rule)
        train_runs_by_rule[rule] = tr
        val_runs_by_rule[rule] = va

    rule_to_label = {r: i for i, r in enumerate(rules)}
    label_to_rule = {i: r for r, i in rule_to_label.items()}

    # Token vocab: 0..255 for patches, and 256 for CLS
    patch_vocab = 2 ** cfg.patch_size
    cls_token_id = patch_vocab
    vocab_size = patch_vocab + 1

    S = cfg.lattice_width // cfg.patch_size
    seq_len = 1 + cfg.window_T * S
    time_size = cfg.window_T + 1
    space_size = S + 1

    train_ds = CARuleWindowDataset(
        runs_by_rule=train_runs_by_rule,
        rule_to_label=rule_to_label,
        lattice_width=cfg.lattice_width,
        patch_size=cfg.patch_size,
        window_T=cfg.window_T,
        cls_token_id=cls_token_id,
    )
    val_ds = CARuleWindowDataset(
        runs_by_rule=val_runs_by_rule,
        rule_to_label=rule_to_label,
        lattice_width=cfg.lattice_width,
        patch_size=cfg.patch_size,
        window_T=cfg.window_T,
        cls_token_id=cls_token_id,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = TransformerRuleClassifier(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_classes=len(rules),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        time_size=time_size,
        space_size=space_size,
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x, tpos, spos, y in train_loader:
            x, tpos, spos, y = x.to(cfg.device), tpos.to(cfg.device), spos.to(cfg.device), y.to(cfg.device)
            logits = model(x, tpos, spos)

            # C) label smoothing
            loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        metrics = eval_topk(model, val_loader, cfg.device, k_list=(1, 3, 5))
        print(
            f"epoch {epoch:02d} | "
            f"top1={metrics[1]*100:.1f}% | top3={metrics[3]*100:.1f}% | top5={metrics[5]*100:.1f}%"
        )

    # ----------------------------
    # Demo 1: single-window inference (same as before, but probabilities should be less extreme now)
    # ----------------------------
    model.eval()
    x, tpos, spos, y = next(iter(val_loader))
    x = x[:1].to(cfg.device)
    tpos = tpos[:1].to(cfg.device)
    spos = spos[:1].to(cfg.device)
    y_true = y[0].item()

    top_i, top_p = model.predict_topk(x, tpos, spos, k=5)
    top_i = top_i[0].tolist()
    top_p = top_p[0].tolist()

    print("\nExample inference (single window):")
    print("True rule:", label_to_rule[y_true])
    for rank, (li, pr) in enumerate(zip(top_i, top_p), start=1):
        print(f"  #{rank}: rule {label_to_rule[li]}  prob={pr:.3f}")

    # ----------------------------
    # Demo 2: B) aggregated inference from an entire held-out run
    # ----------------------------
    # pick one rule and one validation run for that rule (deterministic pick)
    demo_rule = rules[0]  # e.g., 100
    demo_run = val_runs_by_rule[demo_rule][0]["output"]
    true_rule = demo_rule

    top_labels, top_probs, _avg = predict_rules_from_run_output(
        model,
        demo_run,
        patch_size=cfg.patch_size,
        window_T=cfg.window_T,
        lattice_width=cfg.lattice_width,
        cls_token_id=cls_token_id,
        device=cfg.device,
        n_windows=50,
        top_k=5,
        seed=cfg.seed,
    )

    print("\nExample inference (aggregated over 50 windows from one run):")
    print("True rule:", true_rule)
    for rank, (lab, pr) in enumerate(zip(top_labels, top_probs), start=1):
        print(f"  #{rank}: rule {label_to_rule[lab]}  prob={pr:.3f}")


if __name__ == "__main__":
    cfg = Config(
        data_dir="GENERATED_DATA",
        rule_start=100,
        rule_end_exclusive=140,
        lattice_width=32,
        patch_size=8,
        window_T=32,       # A)
        epochs=15,
        label_smoothing=0.05,  # C)
    )
    main(cfg)
    print("Done.")
