"""Transformer-based ECA rule classifier with decomposed time/space embeddings.

Refactored from train_ca_rule_classifier_transformer.py.
Added: extract_embeddings(), TemperatureScaledModel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerRuleClassifier(nn.Module):
    """Transformer encoder classifier with separate time and space embeddings.

    Architecture:
        - Token embedding for patch values (vocab_size includes CLS token)
        - Separate learned embeddings for temporal and spatial positions
        - Standard Transformer encoder (pre-norm, GELU)
        - [CLS] token pooling -> classification head
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_classes: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        time_size: int,
        space_size: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.d_model = d_model

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
        """Forward pass returning logits.

        Args:
            x: Token indices (B, L)
            tpos: Time position indices (B, L)
            spos: Space position indices (B, L)

        Returns: Logits (B, num_classes)
        """
        h = self.tok_emb(x) + self.time_emb(tpos) + self.space_emb(spos)
        h = self.encoder(h)
        h = self.ln_f(h)
        cls = h[:, 0, :]
        return self.head(cls)

    def extract_embeddings(self, x: torch.Tensor, tpos: torch.Tensor, spos: torch.Tensor) -> torch.Tensor:
        """Extract [CLS] token embeddings before the classification head.

        Returns: Embeddings (B, d_model)
        """
        h = self.tok_emb(x) + self.time_emb(tpos) + self.space_emb(spos)
        h = self.encoder(h)
        h = self.ln_f(h)
        return h[:, 0, :]

    @torch.no_grad()
    def probs(self, x: torch.Tensor, tpos: torch.Tensor, spos: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x, tpos, spos)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_topk(self, x: torch.Tensor, tpos: torch.Tensor, spos: torch.Tensor, k: int = 5):
        """Return top-k predictions and probabilities."""
        probs = self.probs(x, tpos, spos)
        top_p, top_i = torch.topk(probs, k=min(k, probs.size(-1)), dim=-1)
        return top_i, top_p

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerRuleClassifierStdPos(nn.Module):
    """Ablation variant: standard positional encoding instead of time/space decomposition."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_classes: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        **kwargs,  # Accept but ignore time_size, space_size
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

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

    def forward(self, x: torch.Tensor, tpos: torch.Tensor = None, spos: torch.Tensor = None) -> torch.Tensor:
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.tok_emb(x) + self.pos_emb(positions)
        h = self.encoder(h)
        h = self.ln_f(h)
        cls = h[:, 0, :]
        return self.head(cls)

    def extract_embeddings(self, x: torch.Tensor, tpos: torch.Tensor = None, spos: torch.Tensor = None) -> torch.Tensor:
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.tok_emb(x) + self.pos_emb(positions)
        h = self.encoder(h)
        h = self.ln_f(h)
        return h[:, 0, :]

    @torch.no_grad()
    def probs(self, x: torch.Tensor, tpos: torch.Tensor = None, spos: torch.Tensor = None) -> torch.Tensor:
        logits = self.forward(x, tpos, spos)
        return F.softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TemperatureScaledModel(nn.Module):
    """Wrapper that applies temperature scaling for post-hoc calibration.

    Reference: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, tpos: torch.Tensor, spos: torch.Tensor) -> torch.Tensor:
        logits = self.model(x, tpos, spos)
        return logits / self.temperature

    @torch.no_grad()
    def probs(self, x: torch.Tensor, tpos: torch.Tensor, spos: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, tpos, spos)
        return F.softmax(logits, dim=-1)

    def extract_embeddings(self, x: torch.Tensor, tpos: torch.Tensor, spos: torch.Tensor) -> torch.Tensor:
        return self.model.extract_embeddings(x, tpos, spos)
