"""Model factory: build models from config with size presets."""

from typing import Dict, Any

from genmod.models.transformer import TransformerRuleClassifier, TransformerRuleClassifierStdPos


MODEL_SIZE_PRESETS: Dict[str, Dict[str, int]] = {
    "tiny":   {"d_model": 64,  "n_heads": 2, "n_layers": 2},
    "small":  {"d_model": 128, "n_heads": 4, "n_layers": 2},
    "medium": {"d_model": 128, "n_heads": 4, "n_layers": 4},
    "base":   {"d_model": 256, "n_heads": 8, "n_layers": 4},
    "large":  {"d_model": 384, "n_heads": 8, "n_layers": 6},
}


def build_model(
    num_classes: int,
    vocab_size: int,
    seq_len: int,
    time_size: int,
    space_size: int,
    model_size: str = "base",
    dropout: float = 0.1,
    positional_encoding: str = "time_space",
) -> TransformerRuleClassifier:
    """Build a Transformer classifier from explicit tokenization params.

    System-agnostic: works for ECA, logistic map, Schelling, or any system
    that provides vocab_size, seq_len, time_size, space_size.
    """
    if model_size not in MODEL_SIZE_PRESETS:
        raise ValueError(f"Unknown model_size: {model_size}. Choose from {list(MODEL_SIZE_PRESETS)}")

    preset = MODEL_SIZE_PRESETS[model_size]
    model_cls = TransformerRuleClassifier if positional_encoding == "time_space" else TransformerRuleClassifierStdPos

    return model_cls(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_classes=num_classes,
        d_model=preset["d_model"],
        n_heads=preset["n_heads"],
        n_layers=preset["n_layers"],
        dropout=dropout,
        time_size=time_size,
        space_size=space_size,
    )


def build_model_for_eca(
    num_classes: int,
    lattice_width: int = 32,
    patch_size: int = 8,
    window_T: int = 32,
    **kwargs,
) -> TransformerRuleClassifier:
    """Convenience wrapper for ECA (backward compatibility)."""
    tok = get_eca_tokenization_params(lattice_width, patch_size, window_T)
    return build_model(num_classes=num_classes, **tok, **kwargs)


def get_eca_tokenization_params(lattice_width: int = 32, patch_size: int = 8, window_T: int = 32) -> dict:
    """Compute tokenization parameters for ECA."""
    patch_vocab = 2 ** patch_size
    cls_token_id = patch_vocab
    vocab_size = patch_vocab + 1
    S = lattice_width // patch_size
    seq_len = 1 + window_T * S
    return {
        "cls_token_id": cls_token_id,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "time_size": window_T + 1,
        "space_size": S + 1,
    }


def get_tokenization_params(system: str, **kwargs) -> dict:
    """Dispatch tokenization params by system name."""
    if system == "eca":
        return get_eca_tokenization_params(**kwargs)
    elif system == "logistic_map":
        from genmod.data.logistic_dataset import get_logistic_tokenization_params
        return get_logistic_tokenization_params(**kwargs)
    elif system == "schelling":
        from genmod.data.schelling_dataset import get_schelling_tokenization_params
        return get_schelling_tokenization_params(**kwargs)
    else:
        raise ValueError(f"Unknown system: {system}")
