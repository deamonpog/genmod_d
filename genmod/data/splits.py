"""Data splitting logic for train/val/test and rule-level generalization.

Supports:
- Run-level 3-way splits (train/val/test) within each rule
- Rule-level splits for generalization experiments with equivalence awareness
"""

from typing import Dict, FrozenSet, List, Optional, Tuple

import torch

from genmod.data.wolfram import get_equivalence_class, get_wolfram_class, EQUIVALENCE_CLASSES


def split_runs(
    runs: list,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[list, list, list]:
    """Three-way split of runs for a single rule.

    Returns: (train_runs, val_runs, test_runs)
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(runs), generator=g).tolist()
    n = len(runs)
    n_train = max(1, int(train_frac * n))
    n_val = max(1, int(val_frac * n))
    # Ensure test has at least 1 run if possible
    if n_train + n_val >= n and n > 2:
        n_val = max(1, n - n_train - 1)

    train = [runs[i] for i in perm[:n_train]]
    val = [runs[i] for i in perm[n_train : n_train + n_val]]
    test = [runs[i] for i in perm[n_train + n_val :]]

    # Ensure val and test are non-empty
    if len(val) == 0 and len(train) > 1:
        val = [train.pop()]
    if len(test) == 0 and len(train) > 1:
        test = [train.pop()]

    return train, val, test


def split_rules_for_generalization(
    all_rules: List[int],
    held_out_fraction: float = 0.25,
    strategy: str = "equivalence",
    seed: int = 42,
    ensure_class_coverage: bool = True,
) -> Tuple[List[int], List[int]]:
    """Split rules into train and held-out sets.

    Args:
        all_rules: List of all rule numbers to split.
        held_out_fraction: Fraction of rules to hold out.
        strategy: Splitting strategy.
            - "random": Uniform random split (may leak via equivalent rules).
            - "equivalence": Hold out entire equivalence classes (no leakage).
            - "by_wolfram_class": Hold out entire Wolfram classes.
        seed: Random seed.
        ensure_class_coverage: If True, ensure at least one rule from each
            Wolfram class in both train and held-out sets (when possible).

    Returns: (train_rules, held_out_rules)
    """
    g = torch.Generator().manual_seed(seed)
    rule_set = set(all_rules)

    if strategy == "random":
        perm = torch.randperm(len(all_rules), generator=g).tolist()
        n_held = max(1, int(held_out_fraction * len(all_rules)))
        held_out = [all_rules[i] for i in perm[:n_held]]
        train = [all_rules[i] for i in perm[n_held:]]

    elif strategy == "equivalence":
        # Group rules by equivalence class, only including rules in our set
        eq_classes_in_set: List[FrozenSet[int]] = []
        seen = set()
        for rule in all_rules:
            if rule not in seen:
                eq = get_equivalence_class(rule)
                present = frozenset(r for r in eq if r in rule_set)
                eq_classes_in_set.append(present)
                seen.update(present)

        n_held_classes = max(1, int(held_out_fraction * len(eq_classes_in_set)))
        perm = torch.randperm(len(eq_classes_in_set), generator=g).tolist()

        held_classes = [eq_classes_in_set[i] for i in perm[:n_held_classes]]
        train_classes = [eq_classes_in_set[i] for i in perm[n_held_classes:]]

        held_out = sorted(r for cls in held_classes for r in cls)
        train = sorted(r for cls in train_classes for r in cls)

    elif strategy == "by_wolfram_class":
        # Hold out entire Wolfram classes
        classes_present = sorted(set(get_wolfram_class(r) for r in all_rules))
        n_held = max(1, int(held_out_fraction * len(classes_present)))
        perm = torch.randperm(len(classes_present), generator=g).tolist()
        held_classes = {classes_present[i] for i in perm[:n_held]}
        held_out = [r for r in all_rules if get_wolfram_class(r) in held_classes]
        train = [r for r in all_rules if get_wolfram_class(r) not in held_classes]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if ensure_class_coverage and strategy != "by_wolfram_class":
        # Ensure both sets have representation from each Wolfram class
        train_classes = set(get_wolfram_class(r) for r in train)
        held_classes = set(get_wolfram_class(r) for r in held_out)
        all_classes = train_classes | held_classes

        for cls in all_classes:
            if cls not in train_classes:
                # Move one rule of this class from held_out to train
                for r in held_out:
                    if get_wolfram_class(r) == cls:
                        held_out.remove(r)
                        train.append(r)
                        break
            if cls not in held_classes:
                # Move one rule of this class from train to held_out
                for r in train:
                    if get_wolfram_class(r) == cls:
                        train.remove(r)
                        held_out.append(r)
                        break

    return sorted(train), sorted(held_out)


def prepare_splits(
    all_runs_by_rule: Dict[int, list],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict[int, list], Dict[int, list], Dict[int, list]]:
    """Split runs within each rule into train/val/test.

    Returns: (train_runs_by_rule, val_runs_by_rule, test_runs_by_rule)
    """
    train_by_rule: Dict[int, list] = {}
    val_by_rule: Dict[int, list] = {}
    test_by_rule: Dict[int, list] = {}

    for rule, runs in all_runs_by_rule.items():
        tr, va, te = split_runs(runs, train_frac, val_frac, seed=seed + rule)
        train_by_rule[rule] = tr
        val_by_rule[rule] = va
        test_by_rule[rule] = te

    return train_by_rule, val_by_rule, test_by_rule
