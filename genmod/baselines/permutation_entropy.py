"""Permutation entropy computation and kNN baseline classifier.

Based on Bandt & Pompe (2002) and Garland et al. (2014) "Model-free
quantification of time-series predictability."

PE measures the complexity of a time series by examining the frequency
of ordinal patterns (permutations) in sliding windows.
"""

from collections import Counter
from itertools import permutations
from math import factorial, log
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def permutation_entropy(
    time_series: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """Compute permutation entropy of a 1D time series.

    Args:
        time_series: 1D array of values.
        order: Embedding dimension (length of ordinal patterns).
        delay: Time delay between consecutive elements in patterns.
        normalize: If True, normalize by log(order!) to get values in [0, 1].

    Returns: Permutation entropy value.
    """
    n = len(time_series)
    if n < order * delay:
        return 0.0

    # Extract ordinal patterns
    pattern_counts = Counter()
    for i in range(n - (order - 1) * delay):
        window = [time_series[i + j * delay] for j in range(order)]
        # Convert to ordinal pattern (rank ordering)
        pattern = tuple(np.argsort(window))
        pattern_counts[pattern] += 1

    total = sum(pattern_counts.values())
    if total == 0:
        return 0.0

    # Shannon entropy of pattern distribution
    entropy = 0.0
    for count in pattern_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * log(p)

    if normalize:
        max_entropy = log(factorial(order))
        if max_entropy > 0:
            entropy /= max_entropy

    return entropy


def weighted_permutation_entropy(
    time_series: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """Compute weighted permutation entropy (Fadlallah et al., 2013).

    Weights patterns by the variance of the corresponding subsequence,
    giving more weight to patterns from high-amplitude regions.
    """
    n = len(time_series)
    if n < order * delay:
        return 0.0

    pattern_weights: Dict[tuple, float] = {}
    for i in range(n - (order - 1) * delay):
        window = [time_series[i + j * delay] for j in range(order)]
        pattern = tuple(np.argsort(window))
        weight = np.var(window) if np.var(window) > 0 else 1e-10
        pattern_weights[pattern] = pattern_weights.get(pattern, 0.0) + weight

    total_weight = sum(pattern_weights.values())
    if total_weight == 0:
        return 0.0

    entropy = 0.0
    for w in pattern_weights.values():
        p = w / total_weight
        if p > 0:
            entropy -= p * log(p)

    if normalize:
        max_entropy = log(factorial(order))
        if max_entropy > 0:
            entropy /= max_entropy

    return entropy


def compute_pe_features(
    spacetime_grid: np.ndarray,
    orders: List[int] = [3, 4, 5],
    delays: List[int] = [1, 2],
    include_weighted: bool = True,
) -> np.ndarray:
    """Compute a feature vector of PE values for a space-time window.

    Extracts PE along temporal columns, spatial rows, and diagonals.
    Aggregates with mean and std across columns/rows.

    Args:
        spacetime_grid: (T, W) binary array.
        orders: List of embedding dimensions to use.
        delays: List of time delays.
        include_weighted: Also compute weighted PE.

    Returns: 1D feature vector.
    """
    T, W = spacetime_grid.shape
    features = []

    for order in orders:
        for delay in delays:
            # Temporal PE: compute PE for each column, then aggregate
            col_pe = []
            col_wpe = []
            for col in range(W):
                ts = spacetime_grid[:, col].astype(float)
                col_pe.append(permutation_entropy(ts, order, delay))
                if include_weighted:
                    col_wpe.append(weighted_permutation_entropy(ts, order, delay))

            features.extend([np.mean(col_pe), np.std(col_pe)])
            if include_weighted:
                features.extend([np.mean(col_wpe), np.std(col_wpe)])

            # Spatial PE: compute PE for each row, then aggregate
            row_pe = []
            row_wpe = []
            for row in range(T):
                ts = spacetime_grid[row, :].astype(float)
                row_pe.append(permutation_entropy(ts, order, delay))
                if include_weighted:
                    row_wpe.append(weighted_permutation_entropy(ts, order, delay))

            features.extend([np.mean(row_pe), np.std(row_pe)])
            if include_weighted:
                features.extend([np.mean(row_wpe), np.std(row_wpe)])

    # Global density features
    features.append(spacetime_grid.mean())
    features.append(spacetime_grid.std())

    # Density change rate (temporal)
    row_densities = spacetime_grid.mean(axis=1)
    if len(row_densities) > 1:
        features.append(np.mean(np.abs(np.diff(row_densities))))
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float64)


def compute_pe_features_for_runs(
    runs_by_rule: Dict[int, list],
    rule_to_label: Dict[int, int],
    window_T: int = 32,
    orders: List[int] = [3, 4, 5],
    delays: List[int] = [1, 2],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PE features for all runs across all rules.

    For each run, extracts features from multiple windows and averages.

    Returns: (features, labels)
    """
    all_features = []
    all_labels = []

    for rule, runs in runs_by_rule.items():
        label = rule_to_label[rule]
        for run in runs:
            grid = np.array(run["output"], dtype=np.float64)
            T_total, W = grid.shape
            if T_total < window_T:
                continue

            # Sample windows and average features
            window_features = []
            for start in range(0, T_total - window_T + 1, max(1, (T_total - window_T) // 5)):
                window = grid[start : start + window_T]
                feat = compute_pe_features(window, orders=orders, delays=delays)
                window_features.append(feat)

            if window_features:
                avg_feat = np.mean(window_features, axis=0)
                all_features.append(avg_feat)
                all_labels.append(label)

    return np.array(all_features), np.array(all_labels)


def pe_knn_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """kNN classifier on PE features.

    Returns: (predictions, probability_estimates)
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)

    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(X_train, train_labels)

    predictions = knn.predict(X_test)
    probabilities = knn.predict_proba(X_test)

    return predictions, probabilities
