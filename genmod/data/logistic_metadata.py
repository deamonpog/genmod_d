"""Logistic map regime metadata.

The logistic map x_{n+1} = r*x*(1-x) exhibits different dynamical regimes
depending on the parameter r:
  - r < 3.0:        Fixed point (converges to (r-1)/r)
  - 3.0 < r < 3.449: Period-2 oscillation
  - 3.449 < r < 3.544: Period-4 and higher period-doubling
  - 3.544 < r < 4.0:  Chaos (with periodic windows)
  - ~3.828 < r < ~3.857: Period-3 window (within chaos)
"""

from typing import Dict, List


REGIME_BOUNDARIES = {
    "fixed_point": (0.0, 3.0),
    "period_2": (3.0, 3.449),
    "period_4": (3.449, 3.544),
    "chaos": (3.544, 4.0),
}

# Period-3 window is a sub-interval within chaos
PERIOD_3_WINDOW = (3.8284, 3.8570)


def get_logistic_regime(r: float) -> str:
    """Return the dynamical regime for a given r value."""
    if PERIOD_3_WINDOW[0] <= r <= PERIOD_3_WINDOW[1]:
        return "period_3_window"
    for regime, (lo, hi) in REGIME_BOUNDARIES.items():
        if lo <= r < hi:
            return regime
    if r >= 4.0:
        return "chaos"
    return "fixed_point"


def get_regime_index(r: float) -> int:
    """Return integer index for the regime (for probing/grouping)."""
    regime = get_logistic_regime(r)
    mapping = {"fixed_point": 0, "period_2": 1, "period_4": 2, "chaos": 3, "period_3_window": 4}
    return mapping[regime]


def get_regime_name(idx: int) -> str:
    """Human-readable name for a regime index."""
    names = {0: "Fixed Point", 1: "Period-2", 2: "Period-4", 3: "Chaos", 4: "Period-3 Window"}
    return names.get(idx, "Unknown")


def regimes_for_r_values(r_values: List[float]) -> Dict[str, List[int]]:
    """Group label indices by regime."""
    result: Dict[str, List[int]] = {}
    for label, r in enumerate(r_values):
        regime = get_logistic_regime(r)
        result.setdefault(regime, []).append(label)
    return result
