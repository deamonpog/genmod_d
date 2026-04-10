"""Schelling segregation model metadata.

Different tolerance thresholds produce qualitatively different levels
of segregation:
  - tau = 0.0:  No segregation pressure (random mixing)
  - tau ~ 0.25: Mild clustering
  - tau ~ 0.5:  Moderate segregation (the classic Schelling result)
  - tau ~ 0.75: High segregation
  - tau = 1.0:  Complete segregation (agents want 100% same-type neighbors)
"""

from typing import Dict, List


def get_segregation_level(threshold: float) -> str:
    """Return qualitative segregation level for a threshold."""
    if threshold <= 0.125:
        return "none"
    elif threshold <= 0.3:
        return "mild"
    elif threshold <= 0.55:
        return "moderate"
    elif threshold <= 0.8:
        return "high"
    else:
        return "complete"


def get_segregation_index(threshold: float) -> int:
    """Return integer index for segregation level (for probing/grouping)."""
    level = get_segregation_level(threshold)
    mapping = {"none": 0, "mild": 1, "moderate": 2, "high": 3, "complete": 4}
    return mapping[level]


def get_segregation_name(idx: int) -> str:
    """Human-readable name for a segregation index."""
    names = {0: "None", 1: "Mild", 2: "Moderate", 3: "High", 4: "Complete"}
    return names.get(idx, "Unknown")


DEFAULT_THRESHOLDS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]


def levels_for_thresholds(thresholds: List[float]) -> Dict[str, List[int]]:
    """Group label indices by segregation level."""
    result: Dict[str, List[int]] = {}
    for label, tau in enumerate(thresholds):
        level = get_segregation_level(tau)
        result.setdefault(level, []).append(label)
    return result
