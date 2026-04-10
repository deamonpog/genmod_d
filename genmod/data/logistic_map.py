"""Logistic map simulation engine.

The logistic map x_{n+1} = r * x_n * (1 - x_n) produces dynamics ranging
from fixed points to chaos depending on the parameter r ∈ [0, 4].
"""

import json
import random
from pathlib import Path
from typing import List, Optional


def run_logistic_map(
    r: float,
    steps: int = 300,
    transient: int = 100,
    x0: Optional[float] = None,
    seed: Optional[int] = None,
) -> dict:
    """Run a single logistic map simulation.

    Args:
        r: Growth rate parameter.
        steps: Total iterations (transient + recorded).
        transient: Iterations to discard before recording.
        x0: Initial condition. If None, random in (0.01, 0.99).
        seed: Random seed for x0.

    Returns: Dict with r_value, x0, output (recorded time series).
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    if x0 is None:
        x0 = rng.uniform(0.01, 0.99)

    x = x0
    # Run transient
    for _ in range(transient):
        x = r * x * (1 - x)
        x = max(0.0, min(1.0, x))  # clip for numerical stability

    # Record
    output = [x]
    for _ in range(steps - 1):
        x = r * x * (1 - x)
        x = max(0.0, min(1.0, x))
        output.append(x)

    return {
        "r_value": r,
        "x0": x0,
        "output": output,
    }


def generate_r_class_data(
    r_value: float,
    label: int,
    num_runs: int = 100,
    steps: int = 200,
    transient: int = 100,
    output_dir: str = "GENERATED_DATA/logistic_map",
) -> None:
    """Generate multiple runs for a single r value and save to file."""
    runs = []
    for i in range(num_runs):
        run = run_logistic_map(r_value, steps=steps, transient=transient, seed=i)
        run["label"] = label
        runs.append(run)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = out_path / f"r_{label:03d}.json"
    with open(filename, "w") as f:
        json.dump(runs, f)
    print(f"  r={r_value:.4f} (label {label}): {len(runs)} runs -> {filename}")


def generate_all_logistic(
    r_values: Optional[List[float]] = None,
    num_runs: int = 100,
    steps: int = 200,
    transient: int = 100,
    output_dir: str = "GENERATED_DATA/logistic_map",
) -> List[float]:
    """Generate data for all r values.

    Args:
        r_values: List of r values. If None, uses 40 evenly-spaced in [2.5, 4.0].

    Returns: The list of r values used.
    """
    if r_values is None:
        import numpy as np
        r_values = np.linspace(2.5, 4.0, 40).tolist()

    print(f"Generating logistic map data for {len(r_values)} r values...")
    for label, r in enumerate(r_values):
        generate_r_class_data(r, label, num_runs=num_runs, steps=steps,
                              transient=transient, output_dir=output_dir)
    print("Done.")
    return r_values


# Default r values for experiments
def get_default_r_values(n: int = 40) -> List[float]:
    """Return n evenly-spaced r values in [2.5, 4.0]."""
    return [2.5 + i * 1.5 / (n - 1) for i in range(n)]
