"""Elementary Cellular Automata simulation engine.

Refactored from gen_ca_data.py. Core functions (rule_to_dict, next_gen,
run_automaton) are preserved exactly.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def rule_to_dict(rule_number: int) -> Dict[Tuple[int, int, int], int]:
    """Convert rule number (0-255) to a mapping from 3-bit neighborhood to output."""
    rule_bin = f"{rule_number:08b}"
    keys = [
        (1, 1, 1),
        (1, 1, 0),
        (1, 0, 1),
        (1, 0, 0),
        (0, 1, 1),
        (0, 1, 0),
        (0, 0, 1),
        (0, 0, 0),
    ]
    return {k: int(v) for k, v in zip(keys, rule_bin)}


def next_gen(cells: List[int], rule_map: Dict[Tuple[int, int, int], int],
             boundary: str = "fixed") -> List[int]:
    """Compute next generation of cells.

    Args:
        cells: Current cell states (0 or 1).
        rule_map: Mapping from 3-bit neighborhood to output.
        boundary: "fixed" (0-padded) or "periodic" (wrap-around).
    """
    n = len(cells)
    new_cells = []
    for i in range(n):
        if boundary == "periodic":
            left = cells[(i - 1) % n]
            right = cells[(i + 1) % n]
        else:
            left = cells[i - 1] if i > 0 else 0
            right = cells[i + 1] if i < n - 1 else 0
        center = cells[i]
        new_cells.append(rule_map[(left, center, right)])
    return new_cells


def cells_to_int(cells: List[int]) -> int:
    """Convert binary cell array to base-10 integer."""
    return int("".join(str(c) for c in cells), 2)


def run_automaton(
    rule_number: int,
    size: int = 32,
    steps: int = 100,
    seed: Optional[List[int]] = None,
    boundary: str = "fixed",
    verbose: bool = False,
) -> dict:
    """Run a single automaton simulation and return the data."""
    rule_map = rule_to_dict(rule_number)
    if verbose:
        print("Neighborhoods and outputs:")
        for k in sorted(rule_map.keys(), reverse=True):
            neighborhood = "".join(str(x) for x in k)
            print(f"  {neighborhood} -> {rule_map[k]}")
    if seed is None:
        cells = [0] * size
        cells[size // 2] = 1
    else:
        cells = seed[:size] + [0] * (size - len(seed))
    output = [cells.copy()]
    for _ in range(steps):
        if verbose:
            print_cells(cells)
        cells = next_gen(cells, rule_map, boundary=boundary)
        output.append(cells.copy())
    output_integers = [cells_to_int(gen) for gen in output]
    data = {
        "rule": rule_number,
        "size": size,
        "steps": steps,
        "seed": seed,
        "output": output,
        "output_as_integers": output_integers,
    }
    return data


def print_cells(cells: List[int]) -> None:
    print("".join(["#" if c else " " for c in cells]))


def generate_rule_data(
    rule_number: int,
    size: int = 32,
    steps: int = 100,
    num_runs: int = 100,
    output_dir: str = "GENERATED_DATA",
    boundary: str = "fixed",
) -> None:
    """Generate multiple runs for a rule with different initial conditions and save to file."""
    runs = []
    # Run 1: Single center cell (default)
    runs.append(run_automaton(rule_number, size=size, steps=steps, seed=None, boundary=boundary))
    # Additional runs with random seeds
    for _ in range(num_runs - 1):
        random_seed = [random.randint(0, 1) for _ in range(size)]
        runs.append(run_automaton(rule_number, size=size, steps=steps, seed=random_seed, boundary=boundary))
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = out_path / f"rule_{rule_number}.json"
    with open(filename, "w") as f:
        json.dump(runs, f)
    print(f"  Rule {rule_number}: {len(runs)} runs -> {filename}")


def generate_all_rules(
    rule_range: range = range(256),
    size: int = 32,
    steps: int = 100,
    num_runs: int = 100,
    output_dir: str = "GENERATED_DATA",
    boundary: str = "fixed",
) -> None:
    """Generate data for all rules in range."""
    print(f"Generating data for {len(rule_range)} rules...")
    for rule in rule_range:
        generate_rule_data(rule, size=size, steps=steps, num_runs=num_runs,
                           output_dir=output_dir, boundary=boundary)
    print("Done.")
