"""Schelling segregation model simulation engine.

Two agent types on a 2D grid. Each agent wants at least a fraction τ
(tolerance threshold) of its neighbors to be the same type. Unhappy agents
move to random vacancies. Different thresholds produce qualitatively
different segregation patterns.
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple


def create_grid(
    size: int = 20,
    occupancy: float = 0.9,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """Create initial random grid.

    Cell values: 0=vacant, 1=type A, 2=type B.
    Approximately equal numbers of A and B agents.
    """
    rng = random.Random(seed)
    n_cells = size * size
    n_occupied = int(n_cells * occupancy)
    n_a = n_occupied // 2
    n_b = n_occupied - n_a
    n_vacant = n_cells - n_occupied

    cells = [1] * n_a + [2] * n_b + [0] * n_vacant
    rng.shuffle(cells)

    grid = []
    for i in range(size):
        grid.append(cells[i * size : (i + 1) * size])
    return grid


def get_neighbors(grid: List[List[int]], row: int, col: int) -> List[int]:
    """Get values of all occupied neighbors (Moore neighborhood)."""
    size = len(grid)
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < size and 0 <= c < size and grid[r][c] != 0:
                neighbors.append(grid[r][c])
    return neighbors


def is_happy(grid: List[List[int]], row: int, col: int, threshold: float) -> bool:
    """Check if an agent at (row, col) is happy with its neighborhood."""
    agent = grid[row][col]
    if agent == 0:
        return True  # vacant cells are always "happy"
    neighbors = get_neighbors(grid, row, col)
    if len(neighbors) == 0:
        return True  # no neighbors = happy
    same = sum(1 for n in neighbors if n == agent)
    return same / len(neighbors) >= threshold


def step(
    grid: List[List[int]],
    threshold: float,
    rng: random.Random,
) -> Tuple[List[List[int]], int]:
    """Execute one simulation step: move all unhappy agents to random vacancies.

    Returns: (new_grid, num_moves)
    """
    size = len(grid)

    # Find unhappy agents and vacancies
    unhappy = []
    vacancies = []
    for r in range(size):
        for c in range(size):
            if grid[r][c] == 0:
                vacancies.append((r, c))
            elif not is_happy(grid, r, c, threshold):
                unhappy.append((r, c))

    if not unhappy or not vacancies:
        return grid, 0

    # Shuffle for random order
    rng.shuffle(unhappy)
    rng.shuffle(vacancies)

    # Make a copy
    new_grid = [row[:] for row in grid]

    moves = 0
    vac_idx = 0
    for r, c in unhappy:
        if vac_idx >= len(vacancies):
            break
        vr, vc = vacancies[vac_idx]
        # Move agent to vacancy
        new_grid[vr][vc] = new_grid[r][c]
        new_grid[r][c] = 0
        # The old position becomes a vacancy
        vacancies.append((r, c))
        vac_idx += 1
        moves += 1

    return new_grid, moves


def run_schelling(
    grid_size: int = 20,
    threshold: float = 0.5,
    occupancy: float = 0.9,
    max_steps: int = 500,
    snapshot_interval: int = 10,
    seed: Optional[int] = None,
) -> dict:
    """Run a Schelling segregation simulation.

    Returns dict with threshold, grid_size, snapshots (list of grids),
    and steps_to_equilibrium.
    """
    rng = random.Random(seed)
    grid = create_grid(grid_size, occupancy, seed=seed)

    snapshots = [grid]
    steps_taken = 0

    for s in range(1, max_steps + 1):
        grid, num_moves = step(grid, threshold, rng)
        steps_taken = s

        if s % snapshot_interval == 0:
            snapshots.append([row[:] for row in grid])

        if num_moves == 0:
            # Equilibrium reached
            if s % snapshot_interval != 0:
                snapshots.append([row[:] for row in grid])
            break

    return {
        "threshold": threshold,
        "grid_size": grid_size,
        "snapshots": snapshots,
        "steps_to_equilibrium": steps_taken,
        "seed": seed,
    }


def generate_threshold_data(
    threshold: float,
    label: int,
    grid_size: int = 20,
    num_runs: int = 100,
    max_steps: int = 500,
    snapshot_interval: int = 10,
    output_dir: str = "GENERATED_DATA/schelling",
) -> None:
    """Generate multiple runs for one threshold and save."""
    runs = []
    for i in range(num_runs):
        run = run_schelling(
            grid_size=grid_size,
            threshold=threshold,
            max_steps=max_steps,
            snapshot_interval=snapshot_interval,
            seed=i,
        )
        run["label"] = label
        runs.append(run)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = out_path / f"threshold_{label:02d}.json"
    with open(filename, "w") as f:
        json.dump(runs, f)
    print(f"  tau={threshold:.3f} (label {label}): {len(runs)} runs -> {filename}")


DEFAULT_THRESHOLDS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]


def generate_all_schelling(
    thresholds: Optional[List[float]] = None,
    num_runs: int = 100,
    grid_size: int = 20,
    max_steps: int = 500,
    snapshot_interval: int = 10,
    output_dir: str = "GENERATED_DATA/schelling",
) -> List[float]:
    """Generate data for all thresholds."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    print(f"Generating Schelling data for {len(thresholds)} thresholds...")
    for label, tau in enumerate(thresholds):
        generate_threshold_data(
            tau, label, grid_size=grid_size, num_runs=num_runs,
            max_steps=max_steps, snapshot_interval=snapshot_interval,
            output_dir=output_dir,
        )
    print("Done.")
    return thresholds
