"""Generate Schelling segregation data for all threshold classes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genmod.data.schelling import generate_all_schelling, DEFAULT_THRESHOLDS


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--snapshot_interval", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="GENERATED_DATA/schelling")
    args = parser.parse_args()

    generate_all_schelling(
        thresholds=DEFAULT_THRESHOLDS,
        num_runs=args.num_runs,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        snapshot_interval=args.snapshot_interval,
        output_dir=args.output_dir,
    )
