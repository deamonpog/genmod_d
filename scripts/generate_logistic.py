"""Generate logistic map data for all r-value classes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genmod.data.logistic_map import generate_all_logistic, get_default_r_values


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=40)
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--transient", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="GENERATED_DATA/logistic_map")
    args = parser.parse_args()

    r_values = get_default_r_values(args.n_classes)
    generate_all_logistic(
        r_values=r_values,
        num_runs=args.num_runs,
        steps=args.steps,
        transient=args.transient,
        output_dir=args.output_dir,
    )
