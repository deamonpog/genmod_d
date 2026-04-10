"""Generate ECA data for all 256 rules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genmod.data.eca import generate_all_rules


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=256)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="GENERATED_DATA")
    args = parser.parse_args()

    generate_all_rules(
        rule_range=range(args.start, args.end),
        size=args.size,
        steps=args.steps,
        num_runs=args.num_runs,
        output_dir=args.output_dir,
    )
