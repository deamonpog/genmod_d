"""Generate a library of rule trees and their Schelling simulation data.

This is the data-generation entry point for Option D. It performs:
  1. Generate n_candidates rule trees with the chosen method
  2. Deduplicate structurally and behaviorally
  3. Save the deduplicated tree library to tree_library.json
  4. Run num_runs Schelling simulations for every tree
  5. Save runs to ruletree_NNNN.json (one file per tree)

This is a heavy job and is intended to run on HPC, not locally.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genmod.data.rule_trees import save_tree_library
from genmod.data.tree_generators import generate_tree_library
from genmod.data.schelling_ruletree import generate_all_ruletrees


def parse_int_list(s):
    """Parse a comma-separated list of integers like '100,200,300'."""
    return [int(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Generate rule-tree dataset")
    parser.add_argument("--n_candidates", type=int, default=10000,
                        help="Number of candidate trees to generate before dedup")
    parser.add_argument("--max_depth", type=int, default=3,
                        help="Maximum tree depth")
    parser.add_argument("--method", type=str, default="mixed",
                        choices=["pseudo", "quasi", "mixed"],
                        help="Tree generation method")
    parser.add_argument("--num_runs", type=int, default=100,
                        help="Schelling simulation runs per tree")
    parser.add_argument("--grid_size", type=int, default=50)
    parser.add_argument("--occupancy", type=float, default=0.95)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--snapshot_ticks", type=str, default="100,200,300,400,500",
                        help="Comma-separated list of ticks at which to snapshot")
    parser.add_argument("--output_dir", type=str, default="GENERATED_DATA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of multiprocessing workers for fingerprinting "
                             "and simulation. 1 = serial. Set to the number of "
                             "available CPU cores on HPC.")
    args = parser.parse_args()

    snapshot_ticks = parse_int_list(args.snapshot_ticks)
    out_root = Path(args.output_dir) / "rule_trees"

    print("=" * 60)
    print("Rule-tree data generation")
    print("=" * 60)
    print("  candidates  : {}".format(args.n_candidates))
    print("  max_depth   : {}".format(args.max_depth))
    print("  method      : {}".format(args.method))
    print("  num_runs    : {}".format(args.num_runs))
    print("  grid_size   : {}".format(args.grid_size))
    print("  max_steps   : {}".format(args.max_steps))
    print("  snapshots   : {}".format(snapshot_ticks))
    print("  output_dir  : {}".format(out_root))
    print("  seed        : {}".format(args.seed))
    print("  workers     : {}".format(args.workers))
    print()

    # Step 1-3: generate and deduplicate the tree library
    library = generate_tree_library(
        n_candidates=args.n_candidates,
        max_depth=args.max_depth,
        seed=args.seed,
        method=args.method,
        n_workers=args.workers,
    )
    save_tree_library(library, out_root / "tree_library.json")

    # Step 4-5: simulate every tree
    generate_all_ruletrees(
        library,
        grid_size=args.grid_size,
        num_runs=args.num_runs,
        occupancy=args.occupancy,
        max_steps=args.max_steps,
        snapshot_ticks=snapshot_ticks,
        output_dir=str(out_root),
        n_workers=args.workers,
    )

    print()
    print("Data generation complete: {} trees, {} runs each".format(
        len(library), args.num_runs))


if __name__ == "__main__":
    main()
