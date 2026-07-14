"""Run the ablation grid.

Each ablation changes exactly ONE thing relative to the primary configuration
(configs/fish_transformer.yaml), so any difference in the result is
attributable to that change and nothing else.

    window       T in {16, 32, 64, 128, 256, 512}
                 How much observation is needed before the rule becomes
                 identifiable, and where does identifiability saturate? The
                 aggregate baseline already reaches 0.962 at T=512, so this
                 curve is a result in its own right, not just a sweep.

    features     all (20) vs basic (11, single-agent kinematics only)
                 Do the relational features carry the rule signal, or can the
                 Transformer recover it from raw kinematics by itself?

    augment      each symmetry switched off in turn
                 Rotation and reflection are genuine symmetries of a circular
                 arena. Agent permutation is REDUNDANT with the architecture
                 (the model is exactly permutation invariant for any weights,
                 see verify_permutation_invariance.py), so switching it off
                 should change nothing. If it does, the architecture claim is
                 wrong and we want to know.

Configs are written to disk before running, so each ablation is reproducible
on its own:

    python scripts/fish/09_train_fish_transformer.py --config <written config>

Usage:
    python scripts/fish/12_run_fish_ablations.py --list
    python scripts/fish/12_run_fish_ablations.py --only window
    python scripts/fish/12_run_fish_ablations.py --only window --which T256
"""

import argparse
import copy
import os
import subprocess
import sys

import yaml

BASE_CONFIG = os.path.join("configs", "fish_transformer.yaml")
GEN_DIR = os.path.join("configs", "fish_ablations")


def build_grid(base):
    """Every ablation, as (group, name, config). One change each."""
    grid = []

    for T in [16, 32, 64, 128, 256, 512]:
        cfg = copy.deepcopy(base)
        cfg["window"] = T
        cfg["tag"] = "abl_window_T%d" % T
        # Attention cost is O(A*T^2 + T*A^2) per window and the window count
        # falls as 1/T, so cost grows about linearly in T. Long windows also
        # yield far fewer windows, so the batch can stay put but the epoch
        # budget is better spent with a shorter schedule at large T.
        if T >= 256:
            cfg["batch_size"] = 32
        grid.append(("window", "T%d" % T, cfg))

    for feats in ["all", "basic"]:
        cfg = copy.deepcopy(base)
        cfg["features"] = feats
        cfg["tag"] = "abl_features_%s" % feats
        grid.append(("features", feats, cfg))

    variants = {
        "none": dict(enabled=False, rotate=False, reflect=False, permute=False),
        "rot": dict(enabled=True, rotate=True, reflect=False, permute=False),
        "rot_refl": dict(enabled=True, rotate=True, reflect=True, permute=False),
        "all": dict(enabled=True, rotate=True, reflect=True, permute=True),
    }
    for name, aug in variants.items():
        cfg = copy.deepcopy(base)
        cfg["augment"] = aug
        cfg["tag"] = "abl_augment_%s" % name
        grid.append(("augment", name, cfg))

    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    choices=["window", "features", "augment"])
    ap.add_argument("--which", default=None, help="a single variant name")
    ap.add_argument("--list", action="store_true", help="print the grid and exit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(BASE_CONFIG) as fh:
        base = yaml.safe_load(fh)

    grid = build_grid(base)
    if args.only:
        grid = [g for g in grid if g[0] == args.only]
    if args.which:
        grid = [g for g in grid if g[1] == args.which]

    os.makedirs(GEN_DIR, exist_ok=True)

    print("%d ablation runs" % len(grid))
    for group, name, cfg in grid:
        print("  %-9s %-10s tag=%-22s T=%-4d features=%-6s augment=%s"
              % (group, name, cfg["tag"], cfg["window"], cfg["features"],
                 cfg["augment"]["enabled"]))
    if args.list:
        return

    for group, name, cfg in grid:
        path = os.path.join(GEN_DIR, "%s.yaml" % cfg["tag"])
        with open(path, "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)

        cmd = [sys.executable, os.path.join("scripts", "fish",
                                            "09_train_fish_transformer.py"),
               "--config", path]
        print("\n" + "=" * 70)
        print("RUN %s   (%s)" % (cfg["tag"], " ".join(cmd)))
        print("=" * 70)
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)

        subprocess.run([sys.executable,
                        os.path.join("scripts", "fish",
                                     "10_calibrate_fish_raps.py"),
                        "--tag", cfg["tag"]], check=True)


if __name__ == "__main__":
    main()
