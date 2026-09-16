#!/usr/bin/env python3
"""
Train/validation/test N-Z figure for each extrapolation separation, in the
same style as Fig. 1 (prepare_data.draw_split_figure).

Reads the splits written by run_extrapolation.py:
    <outdir>/<separation>/seed_<seed>/{train,val,test}.csv

    python plot_splits.py --outdir results_extrap_ame --seed 42
    -> results_extrap_ame/NZ_distribution_neutron_rich_s42.png
       results_extrap_ame/NZ_distribution_heavy_s42.png
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw(train, val, test, outpath, z_cut=None):
    plt.figure(figsize=(10.5, 7.5))
    plt.scatter(train["N"], train["Z"], s=24, alpha=0.30, color="gray", label="Train")
    plt.scatter(val["N"], val["Z"], s=28, alpha=0.75, marker="^",
                color="indigo", label="Validation")
    plt.scatter(test["N"], test["Z"], s=42, alpha=0.95, marker="D",
                color="darkorange", label="Test")
    if z_cut is not None:
        plt.axhline(z_cut + 0.5, color="black", ls="--", lw=1.2)

    plt.xlabel("Neutron Number (N)", fontsize=24)
    plt.ylabel("Proton Number (Z)", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.grid(alpha=0.30)
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.savefig(outpath, dpi=350, bbox_inches="tight")
    plt.close()


def main(args):
    root = Path(args.outdir)
    folders = [root / s for s in args.splits] if args.splits else \
              sorted(p for p in root.iterdir() if (p / f"seed_{args.seed}").is_dir())
    for folder in folders:
        run = folder / f"seed_{args.seed}"
        train, val, test = (pd.read_csv(run / f"{k}.csv") for k in ("train", "val", "test"))
        out = root / f"NZ_distribution_{folder.name}_s{args.seed}.png"
        draw(train, val, test, out, args.z_cut if folder.name == "heavy" else None)
        print(f"{folder.name:<14} train={len(train):5d} val={len(val):5d} "
              f"test={len(test):5d}  -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_extrapolation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", nargs="*", help="default: every separation found")
    ap.add_argument("--z-cut", type=int, default=80, help="dashed line for 'heavy'")
    main(ap.parse_args())
