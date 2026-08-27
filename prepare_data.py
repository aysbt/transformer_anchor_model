#!/usr/bin/env python3
"""
Prepare chronological AME2016 -> AME2020 datasets and draw the N-Z split figure.

Outputs
-------
data/trainval.csv
data/test.csv
data/train_val_test_NZ_distribution.png

The train/validation visualization uses:
  - N/Z bins of width 5
  - bins with fewer than 2 nuclei removed
  - 70/30 stratified split
  - random seed 42
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ame_parser import parse_mass_table, validate

MAGIC = [2, 8, 20, 28, 50, 82, 126, 184]
TARGET = "Mass Excess (keV)"


def _shell_index(x):
    return int(np.searchsorted(MAGIC, x, side="right"))


def _particles_holes(x):
    lo, hi = 0, MAGIC[-1]
    for m in MAGIC:
        if x >= m:
            lo = m
    for m in reversed(MAGIC):
        if x <= m:
            hi = m
    return float(x - lo), float(hi - x)


def _valence(x):
    p, h = _particles_holes(x)
    return float(min(p, h))


def _dist_to_magic(x):
    return float(min(abs(x - m) for m in MAGIC))


def add_physics_features(df):
    df = df.copy()
    N = df["N"].astype(float)
    Z = df["Z"].astype(float)
    A = df["A"].astype(float)

    df["A^2/3"] = A ** (2.0 / 3.0)
    df["Z(Z-1)/A^1/3"] = Z * (Z - 1) / A ** (1.0 / 3.0)
    df["(N-Z)^2/A"] = (N - Z) ** 2 / A
    df["(N-Z)/A"] = (N - Z) / A
    df["N/Z"] = np.where(Z > 0, N / np.maximum(Z, 1e-9), 0.0)

    df["ZEO"] = (Z % 2).astype(int)
    df["NEO"] = (N % 2).astype(int)

    df["Zshell_category"] = df["Z"].apply(_shell_index)
    df["Nshell_category"] = df["N"].apply(_shell_index)

    df["deltaZ"] = df["Z"].apply(_dist_to_magic)
    df["deltaN"] = df["N"].apply(_dist_to_magic)

    ph_z = df["Z"].apply(_particles_holes)
    ph_n = df["N"].apply(_particles_holes)
    df["proton_particles"] = [p for p, _ in ph_z]
    df["proton_holes"] = [h for _, h in ph_z]
    df["neutron_particles"] = [p for p, _ in ph_n]
    df["neutron_holes"] = [h for _, h in ph_n]

    df["nu_Z"] = df["Z"].apply(_valence)
    df["nu_N"] = df["N"].apply(_valence)
    denom = df["nu_Z"] + df["nu_N"]
    df["promiscuity"] = np.where(
        denom > 0,
        df["nu_Z"] * df["nu_N"] / np.maximum(denom, 1e-9),
        0.0,
    )
    return df


def add_neighbor_existence(df, known_keys):
    df = df.copy()
    offsets = {
        "neighbor_N_plus_1": (1, 0),
        "neighbor_N_minus_1": (-1, 0),
        "neighbor_Z_plus_1": (0, 1),
        "neighbor_Z_minus_1": (0, -1),
    }

    for name, (dn, dz) in offsets.items():
        df[name + "_exists"] = [
            int((int(n) + dn, int(z) + dz) in known_keys)
            for n, z in zip(df["N"], df["Z"])
        ]
    return df


def chronological_split(old, new):
    old_keys = set(zip(old.N, old.Z))

    tagged = new.copy()
    tagged["is_new"] = [
        (int(n), int(z)) not in old_keys
        for n, z in zip(tagged.N, tagged.Z)
    ]

    # Preserve the semantics of the previous pipeline:
    # common nuclei use AME2020 evaluated target values.
    trainval = tagged[~tagged.is_new].drop(columns=["is_new"]).reset_index(drop=True)
    test = tagged[tagged.is_new].drop(columns=["is_new"]).reset_index(drop=True)

    return trainval, test


def split_for_plot(trainval, seed=42):
    df = trainval.copy()

    # 5-unit intervals in N and Z
    df["_stratum"] = (
        (df["N"] // 5).astype(str)
        + "_"
        + (df["Z"] // 5).astype(str)
    )

    counts = df["_stratum"].value_counts()
    good_strata = counts[counts >= 2].index
    usable = df[df["_stratum"].isin(good_strata)].copy()
    excluded = df[~df["_stratum"].isin(good_strata)].copy()

    train, val = train_test_split(
        usable,
        test_size=0.30,
        random_state=seed,
        stratify=usable["_stratum"],
    )

    return (
        train.drop(columns=["_stratum"]),
        val.drop(columns=["_stratum"]),
        excluded.drop(columns=["_stratum"]),
    )


def draw_split_figure(trainval, test, outpath, seed=42):
    train, val, excluded = split_for_plot(trainval, seed=seed)

    plt.figure(figsize=(10.5, 7.5))

    plt.scatter(
        train["N"], train["Z"],
        s=24, alpha=0.30, color="gray",
        label=f"Train",
    )
    plt.scatter(
        val["N"], val["Z"],
        s=28, alpha=0.75, marker="^",
        label=f"Validation", color="indigo",
    )
    plt.scatter(
        test["N"], test["Z"],
        s=42, alpha=0.95, marker="D", color="darkorange",
        label=f"Test",
    )

    plt.xlabel("Neutron Number (N)", fontsize=18)
    plt.ylabel("Proton Number (Z)", fontsize=18)
    #plt.title("Train, Validation, and Test Distribution in (N, Z)")
    plt.grid(alpha=0.30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=250, bbox_inches="tight")
    plt.close()

    return train, val, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="data/mass16.txt")
    ap.add_argument("--new", default="data/mass20.txt")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--min-nz", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 78)
    print("PREPARING DATA FROM RAW AME TABLES")
    print("=" * 78)

    old = parse_mass_table(args.old)
    new = parse_mass_table(args.new)

    print(f"\nAME2016 measured nuclei: {len(old)}")
    validate(old, "AME2016")
    print(f"\nAME2020 measured nuclei: {len(new)}")
    validate(new, "AME2020")

    trainval, test = chronological_split(old, new)

    if args.min_nz > 0:
        c = args.min_nz
        before_tv, before_te = len(trainval), len(test)

        trainval = trainval[
            (trainval.N >= c) & (trainval.Z >= c)
        ].reset_index(drop=True)

        test = test[
            (test.N >= c) & (test.Z >= c)
        ].reset_index(drop=True)

        print(f"\nApplied cut N >= {c}, Z >= {c}")
        print(f"  train/val: {before_tv} -> {len(trainval)}")
        print(f"  test     : {before_te} -> {len(test)}")

    trainval = trainval.rename(columns={"mass_excess_keV": TARGET})
    test = test.rename(columns={"mass_excess_keV": TARGET})

    # Neighbor-existence flags use only the historical/common pool.
    known_keys = set(zip(trainval.N, trainval.Z))

    trainval = add_neighbor_existence(
        add_physics_features(trainval),
        known_keys,
    )
    test = add_neighbor_existence(
        add_physics_features(test),
        known_keys,
    )

    os.makedirs(args.outdir, exist_ok=True)

    tv_path = os.path.join(args.outdir, "trainval.csv")
    te_path = os.path.join(args.outdir, "test.csv")
    fig_path = os.path.join(args.outdir, "train_val_test_NZ_distribution.png")

    trainval.drop(columns=["extrapolated"], errors="ignore").to_csv(tv_path, index=False)
    test.drop(columns=["extrapolated"], errors="ignore").to_csv(te_path, index=False)

    train, val, excluded = draw_split_figure(
        trainval, test, fig_path, seed=args.seed
    )

    print("\nFinal dataset sizes")
    print(f"  trainval.csv : {len(trainval)}")
    print(f"  test.csv     : {len(test)}")
    print(f"  plot train   : {len(train)}")
    print(f"  plot val     : {len(val)}")
    print(f"  rare-bin rows excluded from train/val split: {len(excluded)}")

    print("\nTest range")
    print(f"  N: {test.N.min()}-{test.N.max()}")
    print(f"  Z: {test.Z.min()}-{test.Z.max()}")
    print(f"  A: {test.A.min()}-{test.A.max()}")

    print(f"\nWrote {tv_path}")
    print(f"Wrote {te_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()