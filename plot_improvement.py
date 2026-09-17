#!/usr/bin/env python3
"""
Improvement index eta = RMSE(model) / RMSE(LSMF), in the style of
Navarro Perez & Schunck, PLB 833 (2022) 137336, Fig. 3.

One panel per model, bars grouped by Validation / Test, one colour per split.
eta < 1: the model improves on LSMF; eta > 1: it degrades LSMF.
Mean +/- std over seeds, eta computed per seed.

    python plot_improvement.py \
        --chrono results --chrono-model AnchoredFullModel \
        --extrap results_extrap_ame
"""
import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["LSMF+Transformer", "LSMF+anchor+Transformer"]
SPLITS = [("chronological", "AME2016 → AME2020", "tab:blue"),
          ("neutron_rich", "Neutron-rich", "tab:orange"),
          ("heavy", "Heavy ($Z>82$)", "tab:green"),
          ("unmeasured", "Measured → unmeasured", "tab:red")]
GROUPS = [("val", "Validation"), ("test", "Test")]


def chrono_rows(folder, model):
    """Per-seed rows from train.py / test.py outputs, same columns as metrics.csv."""
    rows = []
    for run, label in [(f"{model}_noanchor", "LSMF+Transformer"),
                       (model, "LSMF+anchor+Transformer")]:
        for split, pattern in [("val", "metrics_val"), ("test", "metrics_test")]:
            for path in glob.glob(f"{folder}/{pattern}_{run}_[0-9]*.csv"):
                df = pd.read_csv(path)
                if "regime" in df:
                    df = df[df.regime == "historical"]
                for _, r in df.iterrows():
                    rows.append(dict(separation="chronological", seed=r.seed,
                                     split=split, model=label, rmse_keV=r.rmse_keV))
                    if "rmse_s1_keV" in r and pd.notna(r.rmse_s1_keV):
                        rows.append(dict(separation="chronological", seed=r.seed,
                                         split=split, model="LSMF", rmse_keV=r.rmse_s1_keV))
    # LSMF is identical in both runs (same seed -> same split); keep one per seed
    return pd.DataFrame(rows).drop_duplicates(["seed", "split", "model"])


def improvement(df):
    base = df[df.model == "LSMF"].set_index(["separation", "seed", "split"]).rmse_keV
    out = df[df.model.isin(MODELS)].copy()
    out["eta"] = out.rmse_keV / base.reindex(
        pd.MultiIndex.from_frame(out[["separation", "seed", "split"]])).to_numpy()
    return (out.dropna(subset=["eta"])
               .groupby(["separation", "split", "model"]).eta
               .agg(["mean", "std", "count"]).reset_index())


def main(args):
    SPLITS[2] = ("heavy", f"Heavy ($Z>{args.z_cut}$)", "tab:green")
    frames = []
    if args.chrono:
        frames.append(chrono_rows(args.chrono, args.chrono_model))
    for folder in args.extrap:
        frames.append(pd.read_csv(Path(folder) / "metrics.csv")
                      [["separation", "seed", "split", "model", "rmse_keV"]])
    eta = improvement(pd.concat(frames, ignore_index=True))
    eta.to_csv(args.out.with_suffix(".csv"), index=False)
    print(eta.round(3).to_string(index=False))

    splits = [s for s in SPLITS if s[0] in set(eta.separation)]
    width = 0.8 / len(splits)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(6.5 * len(MODELS), 4.8), sharey=True)
    for ax, model in zip(np.atleast_1d(axes), MODELS):
        for j, (key, label, color) in enumerate(splits):
            for i, (split, _) in enumerate(GROUPS):
                r = eta[(eta.separation == key) & (eta.split == split) & (eta.model == model)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.bar(i + (j - (len(splits) - 1) / 2) * width, r["mean"], width,
                       yerr=r["std"] if r["count"] > 1 else None, capsize=4,
                       color=color, label=label if i == 0 else None)
        ax.axhline(1.0, color="black", ls="--", lw=1)
        ax.set_xticks(range(len(GROUPS)), [g[1] for g in GROUPS], fontsize=16)
        ax.set_title(model.replace("+", " + "), fontsize=16)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(alpha=0.3, axis="y")
    np.atleast_1d(axes)[0].set_ylabel(r"Improvement index $\eta$", fontsize=18)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(splits), fontsize=14,
               frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"\nwrote {args.out} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrono", help="results/ folder of the chronological runs")
    ap.add_argument("--chrono-model", default="AnchoredFullModel")
    ap.add_argument("--extrap", nargs="*", default=[],
                    help="run_extrapolation.py output folders")
    ap.add_argument("--z-cut", type=int, default=80)
    ap.add_argument("--out", type=Path, default=Path("improvement_index.png"))
    main(ap.parse_args())
