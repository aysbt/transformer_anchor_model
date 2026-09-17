#!/usr/bin/env python3
"""
Theory-table figures in the style of Lovell et al., PRC 106, 014305 (2022).

  rmse   RMSE of LSMF and LSMF+Transformer in train / validation / test, one
         panel per theory table, one group per separation, dots = seeds
         (their Fig. 3 shows the same quantity as a distribution over runs).
  chains predicted minus theory mass excess along isotopic chains
         (their Fig. 2 chains: Fe, Mo, Sn, W, Pb, Sg).

    python plot_theory.py --runs FRDM95=results_frdm95 HFB-14=results_hfb14
    python plot_theory.py --runs FRDM95=results_frdm95 --separation heavy --seed 42
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = "Mass Excess (keV)"
SEPARATIONS = [("random", "Random"), ("neutron_rich", "Neutron-rich"),
               ("heavy", "Heavy"), ("unmeasured", "Measured → unmeasured")]
SPLITS = [("train", "Train", "tab:blue"), ("val", "Validation", "tab:orange"),
          ("test", "Test", "tab:green")]
CHAINS = [(26, "Fe"), (42, "Mo"), (50, "Sn"), (74, "W"), (82, "Pb"), (106, "Sg")]


def plot_rmse(runs, out):
    fig, axes = plt.subplots(1, len(runs), figsize=(7.5 * len(runs), 5.2),
                             sharey=True, squeeze=False)
    for ax, (table, folder) in zip(axes[0], runs.items()):
        m = pd.read_csv(Path(folder) / "metrics.csv")
        seps = [s for s in SEPARATIONS if s[0] in set(m.separation)]
        for i, (sep, _) in enumerate(seps):
            net = m[(m.separation == sep) & (m.model == "LSMF+Transformer")]
            for j, (split, label, color) in enumerate(SPLITS):
                x = i + (j - 1) * 0.26
                v = net[net.split == split].rmse_keV / 1000
                if v.empty:
                    continue
                ax.bar(x, v.mean(), 0.24, color=color, alpha=0.75,
                       label=label if i == 0 else None)
                ax.scatter(np.full(len(v), x), v, s=14, color="black", zorder=3)
            # LSMF test RMSE as reference line over the test bar
            base = m[(m.separation == sep) & (m.model == "LSMF") & (m.split == "test")]
            ax.hlines(base.rmse_keV.mean() / 1000, i + 0.13, i + 0.39, color="black",
                      lw=2.5, label="LSMF, test" if i == 0 else None)
        ax.set_xticks(range(len(seps)), [s[1] for s in seps], fontsize=14)
        ax.set_yscale("log")
        ax.set_title(table, fontsize=18)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(alpha=0.3, axis="y", which="both")
    axes[0, 0].set_ylabel(r"$\sigma_{\rm RMS}$ (MeV)", fontsize=18)
    axes[0, 0].legend(fontsize=13, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_chains(table, folder, separation, seed, out):
    run = Path(folder) / separation / f"seed_{seed}" / "noanchor"
    df = pd.concat([pd.read_csv(run / f"{s}_predictions.csv").assign(split=s)
                    for s, _, _ in SPLITS])
    chains = [c for c in CHAINS if c[0] in set(df.Z)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False)
    for ax, (z, symbol) in zip(axes.flat, chains):
        c = df[df.Z == z].sort_values("N")
        ax.axhline(0, color="gray", lw=1)
        ax.plot(c.N, (c.static_keV - c[TARGET]) / 1000, color="gray", lw=1.5,
                label="LSMF")
        ax.plot(c.N, (c.predicted_keV - c[TARGET]) / 1000, color="tab:red", lw=1.5,
                label="LSMF + Transformer")
        for split, label, color in SPLITS:
            p = c[c.split == split]
            ax.scatter(p.N, (p.predicted_keV - p[TARGET]) / 1000, s=14, color=color,
                       zorder=3, label=label)
        ax.set_title(f"{symbol} (Z = {z})", fontsize=15)
        ax.set_xlabel("Neutron number N", fontsize=13)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(chains):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\delta M_{\rm pred} - \delta M_{\rm " + table + "}$ (MeV)",
                      fontsize=13)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=13,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main(args):
    runs = dict(r.split("=", 1) for r in args.runs)
    SEPARATIONS[2] = ("heavy", f"Heavy ($Z>{args.z_cut}$)")
    plot_rmse(runs, Path(args.outdir) / "theory_rmse.png")
    for table, folder in runs.items():
        seps = [args.separation] if args.separation else \
               [s for s, _ in SEPARATIONS if (Path(folder) / s).is_dir()]
        for sep in seps:
            plot_chains(table, folder, sep, args.seed,
                        Path(args.outdir) / f"theory_chains_{table}_{sep}_s{args.seed}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="NAME=run_extrapolation output folder")
    ap.add_argument("--separation", help="chain plots for one separation only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--z-cut", type=int, default=80)
    main(ap.parse_args())
