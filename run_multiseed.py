#!/usr/bin/env python3
"""
run_multiseed.py
================

Runs the whole train -> test cycle once per seed and aggregates the results into
mean +/- standard deviation. This is where the uncertainty in the reported
numbers comes from: it is the spread over independent trainings, which is what a
referee expects when a table says "0.42 +/- 0.05 MeV".

   

Each seed is launched as a SEPARATE PROCESS. That is deliberate: it guarantees no
state leaks between runs (RNG, cached tensors, torch global settings), so the
seed-to-seed spread measures real training variability and nothing else.

What varies with the seed
-------------------------
  * network initialisation
  * batch ordering
  * the stratified train/validation split

The chronological test set never changes -- it is fixed by the AME2016/AME2020
chronology, not by the seed. So the spread on the test numbers is pure model
variability, which is exactly what should be quoted.

Outputs
-------
    results/summary_val_<model>.csv        mean +/- std, validation
    results/summary_test_<model>.csv       mean +/- std, per regime
    results/summary_regions_<model>.csv    mean +/- std, per region
    results/latex_tables_<model>.txt       ready to paste into the paper
    plots/<model>_seed_spread.png
"""

import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGION_ORDER = ["very light", "light", "medium", "heavy", "heavy exotic"]


def banner(text, ch="="):
    print("\n" + ch * 78)
    print(text)
    print(ch * 78)


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-4000:])
        print(res.stderr[-4000:])
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return res.stdout


def mean_std(df, group_cols, value_cols):
    g = df.groupby(group_cols)
    rows = []
    for key, sub in g:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["n_seeds"] = len(sub)
        for c in value_cols:
            row[f"{c}_mean"] = float(sub[c].mean())
            row[f"{c}_std"] = float(sub[c].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main(args):
    t0 = time.time()
    seeds = [int(s) for s in args.seeds.split(",")]
    model = args.model
    py = sys.executable

    banner(f"MULTI-SEED EXPERIMENT   model={model}   seeds={seeds}")
    print(f"   epochs per seed : {args.num_epochs}")
    print(f"   regimes         : {args.regimes}")
    print(f"   each seed runs in its own process (no state carry-over)")

    if not args.skip_runs:
        for i, seed in enumerate(seeds, 1):
            banner(f"SEED {seed}   ({i}/{len(seeds)})", "-")
            run([py, "train.py", "--model", model, "--seed", str(seed),
                 "--num_epochs", str(args.num_epochs),
                 "--batch_size", str(args.batch_size),
                 "--verbose_every", str(args.verbose_every),
                 "--csv_path", args.trainval_csv])
            run([py, "test.py", "--model", model, "--seed", str(seed),
                 "--trainval_csv", args.trainval_csv,
                 "--test_csv", args.test_csv, "--regimes", args.regimes])
            print(f"   seed {seed} complete")

    # ------------------------------------------------------------ aggregate
    banner("AGGREGATING ACROSS SEEDS")

    val = pd.concat([pd.read_csv(p) for p in
                     glob.glob(f"results/metrics_val_{model}_*.csv")], ignore_index=True)
    test = pd.concat([pd.read_csv(p) for p in
                      glob.glob(f"results/metrics_test_{model}_*.csv")], ignore_index=True)
    regs = pd.concat([pd.read_csv(p) for p in
                      glob.glob(f"results/test_region_metrics_{model}_*.csv")],
                     ignore_index=True)

    n_seeds_actual = int(test.seed.nunique())
    metric_cols = ["rmse_keV", "mae_keV", "bias_keV", "median_abs_keV",
                   "within_250keV", "within_500keV"]

    val_s = mean_std(val, ["model"], metric_cols)
    test_cols = metric_cols + [c for c in ("rmse_s1_keV", "rmse_s2_keV") if c in test]
    test_s = mean_std(test, ["regime"], test_cols)
    reg_s = mean_std(regs[regs.regime == args.headline_regime],
                     ["region"], ["rmse_keV", "mae_keV"])
    reg_s["order"] = reg_s.region.apply(
        lambda r: REGION_ORDER.index(r) if r in REGION_ORDER else 99)
    reg_s = reg_s.sort_values("order").drop(columns="order")

    # ---------------------------------------------------------------- print
    print(f"\n   seeds aggregated : {sorted(val.seed.unique().tolist())}")

    print(f"\n   VALIDATION  (AME2016 held-out split)")
    print(f"   {'metric':<22}{'mean':>12}{'std':>10}")
    print("   " + "-" * 44)
    r = val_s.iloc[0]
    for c, lab in [("rmse_keV", "RMSE (keV)"), ("mae_keV", "MAE (keV)"),
                   ("bias_keV", "bias (keV)"), ("median_abs_keV", "median |e| (keV)"),
                   ("within_250keV", "within 250 keV"), ("within_500keV", "within 500 keV")]:
            print(f"   {lab:<22}{r[f'{c}_mean']:>12.2f}{r[f'{c}_std']:>10.2f}")

    print(f"\n   CHRONOLOGICAL TEST  (new AME2020 nuclei), by anchor dictionary")
    print(f"   {'regime':<13}{'RMSE mean':>12}{'RMSE std':>11}{'MAE mean':>11}"
          f"{'MAE std':>10}{'<250keV':>10}")
    print("   " + "-" * 67)
    for _, r in test_s.iterrows():
        print(f"   {r['regime']:<13}{r['rmse_keV_mean']:>12.2f}{r['rmse_keV_std']:>11.2f}"
              f"{r['mae_keV_mean']:>11.2f}{r['mae_keV_std']:>10.2f}"
              f"{r['within_250keV_mean']*100:>9.1f}%")

    if "rmse_s1_keV_mean" in test_s:
        print(f"\n   STAGE DECOMPOSITION  (regime '{args.headline_regime}')")
        h = test_s[test_s.regime == args.headline_regime].iloc[0]
        print(f"   {'S1  LSMF only':<28}{h['rmse_s1_keV_mean']:>10.1f} "
              f"+/- {h['rmse_s1_keV_std']:<8.1f} keV")
        print(f"   {'S1+S2  + local anchor':<28}{h['rmse_s2_keV_mean']:>10.1f} "
              f"+/- {h['rmse_s2_keV_std']:<8.1f} keV")
        print(f"   {'S1+S2+S3  + transformer':<28}{h['rmse_keV_mean']:>10.1f} "
              f"+/- {h['rmse_keV_std']:<8.1f} keV")
        gain = h["rmse_s2_keV_mean"] - h["rmse_keV_mean"]
        print(f"   network contribution        {gain:>10.1f} keV "
              f"({100*gain/max(h['rmse_s2_keV_mean'],1e-9):.1f}%)")

    print(f"\n   REGIONAL  (regime '{args.headline_regime}')")
    print(f"   {'region':<15}{'n_seeds':>9}{'RMSE mean':>12}{'RMSE std':>11}"
          f"{'MAE mean':>11}")
    print("   " + "-" * 58)
    for _, r in reg_s.iterrows():
        print(f"   {r['region']:<15}{int(r['n_seeds']):>9}{r['rmse_keV_mean']:>12.1f}"
              f"{r['rmse_keV_std']:>11.1f}{r['mae_keV_mean']:>11.1f}")

    # ---------------------------------------------------------------- save
    os.makedirs("results", exist_ok=True)
    val_s.to_csv(f"results/summary_val_{model}.csv", index=False)
    test_s.to_csv(f"results/summary_test_{model}.csv", index=False)
    reg_s.to_csv(f"results/summary_regions_{model}.csv", index=False)

    # ---------------------------------------------------------- latex tables
    lines = []
    lines.append(r"% ---- Table: performance by anchor dictionary ----")
    lines.append(r"\begin{table}[htbp]\centering")
    lines.append(r"\caption{Performance on the %d nuclei first measured in AME2020, "
                 r"as a function of the neighbour masses the local anchor is permitted "
                 r"to use. Values are mean $\pm$ standard deviation over %d independent "
                 r"trainings. The \emph{historical} regime uses only masses known in "
                 r"AME2016 and is the prospective result; \emph{leave-one-out} allows "
                 r"test nuclei to support one another and is reported for reference "
                 r"only.}" % (int(test.n.iloc[0]), n_seeds_actual))
    lines.append(r"\label{tab:regimes}")
    lines.append(r"\begin{tabular}{lccc}\hline\hline")
    lines.append(r"Neighbour dictionary & RMSE (keV) & MAE (keV) & within 250 keV \\ \hline")
    pretty = {"historical": "Historical (AME2016 only)", "train": "Training split only",
              "trainval": "Training + validation", "loo": "Leave-one-out"}
    for _, r in test_s.iterrows():
        lines.append(f"{pretty.get(r['regime'], r['regime'])} & "
                     f"${r['rmse_keV_mean']:.0f} \\pm {r['rmse_keV_std']:.0f}$ & "
                     f"${r['mae_keV_mean']:.0f} \\pm {r['mae_keV_std']:.0f}$ & "
                     f"${r['within_250keV_mean']*100:.0f}\\%$ \\\\")
    lines.append(r"\hline\hline\end{tabular}\end{table}")
    lines.append("")

    lines.append(r"% ---- Table: regional performance ----")
    lines.append(r"\begin{table}[htbp]\centering")
    lines.append(r"\caption{Regional RMSE on the chronological test set "
                 r"(historical anchor dictionary), mean $\pm$ standard deviation "
                 r"over %d trainings.}" % n_seeds_actual)
    lines.append(r"\label{tab:regions}")
    lines.append(r"\begin{tabular}{lcc}\hline\hline")
    lines.append(r"Region & RMSE (keV) & MAE (keV) \\ \hline")
    for _, r in reg_s.iterrows():
        lines.append(f"{r['region'].title()} & "
                     f"${r['rmse_keV_mean']:.0f} \\pm {r['rmse_keV_std']:.0f}$ & "
                     f"${r['mae_keV_mean']:.0f} \\pm {r['mae_keV_std']:.0f}$ \\\\")
    lines.append(r"\hline\hline\end{tabular}\end{table}")

    tex_path = f"results/latex_tables_{model}.txt"
    with open(tex_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\n   saved {tex_path}")

    # ---------------------------------------------------------------- plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for regime in test.regime.unique():
        sub = test[test.regime == regime].sort_values("seed")
        ax.plot(sub.seed.astype(str), sub.rmse_keV, "o-", label=regime)
    ax.set_xlabel("seed"); ax.set_ylabel("test RMSE (keV)")
    ax.set_title("Seed-to-seed spread by anchor dictionary")
    ax.legend(); ax.grid(alpha=.25)

    ax = axes[1]
    labels = test_s.regime.tolist()
    means = test_s.rmse_keV_mean.to_numpy()
    stds = test_s.rmse_keV_std.to_numpy()
    ax.bar(labels, means, yerr=stds, capsize=5, color="#1d4ed8", alpha=.85)
    ax.set_ylabel("test RMSE (keV)")
    ax.set_title(f"mean $\\pm$ std over {n_seeds_actual} seeds")
    ax.grid(alpha=.25, axis="y")
    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    p = f"plots/{model}_seed_spread.png"
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"   saved {p}")

    banner(f"MULTI-SEED EXPERIMENT COMPLETE   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CoreModel")
    ap.add_argument("--seeds", default="12,17,33,42,89")
    ap.add_argument("--num_epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--verbose_every", type=int, default=5)
    ap.add_argument("--trainval_csv", default="data/trainval.csv")
    ap.add_argument("--test_csv", default="data/test.csv")
    ap.add_argument("--regimes", default="historical,train,loo")
    ap.add_argument("--headline_regime", default="historical")
    ap.add_argument("--skip_runs", action="store_true",
                    help="only aggregate results already on disk")
    main(ap.parse_args())
