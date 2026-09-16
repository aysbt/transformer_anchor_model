#!/usr/bin/env python3
"""Aggregate and plot feature-level attention scores from multiple seeds.

Expected input filenames:
    attention_importance_AnchoredFullModel_12.csv
    attention_importance_AnchoredFullModel_17.csv
    attention_importance_AnchoredFullModel_33.csv
    attention_importance_AnchoredFullModel_42.csv
    attention_importance_AnchoredFullModel_89.csv

Each CSV must contain the columns ``feature`` and ``attention_score``.
The script saves:
    attention_importance_AnchoredFullModel_all_seeds.csv
    attention_importance_AnchoredFullModel_all_seeds.png
    attention_importance_AnchoredFullModel_all_seeds.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = (12, 17, 33, 42, 89)

# Publication labels. Symbols must match the manuscript (Sec. 2.2 and Eq. 5, 16).
# Unmapped feature names are displayed unchanged.
FEATURE_LABELS = {
    "N": r"$N$",
    "Z": r"$Z$",
    "A": r"$A$",
    "deltaN": r"$\Delta N$",
    "deltaZ": r"$\Delta Z$",
    "NEO": r"$NEO}$",
    "ZEO": r"$ZEO}$",
    "Nshell_category": r"$S_N$",
    "Zshell_category": r"$S_Z$",
    "A^2/3": r"$A^{2/3}$",
    "Z(Z-1)/A^1/3": r"$Z(Z-1)/A^{1/3}$",
    "(N-Z)^2/A": r"$(N-Z)^2/A$",
    "(N-Z)/A": r"$(N-Z)/A$",
    "N/Z": r"$N/Z$",
    "promiscuity": r"$P$",
    "neighbor_N_minus_1_exists": r"$I_{N-1}$",
    "neighbor_N_plus_1_exists": r"$I_{N+1}$",
    "neighbor_Z_minus_1_exists": r"$I_{Z-1}$",
    "neighbor_Z_plus_1_exists": r"$I_{Z+1}$",
    "anchor": r"$a_{\mathrm{anc}}$",
    "anchor_has": r"$I_{\mathrm{anc}}$",
    "anchor_n": r"$n_{\mathrm{anc}}$",
    "anchor_mean_dist": r"$\overline{d}_{\mathrm{anc}}$",
    "anchor_max_dist": r"$d_{\mathrm{anc}}^{\max}$",
    "anchor_scatter": r"$\sigma_{\mathrm{anc}}$",
    "anchor_parity_match": r"$f_{\mathrm{anc}}^{\mathrm{parity}}$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the mean and seed-to-seed standard deviation of "
            "attention scores and create a horizontal bar plot."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing the five attention CSV files (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("attention_plot"),
        help="Output directory (default: the input directory).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=1200,
        help="PNG resolution in dots per inch (default: 1200).",
    )
    return parser.parse_args()


def load_seed_file(path: Path, seed: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing seed file: {path}")

    frame = pd.read_csv(path)
    required = {"feature", "attention_score"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    frame = frame.loc[:, ["feature", "attention_score"]].copy()
    frame["feature"] = frame["feature"].astype(str).str.strip()
    frame["attention_score"] = pd.to_numeric(
        frame["attention_score"], errors="raise"
    )

    duplicated = frame.loc[frame["feature"].duplicated(), "feature"].tolist()
    if duplicated:
        raise ValueError(
            f"{path.name} contains duplicate features: {sorted(set(duplicated))}"
        )
    if frame["attention_score"].isna().any():
        raise ValueError(f"{path.name} contains missing attention scores")

    return frame.rename(columns={"attention_score": f"seed_{seed}"})


def aggregate_attention(input_dir: Path) -> pd.DataFrame:
    seed_frames = []
    reference_features: set[str] | None = None

    for seed in SEEDS:
        filename = f"attention_importance_AnchoredFullModel_{seed}.csv"
        frame = load_seed_file(input_dir / filename, seed)
        current_features = set(frame["feature"])

        if reference_features is None:
            reference_features = current_features
        elif current_features != reference_features:
            missing = sorted(reference_features - current_features)
            extra = sorted(current_features - reference_features)
            raise ValueError(
                f"Feature mismatch in {filename}. Missing: {missing}; extra: {extra}"
            )

        seed_frames.append(frame)

    combined = seed_frames[0]
    for frame in seed_frames[1:]:
        combined = combined.merge(frame, on="feature", how="inner", validate="one_to_one")

    score_columns = [f"seed_{seed}" for seed in SEEDS]
    combined["mean_attention"] = combined[score_columns].mean(axis=1)
    combined["std_attention"] = combined[score_columns].std(axis=1, ddof=1)
    combined["sem_attention"] = combined["std_attention"] / np.sqrt(len(SEEDS))
    combined["min_attention"] = combined[score_columns].min(axis=1)
    combined["max_attention"] = combined[score_columns].max(axis=1)
    combined = combined.sort_values("mean_attention", ascending=False).reset_index(drop=True)
    combined.insert(1, "plot_label", combined["feature"].map(FEATURE_LABELS).fillna(combined["feature"]))
    combined.insert(2, "rank", np.arange(1, len(combined) + 1))
    return combined


def plot_attention(summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    # Descending means are shown from top to bottom.
    y = np.arange(len(summary))
    means = summary["mean_attention"].to_numpy()
    stds = summary["std_attention"].to_numpy()

    fig_height = max(8.0, 0.38 * len(summary))
    fig, ax = plt.subplots(figsize=(9.0, fig_height))

    ax.barh(
        y,
        means,
        xerr=stds,
        height=0.68,
        color="#3F6F8F",
        edgecolor="black",
        linewidth=0.45,
        error_kw={
            "ecolor": "#252525",
            "elinewidth": 1.0,
            "capsize": 2.5,
            "capthick": 1.0,
        },
    )

    ax.set_yticks(y)
    ax.set_yticklabels(summary["plot_label"], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Mean attention score", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    # No plot title: the manuscript caption should describe the figure.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.995,
        0.015,
        r"Error bars: $\pm 1$ SD across five seeds",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#333333",
    )

    fig.tight_layout()
    stem = output_dir / "attention_importance_AnchoredFullModel_all_seeds"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or args.input_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = aggregate_attention(input_dir)
    score_columns = [f"seed_{seed}" for seed in SEEDS]

    output_csv = output_dir / "attention_importance_AnchoredFullModel_all_seeds.csv"
    summary.to_csv(output_csv, index=False, float_format="%.10f")
    plot_attention(summary, output_dir, args.dpi)

    print("Attention-score sums by seed:")
    for column in score_columns:
        print(f"  {column}: {summary[column].sum():.8f}")
    print(f"  mean:   {summary['mean_attention'].sum():.8f}")
    print(f"\nSaved summary: {output_csv}")
    print(
        "Saved figure:  "
        f"{output_dir / 'attention_importance_AnchoredFullModel_all_seeds.png'}"
    )
    print(
        "Saved vector:  "
        f"{output_dir / 'attention_importance_AnchoredFullModel_all_seeds.pdf'}"
    )


if __name__ == "__main__":
    main()