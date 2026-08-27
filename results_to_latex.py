#!/usr/bin/env python3
"""Aggregate nuclear-mass model metrics over seeds and create LaTeX tables.

Examples
--------
python aggregate_results_to_latex.py results
python aggregate_results_to_latex.py results --output-dir results/paper_tables

The script reads only raw metric files (not existing ``summary_*.csv`` files):
  * metrics_val_*.csv
  * metrics_test_*.csv
  * region_metrics_*.csv
  * test_region_metrics_*.csv

Means and sample standard deviations are calculated across seeds separately for
each model and, where applicable, each test regime and nuclear-mass region.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ID_COLUMNS = {
    "model", "seed", "split", "regime", "region", "use_anchor", "n", "count"
}

PRETTY = {
    "n": r"$N_{\mathrm{nuc}}$",
    "count": r"$N_{\mathrm{nuc}}$",
    "rmse_keV": "RMSE (keV)",
    "mae_keV": "MAE (keV)",
    "bias_keV": "Bias (keV)",
    "median_abs_keV": "Median $|e|$ (keV)",
    "p90_abs_keV": "$P_{90}(|e|)$ (keV)",
    "max_abs_keV": "Max. $|e|$ (keV)",
    "within_100keV": "$f_{100}$",
    "within_250keV": "$f_{250}$",
    "within_500keV": "$f_{500}$",
    "rmse_s1_keV": "Stage-1 RMSE (keV)",
    "rmse_s2_keV": "Stage-2 RMSE (keV)",
}

MODEL_ORDER = [
    "CoreModel", "ShellModel", "ZEOModel", "MagicModel",
    "LiquidDropModel", "ValenceModel", "AnchoredFullModel",
]
REGIME_ORDER = ["historical", "train", "loo"]
REGION_ORDER = ["very light", "light", "medium", "heavy", "heavy exotic"]


def read_family(results_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(results_dir.rglob(pattern))
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"WARNING: skipping {path}: {exc}")
            continue
        if frame.empty:
            print(f"WARNING: skipping empty file {path}")
            continue
        frame["source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def numeric_metrics(df: pd.DataFrame) -> list[str]:
    candidates = [c for c in df.columns if c not in ID_COLUMNS | {"source_file"}]
    return [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing grouping columns: {missing}")

    metrics = numeric_metrics(df)
    grouped = df.groupby(group_cols, dropna=False, sort=False)
    pieces = [grouped.size().rename("n_seeds")]

    # n/count describe dataset size and are expected to be invariant by seed.
    for size_col in ("n", "count"):
        if size_col in df.columns:
            pieces.append(grouped[size_col].first().rename(size_col))

    for metric in metrics:
        pieces.append(grouped[metric].mean().rename(f"{metric}_mean"))
        pieces.append(grouped[metric].std(ddof=1).fillna(0.0).rename(f"{metric}_std"))
    return pd.concat(pieces, axis=1).reset_index()


def fmt_value(mean: float, std: float, metric: str) -> str:
    if pd.isna(mean):
        return "--"
    if metric.startswith("within_"):
        return f"{100 * mean:.1f} $\\pm$ {100 * std:.1f}"
    return f"{mean:.1f} $\\pm$ {std:.1f}"


def latex_escape(value: object) -> str:
    """Escape ordinary text placed in a LaTeX table cell."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    orders = {"model": MODEL_ORDER, "regime": REGIME_ORDER, "region": REGION_ORDER}
    for col, order in orders.items():
        if col in out.columns:
            extras = [x for x in out[col].dropna().unique() if x not in order]
            out[col] = pd.Categorical(out[col], categories=order + sorted(extras), ordered=True)
    sort_cols = [c for c in ("model", "regime", "region") if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out


def latex_table(
    summary: pd.DataFrame,
    group_cols: list[str],
    caption: str,
    label: str,
    selected_metrics: Iterable[str],
) -> str:
    if summary.empty:
        return f"% No data found for {label}.\n"

    selected = [m for m in selected_metrics if f"{m}_mean" in summary.columns]
    rows = []
    for _, row in ordered(summary).iterrows():
        item = {c.title(): latex_escape(row[c]) for c in group_cols}
        if "n" in summary.columns:
            item[PRETTY["n"]] = str(int(row["n"]))
        elif "count" in summary.columns:
            item[PRETTY["count"]] = str(int(row["count"]))
        item["Seeds"] = str(int(row["n_seeds"]))
        for metric in selected:
            item[PRETTY.get(metric, metric)] = fmt_value(
                row[f"{metric}_mean"], row[f"{metric}_std"], metric
            )
        rows.append(item)

    display = pd.DataFrame(rows)
    column_format = "l" * len(group_cols) + "r" * (len(display.columns) - len(group_cols))
    header = " & ".join(display.columns) + r" \\" + "\n"
    body = "".join(
        " & ".join(str(value) for value in record) + r" \\" + "\n"
        for record in display.itertuples(index=False, name=None)
    )
    tabular = (
        f"\\begin{{tabular}}{{{column_format}}}\n"
        "\\toprule\n" + header + "\\midrule\n" + body +
        "\\bottomrule\n\\end{tabular}\n"
    )
    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        f"{tabular}"
        f"\\caption{{{caption} Values are the mean $\\pm$ sample standard deviation across random seeds.}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table*}\n"
    )


def save_family(
    df: pd.DataFrame,
    groups: list[str],
    stem: str,
    caption: str,
    label: str,
    metrics: list[str],
    output_dir: Path,
) -> None:
    if df.empty:
        print(f"WARNING: no files found for {stem}")
        return
    summary = aggregate(df, groups)
    summary = ordered(summary)
    summary.to_csv(output_dir / f"{stem}_mean_std.csv", index=False)
    (output_dir / f"{stem}.tex").write_text(
        latex_table(summary, groups, caption, label, metrics), encoding="utf-8"
    )
    print(f"Created {stem}_mean_std.csv and {stem}.tex ({len(df)} input rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Directory containing result CSV files")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: RESULTS_DIR/paper_tables)")
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    output_dir = (args.output_dir or results_dir / "paper_tables").expanduser().resolve()
    if not results_dir.is_dir():
        parser.error(f"Results directory does not exist: {results_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    common = ["rmse_keV", "mae_keV", "bias_keV", "median_abs_keV",
              "p90_abs_keV", "max_abs_keV", "within_100keV",
              "within_250keV", "within_500keV"]

    # Exact patterns avoid accidentally ingesting predictions or old summaries.
    val = read_family(results_dir, "metrics_val_*.csv")
    test = read_family(results_dir, "metrics_test_*.csv")
    val_regions = read_family(results_dir, "region_metrics_*.csv")
    # Remove test_region_metrics files also matched by **/region_metrics_*.csv
    if not val_regions.empty:
        val_regions = val_regions[~val_regions["source_file"].map(
            lambda p: Path(p).name.startswith("test_region_metrics_")
        )].copy()
    test_regions = read_family(results_dir, "test_region_metrics_*.csv")

    save_family(val, ["model"], "validation_metrics",
                "Validation performance of the investigated models.",
                "tab:validation_metrics", common, output_dir)
    save_family(test, ["model", "regime"], "test_metrics",
                "Independent-test performance under each anchoring regime.",
                "tab:test_metrics", common + ["rmse_s1_keV", "rmse_s2_keV"], output_dir)
    save_family(val_regions, ["model", "region"], "validation_region_metrics",
                "Validation performance by nuclear-mass region.",
                "tab:validation_region_metrics", ["rmse_keV", "mae_keV", "bias_keV", "max_abs_keV"], output_dir)
    save_family(test_regions, ["model", "regime", "region"], "test_region_metrics",
                "Independent-test performance by anchoring regime and nuclear-mass region.",
                "tab:test_region_metrics", ["rmse_keV", "mae_keV", "bias_keV", "max_abs_keV"], output_dir)

    master = "\n\n".join(
        f"\\input{{{name}}}" for name in (
            "validation_metrics.tex", "test_metrics.tex",
            "validation_region_metrics.tex", "test_region_metrics.tex"
        ) if (output_dir / name).exists()
    ) + "\n"
    (output_dir / "all_tables.tex").write_text(master, encoding="utf-8")
    print(f"All outputs written to {output_dir}")


if __name__ == "__main__":
    main()