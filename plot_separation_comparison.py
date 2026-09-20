#!/usr/bin/env python3
"""
Publication-quality comparison of Validation vs. Test RMSE (keV)
across three extrapolation separations and four model variants.

Layout  : 1 × 3 panels (one per separation)
X-axis  : four model variants
Bars    : two bars per model — Validation (solid) and Test (hatched)
Error   : ±1 σ over 5 random seeds

Usage:
    python plot_separation_comparison.py [--metric rmse|mae]
                                         [--out plots/comparison_separations.pdf]
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
from matplotlib.ticker import MaxNLocator

# ── global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          11,
    "axes.linewidth":     0.9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "legend.frameon":     True,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#bbbbbb",
    "figure.dpi":         150,
})

# ── model labels and colours (colorblind-friendly, Tableau-derived) ────────────
MODEL_KEYS = [
    "LSMF",
    "LSMF+anchor",
    "LSMF+Transformer",
    "LSMF+anchor+Transformer",
]
MODEL_LABELS = [
    "LSMF",
    "LSMF\n+anchor",
    "LSMF\n+Transformer",
    "LSMF\n+anchor\n+Transformer",
]
# one colour per model; val = solid fill, test = same colour + hatching
MODEL_COLORS = ["#4878CF", "#D55E00", "#009E73", "#CC79A7"]  # blue, vermillion, teal, pink

BAR_W   = 0.32   # width of each individual bar
GROUP_W = 0.78   # total group footprint
X_POS   = np.arange(len(MODEL_KEYS))  # group centres


# ── data loaders ──────────────────────────────────────────────────────────────

def load_sep1(metric: str) -> tuple[dict, dict]:
    """Standard AME2016→AME2020.  Returns (val_dict, test_dict)."""
    base = Path("results2")
    anch_v  = pd.read_csv(base / "summary_val_AnchoredFullModel.csv").iloc[0]
    nach_v  = pd.read_csv(base / "summary_val_AnchoredFullModel_noanchor.csv").iloc[0]
    anch_t  = pd.read_csv(base / "summary_test_AnchoredFullModel.csv")
    nach_t  = pd.read_csv(base / "summary_test_AnchoredFullModel_noanchor.csv")
    anch_t  = anch_t[anch_t["regime"] == "historical"].iloc[0]
    nach_t  = nach_t[nach_t["regime"] == "historical"].iloc[0]

    def row2dict(anch, nach, col, col_s, s1, s1s, s2, s2s):
        return {
            "LSMF":                    (float(anch[s1]),  float(anch[s1s])),
            "LSMF+anchor":             (float(anch[s2]),  float(anch[s2s])),
            "LSMF+Transformer":        (float(nach[col]), float(nach[col_s])),
            "LSMF+anchor+Transformer": (float(anch[col]), float(anch[col_s])),
        }

    m = metric
    val  = row2dict(anch_v, nach_v,
                    f"{m}_keV_mean", f"{m}_keV_std",
                    f"{m}_s1_keV_mean", f"{m}_s1_keV_std",
                    f"{m}_s2_keV_mean", f"{m}_s2_keV_std")
    test = row2dict(anch_t, nach_t,
                    f"{m}_keV_mean", f"{m}_keV_std",
                    f"{m}_s1_keV_mean", f"{m}_s1_keV_std",
                    f"{m}_s2_keV_mean", f"{m}_s2_keV_std")
    return val, test


def load_nrich(metric: str) -> tuple[dict, dict]:
    df = pd.read_csv("results_nrich2/summary.csv")
    def _extract(split):
        sub = df[df["split"] == split]
        return {k: (float(sub[sub["model"] == k][f"{metric}_mean"].values[0]),
                    float(sub[sub["model"] == k][f"{metric}_std"].values[0]))
                for k in MODEL_KEYS}
    return _extract("val"), _extract("test")


def load_heavy(metric: str) -> tuple[dict, dict]:
    df  = pd.read_csv("results_heavy/metrics.csv")
    col = f"{metric}_keV"
    def _extract(split):
        sub = df[df["split"] == split]
        return {k: (float(sub[sub["model"] == k][col].mean()),
                    float(sub[sub["model"] == k][col].std(ddof=1)))
                for k in MODEL_KEYS}
    return _extract("val"), _extract("test")


# ── plotting ──────────────────────────────────────────────────────────────────

def _bar_pair(ax, x_centre, val_mean, val_std, test_mean, test_std, color):
    """Draw one val+test pair at position x_centre."""
    offset = BAR_W / 2 + 0.02
    ekw = dict(elinewidth=1.0, capthick=1.0, capsize=3.5, ecolor="#333333")

    # Validation — solid fill
    ax.bar(x_centre - offset, val_mean, BAR_W,
           yerr=val_std, color=color, edgecolor="#222222", linewidth=0.7,
           error_kw=ekw, zorder=3)

    # Test — hatched fill (same colour, lighter face)
    ax.bar(x_centre + offset, test_mean, BAR_W,
           yerr=test_std,
           color=color, alpha=0.45, hatch="////",
           edgecolor=color, linewidth=0.7,
           error_kw=ekw, zorder=3)


def _label_bar(ax, x, mean_mev, std_mev, max_val):
    """Place a compact numeric label above a bar (values already in MeV)."""
    ypos = mean_mev + std_mev + 0.018 * max_val
    ax.text(x, ypos, f"{mean_mev:.2f}", ha="center", va="bottom",
            fontsize=7.5, color="#222222")


def draw_panel(ax, val_data: dict, test_data: dict, title: str, metric: str,
               show_ylabel: bool) -> None:
    # convert keV → MeV
    val_mev  = {k: (v[0] / 1000, v[1] / 1000) for k, v in val_data.items()}
    test_mev = {k: (v[0] / 1000, v[1] / 1000) for k, v in test_data.items()}

    all_tops = [v[0] + v[1] for v in list(val_mev.values()) + list(test_mev.values())]
    max_val  = max(v[0] for v in list(val_mev.values()) + list(test_mev.values()))
    offset   = BAR_W / 2 + 0.02

    for i, (key, color) in enumerate(zip(MODEL_KEYS, MODEL_COLORS)):
        vm, vs   = val_mev[key]
        tm, ts   = test_mev[key]
        _bar_pair(ax, X_POS[i], vm, vs, tm, ts, color)
        _label_bar(ax, X_POS[i] - offset, vm, vs, max_val)
        _label_bar(ax, X_POS[i] + offset, tm, ts, max_val)

    ax.set_xticks(X_POS)
    ax.set_xticklabels(MODEL_LABELS, fontsize=9.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="y", labelsize=9.5)
    ax.set_ylim(0, max(all_tops) * 1.25)
    ax.yaxis.grid(True, linewidth=0.5, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    if show_ylabel:
        label = "RMSE (MeV)" if metric == "rmse" else "MAE (MeV)"
        ax.set_ylabel(label, fontsize=11)

    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=6)


def build_legend(fig, axs):
    # Model legend (colours)
    model_patches = [mpatches.Patch(facecolor=c, edgecolor="#222222", linewidth=0.7,
                                    label=l.replace("\n", " "))
                     for c, l in zip(MODEL_COLORS,
                                     ["LSMF", "LSMF+anchor",
                                      "LSMF+Transformer", "LSMF+anchor+Transformer"])]
    # Val / Test legend (pattern)
    val_patch  = mpatches.Patch(facecolor="#999999", edgecolor="#222222",
                                linewidth=0.7, label="Validation")
    test_patch = mpatches.Patch(facecolor="#999999", alpha=0.45, hatch="////",
                                edgecolor="#999999", linewidth=0.7, label="Test")

    # Split legend → Separation 1 (upper right, plenty of space there)
    axs[0].legend(handles=[val_patch, test_patch],
                  loc="upper right", fontsize=8.5,
                  title="Split", title_fontsize=8.5,
                  handlelength=1.6, handleheight=1.0, borderpad=0.7)

    # Model legend → Separation 2 (upper right)
    axs[1].legend(handles=model_patches, loc="upper right",
                  fontsize=8.5, title="Model", title_fontsize=8.5,
                  handlelength=1.4, handleheight=1.0, borderpad=0.7)


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    metric = args.metric

    val1, test1 = load_sep1(metric)
    val2, test2 = load_nrich(metric)
    val3, test3 = load_heavy(metric)

    panels = [
        (val1, test1, "Separation 1\n(AME2016 -> AME2020)"),
        (val2, test2, "Separation 2\n(Neutron-Rich)"),
        (val3, test3, "Separation 3\n(Heavy Nuclei, Z > 80)"),
    ]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), sharey=False)
    fig.subplots_adjust(wspace=0.34)

    for idx, (ax, (val, test, title)) in enumerate(zip(axs, panels)):
        draw_panel(ax, val, test, title, metric, show_ylabel=(idx == 0))

    build_legend(fig, axs)

    metric_str = "RMSE" if metric == "rmse" else "MAE"
    fig.suptitle(
        f"Validation vs. Test {metric_str} (MeV) across Extrapolation Scenarios\n"
        r"Mean $\pm$ 1$\sigma$ over 5 random seeds",
        fontsize=12.5, fontweight="bold", y=1.03,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="rmse", choices=["rmse", "mae"])
    ap.add_argument("--out",    default="plots/comparison_separations.png")
    main(ap.parse_args())
