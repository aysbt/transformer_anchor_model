"""
evaluate.py
===========

Everything here works in PHYSICAL keV. Predictions come out of the network in
standardised anchored-residual space and are converted exactly once, in
`evaluate_model`, via

    mass_excess = residual_std * target_std + target_mean + offset

No metric and no plot is ever computed on a transformed quantity.
"""

import os

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "figure.facecolor": "white", "savefig.bbox": "tight"})

PLOTS = "plots"
RESULTS = "results"


# --------------------------------------------------------------- inference
@torch.no_grad()
def evaluate_model(model, dataloader, device, target_mean, target_std):
    """
    Runs the model and returns everything already in keV.

    Returns
    -------
    N, Z          neutron / proton numbers
    preds_keV     predicted mass excess
    acts_keV      experimental mass excess
    offsets       LSMF + anchor contribution (keV), for stage decomposition
    attn          attention weights of the final batch stack
    """
    model.eval()
    N_all, Z_all, p_all, a_all, off_all, attn_all = [], [], [], [], [], []

    for batch in dataloader:
        cat, cont, target, N, Z, offset = batch
        cat, cont = cat.to(device), cont.to(device)
        preds, attn = model(cat, cont)

        p = preds.cpu().numpy() * target_std + target_mean + offset.numpy()
        a = target.numpy() * target_std + target_mean + offset.numpy()

        p_all.extend(p); a_all.extend(a)
        N_all.extend(N.numpy()); Z_all.extend(Z.numpy())
        off_all.extend(offset.numpy())

        if attn is not None and len(attn) > 0:
            stacked = torch.stack(attn, dim=1)
            for m in stacked:
                attn_all.append(m.cpu().numpy())

    return (np.array(N_all), np.array(Z_all), np.array(p_all),
            np.array(a_all), np.array(off_all), attn_all)


# ----------------------------------------------------------------- metrics
def compute_metrics(preds_keV, acts_keV):
    """All inputs already in keV."""
    p = np.asarray(preds_keV, float)
    a = np.asarray(acts_keV, float)
    e = p - a
    return {
        "n": int(len(e)),
        "rmse_keV": float(np.sqrt(np.mean(e ** 2))),
        "mae_keV": float(np.mean(np.abs(e))),
        "bias_keV": float(np.mean(e)),
        "median_abs_keV": float(np.median(np.abs(e))),
        "p90_abs_keV": float(np.percentile(np.abs(e), 90)),
        "max_abs_keV": float(np.max(np.abs(e))),
        "within_100keV": float(np.mean(np.abs(e) < 100)),
        "within_250keV": float(np.mean(np.abs(e) < 250)),
        "within_500keV": float(np.mean(np.abs(e) < 500)),
    }


def print_metrics(m, title):
    print(f"\n{'-' * 62}")
    print(f" {title}")
    print(f"{'-' * 62}")
    print(f"   nuclei evaluated      : {m['n']}")
    print(f"   RMSE                  : {m['rmse_keV']:10.2f} keV")
    print(f"   MAE                   : {m['mae_keV']:10.2f} keV")
    print(f"   mean signed error     : {m['bias_keV']:+10.2f} keV")
    print(f"   median |error|        : {m['median_abs_keV']:10.2f} keV")
    print(f"   90th pct |error|      : {m['p90_abs_keV']:10.2f} keV")
    print(f"   max |error|           : {m['max_abs_keV']:10.2f} keV")
    print(f"   within 100 keV        : {m['within_100keV']*100:9.1f} %")
    print(f"   within 250 keV        : {m['within_250keV']*100:9.1f} %")
    print(f"   within 500 keV        : {m['within_500keV']*100:9.1f} %")


def fmt_row(m, label, width=30):
    return (f"   {label:<{width}s} n={m['n']:5d}  RMSE={m['rmse_keV']:9.1f}  "
            f"MAE={m['mae_keV']:8.1f}  bias={m['bias_keV']:+8.1f}  "
            f"med={m['median_abs_keV']:7.1f}  <250keV={m['within_250keV']*100:5.1f}%")


# ------------------------------------------------------------- region split
def region_label(N, Z):
    A = N + Z
    if Z > 100 and N > 150:
        return "heavy exotic"
    if A < 20:
        return "very light"
    if A < 60:
        return "light"
    if A <= 200:
        return "medium"
    return "heavy"


REGION_ORDER = ["very light", "light", "medium", "heavy", "heavy exotic"]


def compute_region_metrics(N, Z, preds_keV, acts_keV):
    labels = np.array([region_label(n, z) for n, z in zip(N, Z)])
    rows = []
    for reg in REGION_ORDER:
        sel = labels == reg
        if sel.sum() == 0:
            continue
        m = compute_metrics(preds_keV[sel], acts_keV[sel])
        rows.append({"region": reg, "count": int(sel.sum()),
                     "rmse_keV": m["rmse_keV"], "mae_keV": m["mae_keV"],
                     "bias_keV": m["bias_keV"], "max_abs_keV": m["max_abs_keV"]})
    return pd.DataFrame(rows)


def print_region_table(df, title="Regional performance"):
    print(f"\n{'-' * 78}")
    print(f" {title}")
    print(f"{'-' * 78}")
    print(f"   {'region':<14}{'count':>7}{'RMSE (keV)':>14}{'MAE (keV)':>13}"
          f"{'bias (keV)':>13}{'max |e|':>12}")
    print("   " + "-" * 73)
    for _, r in df.iterrows():
        print(f"   {r['region']:<14}{int(r['count']):>7}{r['rmse_keV']:>14.1f}"
              f"{r['mae_keV']:>13.1f}{r['bias_keV']:>+13.1f}{r['max_abs_keV']:>12.1f}")


# ------------------------------------------------------------------- plots
def _save(fig, name):
    os.makedirs(PLOTS, exist_ok=True)
    path = os.path.join(PLOTS, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"   saved {path}")
    return path


def plot_error_diagnostics(N, Z, preds_keV, acts_keV, model_name, clip=None):
    err = np.asarray(preds_keV) - np.asarray(acts_keV)
    if clip is None:
        clip = max(np.percentile(np.abs(err), 98), 1e-6)

    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1],
                           wspace=0.06, hspace=0.08)
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(N, Z, c=err, cmap="coolwarm", s=18, vmin=-clip, vmax=clip,
                    edgecolors="none")
    ax.set_xlabel("Neutron number $N$"); ax.set_ylabel("Proton number $Z$")
    #ax.set_title(f"{model_name}")
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(r"$\Delta M_{\rm pred}-\Delta M_{\rm exp}$ (keV)")

    axl = fig.add_subplot(gs[0, 0], sharey=ax)
    axl.scatter(err, Z, s=8, alpha=0.45, color="tab:blue")
    axl.axvline(0, color="k", lw=0.8); axl.set_xlabel("error (keV)")
    axl.set_xlim(-clip * 1.6, clip * 1.6); axl.invert_xaxis()

    axb = fig.add_subplot(gs[1, 1], sharex=ax)
    axb.scatter(N, err, s=8, alpha=0.45, color="tab:green")
    axb.axhline(0, color="k", lw=0.8); axb.set_ylabel("error (keV)")
    axb.set_xlabel("Neutron number $N$"); axb.set_ylim(-clip * 1.6, clip * 1.6)
    plt.setp(ax.get_xticklabels(), visible=False)
    fig.add_subplot(gs[1, 0]).axis("off")
    return _save(fig, f"{model_name}_signed_error.png")


def plot_error_distribution(preds_keV, acts_keV, model_name):
    err = np.asarray(preds_keV) - np.asarray(acts_keV)
    lim = np.percentile(np.abs(err), 99)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sns.histplot(err, bins=60, binrange=(-lim, lim), kde=True, color="purple", ax=ax)
    ax.axvline(0, color="crimson", ls="--", lw=1.4, label="zero error")
    ax.set_xlabel(r"$\Delta M_{\rm pred}-\Delta M_{\rm exp}$ (keV)")
    ax.set_ylabel("count"); ax.legend()
    return _save(fig, f"{model_name}_error_distribution.png")


def plot_predictions(N, Z, preds_keV, acts_keV, model_name):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(N, Z, preds_keV, color="gray", alpha=0.55, s=14, label="predicted")
    ax.scatter(N, Z, acts_keV, color="red", alpha=0.25, s=14, label="experimental")
    ax.set_xlabel("Neutron number $N$"); ax.set_ylabel("Proton number $Z$")
    ax.set_zlabel("Mass excess (keV)", labelpad=14)
    #ax.set_title(f"Predicted vs experimental mass excess — {model_name}")
    ax.legend()
    return _save(fig, f"{model_name}_3D_predictions.png")


def plot_pred_vs_true(preds_keV, acts_keV, model_name):
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    lim = [min(acts_keV.min(), preds_keV.min()) - 5000,
           max(acts_keV.max(), preds_keV.max()) + 5000]
    ax.plot(np.array(lim) / 1000, np.array(lim) / 1000, "k--", lw=1, label="perfect")
    ax.scatter(acts_keV / 1000, preds_keV / 1000, s=20, alpha=0.6,
               color="#0ea5e9", edgecolors="none")
    ax.set_xlabel("experimental mass excess (MeV)")
    ax.set_ylabel("predicted mass excess (MeV)")
    #ax.set_title(f"{model_name}"); ax.legend(loc="upper left")
    return _save(fig, f"{model_name}_pred_vs_true.png")


def plot_training_logs(logs, model_name):
    ep = range(1, len(logs["train_losses"]) + 1)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(ep, logs["train_losses"], "o-", ms=3, label="train")
    ax.plot(ep, logs["val_losses"], "s-", ms=3, label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (Huber, standardised residual)")
    ax.legend(); 
    #ax.set_title(f"{model_name} — loss")
    _save(fig, f"{model_name}_loss.png")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(ep, logs["train_rmse_keV"], "o-", ms=3, label="train")
    ax.plot(ep, logs["val_rmse_keV"], "s-", ms=3, label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("RMSE (keV)"); ax.set_yscale("log")
    ax.legend(); 
    #ax.set_title(f"{model_name} — RMSE in physical units")
    return _save(fig, f"{model_name}_rmse_keV.png")


def plot_stage_waterfall(stage_rmse, model_name):
    labels = [s[0] for s in stage_rmse]
    vals = [s[1] for s in stage_rmse]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(labels, vals, color=["#94a3b8", "#60a5fa", "#1d4ed8"], width=0.6)
    ax.set_yscale("log"); ax.set_ylabel("RMSE (keV, log scale)")
    #ax.set_title(f"{model_name} — error reduction by stage")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.06, f"{v:,.0f}",
                ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(min(vals) * 0.55, max(vals) * 2.2)
    return _save(fig, f"{model_name}_stage_waterfall.png")


def visualize_attention_weights(attention_weights, feature_names, model_name):
    if not attention_weights:
        print("   no attention weights to visualise")
        return None
    first = np.stack([np.asarray(a) for a in attention_weights[:512]])
    mean_attn = first.mean(axis=(0, 1))
    if mean_attn.ndim == 3:
        mean_attn = mean_attn.mean(axis=0)
    k = mean_attn.shape[0]
    names = feature_names[:k] if len(feature_names) >= k else \
        feature_names + [f"f{i}" for i in range(len(feature_names), k)]

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * k), max(6.5, 0.5 * k)))
    sns.heatmap(mean_attn, xticklabels=names, yticklabels=names, cmap="viridis",
                annot=(k <= 12), fmt=".2f", ax=ax)
    #ax.set_title(f"{model_name} — mean attention weights")
    ax.set_xlabel("key (feature)"); ax.set_ylabel("query (feature)")
    plt.xticks(rotation=45, ha="right")
    return _save(fig, f"{model_name}_attention.png")


def compute_feature_importance_from_attention(attention_weights, feature_names,
                                              model_name, seed):
    if not attention_weights:
        return None
    first = np.stack([np.asarray(a) for a in attention_weights[:512]])
    mean_attn = first.mean(axis=(0, 1))
    if mean_attn.ndim == 3:
        mean_attn = mean_attn.mean(axis=0)
    score = mean_attn.mean(axis=0)
    k = len(score)
    names = feature_names[:k] if len(feature_names) >= k else \
        feature_names + [f"f{i}" for i in range(len(feature_names), k)]
    df = pd.DataFrame({"feature": names, "attention_score": score})
    df = df.sort_values("attention_score", ascending=False).reset_index(drop=True)
    os.makedirs(RESULTS, exist_ok=True)
    path = f"{RESULTS}/attention_importance_{model_name}_{seed}.csv"
    df.to_csv(path, index=False)
    print(f"   saved {path}")
    return df
