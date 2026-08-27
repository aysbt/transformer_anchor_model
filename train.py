#!/usr/bin/env python3
"""
train.py
========

Trains one model, for one seed, on the anchored residual.

    python train.py --csv_path data/trainval.csv --seed 42

Writes
------
    model_checkpoint/best_<experiment>.pth      network weights
    model_checkpoint/preprocess_<experiment>.pth  normalisation + anchor state
    results/metrics_val_<model>_<seed>.csv      validation metrics, keV
    results/region_metrics_<model>_<seed>.csv   per-region metrics, keV
    plots/...                                    figures
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import get_dataloader, get_eval_dataloader
from model import TransformerMassExcessPredictor
from trainer import train_model
from anchor import ANCHOR_FEATURES
import evaluate as ev

for d in ("plots", "results", "model_checkpoint"):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- features
BASE_PHYSICS = [
    "N", "Z", "A",
    "Zshell_category", "Nshell_category",
    "ZEO", "NEO",
    "deltaN", "deltaZ",
    "A^2/3", "Z(Z-1)/A^1/3", "(N-Z)^2/A",
    "(N-Z)/A", "N/Z",
    "promiscuity",
    "neighbor_N_plus_1_exists", "neighbor_N_minus_1_exists",
    "neighbor_Z_plus_1_exists", "neighbor_Z_minus_1_exists",
]

FEATURE_SETS = {
    "CoreModel": ["N", "Z", "A"],
    "ShellModel": ["N", "Z", "A", "Zshell_category", "Nshell_category"],
    "ZEOModel": ["N", "Z", "A", "ZEO", "NEO"],
    "MagicModel": ["N", "Z", "A", "deltaN", "deltaZ"],
    "LiquidDropModel": ["N", "Z", "A", "A^2/3", "Z(Z-1)/A^1/3", "(N-Z)^2/A"],
    "ValenceModel": ["N", "Z", "A", "nu_N", "nu_Z", "promiscuity",
                     "proton_particles", "proton_holes",
                     "neutron_particles", "neutron_holes"],
    "AnchoredFullModel": BASE_PHYSICS,
}

ARCH = dict(d_model=128, num_heads=8, d_ff=512, num_layers=4,
            dropout=0.12, pooling_type="gated_attention")


def banner(text, ch="="):
    print("\n" + ch * 78)
    print(text)
    print(ch * 78)


def main(args):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_name = args.model
    selected = FEATURE_SETS[model_name]

    banner(f"TRAINING  {model_name}   seed {args.seed}")
    print(f"   device              : {device}")
    print(f"   dataset             : {args.csv_path}")
    print(f"   anchoring           : {'ON' if args.use_anchor else 'OFF'}")
    if args.use_anchor:
        print(f"   anchor radius       : {args.anchor_radius}")
        print(f"   anchor min neighbours: {args.anchor_min_neighbors}")
    print(f"   physics features    : {len(selected)}")

    # ------------------------------------------------------------- loaders
    train_loader = get_dataloader(
        args.csv_path, selected, batch_size=args.batch_size, shuffle=True,
        split="train", use_anchor=args.use_anchor, seed=args.seed,
        anchor_max_radius=args.anchor_radius,
        anchor_min_neighbors=args.anchor_min_neighbors)

    stats = train_loader.dataset.preprocess_stats
    anchor_ctx = train_loader.dataset.anchor_context

    val_loader = get_dataloader(
        args.csv_path, selected, batch_size=args.batch_size, shuffle=False,
        split="val", preprocess_stats=stats, anchor_context=anchor_ctx,
        use_anchor=args.use_anchor, seed=args.seed)

    eval_loader = get_eval_dataloader(
        args.csv_path, selected, batch_size=args.batch_size, shuffle=False,
        split="val", preprocess_stats=stats, anchor_context=anchor_ctx,
        use_anchor=args.use_anchor, seed=args.seed)

    cont_feats = train_loader.dataset.continuous_features
    cat_feats = train_loader.dataset.categorical_features
    cat_sizes = [train_loader.dataset.category_sizes[c] for c in cat_feats]
    target_mean = train_loader.dataset.target_mean
    target_std = train_loader.dataset.target_std

    banner("DATA AND ANCHOR SUMMARY", "-")
    print(f"   training nuclei     : {len(train_loader.dataset)}")
    print(f"   validation nuclei   : {len(val_loader.dataset)}")
    print(f"   continuous features : {len(cont_feats)}")
    print(f"     {cont_feats}")
    print(f"   categorical features: {cat_feats}  sizes {cat_sizes}")
    if args.use_anchor:
        N_tr = train_loader.dataset.data["original_N"].to_numpy(int)
        Z_tr = train_loader.dataset.data["original_Z"].to_numpy(int)
        print(f"\n   Stage 1  LSMF train RMSE      : {anchor_ctx.baseline_rmse_:10.1f} keV")
        print(f"   Stage 2  anchor dictionary    : {anchor_ctx.dict_size_} nuclei "
              f"(regime='{anchor_ctx.regime}')")
        off = train_loader.dataset.offset
        y = train_loader.dataset.data["actual_mass"].to_numpy(float)
        print(f"   Stage 1+2 train RMSE          : "
              f"{np.sqrt(np.mean((y - off)**2)):10.1f} keV")
        _, _af = anchor_ctx.offset_and_features(N_tr, Z_tr)
        print(f"   nuclei with an anchor         : "
              f"{int(_af['anchor_has'].sum())} / {len(N_tr)}")
        print(f"   mean neighbours used          : {_af['anchor_n'].mean():10.1f}")
    print(f"\n   target = anchored residual, standardised")
    print(f"     residual mean     : {target_mean:10.2f} keV")
    print(f"     residual std      : {target_std:10.2f} keV")

    # --------------------------------------------------------------- model
    pooling = ARCH["pooling_type"]
    experiment = f"{model_name}_{pooling}_anchored_s{args.seed}"
    ckpt = f"model_checkpoint/best_{experiment}.pth"
    stats_path = f"model_checkpoint/preprocess_{experiment}.pth"

    model = TransformerMassExcessPredictor(
        num_cont_features=len(cont_feats),
        categorical_feature_sizes=cat_sizes,
        categorical_features=cat_feats,
        **ARCH).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n   architecture        : d_model={ARCH['d_model']}, heads={ARCH['num_heads']}, "
          f"layers={ARCH['num_layers']}, d_ff={ARCH['d_ff']}, pooling={pooling}")
    print(f"   trainable parameters: {n_params:,}")

    torch.save({"preprocess_stats": stats,
                "anchor_state": anchor_ctx.state_dict() if anchor_ctx else None,
                "selected_features": selected,
                "continuous_features": cont_feats,
                "categorical_features": cat_feats,
                "category_sizes": cat_sizes,
                "arch": ARCH,
                "use_anchor": args.use_anchor}, stats_path)
    print(f"   saved preprocessing + anchor state -> {stats_path}")

    criterion = nn.SmoothL1Loss(beta=0.25)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                     factor=0.5, patience=6)

    banner("TRAINING", "-")
    model, logs = train_model(train_loader, val_loader, model, criterion,
                              optimizer, scheduler, checkpoint_path=ckpt,
                              num_epochs=args.num_epochs, target_std=target_std,
                              patience=args.patience,
                              verbose_every=args.verbose_every)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # ---------------------------------------------------------- evaluation
    banner(f"VALIDATION RESULTS  {model_name}  seed {args.seed}")
    N, Z, preds, acts, offsets, attn = ev.evaluate_model(
        model, eval_loader, device, target_mean, target_std)

    m_full = ev.compute_metrics(preds, acts)
    ev.print_metrics(m_full, f"{model_name} — validation set (keV)")

    if args.use_anchor:
        m_s1 = ev.compute_metrics(
            anchor_ctx.baseline.predict(N.astype(int), Z.astype(int)), acts)
        m_s2 = ev.compute_metrics(offsets, acts)
        print("\n   stage decomposition")
        print(ev.fmt_row(m_s1, "S1  LSMF only"))
        print(ev.fmt_row(m_s2, "S1+S2  + local anchor"))
        print(ev.fmt_row(m_full, "S1+S2+S3  + transformer"))
        ev.plot_stage_waterfall(
            [("S1", m_s1["rmse_keV"]), ("S1+S2", m_s2["rmse_keV"]),
             ("S1+S2+S3", m_full["rmse_keV"])], f"{model_name}_s{args.seed}")

    reg = ev.compute_region_metrics(N, Z, preds, acts)
    ev.print_region_table(reg, f"{model_name} — regional performance (validation)")

    # ------------------------------------------------------------- outputs
    pd.DataFrame([{**m_full, "model": model_name, "seed": args.seed,
                   "split": "val", "use_anchor": args.use_anchor}]).to_csv(
        f"results/metrics_val_{model_name}_{args.seed}.csv", index=False)
    reg.assign(model=model_name, seed=args.seed).to_csv(
        f"results/region_metrics_{model_name}_{args.seed}.csv", index=False)
    pd.DataFrame({"N": N, "Z": Z, "A": N + Z, "actual_keV": acts,
                  "predicted_keV": preds, "offset_keV": offsets,
                  "error_keV": preds - acts}).to_csv(
        f"results/val_predictions_{model_name}_{args.seed}.csv", index=False)

    banner("PLOTS", "-")
    tag = f"{model_name}_s{args.seed}"
    ev.plot_training_logs(logs, tag)
    ev.plot_error_diagnostics(N, Z, preds, acts, tag)
    ev.plot_error_distribution(preds, acts, tag)
    ev.plot_predictions(N, Z, preds, acts, tag)
    ev.plot_pred_vs_true(preds, acts, tag)
    names = cat_feats + cont_feats
    ev.visualize_attention_weights(attn, names, tag)
    ev.compute_feature_importance_from_attention(attn, names, model_name, args.seed)

    banner(f"DONE  {model_name}  seed {args.seed}   "
           f"val RMSE {m_full['rmse_keV']:,.1f} keV   ({time.time()-t0:.0f}s)")
    return m_full


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default="data/trainval.csv")
    ap.add_argument("--model", default="AnchoredFullModel", choices=list(FEATURE_SETS))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--verbose_every", type=int, default=5)
    ap.add_argument("--use_anchor", type=int, default=1)
    ap.add_argument("--anchor_radius", type=int, default=3)
    ap.add_argument("--anchor_min_neighbors", type=int, default=10)
    a = ap.parse_args()
    a.use_anchor = bool(a.use_anchor)
    main(a)
