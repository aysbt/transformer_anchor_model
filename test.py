#!/usr/bin/env python3
"""
test.py
=======

Evaluates a trained model on the chronological AME2020 test set under FOUR
different anchor dictionaries. The network is identical in all four; only the
set of neighbour masses the anchor is allowed to see changes.

    python test.py --model AnchoredFullModel --seed 42

Regimes
-------
  historical   dictionary = AME2016 masses only.
               PROSPECTIVE. This is the number to report as the headline
               extrapolation result: it uses exactly the information that
               existed before the AME2020 measurements were made.

  train        dictionary = the training split only (a subset of AME2016).
               Slightly more conservative than historical.

  trainval     dictionary = training + validation (= the full AME2016 pool).
               Same as historical here; kept because they differ if you change
               the split.

  loo          dictionary = AME2016 plus the test nuclei themselves, minus the
               target. LEAVE-ONE-OUT INTERPOLATION, not prediction. Test nuclei
               support each other, so this is optimistic. Report separately and
               label it clearly -- never as the extrapolation result.

Also reported
-------------
  * stage decomposition: LSMF alone, LSMF+anchor, LSMF+anchor+transformer.
    If the network gain is small the accuracy is coming from interpolation,
    which the reader deserves to know.
  * stratification by whether an anchor was available at all.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from dataloader import get_eval_dataloader
from model import TransformerMassExcessPredictor
from anchor import AnchorContext
import evaluate as ev

for d in ("plots", "results"):
    os.makedirs(d, exist_ok=True)

TARGET = "Mass Excess (keV)"
REGIME_NOTES = {
    "historical": "AME2016 only  -- PROSPECTIVE, report this one",
    "train": "training split only",
    "trainval": "training + validation",
    "loo": "all except target -- LEAVE-ONE-OUT, optimistic",
}


def banner(text, ch="="):
    print("\n" + ch * 78)
    print(text)
    print(ch * 78)


def build_context(regime, base_state, trainval_df, test_df, seed, split_frac=0.30):
    """Construct an AnchorContext whose dictionary matches the requested regime."""
    ctx = AnchorContext.from_state_dict(base_state)   # keeps the trained LSMF
    ctx.regime = regime

    Ntv = trainval_df["N"].to_numpy(int)
    Ztv = trainval_df["Z"].to_numpy(int)
    ytv = trainval_df[TARGET].to_numpy(float)

    if regime == "train":
        return ctx                                     # already the train dictionary

    if regime in ("historical", "trainval"):
        ctx.set_dictionary(Ntv, Ztv, ytv)
        return ctx

    if regime == "loo":
        N = np.concatenate([Ntv, test_df["N"].to_numpy(int)])
        Z = np.concatenate([Ztv, test_df["Z"].to_numpy(int)])
        y = np.concatenate([ytv, test_df[TARGET].to_numpy(float)])
        ctx.set_dictionary(N, Z, y)                    # (0,0) drop makes it LOO
        return ctx

    raise ValueError(regime)


def main(args):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment = f"{args.model}_gated_attention_anchored_s{args.seed}"
    ckpt = f"model_checkpoint/best_{experiment}.pth"
    stats_path = f"model_checkpoint/preprocess_{experiment}.pth"

    banner(f"TESTING  {args.model}   seed {args.seed}")
    print(f"   checkpoint : {ckpt}")
    print(f"   test set   : {args.test_csv}")

    bundle = torch.load(stats_path, map_location="cpu", weights_only=False)
    stats = bundle["preprocess_stats"]
    selected = bundle["selected_features"]
    cat_feats = bundle["categorical_features"]
    cat_sizes = bundle["category_sizes"]
    cont_feats = bundle["continuous_features"]
    use_anchor = bundle["use_anchor"]
    arch = bundle["arch"]

    model = TransformerMassExcessPredictor(
        num_cont_features=len(cont_feats),
        categorical_feature_sizes=cat_sizes,
        categorical_features=cat_feats, **arch).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    trainval_df = pd.read_csv(args.trainval_csv)
    test_df = pd.read_csv(args.test_csv)
    print(f"   AME2016 pool : {len(trainval_df)} nuclei")
    print(f"   AME2020 new  : {len(test_df)} nuclei  (never seen in training)")

    # sanity: no test nucleus may appear in the training pool
    tv_keys = set(zip(trainval_df.N, trainval_df.Z))
    overlap = sum((n, z) in tv_keys for n, z in zip(test_df.N, test_df.Z))
    print(f"   overlap check: {overlap} test nuclei found in the AME2016 pool "
          f"(must be 0)")

    target_mean, target_std = stats["target_mean"], stats["target_std"]
    regimes = args.regimes.split(",")
    summary, per_regime = [], {}

    for regime in regimes:
        banner(f"REGIME '{regime}'  --  {REGIME_NOTES[regime]}", "-")
        ctx = build_context(regime, bundle["anchor_state"], trainval_df,
                            test_df, args.seed) if use_anchor else None
        if ctx is not None:
            print(f"   anchor dictionary size : {ctx.dict_size_} nuclei")

        loader = get_eval_dataloader(
            args.test_csv, selected, batch_size=args.batch_size, shuffle=False,
            split="test", preprocess_stats=stats, anchor_context=ctx,
            use_anchor=use_anchor, seed=args.seed)

        N, Z, preds, acts, offsets, attn = ev.evaluate_model(
            model, loader, device, target_mean, target_std)

        m_full = ev.compute_metrics(preds, acts)
        ev.print_metrics(m_full, f"{args.model} — test set, regime '{regime}' (keV)")

        if use_anchor:
            m_s1 = ev.compute_metrics(ctx.baseline.predict(N.astype(int),
                                                           Z.astype(int)), acts)
            m_s2 = ev.compute_metrics(offsets, acts)
            print("\n   stage decomposition")
            print(ev.fmt_row(m_s1, "S1  LSMF only"))
            print(ev.fmt_row(m_s2, "S1+S2  + local anchor"))
            print(ev.fmt_row(m_full, "S1+S2+S3  + transformer"))
            gain = m_s2["rmse_keV"] - m_full["rmse_keV"]
            print(f"\n   network contribution   : {gain:,.1f} keV RMSE reduction "
                  f"({100*gain/max(m_s2['rmse_keV'],1e-9):.1f}% of the anchor-only error)")

            has = loader.dataset.data["anchor_has"].to_numpy() > 0
            # anchor_has was standardised; recover the flag from the raw context
            _, feats = ctx.offset_and_features(N.astype(int), Z.astype(int))
            has = feats["anchor_has"] > 0
            print(f"\n   nuclei with an anchor  : {int(has.sum())} / {len(has)}")
            if has.sum() and (~has).sum():
                print(ev.fmt_row(ev.compute_metrics(preds[has], acts[has]),
                                 "  with anchor"))
                print(ev.fmt_row(ev.compute_metrics(preds[~has], acts[~has]),
                                 "  no anchor (pure ML)"))

        reg = ev.compute_region_metrics(N, Z, preds, acts)
        ev.print_region_table(reg, f"regional performance — regime '{regime}'")

        tag = f"{args.model}_s{args.seed}_test_{regime}"
        ev.plot_error_diagnostics(N, Z, preds, acts, tag)
        ev.plot_error_distribution(preds, acts, tag)
        ev.plot_pred_vs_true(preds, acts, tag)

        pd.DataFrame({"N": N, "Z": Z, "A": N + Z, "actual_keV": acts,
                      "predicted_keV": preds, "offset_keV": offsets,
                      "error_keV": preds - acts}).to_csv(
            f"results/test_predictions_{args.model}_{args.seed}_{regime}.csv",
            index=False)
        reg.assign(model=args.model, seed=args.seed, regime=regime).to_csv(
            f"results/test_region_metrics_{args.model}_{args.seed}_{regime}.csv",
            index=False)

        row = {**m_full, "model": args.model, "seed": args.seed, "regime": regime}
        if use_anchor:
            row["rmse_s1_keV"] = m_s1["rmse_keV"]
            row["rmse_s2_keV"] = m_s2["rmse_keV"]
        summary.append(row)
        per_regime[regime] = m_full

    df = pd.DataFrame(summary)
    df.to_csv(f"results/metrics_test_{args.model}_{args.seed}.csv", index=False)

    banner(f"REGIME COMPARISON  {args.model}  seed {args.seed}")
    print(f"\n   {'regime':<13}{'n':>5}{'RMSE':>10}{'MAE':>10}{'median':>9}"
          f"{'<250keV':>9}   note")
    print("   " + "-" * 92)
    for r in summary:
        print(f"   {r['regime']:<13}{r['n']:>5}{r['rmse_keV']:>10.1f}"
              f"{r['mae_keV']:>10.1f}{r['median_abs_keV']:>9.1f}"
              f"{r['within_250keV']*100:>8.1f}%   {REGIME_NOTES[r['regime']]}")
    print(f"\n   saved results/metrics_test_{args.model}_{args.seed}.csv")
    print(f"   runtime {time.time()-t0:.0f}s")
    return per_regime


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="AnchoredFullModel")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trainval_csv", default="data/trainval.csv")
    ap.add_argument("--test_csv", default="data/test.csv")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--regimes", default="historical,train,loo")
    main(ap.parse_args())
