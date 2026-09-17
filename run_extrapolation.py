#!/usr/bin/env python3
"""Region extrapolation study: the four-row table for every separation.

    LSMF                      Stage 1 only
    LSMF+anchor               Stage 1 + Stage 2
    LSMF+Transformer          Stage 1 + network, no anchor
    LSMF+anchor+Transformer   full anchored model (AnchoredFullModel)

Every seed draws its own train/validation split inside the development region,
as in the chronological study. The test region is fixed by the separation.

    # AME2020, separations (b) and (c)
    python run_extrapolation.py --input data/mass20.txt --format ame \
        --splits neutron_rich heavy --outdir results_extrap_ame

    # theory table only (all masses, splits and fits from the table)
    python run_extrapolation.py --input data/mass-frdm95.dat --format frdm95 \
        --splits random neutron_rich heavy --anchor off --outdir results_frdm95
"""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extrapolation_data import (load_table, sanity_check, split_table, ResidualPreprocessor,
                                distances, TARGET, SEPARATIONS)

ARCH = dict(d_model=128, num_heads=8, d_ff=512, num_layers=4,
            dropout=.12, pooling_type='gated_attention')      # = train.py
MODELS = ['LSMF', 'LSMF+anchor', 'LSMF+Transformer', 'LSMF+anchor+Transformer']
LABELS = {'random': '(a) random', 'neutron_rich': '(b) neutron-rich',
          'heavy': '(c) heavy', 'unmeasured': 'unmeasured'}   # heavy gets Z cut in main()
BINS = [0, 1, 2, 3, 5, 10, 20, np.inf]


def metrics(actual, predicted):
    error = np.asarray(predicted)-np.asarray(actual)
    return dict(n=len(error), rmse_keV=float(np.sqrt(np.mean(error**2))),
                mae_keV=float(np.mean(abs(error))), bias_keV=float(error.mean()))


def predict(model, loader, prep, device):
    import torch
    result = []
    model.eval()
    with torch.no_grad():
        for cat, cont, _ in loader:
            y, _ = model(cat.to(device), cont.to(device))
            result.extend(y.cpu().numpy())
    return np.asarray(result)*prep.target_std+prep.target_mean


def train_network(arrays, prep, run, seed, args):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from model import TransformerMassExcessPredictor
    from trainer import train_model
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = {k: TensorDataset(torch.tensor(a[0], dtype=torch.long),
                             torch.tensor(a[1]), torch.tensor(a[2]))
            for k, a in arrays.items()}
    # unshuffled loaders for prediction; shuffled one only for training
    loaders = {k: DataLoader(d, batch_size=args.batch_size) for k, d in data.items()}
    train_loader = DataLoader(data['train'], batch_size=args.batch_size, shuffle=True)
    model = TransformerMassExcessPredictor(len(prep.cont),
            [len(prep.maps[c])+1 for c in prep.cat], prep.cat, **ARCH).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=6)
    checkpoint = str(run/'best.pth')
    model, logs = train_model(train_loader, loaders['val'], model,
                              torch.nn.SmoothL1Loss(beta=.25), optimizer, scheduler,
                              checkpoint, args.epochs, prep.target_std,
                              patience=args.patience, verbose_every=args.verbose_every)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    pd.DataFrame(logs).to_csv(run/'training.csv', index=False)
    return {k: predict(model, loaders[k], prep, device) for k in ['train', 'val', 'test']}


def plot_splits(all_parts, path, z_cut):
    # same style as the paper split figure (prepare_data.draw_split_figure)
    fig, axes = plt.subplots(len(all_parts), 1, figsize=(10.5, 7.5*len(all_parts)),
                             sharex=True, squeeze=False)
    for ax, (separation, parts) in zip(axes[:, 0], all_parts.items()):
        ax.scatter(parts['train'].N, parts['train'].Z, s=24, alpha=0.30,
                   color='gray', label='Train')
        ax.scatter(parts['val'].N, parts['val'].Z, s=28, alpha=0.75, marker='^',
                   color='indigo', label='Validation')
        ax.scatter(parts['test'].N, parts['test'].Z, s=42, alpha=0.95, marker='D',
                   color='darkorange', label='Test')
        if separation == 'heavy':
            ax.axhline(z_cut + 0.5, color='black', ls='--', lw=1.2)
        ax.text(.97, .05, LABELS[separation], transform=ax.transAxes,
                ha='right', fontsize=24)
        ax.set_ylabel('Proton Number (Z)', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(alpha=0.30)
        ax.legend(fontsize=24, loc='upper left')
    axes[-1, 0].set_xlabel('Neutron Number (N)', fontsize=24)
    fig.tight_layout()
    fig.savefig(path, dpi=350, bbox_inches='tight')
    plt.close(fig)


def write_latex(summary, path, n_seeds, source):
    lines = [r'\begin{table}[htbp]\centering',
             r'\caption{Extrapolation performance on %s for each train/test separation. '
             r'Values are mean $\pm$ standard deviation over %d seeds, each with its own '
             r'random train/validation split inside the development region. The test '
             r'anchor reads only development masses.}' % (source, n_seeds),
             r'\label{tab:extrapolation}',
             r'\begin{tabular}{llcccc}\hline\hline',
             r'Separation & Model & RMSE (keV) & MAE (keV) & RMSE (keV) & MAE (keV) \\',
             r' & & (Validation) & (Validation) & (Test) & (Test) \\ \hline']
    for separation in summary.separation.unique():
        first = True
        for model in MODELS:
            row = []
            for split in ['val', 'test']:
                r = summary[(summary.separation == separation) & (summary.model == model)
                            & (summary.split == split)]
                if r.empty:
                    row += ['--', '--']
                    continue
                r = r.iloc[0]
                row += [f"${r.rmse_mean:.0f} \\pm {r.rmse_std:.0f}$",
                        f"${r.mae_mean:.0f} \\pm {r.mae_std:.0f}$"]
            name = LABELS[separation] if first else ''
            lines.append(f"{name} & {model.replace('+', ' + ')} & " + ' & '.join(row) + r' \\')
            first = False
        lines.append(r'\hline')
    lines += [r'\hline\end{tabular}\end{table}']
    Path(path).write_text('\n'.join(lines) + '\n')


def main(args):
    df = load_table(args.input, args.format, args.min_nz)
    if args.format != 'ame' and not args.skip_sanity_check:
        sanity_check(df)
    measured = (load_table(args.measured, args.measured_format, args.min_nz)
                if args.measured else None)
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'metrics.csv').exists() and not args.prepare_only:
        raise SystemExit(f'{root}/metrics.csv exists: choose a new --outdir')
    variants = [False, True] if args.anchor == 'both' else [args.anchor == 'on']
    (root/'manifest.json').write_text(json.dumps(dict(vars(args), architecture=ARCH,
                                                      n_nuclei=len(df)), indent=2))
    LABELS['heavy'] = f'(c) $Z>{args.z_cut}$'
    print(f'{args.input}: {len(df)} nuclei (N, Z >= {args.min_nz})')

    records, first_parts = [], {}
    for separation in args.splits:
        for seed in args.seeds:
            parts = split_table(df, separation, args.tail_fraction, args.z_cut,
                                args.val_fraction, seed, measured)
            dev = pd.concat([parts['train'], parts['val']])
            pools = {'train': parts['train'], 'val': parts['train'], 'test': dev}
            run = root/separation/f'seed_{seed}'
            run.mkdir(parents=True, exist_ok=True)
            for name, frame in parts.items():
                frame.to_csv(run/f'{name}.csv', index=False)
            first_parts.setdefault(separation, parts)
            print(f'\n=== {separation}  seed {seed}  '
                  + '  '.join(f'{k}={len(v)}' for k, v in parts.items()), flush=True)

            for use_anchor in variants:
                prep = ResidualPreprocessor(use_anchor, args.anchor_radius,
                                            args.anchor_min_neighbors).fit(parts['train'])
                arrays = {k: prep.transform(parts[k], pools[k]) for k in parts}
                static = 'LSMF+anchor' if use_anchor else 'LSMF'
                network = 'LSMF+anchor+Transformer' if use_anchor else 'LSMF+Transformer'
                for name in ['train', 'val', 'test']:
                    records.append(dict(separation=separation, seed=seed, split=name,
                                        model=static, anchor_coverage=arrays[name][4].mean(),
                                        **metrics(parts[name][TARGET], arrays[name][3])))
                if args.prepare_only:
                    continue
                tag = 'anchor' if use_anchor else 'noanchor'
                (run/tag).mkdir(exist_ok=True)
                predicted = train_network({k: a[:3] for k, a in arrays.items()},
                                          prep, run/tag, seed, args)
                for name in ['train', 'val', 'test']:
                    frame = parts[name].copy()
                    frame['static_keV'] = arrays[name][3]
                    frame['predicted_keV'] = arrays[name][3] + predicted[name]
                    frame['error_keV'] = frame.predicted_keV - frame[TARGET]
                    frame['has_anchor'] = arrays[name][4]
                    frame['distance'] = distances(frame, pools[name])
                    frame.to_csv(run/tag/f'{name}_predictions.csv', index=False)
                    m = metrics(frame[TARGET], frame.predicted_keV)
                    records.append(dict(separation=separation, seed=seed, split=name,
                                        model=network, anchor_coverage=arrays[name][4].mean(), **m))
                    print(f'  {network:<25} {name:<5} RMSE {m["rmse_keV"]:8.1f}  '
                          f'MAE {m["mae_keV"]:8.1f} keV', flush=True)
                    grouped = []
                    bins = pd.cut(frame.distance, BINS, include_lowest=True)
                    for interval, group in frame.groupby(bins, observed=True):
                        for label, col in [(static, 'static_keV'), (network, 'predicted_keV')]:
                            grouped.append(dict(distance_bin=str(interval), model=label,
                                                **metrics(group[TARGET], group[col])))
                    pd.DataFrame(grouped).to_csv(run/tag/f'{name}_distance_metrics.csv', index=False)
                pd.DataFrame(records).to_csv(root/'metrics.csv', index=False)

    plot_splits(first_parts, root/'splits.png', args.z_cut)
    result = pd.DataFrame(records)
    result.to_csv(root/'metrics.csv', index=False)
    summary = (result.groupby(['separation', 'split', 'model'], sort=False)
               .agg(n=('n', 'first'), n_seeds=('seed', 'nunique'),
                    rmse_mean=('rmse_keV', 'mean'), rmse_std=('rmse_keV', 'std'),
                    mae_mean=('mae_keV', 'mean'), mae_std=('mae_keV', 'std'),
                    bias_mean=('bias_keV', 'mean'), anchor_coverage=('anchor_coverage', 'mean'))
               .reset_index())
    summary = summary.sort_values(
        ['separation', 'split', 'model'],
        key=lambda c: c.map({**{s: i for i, s in enumerate(args.splits)},
                             'train': -1, 'val': 0, 'test': 1, **{m: i for i, m in enumerate(MODELS)}}))
    summary.to_csv(root/'summary.csv', index=False)
    source = args.table_source or {'ame': 'AME2020', 'hfb14': 'the HFB-14 table',
                                   'frdm95': 'the FRDM95 table'}.get(args.format, args.input)
    write_latex(summary, root/'table_extrapolation.tex', len(args.seeds), source)
    pd.set_option('display.width', 160)
    print('\n' + summary.round(1).to_string(index=False))
    print(f'\nwrote {root}/summary.csv, table_extrapolation.tex, splits.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True)
    p.add_argument('--format', choices=['csv', 'ame', 'hfb14', 'frdm95'], default='csv')
    p.add_argument('--splits', nargs='+', choices=SEPARATIONS, default=['neutron_rich', 'heavy'])
    p.add_argument('--measured', help='only for the unmeasured separation (mixes in AME)')
    p.add_argument('--measured-format', choices=['ame', 'csv'], default='ame')
    p.add_argument('--anchor', choices=['both', 'on', 'off'], default='both')
    p.add_argument('--tail-fraction', type=float, default=.20)
    p.add_argument('--z-cut', type=int, default=80)
    p.add_argument('--val-fraction', type=float, default=.30)
    p.add_argument('--seeds', type=int, nargs='+', default=[12, 17, 33, 42, 89])
    p.add_argument('--min-nz', type=int, default=8)
    p.add_argument('--anchor-radius', type=int, default=3)
    p.add_argument('--anchor-min-neighbors', type=int, default=14)
    # lr / batch / epochs as in train.py; early stopping is active here (patience 20)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--verbose-every', type=int, default=20)
    p.add_argument('--skip-sanity-check', action='store_true',
                   help='skip the AME comparison for non-physical test tables')
    p.add_argument('--table-source', help='name used in the LaTeX caption')
    p.add_argument('--outdir', default='results_extrapolation')
    p.add_argument('--prepare-only', action='store_true',
                   help='splits, figure and the two non-network rows only')
    args = p.parse_args()
    if 'unmeasured' in args.splits and not args.measured:
        p.error('--splits unmeasured needs --measured')
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1:
        p.error('Epochs, patience, batch size and threads must be positive')
    import torch
    torch.set_num_threads(args.threads)
    main(args)
