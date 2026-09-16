#!/usr/bin/env python3
"""Four-row table for the chronological AME2016 -> AME2020 test.

Needs both multi-seed runs first:
    python run_multiseed.py --model AnchoredFullModel --use_anchor 1
    python run_multiseed.py --model AnchoredFullModel --use_anchor 0
    python stage_table.py --model AnchoredFullModel
"""
import argparse
from pathlib import Path
import pandas as pd


def cell(r, rmse, mae):
    return (f"${r[rmse + '_mean']:.0f} \\pm {r[rmse + '_std']:.0f}$",
            f"${r[mae + '_mean']:.0f} \\pm {r[mae + '_std']:.0f}$")


def main(args):
    rows = {}
    for split in ['val', 'test']:
        full = pd.read_csv(f'results/summary_{split}_{args.model}.csv')
        noanchor = pd.read_csv(f'results/summary_{split}_{args.model}_noanchor.csv')
        if split == 'test':
            full = full[full.regime == args.regime]
            noanchor = noanchor[noanchor.regime == 'historical']
        full, noanchor = full.iloc[0], noanchor.iloc[0]
        n_seeds = int(full.n_seeds)
        # S1 is identical in both runs (same seeds, same split); take the anchored one
        rows.setdefault('LSMF', []).extend(cell(full, 'rmse_s1_keV', 'mae_s1_keV'))
        rows.setdefault('LSMF + anchor', []).extend(cell(full, 'rmse_s2_keV', 'mae_s2_keV'))
        rows.setdefault('LSMF + Transformer', []).extend(cell(noanchor, 'rmse_keV', 'mae_keV'))
        rows.setdefault('LSMF + anchor + Transformer', []).extend(cell(full, 'rmse_keV', 'mae_keV'))
    lines = [r'\begin{table}[htbp]\centering',
             r'\caption{Stage decomposition on the chronological split: validation on '
             r'AME2016 and test on the nuclei first measured in AME2020 (%s anchor '
             r'dictionary). Mean $\pm$ standard deviation over %d seeds.}' % (args.regime, n_seeds),
             r'\label{tab:stages_chrono}',
             r'\begin{tabular}{lcccc}\hline\hline',
             r'Model & RMSE (keV) & MAE (keV) & RMSE (keV) & MAE (keV) \\',
             r' & (Validation) & (Validation) & (Test) & (Test) \\ \hline']
    lines += [f'{name} & ' + ' & '.join(v) + r' \\' for name, v in rows.items()]
    lines += [r'\hline\hline\end{tabular}\end{table}']
    out = Path(f'results/table_stages_{args.model}.tex')
    out.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines)); print(f'\nwrote {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='AnchoredFullModel')
    ap.add_argument('--regime', default='historical')
    main(ap.parse_args())
