# Baseline + Transformer: extrapolation experiments

Use `run_extrapolation.py` for this experiment. The original uploaded scripts
are included with import-compatible filenames and are unchanged. Do not use the
old `train.py --use_anchor 0`: that path predicts raw mass excess, removing the
fitted global baseline as well as the anchor.

Prediction in the new runner:

    mass_excess = training-fitted LSMF + Transformer global-residual correction

The original model.py architecture (including its learned linear residual head),
trainer.py training loop, anchor.py LSMFBaseline and physics feature construction
are reused. No local anchor or anchor diagnostic tokens are used. All 19
BASE_PHYSICS descriptors from train.py are retained; category maps and scaling
are fitted only on training rows. Unseen categories use the original reserved
unknown-index policy; their counts are recorded in manifest.json. This allows
inference but does not guarantee extrapolation of unseen shell categories.

## Installation and first run

Requires Python 3.10+, numpy, pandas, scikit-learn, matplotlib and torch.

    python -m pip install numpy pandas scikit-learn matplotlib torch
    python run_extrapolation.py --input data/mass-hfb14.dat --format hfb14 --outdir results_hfb14

Both separations run with seed 42, batch size 16, learning rate .001, up to 200
epochs, patience 30. Architecture: 128 dimensions, 8 heads, 4 layers, FF 512,
dropout .12, gated attention pooling. CUDA is selected when available, otherwise
CPU. Model seeds can be added without changing the split:

    python run_extrapolation.py --input data/mass-hfb14.dat --format hfb14 --seeds 42 89 123 --outdir results_hfb14_multiseed

Other input formats:

    python run_extrapolation.py --input data/trainval.csv --format csv --outdir results_ame_csv
    python run_extrapolation.py --input data/mass20.txt --format ame --outdir results_ame20
    python run_extrapolation.py --input data/mass-frdm95.dat --format frdm95 --outdir results_frdm95

CSV requires N, Z and `Mass Excess (keV)` (or `mass_excess_keV`); A can be derived.
Pass ONE consistent full table, not an already held-out subset. If trainval.csv
contains only a historical subset, the experiment covers only that subset.
AME uses the supplied measured-entry parser. RIPL fixed-width readers explicitly
use theoretical Mth in MeV and multiply by 1000; Mexp is never substituted.
HFB14 and FRDM95 formats are supported, not arbitrary HFB/FRDM versions.
Default selection: N,Z >= 8. No target values or precomputed features are used
in deciding split membership. Input duplicate nuclei, invalid coordinates and
nonfinite targets are rejected.

## Exact split definitions

* Separation 2: for each Z, hold out the largest-N ceil(0.20 * chain size)
  nuclei, retaining at least one development nucleus. Single-row chains stay
  in development. Configure `--tail-fraction`. This is an operational analogue
  of panel (b), not an exact reproduction of an undocumented figure boundary.
* Separation 3: development Z <= 82; test Z > 82. Configure `--z-cut`.
* Validation: 30% random split INSIDE the remaining development pool, fixed by
  `--split-seed 42`; no rare-bin rows are dropped. Configure `--val-fraction`.
  This matches the broad diagram's internal random testing design; it is not
  an additional boundary validation protocol. The excluded test region never
  enters early stopping or scheduler decisions.
* Baseline, feature statistics, category mappings, residual normalization and
  neighbor-existence flags all use training rows only. No train+validation refit
  occurs before test evaluation. Model-seed variation is not split variation.

## Outputs

Each separation stores exact train/val/test CSVs, split.png, manifest.json
(including input hash, settings and unseen-category counts), preprocessing.pkl,
and one folder per model seed. Each seed folder stores best.pth, training.csv,
val/test_predictions.csv and val/test_distance_metrics.csv. Error sign is
prediction minus reference. Distance is Euclidean distance in (N,Z) to the
nearest training nucleus. Metrics compare baseline and baseline+Transformer.
Root metrics.csv and summary.csv contain RMSE, MAE and bias in keV. Baseline seed
-1 denotes the single shared baseline, not an extra random seed. Standard
deviation is undefined with one seed. Existing separation folders are protected
against overwriting: choose a new output directory for another experiment.

The pickle and model checkpoint together with these code files support later
reconstruction; load pickle only from your own trusted run. The preprocessor
provides transform() and contains the baseline, feature order and scalers.

## Verification and limits

    python check_extrapolation.py
    python run_extrapolation.py --input data/mass-hfb14.dat --format hfb14 --prepare-only --outdir prepared_hfb14

Checks passed on an artificial diagnostic fixture: split disjointness and full
coverage, directional boundaries, target isolation from input transformations,
residual reconstruction, Mth field/unit parsing and duplicate rejection.
The prepare-only CLI also passed for both splits, including figures and outputs.
All Python files compile. PyTorch is absent in the editing environment, so
end-to-end neural training is NOT verified. No physical mass data were supplied;
no scientific Transformer test results are included. Diagnostic fixture scores
are not paper results. Before a long run, use a separate output directory and
`--epochs 2` to confirm training on your machine, then run the full experiment.

The inherited valence particle/hole helper has a maximum magic number of 184;
review its physics convention before interpreting inputs beyond N=184. We have
preserved the existing feature definitions for this first comparison.

Data source:
https://www-nds.iaea.org/RIPL-3/masses/mass-hfb14.dat
Documentation:
https://www-nds.iaea.org/RIPL-3/masses/mass-hfb14.readme
