#!/usr/bin/env python3
"""
create_configs.py
=================

Writes one JSON per model describing its feature set and architecture, so an
existing checkpoint can be identified without reading train.py.

    python create_configs.py

Note: categorical_feature_sizes are deliberately NOT stored. They are derived
from the training data at runtime and saved inside
model_checkpoint/preprocess_<experiment>.pth together with the anchor state.
Hard-coding them here is what caused the old index-out-of-range failures.
"""

import json
import os

from train import FEATURE_SETS, ARCH

os.makedirs("model_checkpoint", exist_ok=True)

DESCRIPTIONS = {
    "CoreModel": "Fundamental nuclear quantities only (N, Z, A).",
    "ShellModel": "Adds shell occupancy categories.",
    "ZEOModel": "Adds even-odd parity indicators.",
    "MagicModel": "Adds magic-number proximity (deltaN, deltaZ).",
    "LiquidDropModel": "Liquid-drop terms: surface, Coulomb, asymmetry.",
    "ValenceModel": "Valence nucleons, particles/holes and the Casten-Cakirli promiscuity factor.",
    "AnchoredFullModel": "All physics descriptors; trained on the anchored residual.",
}

for name, feats in FEATURE_SETS.items():
    cfg = {
        "model_name": name,
        "description": DESCRIPTIONS.get(name, ""),
        "selected_features": feats,
        "architecture": ARCH,
        "target": "anchored residual = mass excess - LSMF - local anchor",
        "note": "categorical sizes and anchor state live in preprocess_<experiment>.pth",
    }
    path = f"model_checkpoint/{name}_config.json"
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"saved {path}")
