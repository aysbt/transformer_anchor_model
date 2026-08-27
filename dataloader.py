"""
dataloader.py
=============

The network is trained on the ANCHORED RESIDUAL rather than on the raw (or signed-log) mass excess.

    offset   = LSMF(N,Z) + anchor(N,Z)            keV, computed in anchor.py
    residual = mass_excess - offset                keV
    target   = (residual - resid_mean) / resid_std standardised, what the net sees

    reconstruction:  mass_excess_hat = target_hat * resid_std + resid_mean + offset

Two consequences worth noting:

  * The anchored residual is already a small, roughly symmetric quantity
  * `self.offset` is stored per row and must be carried through to prediction
    time. evaluate.py does this for you.

Anchor-dictionary rules (see anchor.py for the full argument):
  split="train"  -> dictionary is the training rows themselves. The anchor drops
                    the (0,0) self match, so this is leave-one-out and safe.
  split="val"    -> caller MUST pass the train-fitted anchor_context.
  split="test"   -> caller MUST pass a context whose dictionary is the historical
                    (AME2016) pool. Never the test set itself.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from anchor import AnchorContext, ANCHOR_FEATURES


class NuclearMassDataset(Dataset):
    def __init__(
        self,
        csv_path,
        selected_features,
        target_column="Mass Excess (keV)",
        split="train",
        include_nz=False,
        preprocess_stats=None,
        anchor_context=None,
        use_anchor=True,
        anchor_max_radius=3,
        anchor_min_neighbors=14,
        seed=42,
    ):
        self.data = pd.read_csv(csv_path)
        self.include_nz = include_nz
        self.target_column = target_column
        self.use_anchor = use_anchor

        self.data["original_N"] = self.data["N"]
        self.data["original_Z"] = self.data["Z"]
        self.data["actual_mass"] = self.data[target_column]

        # ------------------------------------------------------------
        # 1. Split first, before any statistic is computed
        # ------------------------------------------------------------
        if split in ("train", "val"):
            self.data["bin"] = [f"{int(n/5)}_{int(z/5)}"
                                for n, z in zip(self.data["N"], self.data["Z"])]
            counts = self.data["bin"].value_counts()
            self.data = self.data[self.data["bin"].isin(counts[counts >= 2].index)]

            train_df, val_df = train_test_split(
                self.data, test_size=0.30,
                stratify=self.data["bin"], random_state=seed,
            )
            train_df = train_df.drop(columns=["bin"])
            val_df = val_df.drop(columns=["bin"])
            self.data = (train_df if split == "train" else val_df).reset_index(drop=True)

        elif split == "test":
            self.data = self.data.reset_index(drop=True)
        else:
            raise ValueError("split must be 'train', 'val' or 'test'")

        # ------------------------------------------------------------
        # 2. Feature groups
        # ------------------------------------------------------------
        all_categorical = ["Zshell_category", "Nshell_category", "ZEO", "NEO", "deltaN", "deltaZ"]
        self.categorical_features = [f for f in selected_features if f in all_categorical]
        self.continuous_features = [f for f in selected_features if f not in all_categorical]
        for must in ("N", "Z"):
            if must not in self.continuous_features:
                self.continuous_features.append(must)

        for col in self.continuous_features:
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

        N = self.data["original_N"].to_numpy(int)
        Z = self.data["original_Z"].to_numpy(int)
        y = self.data[target_column].to_numpy(float)

        # ------------------------------------------------------------
        # 3. Anchor context
        # ------------------------------------------------------------
        if self.use_anchor:
            if anchor_context is None:
                if split != "train":
                    raise ValueError(
                        "anchor_context must be supplied for val/test. Only the "
                        "train split may build its own dictionary."
                    )
                anchor_context = AnchorContext(
                    regime="train",
                    max_radius=anchor_max_radius,
                    min_neighbors=anchor_min_neighbors,
                )
                anchor_context.fit_baseline(N, Z, y)      # Stage 1 on train only
                anchor_context.set_dictionary(N, Z, y)    # dictionary = train rows
            self.anchor_context = anchor_context

            offset, anchor_feats = anchor_context.offset_and_features(N, Z)
            for name in ANCHOR_FEATURES:
                self.data[name] = anchor_feats[name]
                if name not in self.continuous_features:
                    self.continuous_features.append(name)
        else:
            self.anchor_context = None
            offset = np.zeros(len(self.data))

        self.offset = offset
        residual = y - offset
        self.data["_residual"] = residual

        # ------------------------------------------------------------
        # 4. Train statistics (computed on train only, reused elsewhere)
        # ------------------------------------------------------------
        if preprocess_stats is None:
            if split != "train":
                raise ValueError("preprocess_stats must be given for val/test.")

            cont_mean = self.data[self.continuous_features].mean()
            cont_std = self.data[self.continuous_features].std().replace(0, 1.0)
            t_mean = float(np.mean(residual))
            t_std = float(np.std(residual)) or 1.0

            category_maps = {c: self._ordered_categories(self.data[c])
                             for c in self.categorical_features}
            self.preprocess_stats = {
                "continuous_mean": cont_mean,
                "continuous_std": cont_std,
                "target_mean": t_mean,
                "target_std": t_std,
                "category_maps": category_maps,
                "category_sizes": {c: len(category_maps[c]) + 1
                                   for c in self.categorical_features},
                "continuous_features": list(self.continuous_features),
                "categorical_features": list(self.categorical_features),
                "use_anchor": self.use_anchor,
            }
        else:
            self.preprocess_stats = preprocess_stats

        self.target_mean = self.preprocess_stats["target_mean"]
        self.target_std = self.preprocess_stats["target_std"]

        # ------------------------------------------------------------
        # 5. Apply train statistics
        # ------------------------------------------------------------
        cm = self.preprocess_stats["continuous_mean"]
        cs = self.preprocess_stats["continuous_std"]
        self.data[self.continuous_features] = self.data[self.continuous_features].fillna(cm)
        self.data[self.continuous_features] = (self.data[self.continuous_features] - cm) / cs

        std_target = (residual - self.target_mean) / self.target_std

        # ------------------------------------------------------------
        # 6. Categorical encoding with the train-fitted maps
        # ------------------------------------------------------------
        maps = self.preprocess_stats["category_maps"]
        for col in self.categorical_features:
            encoded, n_unknown = self._encode_with_map(self.data[col], maps[col])
            self.data[col] = encoded
            if n_unknown:
                print(f"   [{split}] '{col}': {n_unknown}/{len(encoded)} unseen "
                      f"values mapped to reserved index {len(maps[col])}")

        self.category_sizes = {c: self.preprocess_stats["category_sizes"][c]
                               for c in self.categorical_features}

        # ------------------------------------------------------------
        # 7. Tensors
        # ------------------------------------------------------------
        self.categorical = (
            torch.tensor(self.data[self.categorical_features].values, dtype=torch.long)
            if self.categorical_features else None
        )
        self.continuous = torch.tensor(
            self.data[self.continuous_features].values, dtype=torch.float32)
        self.target = torch.tensor(std_target, dtype=torch.float32)

        if include_nz:
            self.N_values = torch.tensor(self.data["original_N"].values, dtype=torch.float32)
            self.Z_values = torch.tensor(self.data["original_Z"].values, dtype=torch.float32)
        self.offset_t = torch.tensor(self.offset, dtype=torch.float32)

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _ordered_categories(series):
        values = series.dropna().unique().tolist()
        try:
            return sorted(values, key=float)
        except (TypeError, ValueError):
            return sorted(values, key=str)

    @staticmethod
    def _encode_with_map(series, categories):
        mapping = {c: i for i, c in enumerate(categories)}
        unknown = len(categories)
        enc = series.map(mapping)
        n_unknown = int(enc.isna().sum())
        return enc.fillna(unknown).astype("int64"), n_unknown

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cat = (self.categorical[idx] if self.categorical_features
               else torch.tensor([], dtype=torch.long))
        if self.include_nz:
            return (cat, self.continuous[idx], self.target[idx],
                    self.N_values[idx], self.Z_values[idx], self.offset_t[idx])
        return cat, self.continuous[idx], self.target[idx]


# ------------------------------------------------------------------ helpers
def get_dataloader(csv_path, selected_features, batch_size, shuffle=True,
                   num_workers=0, split="train", include_nz=False,
                   preprocess_stats=None, anchor_context=None, use_anchor=True,
                   seed=42, **kw):
    ds = NuclearMassDataset(
        csv_path=csv_path, selected_features=selected_features, split=split,
        include_nz=include_nz, preprocess_stats=preprocess_stats,
        anchor_context=anchor_context, use_anchor=use_anchor, seed=seed, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_eval_dataloader(csv_path, selected_features, batch_size, shuffle=False,
                        split="val", preprocess_stats=None, anchor_context=None,
                        use_anchor=True, seed=42, **kw):
    ds = NuclearMassDataset(
        csv_path=csv_path, selected_features=selected_features, split=split,
        include_nz=True, preprocess_stats=preprocess_stats,
        anchor_context=anchor_context, use_anchor=use_anchor, seed=seed, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
