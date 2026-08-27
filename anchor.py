"""
anchor.py
=========

Stage 1 (LSMF)   : a least-squares mass formula fitted to the training split.
Stage 2 (anchor) : a leak-free local estimate of the Stage-1 residual field.

The network then only has to learn

    residual = experimental_mass_excess - LSMF - anchor

and the reconstruction is

    predicted_mass_excess = LSMF + anchor + predicted_residual


Why this is not "residual learning off a theory model"
------------------------------------------------------
The LSMF coefficients are obtained by ordinary least squares ON THE TRAINING
SPLIT. Nothing is imported from FRDM, WS4, HFB or any other published mass model,
so the pipeline never inherits another model's assumptions or its errors. The
direct-prediction framing of the study survives intact.


Leak control -- read this before changing anything
--------------------------------------------------
There are TWO distinct leaks and they need separate defences.

  Leak A -- self label.
      Nucleus X reads X's own mass.
      Defence: LocalResidualAnchor.predict() ALWAYS discards the exact
      (dN, dZ) = (0, 0) match. This makes training anchors leave-one-out by
      construction, so the anchor feature has the same meaning and the same
      difficulty at fit time as at predict time. Never remove that check.

  Leak B -- mutual support between held-out nuclei.
      Nucleus X reads neighbour Y, where Y is also held out and would not have
      been known at prediction time.
      Defence: the CALLER chooses the dictionary. This module never decides for
      you. Use AnchorContext with an explicit regime:

          "train"       dictionary = training split only
          "trainval"    dictionary = training + validation
          "historical"  dictionary = AME2016 masses only        <- prospective
          "loo"         dictionary = everything except the target itself

      "historical" is the defensible protocol for the AME2016 -> AME2020 test.
      "loo" is leave-one-out interpolation and answers a different, easier
      question; report it separately and label it as such.

Diagnostics returned per nucleus (all usable as network features):
    anchor              local estimate of the residual field, 0 where unavailable
    anchor_has          1 if any allowed neighbour was found
    anchor_n            number of neighbours used
    anchor_mean_dist    mean euclidean distance of those neighbours
    anchor_max_dist     max euclidean distance of those neighbours
    anchor_scatter      weighted RMS of the local fit residual (difficulty proxy)
    anchor_parity_match fraction of neighbours sharing the query's (N,Z) parity
"""

import numpy as np

SEMF_TERMS = ["A", "A23", "coulomb", "asym", "pair_term", "A13", "NZ_over_A"]

ANCHOR_FEATURES = [
    "anchor",
    "anchor_has",
    "anchor_n",
    "anchor_mean_dist",
    "anchor_max_dist",
    "anchor_scatter",
    "anchor_parity_match",
]


# --------------------------------------------------------------- Stage 1
def semf_design(N, Z):
    """Small, smooth, physics-shaped basis. Deliberately only 8 columns."""
    N = np.asarray(N, dtype=float)
    Z = np.asarray(Z, dtype=float)
    A = np.maximum(N + Z, 1e-9)

    oddN, oddZ = np.mod(N, 2), np.mod(Z, 2)
    pair_sign = np.where((oddN == 0) & (oddZ == 0), 1.0,
                np.where((oddN == 1) & (oddZ == 1), -1.0, 0.0))

    cols = {
        "A": A,
        "A23": A ** (2.0 / 3.0),
        "coulomb": Z * (Z - 1) / A ** (1.0 / 3.0),
        "asym": (N - Z) ** 2 / A,
        "pair_term": pair_sign / np.sqrt(A),
        "A13": A ** (1.0 / 3.0),
        "NZ_over_A": (N - Z) / A,
    }
    X = np.column_stack([cols[t] for t in SEMF_TERMS])
    return np.column_stack([np.ones(len(X)), X])


class LSMFBaseline:
    """Ordinary least squares mass formula. Eight coefficients, cannot overfit."""

    def __init__(self):
        self.coef_ = None

    def fit(self, N, Z, mass_excess):
        self.coef_, *_ = np.linalg.lstsq(semf_design(N, Z),
                                         np.asarray(mass_excess, float), rcond=None)
        return self

    def predict(self, N, Z):
        return semf_design(N, Z) @ self.coef_

    def state_dict(self):
        return {"coef": self.coef_}

    def load_state_dict(self, state):
        self.coef_ = np.asarray(state["coef"], dtype=float)
        return self


# --------------------------------------------------------------- Stage 2
class LocalResidualAnchor:
    """
    Weighted local linear fit of a scalar field on the integer (N, Z) lattice.

    Besides the constant and the two displacements, the local design matrix
    carries two parity-mismatch indicators. These let the fit absorb any odd-even
    staggering left in the residual field; evaluating at mismatch = 0 then returns
    the value appropriate to the query's own parity.
    """

    def __init__(self, max_radius=3, min_neighbors=14):
        self.max_radius = max_radius
        self.min_neighbors = min_neighbors
        self.field = {}

    def fit(self, N, Z, values):
        self.field = {(int(n), int(z)): float(v)
                      for n, z, v in zip(N, Z, values) if np.isfinite(v)}
        return self

    def _neighbors(self, n, z):
        found = []
        for radius in range(1, self.max_radius + 1):
            for dn in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if max(abs(dn), abs(dz)) != radius:
                        continue                      # only the newly added ring
                    if dn == 0 and dz == 0:
                        continue                      # LEAK A DEFENCE -- keep this
                    val = self.field.get((n + dn, z + dz))
                    if val is not None:
                        found.append((dn, dz, val))
            if len(found) >= self.min_neighbors:
                break
        return found

    def predict(self, N, Z):
        n_q = len(N)
        out = {k: np.zeros(n_q) for k in ANCHOR_FEATURES}

        for i, (n, z) in enumerate(zip(N, Z)):
            n, z = int(n), int(z)
            nb = self._neighbors(n, z)
            if not nb:
                continue                               # anchor stays 0, has = 0

            dn = np.array([b[0] for b in nb], dtype=float)
            dz = np.array([b[1] for b in nb], dtype=float)
            val = np.array([b[2] for b in nb], dtype=float)

            dist = np.hypot(dn, dz)
            scale = max(dist.min(), 1.0)
            w = np.exp(-0.5 * (dist / (1.5 * scale)) ** 2)

            pn, pz = np.mod(dn, 2), np.mod(dz, 2)      # parity mismatch flags

            X = np.column_stack([np.ones_like(dn), dn, dz, pn, pz])
            X = X[:, :min(X.shape[1], max(1, len(nb) - 1))]

            sw = np.sqrt(w)
            try:
                beta, *_ = np.linalg.lstsq(X * sw[:, None], val * sw, rcond=None)
                anchor = float(beta[0])                # value at dN=dZ=0, parity matched
                scatter = float(np.sqrt(np.average((val - X @ beta) ** 2, weights=w)))
            except np.linalg.LinAlgError:
                anchor = float(np.average(val, weights=w))
                scatter = float(np.sqrt(np.average((val - anchor) ** 2, weights=w)))

            out["anchor"][i] = anchor
            out["anchor_has"][i] = 1.0
            out["anchor_n"][i] = len(nb)
            out["anchor_mean_dist"][i] = dist.mean()
            out["anchor_max_dist"][i] = dist.max()
            out["anchor_scatter"][i] = scatter
            out["anchor_parity_match"][i] = float(np.mean((pn == 0) & (pz == 0)))

        return out


# ------------------------------------------------------- dictionary control
REGIMES = ("train", "trainval", "historical", "loo")


class AnchorContext:
    """
    Bundles the Stage-1 baseline with an anchor field built from an EXPLICIT
    dictionary of allowed known masses.

    Parameters
    ----------
    regime : one of REGIMES -- recorded so every produced number is auditable.
    """

    def __init__(self, regime="train", max_radius=3, min_neighbors=14):
        if regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
        self.regime = regime
        self.max_radius = max_radius
        self.min_neighbors = min_neighbors
        self.baseline = LSMFBaseline()
        self.anchor = LocalResidualAnchor(max_radius, min_neighbors)

    def fit_baseline(self, N, Z, mass_excess):
        """Stage 1 must always be fitted on the TRAINING split only."""
        self.baseline.fit(N, Z, mass_excess)
        self.baseline_rmse_ = float(np.sqrt(np.mean(
            (np.asarray(mass_excess, float) - self.baseline.predict(N, Z)) ** 2)))
        return self

    def set_dictionary(self, N, Z, mass_excess):
        """Install the allowed known-mass dictionary for this regime."""
        resid = np.asarray(mass_excess, float) - self.baseline.predict(N, Z)
        self.anchor.fit(N, Z, resid)
        self.dict_size_ = len(self.anchor.field)
        return self

    def offset_and_features(self, N, Z):
        """
        Returns
        -------
        offset : LSMF + anchor, in keV -- what the network does NOT have to learn
        feats  : dict of anchor diagnostics, all usable as network inputs
        """
        base = self.baseline.predict(N, Z)
        feats = self.anchor.predict(np.asarray(N), np.asarray(Z))
        return base + feats["anchor"], feats

    def state_dict(self):
        return {
            "regime": self.regime,
            "max_radius": self.max_radius,
            "min_neighbors": self.min_neighbors,
            "baseline": self.baseline.state_dict(),
            "field": self.anchor.field,
        }

    @classmethod
    def from_state_dict(cls, state):
        ctx = cls(state["regime"], state["max_radius"], state["min_neighbors"])
        ctx.baseline.load_state_dict(state["baseline"])
        ctx.anchor.field = state["field"]
        ctx.dict_size_ = len(ctx.anchor.field)
        return ctx
