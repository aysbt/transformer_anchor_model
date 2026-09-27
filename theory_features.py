"""theory_features.py
====================
Theory-table-aware feature preparation.

The standard pipeline (prepare_data.add_physics_features) uses hardcoded
empirical magic numbers [2, 8, 20, 28, 50, 82, 126, 184].  For superheavy
nuclei this misses the sub-shell closures that FRDM95 / HFB-14 predict
(e.g. N=152, 162 for Z~114).  This module:

  1. Extracts theory-specific magic numbers from two-nucleon shell gaps
     D2n = M(N-2,Z) - 2M(N,Z) + M(N+2,Z)  (and D2p for protons).
  2. Merges them with the empirical list (empirical stays authoritative for
     light / medium nuclei; theory extends the heavy region).
  3. Provides add_theory_physics_features() — a drop-in replacement for
     add_physics_features() with the merged magic-number set.

Usage in run_extrapolation.py (see bottom of this file):
    import extrapolation_data as _ed
    from theory_features import TheoryFeatures
    tf = TheoryFeatures(df)             # df = full theory table
    _ed.add_physics_features = tf.add_features   # monkey-patch once
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prepare_data import MAGIC as EMPIRICAL_MAGIC

TARGET = "Mass Excess (keV)"


# ── shell-gap extraction ──────────────────────────────────────────────────────

def _mean_d2(df: pd.DataFrame, nucleon: str, min_chain: int = 3) -> dict[int, float]:
    """Mean two-nucleon shell gap as a function of N (nucleon='N') or Z (nucleon='Z').

    D2n(N,Z) = M(N-2,Z) - 2*M(N,Z) + M(N+2,Z)
    D2p(N,Z) = M(N,Z-2) - 2*M(N,Z) + M(N,Z+2)

    Only includes values where >= min_chain complementary nucleon numbers contribute.
    """
    other = "Z" if nucleon == "N" else "N"
    tbl = df.set_index(["N", "Z"])[TARGET]
    bucket: dict[int, list[float]] = {}
    for (n, z) in tbl.index:
        x = n if nucleon == "N" else z
        m1 = (n - 2, z) if nucleon == "N" else (n, z - 2)
        m2 = (n + 2, z) if nucleon == "N" else (n, z + 2)
        if m1 in tbl.index and m2 in tbl.index:
            bucket.setdefault(x, []).append(
                float(tbl[m1] - 2 * tbl[(n, z)] + tbl[m2])
            )
    return {x: float(np.mean(v)) for x, v in bucket.items() if len(v) >= min_chain}


def _local_peaks(d2: dict[int, float], half_win: int = 4) -> list[int]:
    """Return keys that are strict local maxima within a ±half_win window."""
    xs = sorted(d2)
    vals = [d2[x] for x in xs]
    peaks = []
    for i, x in enumerate(xs):
        lo = max(0, i - half_win)
        hi = min(len(xs) - 1, i + half_win)
        if vals[i] == max(vals[lo: hi + 1]):
            peaks.append(x)
    return peaks


def extract_theory_magic(
    df: pd.DataFrame,
    nucleon: str = "N",
    threshold_keV: float = 1500.0,
    half_win: int = 4,
    min_chain: int = 3,
) -> list[int]:
    """Shell closures predicted by the theory table for one nucleon type.

    Combines two criteria:
      * mean D2 > threshold_keV  (strong gap)
      * local maximum within ±half_win (structural peak, not a plateau edge)
    """
    d2 = _mean_d2(df, nucleon, min_chain)
    peaks = set(_local_peaks(d2, half_win))
    return sorted(x for x, v in d2.items() if v > threshold_keV and x in peaks)


def merged_magic(
    df: pd.DataFrame,
    threshold_keV: float = 800.0,
    heavy_cut: int = 82,
) -> tuple[list[int], list[int]]:
    """Return (magic_N, magic_Z) = empirical list ∪ theory-predicted closures.

    Theory closures are extracted from the HEAVY region only (Z > heavy_cut)
    with a lower threshold, then only values above the heavy_cut are added.
    This leaves light / medium nucleus features unchanged (empirical stays
    authoritative there) while capturing FRDM95 / HFB-14 sub-shell closures
    like N=152, N=162, Z=114 in the superheavy region.
    """
    heavy = df[df.Z > heavy_cut]
    theory_n = extract_theory_magic(heavy, "N", threshold_keV)
    theory_z = extract_theory_magic(heavy, "Z", threshold_keV)
    # Only add values beyond the well-established empirical shell structure
    extra_n = {n for n in theory_n if n > heavy_cut}
    extra_z = {z for z in theory_z if z > heavy_cut}
    magic_n = sorted(set(EMPIRICAL_MAGIC) | extra_n)
    magic_z = sorted(set(EMPIRICAL_MAGIC) | extra_z)
    return magic_n, magic_z


# ── theory-aware feature computation ─────────────────────────────────────────

def _shell_index(x: float, magic: list[int]) -> int:
    return int(np.searchsorted(magic, x, side="right"))


def _particles_holes(x: float, magic: list[int]) -> tuple[float, float]:
    lo, hi = 0, magic[-1]
    for m in magic:
        if x >= m:
            lo = m
        else:
            hi = m
            break
    return float(x - lo), float(hi - x)


def _valence(x: float, magic: list[int]) -> float:
    p, h = _particles_holes(x, magic)
    return float(min(p, h))


def _dist_to_magic(x: float, magic: list[int]) -> float:
    return float(min(abs(x - m) for m in magic))


def add_theory_physics_features(
    df: pd.DataFrame,
    magic_n: list[int],
    magic_z: list[int],
) -> pd.DataFrame:
    """Drop-in replacement for prepare_data.add_physics_features.

    Identical liquid-drop / composition terms; shell-structure features
    (Nshell_category, Zshell_category, deltaN, deltaZ, promiscuity)
    use the supplied magic-number lists instead of the hardcoded empirical list.
    """
    df = df.copy()
    N = df["N"].astype(float)
    Z = df["Z"].astype(float)
    A = df["A"].astype(float)

    # ── composition / liquid-drop terms (unchanged) ───────────────────────────
    df["A^2/3"]        = A ** (2.0 / 3.0)
    df["Z(Z-1)/A^1/3"] = Z * (Z - 1) / A ** (1.0 / 3.0)
    df["(N-Z)^2/A"]    = (N - Z) ** 2 / A
    df["(N-Z)/A"]      = (N - Z) / A
    df["N/Z"]          = np.where(Z > 0, N / np.maximum(Z, 1e-9), 0.0)
    df["ZEO"]          = (df["Z"] % 2).astype(int)
    df["NEO"]          = (df["N"] % 2).astype(int)

    # ── theory-derived shell-structure features ───────────────────────────────
    df["Zshell_category"] = df["Z"].apply(lambda z: _shell_index(z, magic_z))
    df["Nshell_category"] = df["N"].apply(lambda n: _shell_index(n, magic_n))
    df["deltaZ"]          = df["Z"].apply(lambda z: _dist_to_magic(z, magic_z))
    df["deltaN"]          = df["N"].apply(lambda n: _dist_to_magic(n, magic_n))

    ph_z = df["Z"].apply(lambda z: _particles_holes(z, magic_z))
    ph_n = df["N"].apply(lambda n: _particles_holes(n, magic_n))
    df["proton_particles"]  = [p for p, _ in ph_z]
    df["proton_holes"]      = [h for _, h in ph_z]
    df["neutron_particles"] = [p for p, _ in ph_n]
    df["neutron_holes"]     = [h for _, h in ph_n]

    nu_z  = df["Z"].apply(lambda z: _valence(z, magic_z))
    nu_n  = df["N"].apply(lambda n: _valence(n, magic_n))
    denom = nu_z + nu_n
    df["nu_Z"]       = nu_z
    df["nu_N"]       = nu_n
    df["promiscuity"] = np.where(
        denom > 0, nu_z * nu_n / np.maximum(denom, 1e-9), 0.0
    )
    return df


# ── public adapter ────────────────────────────────────────────────────────────

class TheoryFeatures:
    """Compute and cache theory-derived magic numbers; provide add_features().

    Parameters
    ----------
    df            : full theory table (N, Z, A, 'Mass Excess (keV)')
    threshold_keV : minimum mean D2n/D2p to count as a shell closure (default 1500)

    Quick start
    -----------
    import extrapolation_data as _ed
    from theory_features import TheoryFeatures

    tf = TheoryFeatures(df_theory)
    _ed.add_physics_features = tf.add_features   # one-line monkey-patch
    """

    def __init__(self, df: pd.DataFrame, threshold_keV: float = 800.0) -> None:
        self.magic_n, self.magic_z = merged_magic(df, threshold_keV)
        added_n = sorted(set(self.magic_n) - set(EMPIRICAL_MAGIC))
        added_z = sorted(set(self.magic_z) - set(EMPIRICAL_MAGIC))
        print(f"[TheoryFeatures] empirical magic  : {EMPIRICAL_MAGIC}")
        print(f"[TheoryFeatures] theory-added N   : {added_n}")
        print(f"[TheoryFeatures] theory-added Z   : {added_z}")
        print(f"[TheoryFeatures] final magic_N    : {self.magic_n}")
        print(f"[TheoryFeatures] final magic_Z    : {self.magic_z}")

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop-in for prepare_data.add_physics_features."""
        return add_theory_physics_features(df, self.magic_n, self.magic_z)

    def summary(self) -> pd.DataFrame:
        """Return a small table comparing empirical vs theory magic numbers."""
        rows = []
        all_vals = sorted(set(EMPIRICAL_MAGIC) | set(self.magic_n) | set(self.magic_z))
        for v in all_vals:
            rows.append({
                "value": v,
                "empirical_N": v in EMPIRICAL_MAGIC,
                "theory_N":    v in self.magic_n,
                "empirical_Z": v in EMPIRICAL_MAGIC,
                "theory_Z":    v in self.magic_z,
            })
        return pd.DataFrame(rows)
