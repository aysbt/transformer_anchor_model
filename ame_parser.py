"""
Robust parser for AME2016 / AME2020 mass tables.

The first character of the AME files is a Fortran carriage-control column.
After removing that character, the first fields are:

    N-Z, N, Z, A, EL, [origin flag], MASS EXCESS, ...

AME2016 and AME2020 do not align N/Z/A identically in absolute character
positions, so parsing them with fixed slices can silently drop valid nuclei.
This parser therefore reads the leading fields token-wise.
"""

import re
import pandas as pd

_NUM = re.compile(r"^[+-]?\d+\.?\d*$")


def _first_number(tokens):
    """Return (value, is_extrapolated) for the first numeric token."""
    for tok in tokens:
        extrap = "#" in tok
        clean = tok.replace("#", "")
        if clean in ("", "*"):
            continue
        if _NUM.match(clean):
            return float(clean), extrap
    return None, None


def parse_mass_table(path, keep_extrapolated=False):
    """Parse an AME mass table into N, Z, A and mass excess."""
    rows = []

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if len(line) < 40:
                continue

            # Column 0 is Fortran carriage control.  Removing it makes the
            # leading integer fields stable across AME2016 and AME2020.
            tokens = line[1:].split()
            if len(tokens) < 6:
                continue

            try:
                nz = int(tokens[0])
                N = int(tokens[1])
                Z = int(tokens[2])
                A = int(tokens[3])
            except ValueError:
                continue  # header / comment

            # Strong sanity checks prevent accidental parsing of non-data lines.
            if A <= 0 or A != N + Z or nz != N - Z:
                continue

            # tokens[4] is the element symbol.  tokens[5:] starts with either
            # the optional origin flag or the mass excess itself.
            me, extrap = _first_number(tokens[5:])
            if me is None:
                continue
            if extrap and not keep_extrapolated:
                continue

            rows.append((N, Z, A, me, int(bool(extrap))))

    df = pd.DataFrame(
        rows,
        columns=["N", "Z", "A", "mass_excess_keV", "extrapolated"],
    )
    return df.drop_duplicates(subset=["N", "Z"], keep="first").reset_index(drop=True)


def validate(df, name=""):
    """Basic parser sanity checks."""
    refs = {
        (6, 6): (0.0, 1.0),
        (2, 2): (2424.9, 5.0),
        (30, 26): (-60605.4, 10.0),
        (126, 82): (-21748.6, 10.0),
        (146, 92): (47308.9, 30.0),
        # catches the AME2016 alignment bug that caused the false low-A test pile-up
        (10, 8): (-782.8, 5.0),       # 18O
        (10, 10): (-7041.9, 5.0),     # 20Ne
    }

    ok = True
    print(f"  validation [{name}]")
    for (N, Z), (expected, tol) in refs.items():
        hit = df[(df.N == N) & (df.Z == Z)]
        if hit.empty:
            print(f"    N={N:4d} Z={Z:3d}  MISSING")
            ok = False
            continue

        got = float(hit.mass_excess_keV.iloc[0])
        good = abs(got - expected) <= tol
        ok &= good
        print(
            f"    N={N:4d} Z={Z:3d}  got {got:12.3f}  "
            f"expected {expected:12.3f}  {'OK' if good else 'MISMATCH'}"
        )
    return ok


def build_datasets(path_old, path_new):
    """
    Chronological comparison used by the existing pipeline.

    train/val pool:
        nuclei measured in AME2016 and also measured in AME2020,
        using the AME2020 target value.

    test:
        nuclei measured in AME2020 but absent from the measured AME2016 set.
    """
    old = parse_mass_table(path_old)
    new = parse_mass_table(path_new)

    old_keys = set(zip(old.N, old.Z))

    tagged = new.copy()
    tagged["is_new"] = [
        (int(n), int(z)) not in old_keys
        for n, z in zip(tagged.N, tagged.Z)
    ]

    trainval = tagged[~tagged.is_new].drop(columns=["is_new"]).reset_index(drop=True)
    test = tagged[tagged.is_new].drop(columns=["is_new"]).reset_index(drop=True)

    return old, new, trainval, test