"""Region splits and train-only preprocessing for the extrapolation study.

Separations (letters follow the referee's reference figure):
  random        (a) random hold-out, reference only
  neutron_rich  (b) most neutron-rich tail_fraction of every isotopic chain
  heavy         (c) all nuclei with Z > z_cut (heavier than lead)
  unmeasured    theory tables only: train on nuclei measured in AME,
                predict every other nucleus of the table

Anchor dictionaries (the pool of known masses a split may read):
  train -> train (leave-one-out through the (0,0) drop in anchor.py)
  val   -> train
  test  -> development = train + val. Never the test nuclei.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from anchor import LSMFBaseline, LocalResidualAnchor, ANCHOR_FEATURES
from prepare_data import add_physics_features, add_neighbor_existence

TARGET = 'Mass Excess (keV)'
FEATURES = ['N', 'Z', 'A', 'Zshell_category', 'Nshell_category', 'ZEO', 'NEO',
            'deltaN', 'deltaZ', 'A^2/3', 'Z(Z-1)/A^1/3', '(N-Z)^2/A',
            '(N-Z)/A', 'N/Z', 'promiscuity', 'neighbor_N_plus_1_exists',
            'neighbor_N_minus_1_exists', 'neighbor_Z_plus_1_exists',
            'neighbor_Z_minus_1_exists']          # = AnchoredFullModel in train.py
CATEGORICAL = ['Zshell_category', 'Nshell_category', 'ZEO', 'NEO', 'deltaN', 'deltaZ']
SEPARATIONS = ['random', 'neutron_rich', 'heavy', 'unmeasured']

# AME2020 mass excesses (keV) used to catch a misread theory table
REFERENCE = {(8, 8): -4737.0, (30, 26): -60607.0, (126, 82): -21749.0,
             (146, 92): 47309.0}


def load_table(path, fmt='csv', min_nz=8):
    if fmt == 'csv':
        df = pd.read_csv(path)
        df = df.rename(columns={'mass_excess_keV': TARGET})
    elif fmt == 'ame':
        from ame_parser import parse_mass_table
        df = parse_mass_table(path).rename(columns={'mass_excess_keV': TARGET})
    else:
        # IAEA RIPL-3 mass-hfb14.dat / mass-frdm95.dat:
        # (2i4,1x,a2,1x,a1,3f10.3,...) = Z, A, El, fl, Mexp, Err, Mth  [MeV]
        # Mth, not Mexp, is the synthetic target.
        rows = []
        for line in open(path):
            try:
                z, a = int(line[:4]), int(line[4:8])
            except ValueError:
                continue  # textual header
            try:
                mass = float(line[33:43])
            except ValueError as exc:
                raise ValueError(f'Invalid theoretical Mth record: {line!r}') from exc
            rows.append({'Z': z, 'N': a-z, 'A': a, TARGET: mass*1000})
        df = pd.DataFrame(rows)
    if 'A' not in df and {'N', 'Z'} <= set(df):
        df['A'] = df.N + df.Z
    required = ['N', 'Z', 'A', TARGET]
    if not set(required) <= set(df):
        raise ValueError(f'Required columns: {required}')
    df = df[required].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError('Nonfinite input values')
    if not (df[['N', 'Z', 'A']] % 1 == 0).all().all():
        raise ValueError('N, Z, A must be integers')
    if not (df.A == df.N + df.Z).all() or (df[['N','Z']] < 0).any().any():
        raise ValueError('Invalid nuclear coordinates')
    if df.duplicated(['N', 'Z']).any():
        raise ValueError('Duplicate nuclei: supply one consistent mass table')
    df = df[(df.N >= min_nz) & (df.Z >= min_nz)].copy()
    if df.empty:
        raise ValueError('No nuclei survive the selection')
    df[['N', 'Z', 'A']] = df[['N', 'Z', 'A']].astype(int)
    return df.sort_values(['Z', 'N']).reset_index(drop=True)


def sanity_check(df, tol_keV=3000.):
    """A theory table must be within a few MeV of AME for well-known nuclei."""
    table = dict(zip(zip(df.N, df.Z), df[TARGET]))
    for key, ref in REFERENCE.items():
        if key in table and abs(table[key]-ref) > tol_keV:
            raise ValueError(f'(N,Z)={key}: {table[key]:.0f} keV vs AME {ref:.0f} keV. '
                             'Wrong column or units?')


def split_table(df, separation, tail_fraction=.20, z_cut=82, val_fraction=.30,
                split_seed=42, measured=None):
    if not 0 < tail_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError('Fractions must be strictly between zero and one')
    if separation == 'random':
        mask = df.index.isin(df.sample(frac=tail_fraction, random_state=split_seed).index)
    elif separation == 'neutron_rich':
        test_ids = []
        for _, chain in df.groupby('Z'):
            # Single-row chains remain in development; retain >=1 row per chain.
            k = min(len(chain)-1, int(np.ceil(len(chain)*tail_fraction)))
            if k:
                test_ids.extend(chain.nlargest(k, 'N').index)
        mask = df.index.isin(test_ids)
    elif separation == 'heavy':
        mask = df.Z > z_cut
    elif separation == 'unmeasured':
        if measured is None:
            raise ValueError('unmeasured separation needs the measured AME table')
        known = set(zip(measured.N, measured.Z))
        mask = ~np.array([k in known for k in zip(df.N, df.Z)])
    else:
        raise ValueError(f'separation must be one of {SEPARATIONS}')
    dev, test = df.loc[~mask], df.loc[mask]
    if len(dev) < 4 or test.empty:
        raise ValueError('Split needs at least 4 development nuclei and a nonempty test')
    # Random validation ONLY inside development. No rare-bin rows discarded.
    train, val = train_test_split(dev, test_size=val_fraction, random_state=split_seed)
    return {k: v.sort_values(['Z','N']).reset_index(drop=True)
            for k, v in [('train', train), ('val', val), ('test', test)]}


class ResidualPreprocessor:
    """Stage 1 always; Stage 2 (local anchor) only when use_anchor=True."""

    def __init__(self, use_anchor=False, max_radius=3, min_neighbors=14):
        self.use_anchor = use_anchor
        self.max_radius = max_radius
        self.min_neighbors = min_neighbors

    def fit(self, train):
        self.baseline = LSMFBaseline().fit(train.N, train.Z, train[TARGET])
        f, offset = self.features(train, pool=train)
        self.cat = [x for x in FEATURES if x in CATEGORICAL]
        self.cont = [x for x in FEATURES if x not in CATEGORICAL]
        if self.use_anchor:
            self.cont += ANCHOR_FEATURES
        self.maps = {c: sorted(f[c].unique().tolist()) for c in self.cat}
        self.mean = f[self.cont].mean()
        self.std = f[self.cont].std().replace(0, 1).fillna(1)
        residual = train[TARGET].to_numpy() - offset
        self.target_mean = float(residual.mean())
        self.target_std = float(residual.std()) or 1.
        return self

    def features(self, df, pool):
        """pool = nuclei whose masses this split is allowed to read."""
        f = add_neighbor_existence(add_physics_features(df),
                                   set(zip(pool.N.astype(int), pool.Z.astype(int))))
        offset = self.baseline.predict(df.N, df.Z)
        if self.use_anchor:
            anchor = LocalResidualAnchor(self.max_radius, self.min_neighbors).fit(
                pool.N, pool.Z, pool[TARGET].to_numpy() - self.baseline.predict(pool.N, pool.Z))
            feats = anchor.predict(df.N.to_numpy(int), df.Z.to_numpy(int))
            for name in ANCHOR_FEATURES:
                f[name] = feats[name]
            offset = offset + feats['anchor']
        else:
            f['anchor_has'] = 0.
        return f, offset

    def transform(self, df, pool):
        f, offset = self.features(df, pool)
        cat = np.column_stack([f[c].map({v:i for i,v in enumerate(self.maps[c])})
                               .fillna(len(self.maps[c])).to_numpy(int) for c in self.cat])
        cont = ((f[self.cont]-self.mean)/self.std).to_numpy(np.float32)
        y = (df[TARGET].to_numpy()-offset-self.target_mean)/self.target_std
        return cat, cont, y.astype(np.float32), offset, f['anchor_has'].to_numpy() > 0


def distances(query, known):
    ref = known[['N','Z']].to_numpy(float)
    return np.concatenate([np.sqrt(((b[:,None,:]-ref[None,:,:])**2).sum(2)).min(1)
                           for b in np.array_split(query[['N','Z']].to_numpy(float),
                                                   max(1, int(np.ceil(len(query)/256))))])
