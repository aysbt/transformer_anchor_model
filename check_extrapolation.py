"""Data-path regression checks. Artificial fixture; not physics performance."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from extrapolation_data import (load_table, sanity_check, split_table,
                                ResidualPreprocessor, TARGET)


def main():
    df = pd.DataFrame([{'N':n,'Z':z,'A':n+z,TARGET:float(n*n-z*z+3*n*z)}
                       for z in range(70,91) for n in range(z,z+20)])
    measured = df[df.N < df.Z+12]
    for mode in ['random','neutron_rich','heavy','unmeasured']:
        parts = split_table(df,mode,measured=measured)
        keys = {k:set(zip(v.N,v.Z)) for k,v in parts.items()}
        assert not keys['train'] & keys['test']
        assert not keys['train'] & keys['val']
        assert not keys['val'] & keys['test']
        assert len(set.union(*keys.values())) == len(df)
        dev = pd.concat([parts['train'],parts['val']])
        if mode == 'neutron_rich':
            for z, group in parts['test'].groupby('Z'):
                assert group.N.min() > dev.loc[dev.Z == z,'N'].max()
        elif mode == 'heavy':
            assert dev.Z.max() <= 82 and parts['test'].Z.min() > 82
        elif mode == 'unmeasured':
            assert keys['train'] | keys['val'] == set(zip(measured.N, measured.Z))
        for use_anchor in [False, True]:
            prep = ResidualPreprocessor(use_anchor).fit(parts['train'])
            cat,cont,y,offset,has = prep.transform(parts['test'], dev)
            np.testing.assert_allclose(y*prep.target_std+prep.target_mean+offset,
                                       parts['test'][TARGET],rtol=1e-6,atol=.01)
            # test masses must never reach the inputs or the offset
            altered = parts['test'].copy(); altered[TARGET] += 999999
            cat2,cont2,_,offset2,_ = prep.transform(altered, dev)
            np.testing.assert_array_equal(cat,cat2)
            np.testing.assert_array_equal(cont,cont2)
            np.testing.assert_array_equal(offset,offset2)
            # changing a development mass must move the anchored offset only
            shifted = dev.copy(); shifted[TARGET] += 1000.
            _,_,_,offset3,_ = prep.transform(parts['test'], shifted)
            assert np.any(offset3 != offset) == (use_anchor and has.any())
            assert any('anchor' in c for c in prep.cont) == use_anchor
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)/'fixture.dat'
        p.write_text('header\n'+f'{8:4d}{16:4d} O  2{99.:10.3f}{.1:10.3f}{-4.737:10.3f}\n')
        for fmt in ['hfb14','frdm95']:
            loaded = load_table(p,fmt)
            assert loaded.iloc[0][TARGET] == -4737.
            sanity_check(loaded)
        try:
            sanity_check(loaded.assign(**{TARGET: -4.737}))  # MeV read as keV
        except ValueError:
            pass
        else:
            raise AssertionError('Unit error not caught')
        p = Path(d)/'duplicates.csv'
        pd.concat([df,df.iloc[:1]]).to_csv(p,index=False)
        try:
            load_table(p)
        except ValueError:
            pass
        else:
            raise AssertionError('Duplicate nuclei accepted')
    print('PASS: disjoint/complete splits for all separations, directional holdouts, '
          'target isolation with and without anchor, residual reconstruction, '
          'theoretical Mth parsing and unit check, duplicate rejection')

if __name__ == '__main__':
    main()
