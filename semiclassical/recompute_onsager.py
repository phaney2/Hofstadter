"""
Re-solve the Onsager condition from a saved `*_detail.mat` file.

The `onsager_bfield` stage writes its per-B intermediates (orbit areas,
enclosed Berry curvature, energy grids) to `<out>_detail.mat`.  Those are
the complete inputs to the quantization step, so the LL fan can be rebuilt
without redoing the expensive contour work.

Main use: sweep the Berry curvature prefactor (`--bc-factor`, which is
`term_factors[0]`) — including a sign flip — and see how the levels move.

    python recompute_onsager.py onsager_12_2_detail.mat \
           --ref onsager_12_2.mat --bc-factor 1,-1 --out bcsign.mat

With `--ref` the reference file supplies `nmax` and `onsager_Bmultiplier`
and each recomputed set is diffed against it.  Passing a single
`--bc-factor` reproduces the original key names exactly; passing several
appends `_bcf0`, `_bcf1`, ... in the order given.

`--bc-sign-mode` controls the parity of the Berry phase in B.  `onsager.py`
writes the term as `-f * gamma * |B| / (2*pi*phi0)`, which — after the
condition is divided through by sign(B) — shifts the level index by
`-f * gamma * sign(B) / (2*pi)`: **odd in B**, the production convention
(`odd`, the default).  `fixed` removes that parity so the same sign is used
on both field branches.  Because `B = sign(B)*|B|`, this needs no change to
the solver: passing a per-field `f * sign(B)` is algebraically identical to
replacing `|B|` with `B` in the `base_SB` line.  Diagnostic only — see
"Open issues" in `notes_onsager.tex`.

    python recompute_onsager.py onsager_25_10_unfold_detail.mat \
           --bc-sign-mode fixed --bc-factor 1,-1 --out bcfixed.mat
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import re
import sys

import numpy as np
from scipy.io import loadmat, whosmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onsager import onsager_fan_band
from semiclassical import save_result, load_data


DETAIL_PREFIXES = ('area_', 'enclosedBC_', 'E_levels_')


def scan_detail(path):
    """Return (list of top-level names, {(valley, band)} present)."""
    names = [n for n, _, _ in whosmat(path)]
    combos = set()
    for n in names:
        m = re.match(r'area_(K|Kp)_band(\d+)$', n)
        if m:
            combos.add((m.group(1), int(m.group(2))))
    return names, combos


def read_band(path, valley, band, cache):
    """Load one band's detail arrays, keeping peak memory to a few tens of MB."""
    keys = [f'area_{valley}_band{band}',
            f'enclosedBC_{valley}_band{band}',
            f'E_levels_{valley}_band{band}']
    if cache is not None:
        return tuple(cache[k] for k in keys)
    d = loadmat(path, variable_names=keys)
    return d[keys[0]], d[keys[1]], d[keys[2]]


def fan_for_band(Blist, area, enclosedBC, E_levels, nmax, bc_factor,
                 Bmultiplier, lifshitz_threshold, bc_sign_mode='odd'):
    """Rebuild the (nB, nmax+1) LL arrays for one band, one BC factor.

    Mirrors `_onsager_bfield_worker`: dL/dE is identically zero (the orbital
    moment is already in the dispersion), so the `S` -> `SM` and `SB` -> `SBM`
    renaming applies and the solver's own `SBM` output is discarded.

    Under `bc_sign_mode='fixed'` the factor handed to the solver is scaled by
    sign(B), which cancels the |B| in the solver's Berry term and leaves the
    phase shift with the same sign on both field branches.
    """
    nB = len(Blist)
    nE = E_levels.shape[1]
    zero_dLdE = np.zeros(nE)

    per_B = []
    suffixes = set()
    for iB, B in enumerate(Blist):
        f = bc_factor * np.sign(B) if bc_sign_mode == 'fixed' else bc_factor
        ll = onsager_fan_band([B], nmax, E_levels[iB], area[iB],
                              enclosedBC[iB], zero_dLdE,
                              term_factors=(f, 0.0, 1.0),
                              Bmultiplier=Bmultiplier,
                              lifshitz_threshold=lifshitz_threshold)
        if ll is None:
            per_B.append(None)
            continue
        renamed = {}
        for k, v in ll.items():
            if k.startswith('S_seg') or k == 'S':
                renamed['SM' + k[1:]] = v[0]
            elif k.startswith('SB_seg') or k == 'SB':
                renamed['SBM' + k[2:]] = v[0]
        per_B.append(renamed)
        suffixes.update(renamed.keys())

    out = {s: np.full((nB, nmax + 1), np.nan) for s in suffixes}
    for iB, renamed in enumerate(per_B):
        if renamed is None:
            continue
        for s, row in renamed.items():
            out[s][iB] = row
    return out


def compare(mine, ref_struct, valley, band, tag):
    """Diff one band's recomputed fan against the reference file."""
    prefix = f'LL_{valley}_band{band}_'
    ref_keys = {k[len(prefix):] for k in ref_struct._fieldnames
                if k.startswith(prefix)}
    mine_keys = set(mine)

    worst = 0.0
    nan_mismatch = 0
    for s in sorted(mine_keys & ref_keys):
        a = mine[s]
        b = np.asarray(getattr(ref_struct, prefix + s), dtype=float)
        if a.shape != b.shape:
            print(f'    {tag} {valley} band{band} {s}: shape {a.shape} vs {b.shape}')
            continue
        na, nb = np.isnan(a), np.isnan(b)
        nan_mismatch += int(np.sum(na != nb))
        both = ~na & ~nb
        if both.any():
            worst = max(worst, float(np.max(np.abs(a[both] - b[both]))))

    return {'worst': worst, 'nan_mismatch': nan_mismatch,
            'only_mine': sorted(mine_keys - ref_keys),
            'only_ref': sorted(ref_keys - mine_keys)}


def main():
    ap = argparse.ArgumentParser(
        description='Re-solve the Onsager condition from onsager_bfield '
                    'detail data.')
    ap.add_argument('detail', help='path to <name>_detail.mat')
    ap.add_argument('--ref', default=None,
                    help='original fan .mat; supplies nmax/Bmultiplier '
                         'defaults and is diffed against the result')
    ap.add_argument('--out', default=None,
                    help='output .mat (default <detail base minus _detail>'
                         '_recomp.mat; use "none" to only compare)')
    ap.add_argument('--bc-factor', default='1',
                    help='comma-separated Berry curvature prefactors '
                         '(term_factors[0]).  Default 1.  Use "1,-1" to '
                         'compare both signs.')
    ap.add_argument('--bc-sign-mode', choices=('odd', 'fixed'), default='odd',
                    help='parity of the Berry phase in B.  "odd" (default) '
                         'is the production convention, where the phase '
                         'shift carries sign(B).  "fixed" applies the same '
                         'sign on both field branches.')
    ap.add_argument('--nmax', type=int, default=None)
    ap.add_argument('--Bmultiplier', type=float, default=None)
    ap.add_argument('--lifshitz-threshold', type=float, default=50.0)
    ap.add_argument('--bands', default=None,
                    help='comma-separated band indices (default: all)')
    ap.add_argument('--valleys', default='K,Kp')
    ap.add_argument('--in-memory', action='store_true',
                    help='load the whole detail file once instead of '
                         'per-band (faster, needs several GB)')
    args = ap.parse_args()

    bc_factors = [float(x) for x in args.bc_factor.split(',')]
    valleys = [v.strip() for v in args.valleys.split(',') if v.strip()]

    ref_struct = None
    nmax, Bmultiplier = args.nmax, args.Bmultiplier
    if args.ref:
        ref_struct = loadmat(args.ref, squeeze_me=True,
                             struct_as_record=False)['results']
        if nmax is None:
            nmax = int(ref_struct.nmax)
        if Bmultiplier is None:
            Bmultiplier = float(ref_struct.onsager_Bmultiplier)
    if nmax is None:
        nmax = 50
    if Bmultiplier is None:
        Bmultiplier = 4.0

    names, combos = scan_detail(args.detail)
    if not combos:
        raise SystemExit(
            f'{args.detail} has no area_<valley>_band<n> arrays.  '
            'Is it really an onsager_bfield *_detail.mat file?')

    Blist = loadmat(args.detail, variable_names=['Blist'])['Blist'].ravel()
    cache = load_data(args.detail) if args.in_memory else None

    all_bands = sorted({b for _, b in combos})
    bands = ([int(x) for x in args.bands.split(',')] if args.bands
             else all_bands)

    print(f'  detail   : {args.detail}')
    print(f'  nB={len(Blist)}  nmax={nmax}  Bmultiplier={Bmultiplier}  '
          f'lifshitz_threshold={args.lifshitz_threshold}')
    print(f'  bc_factors={bc_factors}  bc_sign_mode={args.bc_sign_mode}  '
          f'valleys={valleys}  bands={bands}')

    result = {'Blist': Blist, 'nmax': nmax,
              'onsager_Bmultiplier': Bmultiplier,
              'lifshitz_threshold': args.lifshitz_threshold,
              'bc_factors': np.array(bc_factors),
              'bc_sign_mode': args.bc_sign_mode}

    worst_overall = {f: 0.0 for f in bc_factors}
    nan_overall = {f: 0 for f in bc_factors}

    for valley in valleys:
        for band in bands:
            if (valley, band) not in combos:
                continue
            area, bc, elev = read_band(args.detail, valley, band, cache)
            for fi, f in enumerate(bc_factors):
                tag = '' if len(bc_factors) == 1 else f'_bcf{fi}'
                fan = fan_for_band(Blist, area, bc, elev, nmax, f,
                                   Bmultiplier, args.lifshitz_threshold,
                                   bc_sign_mode=args.bc_sign_mode)
                for s, arr in fan.items():
                    result[f'LL_{valley}_band{band}_{s}{tag}'] = arr

                if ref_struct is not None:
                    c = compare(fan, ref_struct, valley, band, tag)
                    worst_overall[f] = max(worst_overall[f], c['worst'])
                    nan_overall[f] += c['nan_mismatch']
                    msg = (f'  {valley} band{band:2d} bc={f:+g}: '
                           f'max|diff|={c["worst"]:.3e} meV  '
                           f'NaN-mismatch={c["nan_mismatch"]}')
                    if c['only_mine'] or c['only_ref']:
                        msg += (f'  keys only-mine={len(c["only_mine"])} '
                                f'only-ref={len(c["only_ref"])}')
                    print(msg)
            del area, bc, elev

    if ref_struct is not None:
        print(f'\n  vs {args.ref}:')
        for f in bc_factors:
            print(f'    bc_factor={f:+g}: max|diff| = {worst_overall[f]:.6e} meV, '
                  f'NaN-pattern mismatches = {nan_overall[f]}')

    if args.out != 'none':
        out = args.out
        if out is None:
            base = args.detail
            if base.endswith('_detail.mat'):
                base = base[:-len('_detail.mat')]
            else:
                base = os.path.splitext(base)[0]
            out = base + '_recomp.mat'
        save_result(result, out, {'source_detail': args.detail,
                                  'bc_factors': np.array(bc_factors),
                                  'bc_sign_mode': args.bc_sign_mode,
                                  'nmax': nmax,
                                  'onsager_Bmultiplier': Bmultiplier,
                                  'lifshitz_threshold': args.lifshitz_threshold})


if __name__ == '__main__':
    main()
