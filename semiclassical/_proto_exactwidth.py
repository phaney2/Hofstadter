"""Exact Landau-level widths from main_v3, for comparison with the
magnetic-breakdown broadening estimate.

A Landau level holds qq magnetic subbands (degeneracy qq/(2pp) per moire cell,
subband weight 1/(2pp)), so only qq = 1 gives one subband per level -- there
the subband width IS the level broadening and the subband spacing IS
hbar*omega_c.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import json
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_v3 import do_calc

B0 = 46.219
WIN = (-170.0, -100.0)
NK = 16

TMPL = """calctype = 'ek'
isparallel = 1
nworkers = 8
theta = .965
qq = 1
pp = {pp}
g0 = 2400
g1 = 340
g3 = 136
g4 = 0
delta = 0
v0 = -6.513942
v1 = 8.979034
moire_psi = 0.644806
w = 110
eta = 2
U = [-10 10]
nk1 = {nk}
nk2 = {nk}
LL_multiplier = 6
Nmax = 500
layer_resolved = 0
valley = {{'K'}}
outputfile = ''
"""

PPS = [12, 10, 9, 8, 7, 6, 5, 4, 3, 2]


def run(pp, nk):
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write(TMPL.format(pp=pp, nk=nk))
        path = f.name
    try:
        return do_calc(path)['bands_K']
    finally:
        os.unlink(path)


if __name__ == '__main__':
    out = []
    for pp in PPS:
        B = B0 / (2 * pp)
        bands = run(pp, NK)
        lo, hi = bands.min(axis=0), bands.max(axis=0)
        ctr = 0.5 * (lo + hi)
        m = (ctr >= WIN[0]) & (ctr <= WIN[1])
        c, w = ctr[m], (hi - lo)[m]
        o = np.argsort(c)
        out.append(dict(B=B, pp=pp, ctr=c[o].tolist(), w=w[o].tolist()))
        with open('_exactwidth.json', 'w') as f:
            json.dump(out, f)
        print(f"[done] pp={pp}  B={B:.2f} T  nLL={len(w)}", flush=True)

    print(f"\nwindow {WIN} meV, nk = {NK}x{NK}, qq = 1 (1 subband = 1 LL)\n")
    print(f"{'B(T)':>6} {'p':>3} {'nLL':>4} {'w_med':>7} {'w_mean':>7} "
          f"{'hw_c':>7} {'w/hw_c':>7}   per-level (E: w)")
    for r in sorted(out, key=lambda r: r['B']):
        c, w = np.array(r['ctr']), np.array(r['w'])
        hw = np.median(np.diff(c))
        print(f"{r['B']:6.2f} {r['pp']:3d} {len(w):4d} {np.median(w):7.2f} "
              f"{w.mean():7.2f} {hw:7.2f} {np.median(w)/hw:7.3f}   "
              + "  ".join(f"{a:.0f}:{b:.1f}" for a, b in zip(c, w)))
