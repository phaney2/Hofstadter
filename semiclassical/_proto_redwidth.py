"""Measure the exact-Hofstadter 'red spread' as a filled fraction.

For a Landau ladder of spacing hbar*omega_c whose levels are broadened to a
width Gamma, the measure of {E : DOS > 0} per unit energy is Gamma/hbar*omega_c.
That is approximant-robust in a way individual subband widths are not.
"""

import glob
import numpy as np
import scipy.io as sio

HBAR = 1.05e-34
QE = 1.6e-19
A_G, A_HBN = 2.46e-10, 2.504e-10
theta = np.radians(0.965)
eps = A_HBN / A_G - 1
LM = (1 + eps) * A_G / np.sqrt(eps**2 + 2 * (1 + eps) * (1 - np.cos(theta)))
uc = np.sqrt(3) / 2 * LM**2
b0 = HBAR * 2 * np.pi / QE / uc

D = "C:/Users/phaney/wrk/hofstadter/calculations/quantum/blackbird/butterfly/0.5moire_Um20_em0.5"
WIN = (-165.0, -105.0)

rows = []
for f in sorted(glob.glob(f"{D}/dos_p*_q*.mat")):
    d = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
    r, p = d['results'], d['params']
    pp, qq = int(p.pp), int(p.qq)
    B = b0 * qq / (2 * pp)
    if not (1.5 <= B <= 13.0) or pp > 20:
        continue
    e = np.atleast_1d(r.elist).ravel()
    dos = np.atleast_1d(r.dos_K).ravel()
    de = e[1] - e[0]
    m = (e >= WIN[0]) & (e <= WIN[1])
    ee, dd = e[m], dos[m]
    on = dd > 0
    edges = np.diff(on.astype(int))
    s = np.where(edges == 1)[0] + 1
    t = np.where(edges == -1)[0] + 1
    if on[0]:
        s = np.r_[0, s]
    if on[-1]:
        t = np.r_[t, len(on)]
    w = ee[t - 1] - ee[s]                       # lower bound (1 bin -> 0)
    W = WIN[1] - WIN[0]
    rows.append((B, pp, qq, len(w), w.sum() / W, (w + de).sum() / W))

rows.sort()
print(f"b0 = {b0:.3f} T,  window {WIN} meV,  pp <= 20\n")
print(f"{'B(T)':>6} {'p':>3} {'q':>3} {'nsub':>5} {'2p':>4} "
      f"{'frac_lo':>8} {'frac_hi':>8}")
for B, pp, qq, n, flo, fhi in rows:
    print(f"{B:6.2f} {pp:3d} {qq:3d} {n:5d} {2*pp:4d} {flo:8.3f} {fhi:8.3f}")

print("\n  binned by B (mean over approximants):")
edges = np.arange(2.0, 13.01, 1.0)
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = [r for r in rows if lo <= r[0] < hi]
    if not sel:
        continue
    print(f"  {lo:4.1f}-{hi:4.1f} T   n={len(sel):3d}   "
          f"frac_lo {np.mean([r[4] for r in sel]):.3f}   "
          f"frac_hi {np.mean([r[5] for r in sel]):.3f}   "
          f"(pp = {sorted(set(r[1] for r in sel))})")
