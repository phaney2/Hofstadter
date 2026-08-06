"""Prototype: magnetic-breakdown broadening of the semiclassical LLs.

Crossings are found by the Bragg condition 2 k.G = |G|^2 directly on the
reciprocal-lattice shells, not by a Wigner-Seitz partition (which mislabels
the boundaries once the orbit outgrows the first zone).
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import numpy as np
import scipy.io as sio
from scipy.ndimage import map_coordinates
from skimage.measure import find_contours

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bandstructure as bs

HBAR = 1.054571817e-34
QE = 1.602176634e-19
ME = 9.1093837015e-31
MEVA_TO_MS = 1e-3 * QE * 1e-10 / HBAR          # meV*Angstrom -> m/s

D = "C:/Users/phaney/wrk/hofstadter/calculations/semiclassical/blackbird/moire_0.5_Um20_Em0.5"


def load(name):
    return sio.loadmat(f"{D}/{name}", squeeze_me=True,
                       struct_as_record=False)['results']


def main():
    rc, rd = load('zf.mat'), load('zf_extended_dominant.mat')
    b = 1                                       # top valence branch
    Ec = rc.E_K[b]
    gap = 2.0 * np.abs(rd.E_K[b] - Ec)   # peaks at E_g on a Bragg plane

    nk1e, nk2e = int(rc.nk1), int(rc.nk2)
    nk1, nk2 = int(rc.nk1_folded), int(rc.nk2_folded)
    off = (nk1e // nk1 - 1) // 2

    _, _, _, _, vb, _ = bs.compute_moire_geometry(np.radians(0.965))
    M = np.array([vb[0, :2], vb[1, :2]])
    Minv = np.linalg.inv(M)

    E2 = Ec.reshape(nk2e, nk1e, order='F')      # axis0 = N2, axis1 = N1
    G2 = gap.reshape(nk2e, nk1e, order='F')
    BZ = (2 * np.pi)**2 / float(rc.vol_M_folded)
    cell_area = BZ / (nk1 * nk2)

    dN2, dN1 = np.gradient(E2)
    grad = np.stack([dN1 * nk1, dN2 * nk2], axis=-1) @ Minv.T   # meV*Angstrom

    # reciprocal-lattice shells
    mm = np.array([[m1, m2] for m1 in range(-3, 4) for m2 in range(-3, 4)
                   if (m1, m2) != (0, 0)])
    Gs = mm @ M
    G2n = (Gs**2).sum(axis=1)

    def orbit(level):
        best = None
        for c in find_contours(E2, level):
            if (np.linalg.norm(c[0] - c[-1]) > 1.0
                    or c[:, 0].min() <= 0 or c[:, 0].max() >= nk2e - 1
                    or c[:, 1].min() <= 0 or c[:, 1].max() >= nk1e - 1):
                continue
            x, y = c[:-1, 0], c[:-1, 1]
            a = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
            if best is None or a > best[0]:
                best = (a, c)
        return best

    def crossings(c):
        """(k, Ghat, Eg) at every Bragg-plane crossing of the closed contour."""
        kx = ((c[:-1, 1] / nk1 - 0.5 - off)[:, None] * vb[0, :2]
              + (c[:-1, 0] / nk2 - 0.5 - off)[:, None] * vb[1, :2])
        s = 2.0 * (kx @ Gs.T) - G2n[None, :]            # (npts, nG)
        sgn = np.sign(s)
        hit = np.where((sgn != np.roll(sgn, -1, axis=0)) & (sgn != 0))
        out = []
        npts = len(kx)
        w = max(3, npts // 60)
        for i, ig in zip(*hit):
            j = (i + 1) % npts
            t = s[i, ig] / (s[i, ig] - s[j, ig])
            ci = c[i] + t * (c[j] - c[i])
            g = np.array([map_coordinates(grad[:, :, d], [[ci[0]], [ci[1]]],
                                          order=1)[0] for d in (0, 1)])
            idx = (np.arange(i - w, i + w + 1)) % npts
            Eg = G2[np.rint(c[idx, 0]).astype(int) % nk2e,
                    np.rint(c[idx, 1]).astype(int) % nk1e].max()
            out.append((ci, kx[i] + t * (kx[j] - kx[i]),
                        Gs[ig] / np.sqrt(G2n[ig]), Eg, g))
        # dedup: same physical point picked up by two shells
        keep = []
        for o in out:
            if all(np.linalg.norm(o[1] - p[1]) > 1e-4 for p in keep):
                keep.append(o)
        return keep

    levels = np.arange(-168.0, -99.0, 3.0)
    Blist = 46.219 / (2 * np.array([12, 10, 9, 8, 7, 6, 5, 4, 3, 2]))

    print(f"{'E':>6} {'A/A_BZ':>7} {'mc/me':>6} {'hw_c(1T)':>8} {'N':>3} "
          f"{'Eg_med':>7} {'B0_med':>7} {'B0_max':>7} | "
          + " ".join(f"{'G(%gT)' % B:>7}" for B in Blist))

    rows = []
    for lvl in levels:
        got = orbit(lvl)
        if got is None:
            continue
        area_px, c = got
        area = area_px * cell_area

        h = 2.0
        a_lo, a_hi = orbit(lvl - h), orbit(lvl + h)
        if a_lo is None or a_hi is None:
            continue
        dAdE = (a_hi[0] - a_lo[0]) * cell_area / (2 * h * 1e-3 * QE)
        mc = HBAR**2 / (2 * np.pi) * abs(dAdE)
        hw1 = HBAR * QE / mc / QE * 1e3          # hbar*omega_c at 1 T, meV

        xs = crossings(c)
        B0s, Egs = [], []
        for ci, k, ghat, Eg, g in xs:
            vperp = abs(g @ ghat) * MEVA_TO_MS
            vpar = np.linalg.norm(g - (g @ ghat) * ghat) * MEVA_TO_MS
            if vperp <= 0 or vpar <= 0 or Eg <= 0:
                continue
            EgJ = Eg * 1e-3 * QE
            B0s.append(np.pi * EgJ**2 / (4 * HBAR * QE * vperp * vpar))
            Egs.append(Eg)
        B0s = np.array(B0s)

        Gam = np.array([hw1 * B / (2 * np.pi)
                        * np.sum(np.sqrt(1 - np.exp(-B0s / B))) for B in Blist])
        rows.append(dict(E=lvl, a=area / BZ, mc=mc / ME, hw1=hw1, n=len(B0s),
                         egm=np.median(Egs) if Egs else np.nan,
                         b0m=np.median(B0s) if len(B0s) else np.nan,
                         b0x=B0s.max() if len(B0s) else np.nan, G=Gam, B0=B0s))

    for r in rows:
        print(f"{r['E']:6.0f} {r['a']:7.3f} {r['mc']:6.4f} {r['hw1']:8.3f} "
              f"{r['n']:3d} {r['egm']:7.2f} {r['b0m']:7.3f} {r['b0x']:7.3f} | "
              + " ".join(f"{g:7.2f}" for g in r['G']))

    print("\n  Gamma / (hbar omega_c)  -- 0 = clean ladder, 1 = washed out")
    print(f"{'E':>6} " + " ".join(f"{'%.2fT' % B:>7}" for B in Blist))
    for r in rows:
        print(f"{r['E']:6.0f} "
              + " ".join(f"{g / (r['hw1'] * B):7.3f}"
                         for g, B in zip(r['G'], Blist)))

    import json
    with open('_breakdown.json', 'w') as f:
        json.dump(dict(Blist=Blist.tolist(),
                       rows=[dict(E=r['E'], hw1=r['hw1'], n=r['n'],
                                  B0=r['B0'].tolist(), G=r['G'].tolist())
                             for r in rows]), f)


main()
