"""
Magnetic-BZ unfolding for folded Hofstadter band structures.

When qfac = gcd(2*pp, qq) is 2 and pp is odd (flux qq/pp = 2/(odd), and also
e.g. (pp,qq) = (7,4)) the Landau gauge forces a rectangular construction cell
on a triangular moire lattice, and the resulting magnetic BZ is a factor of
two too small.  Each physical band then appears as two folded subbands that
overlap in energy, never mix, and swap hi/lo ordering across lines of exact
degeneracy.  Sorting by energy therefore produces two kinked, unphysical
surfaces instead of one smooth band.

Two degeneracy line families separate the subbands (fractional coordinates
f1 = n1/nk1 along G1 = b1/pp, f2 = n2/nk2 along G2 = qfac*b2/pp):

  family 1   f1 - f2 = d_off   (d_off = 1/2, symmetry-enforced)   direction G1+G2
  family 2   f2 = h_off        (h_off band-dependent)             direction G1

The branch label is the parity of the number of lines crossed.  It is only
single-valued on the double cover along G1, which is precisely the statement
that the true reciprocal lattice is <2*G1, G2> and the computed BZ is half of
it.  Unfolding therefore tiles the data twice along n1 and selects one branch.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers on the periodic interval [0, 1)
# ---------------------------------------------------------------------------

def _circ_dist(a, b):
    d = np.abs((np.asarray(a, dtype=float) - b) % 1.0)
    return np.minimum(d, 1.0 - d)


def _circ_mean(vals):
    ang = 2 * np.pi * np.asarray(vals, dtype=float)
    return float(np.arctan2(np.sin(ang).mean(),
                            np.cos(ang).mean()) / (2 * np.pi)) % 1.0


def _circ_rms(vals, ref):
    return float(np.sqrt(np.mean(_circ_dist(vals, ref)**2)))


# ---------------------------------------------------------------------------
# Degeneracy line detection
# ---------------------------------------------------------------------------

def _cone_minima(g, tol):
    """Apexes of the conical minima of a periodic 1-D gap profile.

    The two subbands cross linearly, so the gap behaves like |x| and the apex
    generally falls between grid points.  A parabolic sub-grid fit reports a
    spurious finite gap there; the apex is instead recovered by intersecting
    the two linear flanks.  Returns [(position, gap_at_apex), ...] for apexes
    below `tol`.
    """
    N, out = len(g), []
    for i in range(N):
        if not (g[i] <= g[i - 1] and g[i] <= g[(i + 1) % N]):
            continue
        sl = g[i - 1] - g[i - 2]
        sr = g[(i + 2) % N] - g[(i + 1) % N]
        if sl >= 0 or sr <= 0:
            out.append((float(i), g[i]))
            continue
        x = ((g[(i + 1) % N] - sr * (i + 1))
             - (g[i - 1] - sl * (i - 1))) / (sl - sr)
        out.append((x % N, g[i - 1] + sl * (x - (i - 1))))
    return [(x, v) for x, v in out if v < tol]


def analyze_pair(lo2d, hi2d, rel_tol=3e-3):
    """Locate both degeneracy line families of a candidate folded pair.

    `lo2d`, `hi2d` are (nk2, nk1) energy-sorted subbands.  Returns a dict with
    the fitted offsets and the quality metrics used to decide whether the pair
    really is folded.
    """
    nk2, nk1 = lo2d.shape
    gap = hi2d - lo2d
    tol = rel_tol * (hi2d.max() - lo2d.min())

    # --- family 1: exactly one crossing on every cut at fixed n2 ---
    d_samples = []
    for r in range(nk2):
        m = _cone_minima(gap[r, :], tol)
        if len(m) == 1:
            d_samples.append((m[0][0] / nk1 - r / nk2) % 1.0)

    info = {'frac_rows': len(d_samples) / nk2, 'gap_min': float(gap.min()),
            'd_off': None, 'd_rms': None,
            'h_off': None, 'h_rms': None, 'h_n': 0}
    if not d_samples:
        return info
    info['d_off'] = _circ_mean(d_samples)
    info['d_rms'] = _circ_rms(d_samples, info['d_off'])

    # --- family 2: cuts at fixed n1, discarding the family-1 crossing ---
    excl = max(0.02, 3.0 / nk2)
    h_samples = []
    for c in range(nk1):
        for x, _ in _cone_minima(gap[:, c], tol):
            if _circ_dist(c / nk1 - x / nk2, info['d_off']) > excl:
                h_samples.append(x / nk2)
    if h_samples:
        info['h_off'] = _circ_mean(h_samples)
        info['h_rms'] = _circ_rms(h_samples, info['h_off'])
        info['h_n'] = len(h_samples)
    return info


def _qualifies(info, nk1, min_frac, max_rms):
    return (info['d_off'] is not None
            and info['h_off'] is not None
            and info['frac_rows'] >= min_frac
            and info['d_rms'] <= max_rms
            and info['h_rms'] <= max_rms
            and info['h_n'] >= 0.25 * nk1)


# ---------------------------------------------------------------------------
# Branch labelling and unfolding
# ---------------------------------------------------------------------------

def branch_label(nk1, nk2, d_off, h_off):
    """Parity of the number of degeneracy lines crossed, on the doubled grid.

    Neither parity is separately periodic under f2 -> f2 + 1, but their sum is,
    and the sum is periodic under f1 -> f1 + 2 while flipping under f1 -> f1 + 1.
    That is exactly the branch structure of a band folded once along G1.

    When nk1 == nk2 the family-1 line runs exactly through grid points, whose
    side of the cut would otherwise be decided by the sign of the ~1e-8 fit
    residual.  `eps` (1% of a grid step, far above the fit error and far below
    a step) makes that choice deterministic; the two branches are degenerate
    there, so which one is picked is immaterial.
    """
    eps = 0.01 / max(nk1, nk2)
    f1 = (np.arange(2 * nk1) / nk1)[None, :]
    f2 = (np.arange(nk2) / nk2)[:, None]
    p_diag = np.floor(f1 - f2 - d_off + eps).astype(int) % 2
    p_horz = np.floor(f2 - h_off + eps).astype(int) % 2
    return (p_diag + p_horz) % 2


def unfold_pair(lo2d, hi2d, s):
    """Select one branch of a folded pair on the (nk2, 2*nk1) doubled grid."""
    nk1 = lo2d.shape[1]
    w = np.arange(s.shape[1]) % nk1
    return np.where(s == 0, lo2d[:, w], hi2d[:, w])


def _roughness(Z):
    """Max |second difference| along each axis of a doubly periodic surface."""
    return tuple(float(np.abs(np.roll(Z, -1, ax) - 2 * Z
                              + np.roll(Z, 1, ax)).max()) for ax in (1, 0))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def unfold_bandstructure(result, rel_tol=3e-3, min_frac=0.9, max_rms=0.02,
                         valleys=('K', 'Kp'), verbose=True):
    """Unfold a folded band structure onto the BZ doubled along G1.

    Detects folded pairs from the data (never from qq/pp alone), separates the
    two non-mixing subbands of each pair, and shifts one by G1 so that each
    pair becomes a single continuous band on the doubled BZ.  E, Oz and Lz are
    all carried through the same branch label.

    The unfolded arrays replace the primary keys, with nk1 -> 2*nk1 and
    vol_M -> vol_M/2 so that downstream stages inherit the correct mesh; the
    original arrays are preserved under `*_folded` keys.  Bands with no partner
    inside the window are dropped and reported.

    Returns a new result dict; `result` is not modified.
    """
    nk1 = int(result['nk1'])
    nk2 = int(result['nk2'])
    nbands = result[f'E_{valleys[0]}'].shape[0]

    def as2d(key, n):
        return result[key][n].reshape(nk2, nk1, order='F')

    # --- Per-valley analysis of every adjacent pair ---
    infos = {v: [analyze_pair(as2d(f'E_{v}', n), as2d(f'E_{v}', n + 1), rel_tol)
                 for n in range(nbands - 1)]
             for v in valleys}

    # A pair is folded only if both valleys agree; time reversal guarantees they
    # should, and requiring it keeps the two valleys on a common band indexing.
    ok = [all(_qualifies(infos[v][n], nk1, min_frac, max_rms) for v in valleys)
          for n in range(nbands - 1)]

    pairs, unpaired, n = [], [], 0
    while n < nbands:
        if n < nbands - 1 and ok[n]:
            pairs.append(n)
            n += 2
        else:
            unpaired.append(n)
            n += 1

    if verbose:
        print(f"  Unfold: {len(pairs)} folded pair(s) detected "
              f"among {nbands} bands")
    if not pairs:
        print("  Unfold: WARNING - no folded pairs found; "
              "returning the band structure unchanged.")
        return dict(result)
    if unpaired and verbose:
        print(f"  Unfold: WARNING - bands {unpaired} have no partner in the "
              f"window and are DROPPED (their partner lies outside `bands`).")

    # --- Build the unfolded bands ---
    out = {v: {'E': [], 'Oz': [], 'Lz': []} for v in valleys}
    for n in pairs:
        for v in valleys:
            info = infos[v][n]
            s = branch_label(nk1, nk2, info['d_off'], info['h_off'])
            lo = as2d(f'E_{v}', n)
            A = unfold_pair(lo, as2d(f'E_{v}', n + 1), s)
            out[v]['E'].append(A.ravel(order='F'))
            out[v]['Oz'].append(
                unfold_pair(as2d(f'Oz_{v}', n), as2d(f'Oz_{v}', n + 1),
                            s).ravel(order='F'))
            out[v]['Lz'].append(
                unfold_pair(as2d(f'Lz_{v}', n), as2d(f'Lz_{v}', n + 1),
                            s).ravel(order='F'))
            if verbose:
                r_lo, r_hi = _roughness(lo), _roughness(A)
                print(f"    {v} bands ({n},{n+1}) -> unfolded band "
                      f"{len(out[v]['E'])-1}:  f1-f2={info['d_off']:.6f}  "
                      f"f2={info['h_off']:.6f}  "
                      f"max|d2E| sorted {r_lo[0]:.2e}/{r_lo[1]:.2e} -> "
                      f"unfolded {r_hi[0]:.2e}/{r_hi[1]:.2e} meV (n1/n2)")

    # --- k-mesh doubled along G1 ---
    kp = np.asarray(result['kpoints'])
    kp2 = kp.reshape(nk2, nk1, 2, order='F')
    G1 = nk1 * (kp2[0, 1] - kp2[0, 0])
    kpoints_unf = np.concatenate([kp2, kp2 + G1], axis=1).reshape(
        nk2 * 2 * nk1, 2, order='F')

    new = dict(result)
    for key in ('E', 'Oz', 'Lz'):
        for v in valleys:
            new[f'{key}_{v}_folded'] = result[f'{key}_{v}']
            new[f'{key}_{v}'] = np.array(out[v][key])
    new['kpoints_folded'] = kp
    new['nk1_folded'] = nk1
    new['nk2_folded'] = nk2
    new['vol_M_folded'] = result['vol_M']

    new['kpoints'] = kpoints_unf
    new['nk1'] = 2 * nk1
    new['nk2'] = nk2
    new['vol_M'] = result['vol_M'] / 2.0

    new['unfold'] = 1
    new['unfold_pairs'] = np.array([[n, n + 1] for n in pairs])
    new['unfold_dropped'] = np.array(unpaired, dtype=int)

    if verbose:
        print(f"  Unfold: nk1 {nk1} -> {2*nk1}, vol_M halved, "
              f"{nbands} -> {len(pairs)} bands")
    return new
