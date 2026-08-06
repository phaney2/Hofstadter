"""Magnetic-breakdown broadening of the semiclassical Landau levels.

The extended-zone unfolding (`extended_zone.py`, `centroid` mode) is the
magnetic breakdown limit taken all the way: the carrier tunnels straight
through every moire Bragg plane and the gaps there play no role at all.  That
is why an Onsager fan built on it reproduces a slightly perturbed bilayer, and
misses the extra structure the exact Hofstadter spectrum shows near the moire
BZ boundary.

This module puts the gaps back in, not as a shift of the level but as a width.
At a Bragg plane the semiclassical orbit is a Landau-Zener crossing between the
two branches, with tunnelling probability

    P = exp(-B0/B),    B0 = pi*Eg^2 / (4*hbar*e*v_perp*v_par)

`v_perp` is the velocity component along Ghat -- it sets how fast the two
branches separate -- and `v_par` is the component in the plane: the equation of
motion `hbar*kdot = -e v x B` runs k perpendicular to v, so it is the *parallel*
velocity that carries the state through the crossing, at a rate
`d(delta)/dt = 2*hbar*v_perp*(e*B/hbar)*v_par`.

A crossing that is not fully transparent reflects with amplitude
`r = sqrt(1-P)`, which perturbs the round-trip phase by of order `r` per
crossing.  With `dphi/dE = 2*pi/(hbar*omega_c)` that is a level width

    Gamma(E, B) = (hbar*omega_c / 2*pi) * sum_i sqrt(1 - exp(-B0_i/B))

summed over the crossings of the orbit, with `omega_c = e*B/m_c` and
`m_c = (hbar^2/2*pi)|dA/dE|` from the orbit areas themselves.  There are no
free parameters.

Validated against the exact `main_v3.py` spectrum at `qq = 1`, where one
magnetic subband is one Landau level (LL degeneracy per moire cell is
qq/(2pp), subband weight 1/(2pp), so a level holds qq subbands): over 105
levels spanning 1.9 to 11.6 T the median `w_exact/Gamma` is 0.83, with no
systematic drift beyond a mild one in B (0.76 below 4 T, 0.95 above).

`Gamma` is an *envelope*, not a per-level prediction.  The exact widths
oscillate by a factor of ~4 from one level to the next -- the coherent
interference of the several crossings, which an incoherent sum of reflection
amplitudes cannot carry -- and the rank correlation between `w_exact` and
`Gamma` within a field is near zero at low B.  Resolving the oscillation needs
the full Falicov-Stachowiak coupled-orbit network, which this is not.

Breakdown: Cohen & Falicov, PRL 7, 231 (1961); Blount, PR 126, 1636 (1962)
(also the source of the two-velocity B0); review, Stark & Falicov, Prog. Low
Temp. Phys. 5, 235 (1967).  Networks: Pippard, Phil. Trans. A 256, 317 (1964);
Falicov & Stachowiak, PR 147, 505 (1966).  The `Gamma` above is in none of
them.  Annotated list in `doc_technical.md`.
"""

import numpy as np
from scipy.ndimage import map_coordinates


HBAR = 1.054571817e-34
QE = 1.602176634e-19
MEVA_TO_MS = 1e-3 * QE * 1e-10 / HBAR       # meV*Angstrom -> m/s


def reciprocal_shells(vb, kmax):
    """Moire reciprocal-lattice vectors whose Bragg plane sits within kmax.

    The plane 2 k.G = |G|^2 is reachable at |k| <= kmax only if |G| <= 2*kmax,
    so the shell count follows from the orbit size.  Testing the Bragg
    condition against these directly is what makes the crossing count stable:
    a Wigner-Seitz partition of the extended zone instead reports transitions
    between outer cells once the orbit outgrows the first zone, and those are
    crossed at near-grazing incidence where v_par -> 0 and B0 diverges.
    """
    gmin = min(np.linalg.norm(vb[0]), np.linalg.norm(vb[1]))
    n = max(1, int(np.ceil(2 * kmax / gmin)))
    mm = np.array([[m1, m2] for m1 in range(-n, n + 1)
                   for m2 in range(-n, n + 1) if (m1, m2) != (0, 0)])
    Gs = mm @ np.asarray(vb)
    G2 = (Gs**2).sum(axis=1)
    keep = G2 <= (2 * kmax)**2
    return Gs[keep], G2[keep]


def band_gradient(E_2d, vb, nk1f, nk2f):
    """dE/dk on the extended grid, in meV*Angstrom.

    The extended k-map is k = u1*vb0 + u2*vb1 with u_a stepping by 1/nk_a of
    the *folded* mesh, so dE/du = M dE/dk with M the rows of vb.
    """
    dN2, dN1 = np.gradient(E_2d)
    return (np.stack([dN1 * nk1f, dN2 * nk2f], axis=-1)
            @ np.linalg.inv(np.asarray(vb)).T)


def orbit_breakdown_fields(contour, grad, gap_2d, vb, nk1f, nk2f, off):
    """Landau-Zener fields B0 at every Bragg-plane crossing of one orbit.

    `contour` is in (row, col) = (N2, N1) extended-grid coordinates, as
    returned by `isoenergy_areas(..., return_contours=True)`.

    The gap is read from the `gap` map as a local maximum over a window of
    contour points around the crossing, not at the crossing point itself:
    `gap` is 2|E_dominant - E_centroid|, which for two states split by Eg and
    detuned by delta is a ridge of height Eg sitting on the plane and decaying
    as Eg^2/(2*delta) away from it, so the peak value is the gap and any single
    nearby sample underestimates it.
    """
    nk2e, nk1e = gap_2d.shape
    c = contour[:-1]
    npts = len(c)
    u1 = c[:, 1] / nk1f - 0.5 - off
    u2 = c[:, 0] / nk2f - 0.5 - off
    kk = u1[:, None] * np.asarray(vb)[0] + u2[:, None] * np.asarray(vb)[1]

    Gs, G2n = reciprocal_shells(vb, np.linalg.norm(kk, axis=1).max())
    if len(Gs) == 0:
        return np.zeros(0)

    s = 2.0 * (kk @ Gs.T) - G2n[None, :]
    sgn = np.sign(s)
    hit = np.where((sgn != np.roll(sgn, -1, axis=0)) & (sgn != 0))

    w = max(3, npts // 60)
    B0, kseen = [], []
    for i, ig in zip(*hit):
        j = (i + 1) % npts
        t = s[i, ig] / (s[i, ig] - s[j, ig])
        kc = kk[i] + t * (kk[j] - kk[i])
        if any(np.linalg.norm(kc - p) < 1e-4 for p in kseen):
            continue                     # one point, two equivalent shells
        kseen.append(kc)

        idx = np.arange(i - w, i + w + 1) % npts
        Eg = gap_2d[np.rint(c[idx, 0]).astype(int) % nk2e,
                    np.rint(c[idx, 1]).astype(int) % nk1e].max()
        if Eg <= 0:
            continue

        ci = c[i] + t * (c[j] - c[i])
        g = np.array([map_coordinates(grad[:, :, d], [[ci[0]], [ci[1]]],
                                      order=1)[0] for d in (0, 1)])
        ghat = Gs[ig] / np.sqrt(G2n[ig])
        vperp = abs(g @ ghat) * MEVA_TO_MS
        vpar = np.linalg.norm(g - (g @ ghat) * ghat) * MEVA_TO_MS
        if vperp <= 0 or vpar <= 0:
            continue

        EgJ = Eg * 1e-3 * QE
        B0.append(np.pi * EgJ**2 / (4 * HBAR * QE * vperp * vpar))

    return np.array(B0)


def cyclotron_mass(area, E_levels):
    """m_c = (hbar^2/2*pi)|dA/dE| in kg, from areas in m^-2 and E in meV."""
    area = np.asarray(area, dtype=float)
    E_J = np.asarray(E_levels, dtype=float) * 1e-3 * QE
    mc = HBAR**2 / (2 * np.pi) * np.abs(np.gradient(area, E_J))
    bad = area <= 0
    mc[bad | np.roll(bad, 1) | np.roll(bad, -1)] = np.nan
    return mc


def breakdown_band(E_band, gap_band, E_levels, area, contours,
                   vb, nk1, nk2, ntile):
    """B0 `(nE, ncmax)` and cyclotron mass `(nE,)` for one band.

    `nk1`/`nk2` are the extended mesh dimensions and `ntile` the tiling that
    produced them; the orbit taken at each energy is the largest pocket, the
    same one `onsager_fan_band` quantizes.  Rows are NaN-padded to the largest
    crossing count on the band.
    """
    nk1f, nk2f = nk1 // ntile, nk2 // ntile
    off = (ntile - 1) // 2

    E_2d = np.asarray(E_band).reshape(nk2, nk1, order='F')
    gap_2d = np.asarray(gap_band).reshape(nk2, nk1, order='F')
    grad = band_gradient(E_2d, vb, nk1f, nk2f)

    per_level = [
        orbit_breakdown_fields(contours[i][0], grad, gap_2d,
                               vb, nk1f, nk2f, off)
        if contours[i] else np.zeros(0)
        for i in range(len(E_levels))]

    ncmax = max(max((len(b) for b in per_level), default=0), 1)
    B0 = np.full((len(E_levels), ncmax), np.nan)
    for i, b in enumerate(per_level):
        B0[i, :len(b)] = b

    return B0, cyclotron_mass(np.asarray(area)[:, 0], E_levels)


def level_widths(B0, mc, Blist):
    """Landau-level width `Gamma(E, B)` in meV, shape `(nE, nB)`."""
    B = np.abs(np.atleast_1d(Blist).astype(float))
    hw = HBAR * QE * B[None, :] / np.asarray(mc)[:, None] / (1e-3 * QE)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.sqrt(1.0 - np.exp(-np.asarray(B0)[:, :, None]
                                 / B[None, None, :]))
    return hw / (2 * np.pi) * np.nansum(r, axis=1)


def widths_at(LL, E_levels, G):
    """Interpolate `Gamma(E, B)` onto the level energies of one fan array.

    `LL` is `(nB, nmax+1)` as `onsager_fan_band` returns it, NaN where the
    quantization condition has no root; those entries stay NaN.
    """
    LL = np.asarray(LL)
    W = np.full(LL.shape, np.nan)
    E_levels = np.asarray(E_levels)
    for iB in range(min(LL.shape[0], G.shape[1])):
        ok = np.isfinite(G[:, iB])
        m = np.isfinite(LL[iB])
        if ok.sum() < 2 or not m.any():
            continue
        W[iB, m] = np.interp(LL[iB][m], E_levels[ok], G[ok, iB])
    return W
