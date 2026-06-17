"""
Semiclassical Onsager quantization.

Computes Landau level fan diagram E(B) from orbit areas, enclosed Berry
curvature, and orbital moment derivative dL/dE.  Operates on a single
band at a time.  Always computes cumulative levels: S, S+BC, S+BC+morb,
and (if chi data available) S+BC+morb+chi.
"""

import numpy as np


EC = 1.60217662e-19       # C
HBAR_SI = 1.05457182e-34  # J*s
PHI0 = 2 * np.pi * HBAR_SI / EC


def _solve_onsager(rhs, base, valid, E_levels, rtol=0.05):
    onsager = rhs + base[:, :, np.newaxis]
    onsager[~valid, :, :] = np.inf

    nE = len(E_levels)

    o_lo = np.where(np.isfinite(onsager[:-1]), onsager[:-1], 0.0)
    o_hi = np.where(np.isfinite(onsager[1:]), onsager[1:], 0.0)
    both_finite = np.isfinite(onsager[:-1]) & np.isfinite(onsager[1:])
    has_crossing = (o_lo * o_hi < 0) & both_finite

    any_crossing = np.any(has_crossing, axis=0)
    first_idx = np.argmax(has_crossing, axis=0)

    onsager_clipped = np.where(np.isfinite(onsager), onsager, 0.0)

    f0 = np.take_along_axis(onsager_clipped, first_idx[np.newaxis], axis=0)[0]
    f1 = np.take_along_axis(
        onsager_clipped,
        np.minimum(first_idx + 1, nE - 1)[np.newaxis], axis=0)[0]

    denom = f0 - f1
    safe_denom = np.where(np.abs(denom) > 0, denom, 1.0)
    t = np.where(np.abs(denom) > 0, f0 / safe_denom, 0.5)
    t = np.clip(t, 0.0, 1.0)

    E0 = E_levels[first_idx]
    E1 = E_levels[np.minimum(first_idx + 1, nE - 1)]
    result_interp = E0 + t * (E1 - E0)

    residual = np.abs(onsager)
    ind = np.argmin(residual, axis=0)
    best = np.take_along_axis(residual, ind[np.newaxis], axis=0)[0]
    scale = np.abs(rhs[0, :, :])
    scale = np.where(scale > 0, scale, 1.0)
    result_argmin = E_levels[ind]
    result_argmin[best / scale > rtol] = np.nan

    return np.where(any_crossing, result_interp, result_argmin)


def _find_segments(tarea, lifshitz_threshold):
    """Split area(E) at Lifshitz transitions into monotonic segments.

    Returns list of (start, end) index pairs for slicing.
    """
    dA = np.abs(np.diff(tarea))
    dA_pos = dA[dA > 0]
    if len(dA_pos) == 0:
        return [(0, len(tarea))]

    median_dA = np.median(dA_pos)
    jumps = np.where(dA > lifshitz_threshold * median_dA)[0]

    if len(jumps) == 0:
        return [(0, len(tarea))]

    boundaries = [0] + [j + 1 for j in jumps] + [len(tarea)]
    return [(boundaries[i], boundaries[i + 1])
            for i in range(len(boundaries) - 1)]


def onsager_fan_band(Blist, nmax, E_levels, area, enclosedBC, dL_dE,
                     dChi_dE=None, term_factors=(1.0, 1.0, 1.0),
                     Bmultiplier=1.0, lifshitz_threshold=50):
    """
    Solve the Onsager quantization condition for one band.

    Always computes cumulative LL sets:
      S    — isoenergy area only
      SB   — + enclosed Berry curvature
      SBM  — + dL/dE orbital moment
      SBMC — + chi' susceptibility (only if dChi_dE provided)

    When the orbit area has Lifshitz transitions (discontinuous jumps),
    the energy grid is split into monotonic segments and each is solved
    independently.  Output keys are suffixed with _seg0, _seg1, etc.
    when multiple segments exist.

    Parameters
    ----------
    Blist : (nB,) array
        Magnetic field values in Tesla.
    nmax : int
        Maximum Landau level index.
    E_levels : (nE,) array
        Energy grid for this band.
    area : (nE, npockets) array
        Orbit areas at each energy.
    enclosedBC : (nE, npockets) array
        Enclosed Berry curvature at each energy.
    dL_dE : (nE,) array
        Energy derivative of orbital moment.
    dChi_dE : (nE,) array or None
        Susceptibility derivative (interpolated to this band's grid).
    term_factors : (3,) tuple of float
        (BC_factor, morb_factor, chi_factor).  Multiplicative prefactors
        on the BC, dL/dE, and chi' terms.  Default (1,1,1).
    Bmultiplier : float
        Multiplicative factor on B in the rhs of the Onsager condition.
        Default 1.0.  Diagnostic/testing parameter.
    lifshitz_threshold : float
        Area jump detection threshold: a jump is detected where |ΔA|
        exceeds this factor times the median |ΔA|.  Default 50.

    Returns
    -------
    dict or None
        Keys are cumulative correction labels ('S', 'SB', 'SBM', 'SBMC')
        suffixed with '_seg0', '_seg1', etc. when multiple segments exist.
        Each value is (nB, nmax+1).  Entries with no valid root are NaN.
        None if no orbits found.
    """
    BC_factor, morb_factor, chi_factor = term_factors

    tarea = area[:, 0]
    if np.max(tarea) == 0:
        return None

    tBC = enclosedBC[:, 0]

    n_vec = np.arange(nmax + 1).reshape(1, 1, -1)
    B_vec = np.asarray(Blist).reshape(1, -1, 1)
    rhs = Bmultiplier * B_vec * (n_vec + 0.5) / PHI0

    B2 = B_vec[:, :, 0]
    Bsign = np.sign(B2)

    segments = _find_segments(tarea, lifshitz_threshold)
    multi = len(segments) > 1

    result = {}
    for si, (s0, s1) in enumerate(segments):
        if s1 - s0 < 3:
            continue
        seg_area = tarea[s0:s1]
        if np.max(seg_area) == 0:
            continue

        seg_valid = seg_area > 0
        seg_E = E_levels[s0:s1]
        seg_BC = tBC[s0:s1]
        seg_dLdE = dL_dE[s0:s1]

        sfx = f'_seg{si}' if multi else ''

        base_S = -Bsign * seg_area[:, np.newaxis] / (2 * np.pi)**2
        result[f'S{sfx}'] = _solve_onsager(rhs, base_S, seg_valid, seg_E)

        base_SB = base_S - BC_factor * seg_BC[:, np.newaxis] * B2 / (2 * np.pi * PHI0)
        result[f'SB{sfx}'] = _solve_onsager(rhs, base_SB, seg_valid, seg_E)

        base_SBM = base_SB - morb_factor * seg_dLdE[:, np.newaxis] * B2 / (2 * np.pi * PHI0)
        result[f'SBM{sfx}'] = _solve_onsager(rhs, base_SBM, seg_valid, seg_E)

        if dChi_dE is not None:
            seg_chi = dChi_dE[s0:s1]
            base_SBMC = base_SBM - chi_factor * (2 * np.pi) * seg_chi[:, np.newaxis] * B2**2 / PHI0**2
            result[f'SBMC{sfx}'] = _solve_onsager(rhs, base_SBMC, seg_valid, seg_E)

    return result if result else None
