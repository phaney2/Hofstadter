"""
Semiclassical Onsager quantization.

Computes Landau level fan diagram E(B) from orbit areas, enclosed Berry
curvature, and orbital moment derivative dL/dE.
"""

import numpy as np


EC = 1.60217662e-19       # C
HBAR_SI = 1.05457182e-34  # J*s
PHI0 = 2 * np.pi * HBAR_SI / EC


def onsager_fan(Blist, nmax, E_levels, area, enclosedBC, dL_dE,
                dChi_dE=None, termflags=(1, 1, 1)):
    """
    Solve the Onsager quantization condition for each band.

    S(E)/(2pi)^2 + BCflag * Phi_B*B/(2pi*phi0)
                 + morbflag * (dL/dE)*B/(2pi*phi0)
                 + chiflag * (2pi)*dChi/dE * B^2 / phi0^2
                 = B*(n+1/2)/phi0

    Parameters
    ----------
    Blist : (nB,) array
        Magnetic field values in Tesla.
    nmax : int
        Maximum Landau level index.
    E_levels : (NE,) array
        Energy grid (same units as area/enclosedBC/dL_dE).
    area : (NE, nbands, npockets) array
        Orbit areas at each energy.
    enclosedBC : (NE, nbands, npockets) array
        Enclosed Berry curvature at each energy.
    dL_dE : (NE, nbands) array
        Energy derivative of orbital moment.
    dChi_dE : (NE,) array or None
        Susceptibility derivative. Required if chiflag is set.
    termflags : (3,) tuple of {0, 1}
        (BCflag, morbflag, chiflag) — toggle each correction term.

    Returns
    -------
    LL_all : list of (nB, nmax+1) arrays
        Landau level energies for each band that has nonzero orbits.
        Each entry corresponds to one band.
    band_indices : list of int
        Which band index each LL_all entry corresponds to.
    """
    BCflag, morbflag, chiflag = termflags

    nB = len(Blist)
    NE, nbands, npockets = area.shape

    n_vec = np.arange(nmax + 1).reshape(1, 1, -1)       # 1 x 1 x (nmax+1)
    B_vec = np.asarray(Blist).reshape(1, -1, 1)          # 1 x nB x 1

    rhs = B_vec * (n_vec + 0.5) / PHI0                   # 1 x nB x (nmax+1)

    LL_all = []
    band_indices = []

    for bandc in range(nbands):
        tarea = area[:, bandc, 0]
        if np.max(tarea) == 0:
            continue

        tBC = enclosedBC[:, bandc, 0]
        tdL_dE = dL_dE[:, bandc]

        B2 = B_vec[:, :, 0]                                         # 1 x nB
        base = -tarea[:, np.newaxis] * np.ones_like(B2) / (2 * np.pi)**2  # NE x nB
        if BCflag:
            base -= tBC[:, np.newaxis] * B2 / (2 * np.pi * PHI0)
        if morbflag:
            base -= tdL_dE[:, np.newaxis] * B2 / (2 * np.pi * PHI0)
        if chiflag and dChi_dE is not None:
            base -= (2 * np.pi) * dChi_dE[:, np.newaxis] * B2**2 / PHI0**2

        onsager = rhs + base[:, :, np.newaxis]                      # NE x nB x (nmax+1)
        valid = tarea > 0                                           # NE mask
        onsager[~valid, :, :] = np.inf
        ind = np.argmin(np.abs(onsager), axis=0)                    # nB x (nmax+1)
        LL = E_levels[ind]                                          # nB x (nmax+1)

        LL_all.append(LL)
        band_indices.append(bandc)

    return LL_all, band_indices
