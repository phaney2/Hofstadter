"""
Semiclassical Onsager quantization.

Computes Landau level fan diagram E(B) from orbit areas, enclosed Berry
curvature, and orbital moment derivative dL/dE.  Operates on a single
band at a time.
"""

import numpy as np


EC = 1.60217662e-19       # C
HBAR_SI = 1.05457182e-34  # J*s
PHI0 = 2 * np.pi * HBAR_SI / EC


def onsager_fan_band(Blist, nmax, E_levels, area, enclosedBC, dL_dE,
                     dChi_dE=None, termflags=(1, 1, 1)):
    """
    Solve the Onsager quantization condition for one band.

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
    termflags : (3,) tuple of {0, 1}
        (BCflag, morbflag, chiflag).

    Returns
    -------
    LL : (nB, nmax+1) array or None
        Landau level energies.  None if no orbits found.
    """
    BCflag, morbflag, chiflag = termflags

    tarea = area[:, 0]
    if np.max(tarea) == 0:
        return None

    nB = len(Blist)
    tBC = enclosedBC[:, 0]

    n_vec = np.arange(nmax + 1).reshape(1, 1, -1)       # 1 x 1 x (nmax+1)
    B_vec = np.asarray(Blist).reshape(1, -1, 1)          # 1 x nB x 1

    rhs = B_vec * (n_vec + 0.5) / PHI0                   # 1 x nB x (nmax+1)

    B2 = B_vec[:, :, 0]                                   # 1 x nB
    base = -tarea[:, np.newaxis] * np.ones_like(B2) / (2 * np.pi)**2
    if BCflag:
        base -= tBC[:, np.newaxis] * B2 / (2 * np.pi * PHI0)
    if morbflag:
        base -= dL_dE[:, np.newaxis] * B2 / (2 * np.pi * PHI0)
    if chiflag and dChi_dE is not None:
        base -= (2 * np.pi) * dChi_dE[:, np.newaxis] * B2**2 / PHI0**2

    onsager = rhs + base[:, :, np.newaxis]                 # nE x nB x (nmax+1)
    valid = tarea > 0
    onsager[~valid, :, :] = np.inf
    ind = np.argmin(np.abs(onsager), axis=0)               # nB x (nmax+1)
    LL = E_levels[ind]

    return LL
