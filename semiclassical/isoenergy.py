"""
Contour-based isoenergy orbit detection.

Traces closed orbits at each energy level using marching squares on a
3x3 BZ-tiled energy surface.  Orbit area is computed via the shoelace
formula; enclosed k-points are found by polygon containment.

All functions operate on a single band at a time.
"""

import numpy as np
from skimage.measure import find_contours
from matplotlib.path import Path


def _shoelace_area(verts):
    """Area of a closed polygon (first vertex == last vertex)."""
    x, y = verts[:-1, 0], verts[:-1, 1]
    xp, yp = verts[1:, 0], verts[1:, 1]
    return 0.5 * np.abs(np.dot(x, yp) - np.dot(xp, y))


def isoenergy_areas(E_band, E_levels, vol_M, nk1, nk2):
    """
    Compute k-space orbit areas and enclosed k-point indices for one band.

    Parameters
    ----------
    E_band : (Nk_tot,) array
        Band energies at each k-point.
    E_levels : (nE,) array
        Energy values at which to find orbits.
    vol_M : float
        Real-space moire unit cell area.
    nk1, nk2 : int
        k-mesh dimensions (Nk_tot = nk1 * nk2).

    Returns
    -------
    areas : list of lists
        areas[i] = list of orbit areas at energy i, sorted descending.
    kindices : list of lists
        kindices[i] = list of 1-D index arrays for k-points inside each orbit.
    """
    nE = len(E_levels)
    Nk = nk1 * nk2
    BZ_area = (2 * np.pi)**2 / vol_M
    cell_area = BZ_area / Nk

    bmin = E_band.min()
    bmax = E_band.max()

    E_2d = E_band.reshape(nk2, nk1, order='F')

    areas = [[] for _ in range(nE)]
    kindices = [[] for _ in range(nE)]

    for i in range(nE):
        lvl = E_levels[i]
        if lvl <= bmin or lvl >= bmax:
            continue

        E_tiled = np.tile(E_2d, (3, 3))
        contours = find_contours(E_tiled, lvl)

        orbit_areas = []
        orbit_kidx = []

        for contour in contours:
            if np.linalg.norm(contour[0] - contour[-1]) > 1.0:
                continue
            if not np.allclose(contour[0], contour[-1]):
                contour = np.vstack([contour, contour[:1]])
            if len(contour) < 4:
                continue

            cx, cy = contour[:-1].mean(axis=0)
            if cx < nk2 or cx >= 2 * nk2 or cy < nk1 or cy >= 2 * nk1:
                continue

            area_k = _shoelace_area(contour) * cell_area
            if area_k < cell_area or area_k >= BZ_area - cell_area:
                continue

            r0 = max(int(np.floor(contour[:, 0].min())), 0)
            r1 = min(int(np.ceil(contour[:, 0].max())), 3 * nk2 - 1)
            c0 = max(int(np.floor(contour[:, 1].min())), 0)
            c1 = min(int(np.ceil(contour[:, 1].max())), 3 * nk1 - 1)

            rr, cc = np.meshgrid(
                np.arange(r0, r1 + 1),
                np.arange(c0, c1 + 1),
                indexing='ij')
            pts = np.column_stack([rr.ravel(), cc.ravel()])

            inside = Path(contour).contains_points(pts)

            orig_r = pts[inside, 0] % nk2
            orig_c = pts[inside, 1] % nk1
            orig_lin = np.unique(orig_c * nk2 + orig_r)

            if len(orig_lin) == 0 or len(orig_lin) >= Nk:
                continue

            orbit_areas.append(area_k)
            orbit_kidx.append(orig_lin)

        if orbit_areas:
            order = np.argsort(orbit_areas)[::-1]
            areas[i] = [orbit_areas[j] for j in order]
            kindices[i] = [orbit_kidx[j] for j in order]

    return areas, kindices


def get_energy_resolved_data(kT, E_band, Oz_band, Lz_band,
                             E_levels, vol_M, nk1, nk2):
    """
    Compute orbit areas, enclosed Berry curvature, and dL/dE for one band.

    Parameters
    ----------
    kT : float
        Thermal broadening (same energy units as E_band).
    E_band : (Nk,) array
        Band energies.
    Oz_band : (Nk,) array
        Berry curvature.
    Lz_band : (Nk,) array
        Orbital moment.
    E_levels : (nE,) array
        Energy grid for this band.
    vol_M : float
        Moire cell area.
    nk1, nk2 : int
        k-mesh dimensions.

    Returns
    -------
    area : (nE, max_pockets) array
    enclosedBC : (nE, max_pockets) array
    dL_dE : (nE,) array
    """
    A, K = isoenergy_areas(E_band, E_levels, vol_M, nk1, nk2)
    nE = len(E_levels)
    Nk = nk1 * nk2
    kweight = (2 * np.pi)**2 / (Nk * vol_M)

    max_pockets = max((len(a) for a in A if a), default=1)

    area = np.zeros((nE, max_pockets))
    enclosedBC = np.zeros((nE, max_pockets))

    for i in range(nE):
        if not A[i]:
            continue
        for p, (a_val, k_idx) in enumerate(zip(A[i], K[i])):
            area[i, p] = a_val
            enclosedBC[i, p] = np.sum(Oz_band[k_idx]) * kweight

    E_col = E_levels[:, np.newaxis]
    x = (E_band - E_col) / kT
    dfde = np.exp(-np.abs(x)) / (kT * (1 + np.exp(-np.abs(x)))**2)
    dL_dE = (dfde @ Lz_band) * kweight

    return area, enclosedBC, dL_dE
