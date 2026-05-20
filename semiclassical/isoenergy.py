"""
Contour-based isoenergy orbit detection.

Traces closed orbits at each energy level using marching squares on a
3x3 BZ-tiled energy surface.  Orbit area is computed via the shoelace
formula; enclosed k-points are found by polygon containment.
"""

import numpy as np
from skimage.measure import find_contours
from matplotlib.path import Path


def _shoelace_area(verts):
    """Area of a closed polygon (first vertex == last vertex)."""
    x, y = verts[:-1, 0], verts[:-1, 1]
    xp, yp = verts[1:, 0], verts[1:, 1]
    return 0.5 * np.abs(np.dot(x, yp) - np.dot(xp, y))


def isoenergy_areas(E_bands, E_levels, vol_M, nk1, nk2):
    """
    Compute k-space orbit areas and enclosed k-point indices.

    Parameters
    ----------
    E_bands : (num_bands, Nk_tot) array
        Band energies at each k-point (any units).
    E_levels : (NE,) array
        Energy values at which to find orbits.
    vol_M : float
        Real-space moire unit cell area (same length units as k).
    nk1, nk2 : int
        k-mesh dimensions (E_bands columns = nk1 * nk2).

    Returns
    -------
    areas : list of lists
        areas[n][i] = list of orbit areas for band n at energy i,
        sorted descending.  Empty list if no orbits.
    kindices : list of lists
        kindices[n][i] = list of 1-D index arrays (into E_bands columns)
        for k-points inside each orbit.  Ordering matches areas.
    """
    num_bands = E_bands.shape[0]
    NE = len(E_levels)
    Nk = nk1 * nk2
    BZ_area = (2 * np.pi)**2 / vol_M
    cell_area = BZ_area / Nk

    Emin = E_bands.min(axis=1)
    Emax = E_bands.max(axis=1)

    areas = [[[] for _ in range(NE)] for _ in range(num_bands)]
    kindices = [[[] for _ in range(NE)] for _ in range(num_bands)]

    for n in range(num_bands):
        E_2d = E_bands[n, :].reshape(nk2, nk1, order='F')
        bmin, bmax = Emin[n], Emax[n]

        for i in range(NE):
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

                # F-order linear index: row + col * nk2
                orig_r = pts[inside, 0] % nk2
                orig_c = pts[inside, 1] % nk1
                orig_lin = np.unique(orig_c * nk2 + orig_r)

                if len(orig_lin) == 0 or len(orig_lin) >= Nk:
                    continue

                orbit_areas.append(area_k)
                orbit_kidx.append(orig_lin)

            if orbit_areas:
                order = np.argsort(orbit_areas)[::-1]
                areas[n][i] = [orbit_areas[j] for j in order]
                kindices[n][i] = [orbit_kidx[j] for j in order]

    return areas, kindices


def get_energy_resolved_data(bands_sel, kT, E_bands, Oz, Lz,
                             E_levels, vol_M, nk1, nk2):
    """
    Compute orbit areas, enclosed Berry curvature, and dL/dE.

    Parameters
    ----------
    bands_sel : array-like
        Band offset indices (e.g. [-3,-2,...,3]).
    kT : float
        Thermal broadening (same energy units as E_bands).
    E_bands : (num_bands, Nk) array
        Band energies.
    Oz : (num_bands, Nk) array
        Berry curvature at each (band, k).
    Lz : (num_bands, Nk) array
        Orbital moment at each (band, k).
    E_levels : (NE,) array
        Energy grid.
    vol_M : float
        Moire cell area.
    nk1, nk2 : int
        k-mesh dimensions.

    Returns
    -------
    area : (NE, nbands, max_pockets) array
    enclosedBC : (NE, nbands, max_pockets) array
    dL_dE : (NE, nbands) array
    """
    A, K = isoenergy_areas(E_bands, E_levels, vol_M, nk1, nk2)
    nbands = len(bands_sel)
    NE = len(E_levels)
    Nk = nk1 * nk2
    kweight = (2 * np.pi)**2 / (Nk * vol_M)

    max_pockets = 1
    for n in range(nbands):
        for i in range(NE):
            if A[n][i]:
                max_pockets = max(max_pockets, len(A[n][i]))

    area = np.zeros((NE, nbands, max_pockets))
    enclosedBC = np.zeros((NE, nbands, max_pockets))

    for n in range(nbands):
        for i in range(NE):
            if not A[n][i]:
                continue
            for p, (a_val, k_idx) in enumerate(zip(A[n][i], K[n][i])):
                area[i, n, p] = a_val
                enclosedBC[i, n, p] = np.sum(Oz[n, k_idx]) * kweight

    # Vectorized dL/dE
    dL_dE = np.zeros((NE, nbands))
    E_col = E_levels[:, np.newaxis]
    for n in range(nbands):
        tek = E_bands[n, :]
        x = (tek - E_col) / kT
        dfde = np.exp(-np.abs(x)) / (kT * (1 + np.exp(-np.abs(x)))**2)
        dL_dE[:, n] = (dfde @ Lz[n, :]) * kweight

    return area, enclosedBC, dL_dE
