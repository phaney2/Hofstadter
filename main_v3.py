"""
Magnetic Bloch bands for mono/bilayer graphene on hBN substrate.

Uses a qq-site guiding-center chain (Nq = qq) built on the
centered-rectangular cell of the moire lattice (flux qq/(2*pp) per
primitive moire cell), with corrected moire coupling matrices
(order=[3,1,2], conj=1, psiconj=1).

Reference: "A Quantum Ruler for Orbital Magnetism in Moire Quantum Matter"
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import numpy as np
from scipy import linalg
from multiprocessing import Pool, cpu_count

from constants import HBAR, Q_E, A_GRAPHENE, A_HBN
from parser import parse_input_file
from basis import getindices
from hamiltonian import (get_interbilayerterms_K, get_interbilayerterms_Kp,
                         get_intermonolayerH_K, get_intermonolayerH_Kp,
                         get_intralayerH_K, get_intralayerH_Kp,
                         get_berry_connection_K, get_berry_connection_Kp)

HBAR_EV = 6.582119569e-16  # eV*s

# np.trapezoid is the numpy>=2 name for np.trapz
_trapz = getattr(np, 'trapezoid', np.trapz)


# ---------------------------------------------------------------------------
# Per-k-point solver (serial core + parallel wrapper)
# ---------------------------------------------------------------------------

def _solve_kpoint_core(d, kpt):
    """Solve both valleys at a single k-point.

    Returns (tek_K, tek_Kp, wt_K, wt_Kp).
    wt_K/wt_Kp are top-layer weights per eigenstate when layer_resolved=1,
    otherwise None.
    """
    kpts = kpt - d['M_mag']
    kx_val, ky_val = kpts
    pp, qq = d['pp'], d['qq']
    Lx, Ly = d['Lx'], d['Ly']
    gamma = d['gamma']
    mo = d['moire_offset']
    layer_resolved = d.get('layer_resolved', 0)
    dl = d.get('dim_MLG', 0)

    tek_K = None
    tek_Kp = None
    wt_K = None
    wt_Kp = None

    if 'K' in d['valley']:
        tphase1 = np.exp(1j * (pp / qq) * kx_val * Lx)
        tphase2 = np.exp(-1j * (pp / qq) * kx_val * Lx / 2) * np.exp(1j * ky_val * Ly * (pp / qq))
        tphase3 = np.exp(-1j * (pp / qq) * kx_val * Lx / 2) * np.exp(-1j * ky_val * Ly * (pp / qq))

        V_pq = gamma * tphase1 * d['term1_K'] + tphase2 * d['term2_K'] + tphase3 * d['term3_K']

        Htotal_K = d['H_base_K'].copy()
        Htotal_K[mo:, mo:] += d['v0_eye'] + V_pq + V_pq.T.conj()

        if layer_resolved:
            tek_K, evecs = linalg.eigh(Htotal_K, overwrite_a=True, check_finite=False)
            wt_K = np.sum(np.abs(evecs[:dl, :])**2, axis=0)
        else:
            tek_K = np.sort(linalg.eigvalsh(Htotal_K, overwrite_a=True, check_finite=False))

    if 'Kp' in d['valley']:
        tphase1 = np.exp(-1j * (pp / qq) * kx_val * Lx)
        tphase2 = np.exp(1j * (pp / qq) * kx_val * Lx / 2) * np.exp(-1j * ky_val * Ly * (pp / qq))
        tphase3 = np.exp(1j * (pp / qq) * kx_val * Lx / 2) * np.exp(1j * ky_val * Ly * (pp / qq))

        V_pq = gamma * tphase1 * d['term1_Kp'] + tphase2 * d['term2_Kp'] + tphase3 * d['term3_Kp']

        Htotal_Kp = d['H_base_Kp'].copy()
        Htotal_Kp[mo:, mo:] += d['v0_eye'] + V_pq + V_pq.T.conj()

        if layer_resolved:
            tek_Kp, evecs = linalg.eigh(Htotal_Kp, overwrite_a=True, check_finite=False)
            wt_Kp = np.sum(np.abs(evecs[:dl, :])**2, axis=0)
        else:
            tek_Kp = np.sort(linalg.eigvalsh(Htotal_Kp, overwrite_a=True, check_finite=False))

    return tek_K, tek_Kp, wt_K, wt_Kp


_worker_shared = {}

def _init_kpoint_worker(shared):
    global _worker_shared
    _worker_shared = shared


def _solve_kpoint(args):
    kc, kpt = args
    tek_K, tek_Kp, wt_K, wt_Kp = _solve_kpoint_core(_worker_shared, kpt)
    return kc, tek_K, tek_Kp, wt_K, wt_Kp


# ---------------------------------------------------------------------------
# Transport: per-k-point solver with in-worker Kubo summation
# ---------------------------------------------------------------------------

def _transport_kubo_single_k(E_meV, vx, vy, d):
    """Compute per-k-point contribution to transport sums.

    Returns (sxx, sxy, l12xx, l12xy) each shape (n_gamma, n_mu).
    Prefactors (magnetic cell area, 1/Nk) are applied in the caller.

    In SCBA mode (d['scba_Gamma_E'] is not None), sxx and l12xx include
    the Gamma(eps)^2 factor from the normalized spectral functions, so
    the caller's pf_xx must NOT contain G^2.  In constant mode, sxx is
    G^2-free as before.
    """
    E = E_meV / 1000.0
    all_mu = d['all_mu_eV']
    Gamma_list = d['Gamma_list_eV']
    n_gamma = len(Gamma_list)
    kT = d['kT_eV']

    scba_grid = d.get('scba_E_grid')
    scba_Gamma = d.get('scba_Gamma_E')
    use_scba = scba_Gamma is not None
    scba_xy_constant = d.get('scba_xy_constant', 0)

    vx_sq = np.abs(vx) ** 2
    Omega = np.imag(vx * np.conj(vy))
    D2 = (E[:, None] - E[None, :]) ** 2

    n_mu = len(all_mu)
    sxx = np.zeros((n_gamma, n_mu))
    sxy = np.zeros((n_gamma, n_mu))
    l12xx = np.zeros((n_gamma, n_mu))
    l12xy = np.zeros((n_gamma, n_mu))

    sort_idx = np.argsort(E)
    E_sorted = E[sort_idx]

    if kT > 0:
        margin = 10.0 * kT
        eps_lo = all_mu.min() - margin
        eps_hi = all_mu.max() + margin
        if use_scba:
            G_for_grid = np.min(scba_Gamma)
        else:
            G_for_grid = np.min(Gamma_list)
        d_eps = min(G_for_grid, kT) / 5.0
        n_eps = max(int(np.ceil((eps_hi - eps_lo) / d_eps)) + 1, 50)
        eps_grid = np.linspace(eps_lo, eps_hi, n_eps)
        dE2 = (eps_grid[:, None] - E[None, :]) ** 2

    for ig, G in enumerate(Gamma_list):
        G2 = G * G

        if use_scba and not scba_xy_constant:
            G_n = np.interp(E, scba_grid, scba_Gamma)
            G2_n = G_n ** 2
            inv_D2G2 = 1.0 / (D2 + G2_n[:, None])
        else:
            inv_D2G2 = 1.0 / (D2 + G2)

        K_xy = Omega * inv_D2G2
        np.fill_diagonal(K_xy, 0.0)
        K_n = np.sum(K_xy, axis=1)

        K_cumsum = np.concatenate(([0.0], np.cumsum(K_n[sort_idx])))

        if kT > 0:
            if use_scba:
                G_eps = np.interp(eps_grid, scba_grid, scba_Gamma)
                G2_eps = G_eps ** 2
                L_all = 1.0 / (dE2 + G2_eps[:, None])
                Phi_xx = np.sum((L_all @ vx_sq) * L_all, axis=1) * G2_eps
            else:
                L_all = 1.0 / (dE2 + G2)
                Phi_xx = np.sum((L_all @ vx_sq) * L_all, axis=1)

            bins = np.searchsorted(E_sorted, eps_grid, side='right')
            Phi_xy = K_cumsum[bins]

        for i_mu, mu in enumerate(all_mu):
            if kT > 0:
                x_eps = (eps_grid - mu) / kT
                x_eps_clip = np.clip(x_eps, -500, 500)
                f_eps = 1.0 / (np.exp(x_eps_clip) + 1.0)
                neg_dfde = (1.0 / kT) * f_eps * (1.0 - f_eps)
                sxx[ig, i_mu] = _trapz(neg_dfde * Phi_xx, eps_grid)
                sxy[ig, i_mu] = _trapz(neg_dfde * Phi_xy, eps_grid)
                l12xx[ig, i_mu] = _trapz(
                    (eps_grid - mu) * neg_dfde * Phi_xx, eps_grid)
                l12xy[ig, i_mu] = _trapz(
                    (eps_grid - mu) * neg_dfde * Phi_xy, eps_grid)
            else:
                if use_scba:
                    G_mu = float(np.interp(mu, scba_grid, scba_Gamma))
                    G2_mu = G_mu * G_mu
                    L = 1.0 / ((E - mu) ** 2 + G2_mu)
                    sxx[ig, i_mu] = G2_mu * (L @ vx_sq @ L)
                else:
                    L = 1.0 / ((E - mu) ** 2 + G2)
                    sxx[ig, i_mu] = L @ vx_sq @ L
                sxy[ig, i_mu] = np.sum(K_n[E < mu])

    return sxx, sxy, l12xx, l12xy


def _solve_scba(all_eigs_eV, Gamma0, pp, Nk, mixing=0.3, tol=1e-4,
                maxiter=200, floor_ratio=0.01, anderson_depth=5):
    """Compute self-consistent energy-dependent broadening Gamma(E).

    Iterates the SCBA equation Gamma(E) = pi * Gamma0^2 * rho(E) where
    rho(E) is the physical DOS per primary moire unit cell, computed with
    the current Gamma(E).

    The k-sum covers the magnetic BZ (1/pp of the moire BZ, halved again
    by the doubled real-space cell), so the physical DOS normalization is
    1/(Nk * 2*pp) per eigenvalue sum.

    Uses Anderson/Pulay mixing when anderson_depth > 0 (default 5).
    Falls back to linear mixing with parameter `mixing` for the first
    iteration and whenever the Anderson least-squares problem is singular.

    Returns (E_grid, Gamma_E, niter) where E_grid and Gamma_E are in eV.
    """
    eigs_flat = all_eigs_eV.ravel()
    E_min, E_max = eigs_flat.min(), eigs_flat.max()
    margin = 5.0 * Gamma0 + 0.1 * (E_max - E_min)
    dE = Gamma0 / 10.0
    n_grid = max(int(np.ceil((E_max - E_min + 2 * margin) / dE)) + 1, 500)
    E_grid = np.linspace(E_min - margin, E_max + margin, n_grid)

    Gamma_E = np.full(n_grid, Gamma0)
    Gamma_min = floor_ratio * Gamma0
    n_eigs = len(eigs_flat)
    dos_norm = 1.0 / (np.pi * Nk * 2 * pp)
    chunk = max(1, 50_000_000 // n_eigs)

    import time as _time
    print(f"  SCBA: n_grid={n_grid}, n_eigs={n_eigs}, chunk={chunk}")
    t_start = _time.time()

    M = max(0, anderson_depth)
    hist_x = []
    hist_r = []

    for it in range(maxiter):
        G2 = Gamma_E ** 2
        rho = np.zeros(n_grid)
        for i0 in range(0, n_grid, chunk):
            i1 = min(i0 + chunk, n_grid)
            diffs2 = (E_grid[i0:i1, None] - eigs_flat[None, :]) ** 2
            rho[i0:i1] = np.sum(
                Gamma_E[i0:i1, None] / (diffs2 + G2[i0:i1, None]), axis=1)
        rho *= dos_norm

        Gamma_new = np.pi * Gamma0 ** 2 * rho
        np.maximum(Gamma_new, Gamma_min, out=Gamma_new)

        R = Gamma_new - Gamma_E
        residual = np.max(np.abs(R)) / np.max(Gamma_E)
        dt = _time.time() - t_start
        print(f"  SCBA iter {it+1}: residual = {residual:.2e} ({dt:.1f}s)",
              flush=True)

        if residual < tol:
            print(f"  SCBA converged in {it + 1} iterations (residual = {residual:.2e})")
            return E_grid, Gamma_new, it + 1

        if M > 0 and len(hist_r) > 0:
            m = len(hist_r)
            dR = np.column_stack([R - hist_r[j] for j in range(m)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(dR, R, rcond=None)
                dX = np.column_stack([
                    (Gamma_new - hist_x[j]) for j in range(m)])
                Gamma_E = Gamma_new - dX @ coeffs
                np.maximum(Gamma_E, Gamma_min, out=Gamma_E)
            except np.linalg.LinAlgError:
                Gamma_E = mixing * Gamma_new + (1.0 - mixing) * Gamma_E
        else:
            Gamma_E = mixing * Gamma_new + (1.0 - mixing) * Gamma_E

        hist_x.append(Gamma_new.copy())
        hist_r.append(R.copy())
        if len(hist_x) > M:
            hist_x.pop(0)
            hist_r.pop(0)

    print(f"  SCBA did NOT converge after {maxiter} iterations (residual = {residual:.2e})")
    return E_grid, Gamma_E, maxiter


def _solve_kpoint_eigenvalues_core(d, kpt):
    """Diagonalize H(k) and return eigenvalues only (no eigenvectors).

    Used for the SCBA eigenvalue-collection pass.
    """
    kpts = kpt - d['M_mag']
    kx_val, ky_val = kpts
    pp, qq = d['pp'], d['qq']
    Lx, Ly = d['Lx'], d['Ly']
    gamma = d['gamma']
    mo = d['moire_offset']
    band_sel = d['transport_band_sel']

    E_K = E_Kp = None

    if 'K' in d['valley']:
        tphase1 = np.exp(1j * (pp / qq) * kx_val * Lx)
        tphase2 = np.exp(-1j * (pp / qq) * kx_val * Lx / 2) * np.exp(1j * ky_val * Ly * (pp / qq))
        tphase3 = np.exp(-1j * (pp / qq) * kx_val * Lx / 2) * np.exp(-1j * ky_val * Ly * (pp / qq))
        V_pq = gamma * tphase1 * d['term1_K'] + tphase2 * d['term2_K'] + tphase3 * d['term3_K']
        Htotal_K = d['H_base_K'].copy()
        Htotal_K[mo:, mo:] += d['v0_eye'] + V_pq + V_pq.T.conj()
        ek = np.sort(linalg.eigvalsh(Htotal_K, overwrite_a=True, check_finite=False))
        E_K = ek[band_sel]

    if 'Kp' in d['valley']:
        tphase1 = np.exp(-1j * (pp / qq) * kx_val * Lx)
        tphase2 = np.exp(1j * (pp / qq) * kx_val * Lx / 2) * np.exp(-1j * ky_val * Ly * (pp / qq))
        tphase3 = np.exp(1j * (pp / qq) * kx_val * Lx / 2) * np.exp(1j * ky_val * Ly * (pp / qq))
        V_pq = gamma * tphase1 * d['term1_Kp'] + tphase2 * d['term2_Kp'] + tphase3 * d['term3_Kp']
        Htotal_Kp = d['H_base_Kp'].copy()
        Htotal_Kp[mo:, mo:] += d['v0_eye'] + V_pq + V_pq.T.conj()
        ek = np.sort(linalg.eigvalsh(Htotal_Kp, overwrite_a=True, check_finite=False))
        E_Kp = ek[band_sel]

    return E_K, E_Kp


def _solve_kpoint_eigenvalues(args):
    kc, kpt = args
    E_K, E_Kp = _solve_kpoint_eigenvalues_core(_worker_shared, kpt)
    return kc, E_K, E_Kp


def _solve_kpoint_transport_core(d, kpt):
    """Diagonalize H(k), compute velocity elements, and do Kubo sums.

    Returns per-k partial transport arrays (no large matrices cross the pipe).
    """
    kpts = kpt - d['M_mag']
    kx_val, ky_val = kpts
    pp, qq = d['pp'], d['qq']
    Lx, Ly = d['Lx'], d['Ly']
    gamma = d['gamma']
    mo = d['moire_offset']
    band_sel = d['transport_band_sel']

    n_mu = len(d['all_mu_eV'])
    sxx_K = sxy_K = l12xx_K = l12xy_K = None
    sxx_Kp = sxy_Kp = l12xx_Kp = l12xy_Kp = None
    E_K = E_Kp = None

    if 'K' in d['valley']:
        tphase1 = np.exp(1j * (pp / qq) * kx_val * Lx)
        tphase2 = np.exp(-1j * (pp / qq) * kx_val * Lx / 2) * np.exp(1j * ky_val * Ly * (pp / qq))
        tphase3 = np.exp(-1j * (pp / qq) * kx_val * Lx / 2) * np.exp(-1j * ky_val * Ly * (pp / qq))

        V_pq = gamma * tphase1 * d['term1_K'] + tphase2 * d['term2_K'] + tphase3 * d['term3_K']

        Htotal_K = d['H_base_K'].copy()
        Htotal_K[mo:, mo:] += d['v0_eye'] + V_pq + V_pq.T.conj()

        ek, Psi = linalg.eigh(Htotal_K, overwrite_a=True, check_finite=False)
        E_K = ek[band_sel]
        Psi_sel = Psi[:, band_sel]
        vx_K = Psi_sel.conj().T @ d['Vx_K'] @ Psi_sel
        vy_K = Psi_sel.conj().T @ d['Vy_K'] @ Psi_sel
        sxx_K, sxy_K, l12xx_K, l12xy_K = _transport_kubo_single_k(
            E_K, vx_K, vy_K, d)

    if 'Kp' in d['valley']:
        tphase1 = np.exp(-1j * (pp / qq) * kx_val * Lx)
        tphase2 = np.exp(1j * (pp / qq) * kx_val * Lx / 2) * np.exp(-1j * ky_val * Ly * (pp / qq))
        tphase3 = np.exp(1j * (pp / qq) * kx_val * Lx / 2) * np.exp(1j * ky_val * Ly * (pp / qq))

        V_pq = gamma * tphase1 * d['term1_Kp'] + tphase2 * d['term2_Kp'] + tphase3 * d['term3_Kp']

        Htotal_Kp = d['H_base_Kp'].copy()
        Htotal_Kp[mo:, mo:] += d['v0_eye'] + V_pq + V_pq.T.conj()

        ek, Psi = linalg.eigh(Htotal_Kp, overwrite_a=True, check_finite=False)
        E_Kp = ek[band_sel]
        Psi_sel = Psi[:, band_sel]
        vx_Kp = Psi_sel.conj().T @ d['Vx_Kp'] @ Psi_sel
        vy_Kp = Psi_sel.conj().T @ d['Vy_Kp'] @ Psi_sel
        sxx_Kp, sxy_Kp, l12xx_Kp, l12xy_Kp = _transport_kubo_single_k(
            E_Kp, vx_Kp, vy_Kp, d)

    return (sxx_K, sxy_K, l12xx_K, l12xy_K, E_K,
            sxx_Kp, sxy_Kp, l12xx_Kp, l12xy_Kp, E_Kp)


def _solve_kpoint_transport(args):
    kc, kpt = args
    r = _solve_kpoint_transport_core(_worker_shared, kpt)
    return (kc,) + r


# ---------------------------------------------------------------------------
# Main calculation routine
# ---------------------------------------------------------------------------

def do_calc(filepath):
    """
    Compute magnetic Bloch bands for mono/bilayer graphene on hBN.

    Returns a dict with keys depending on calctype:
      'ek':  kpoints, bands_K, bands_Kp
      'dos': elist, dos_K, dos_Kp
    """
    # --- defaults ---
    eta = 1
    vF = 1e6
    gamma = 1
    Nmax = 5000
    calctype = 'ek'
    valley = ['K', 'Kp']

    # --- read input ---
    inp = parse_input_file(filepath)
    d = dict(inp)

    theta = np.radians(d.get('theta', 0.0))
    qq = int(d['qq'])
    pp = int(d['pp'])
    g0 = d['g0']
    nlayers = int(d.get('nlayers', 2))
    if nlayers == 2:
        g1 = d['g1']
        g3 = d['g3']
        g4 = d['g4']
    else:
        g1 = d.get('g1', 0)
        g3 = d.get('g3', 0)
        g4 = d.get('g4', 0)
    delta = 0 if nlayers == 1 else d.get('delta', 0)
    v0_meV = d['v0']
    v1_meV = d['v1']
    w_meV = d['w']
    eta = d.get('eta', eta)
    U = np.atleast_1d(d.get('U', np.array([0, 0])))
    if len(U) == 1:
        U = np.array([U[0], 0])
    nk1 = int(d.get('nk1', 25))
    nk2 = int(d.get('nk2', 40))
    LL_multiplier = d.get('LL_multiplier', 6)
    Nmax = int(d.get('Nmax', Nmax))
    calctype = d.get('calctype', calctype)
    valley = d.get('valley', valley)
    nebin = int(d.get('nebin', 1000))
    gamma = d.get('gamma', gamma)
    vF = d.get('vF', vF)
    isparallel = int(d.get('isparallel', 0))
    elist = np.asarray(d.get('elist', np.linspace(-300, 300, nebin)))
    layer_resolved = int(d.get('layer_resolved', 0))
    if layer_resolved and nlayers == 1:
        layer_resolved = 0
    stacking_type = int(d.get('stacking_type', 2))
    psi = float(d.get('moire_psi', 0.29))

    if calctype == 'spectrum':
        calctype = 'dos'

    # --- derived quantities ---
    eps = A_HBN / A_GRAPHENE - 1
    L_moire = (1 + eps) * A_GRAPHENE / np.sqrt(eps ** 2 + 2 * (1 + eps) * (1 - np.cos(theta)))

    ktheta = 4 * np.pi / (3 ** 0.5 * L_moire)
    # Primitive moire unit cell area.  The magnetic flux per primitive
    # cell is qq/(2*pp) flux quanta; the factor 2 arises because the
    # Landau-gauge construction is built on the centered-rectangular
    # (two-lattice-point) cell of the triangular moire lattice, which
    # carries flux qq/pp.
    A_uc = 3 ** 0.5 * L_moire ** 2 / 2

    phi_0 = HBAR * 2 * np.pi / Q_E
    B = (qq / (2 * pp)) * phi_0 / A_uc
    lB = (HBAR / (Q_E * B)) ** 0.5

    eneLL = g0 / 1e3 * Q_E * A_GRAPHENE / lB * 2 ** 0.5

    w_J = w_meV / 1e3 * Q_E
    v0 = v0_meV / 1e3 * Q_E
    v1 = v1_meV / 1e3 * Q_E

    N = int(LL_multiplier * round(max(HBAR * vF * ktheta, w_J) / eneLL) ** 2)
    if N > Nmax:
        N = Nmax

    TBGparams = {
        'g0': g0 / 1e3 * Q_E,
        'g1': g1 / 1e3 * Q_E,
        'g3': g3 / 1e3 * Q_E,
        'g4': g4 / 1e3 * Q_E,
        'delta': delta / 1e3 * Q_E,
    }

    Nq = 1 * qq

    Lx = L_moire
    Ly = np.sqrt(3) * L_moire / 2

    print(f"  nlayers = {nlayers}")
    print(f"  N (Landau levels) = {N}")
    print(f"  Nq (chain size) = {Nq}")
    print(f"  B = {B:.6e} T")

    # --- K valley: k-independent Hamiltonian ---
    if 'K' in valley:
        print("  Building K-valley Hamiltonian...")
        term1_K, term2_K, term3_K, qNslabels_K = get_interbilayerterms_K(
            N, Nq, ktheta, lB, v0, v1, eta, qq, pp, theta, psi)
        Hintra_K = get_intralayerH_K(N, 0, B, qNslabels_K, TBGparams, 'A')

        dl = Hintra_K.shape[0]
        if nlayers == 1:
            U_onsite_val = U[0] / 1e3 * Q_E
            H_base_K = Hintra_K + np.eye(dl) * U_onsite_val
        else:
            Utp_val = U[0] / 1e3 * Q_E
            Ubm_val = U[1] / 1e3 * Q_E
            Hinter_K = get_intermonolayerH_K(N, 0, B, qNslabels_K, TBGparams)
            Hintra2_K = get_intralayerH_K(N, 0, B, qNslabels_K, TBGparams, 'B')
            Utp = np.eye(dl) * Utp_val
            Ubm = np.eye(dl) * Ubm_val
            if stacking_type == 1:
                H_base_K = np.block([
                    [Hintra_K + Utp, Hinter_K.T.conj()],
                    [Hinter_K, Hintra2_K + Ubm]
                ])
            else:
                H_base_K = np.block([
                    [Hintra_K + Utp, Hinter_K],
                    [Hinter_K.T.conj(), Hintra2_K + Ubm]
                ])

    # --- K' valley: k-independent Hamiltonian ---
    if 'Kp' in valley:
        print("  Building K'-valley Hamiltonian...")
        term1_Kp, term2_Kp, term3_Kp, qNslabels_Kp = get_interbilayerterms_Kp(
            N, Nq, ktheta, lB, v0, v1, eta, qq, pp, theta, psi)
        Hintra_Kp = get_intralayerH_Kp(N, 0, B, qNslabels_Kp, TBGparams, 'A')

        dl = Hintra_Kp.shape[0]
        if nlayers == 1:
            U_onsite_val = U[0] / 1e3 * Q_E
            H_base_Kp = Hintra_Kp + np.eye(dl) * U_onsite_val
        else:
            Utp_val = U[0] / 1e3 * Q_E
            Ubm_val = U[1] / 1e3 * Q_E
            Hinter_Kp = get_intermonolayerH_Kp(N, 0, B, qNslabels_Kp, TBGparams)
            Hintra2_Kp = get_intralayerH_Kp(N, 0, B, qNslabels_Kp, TBGparams, 'B')
            Utp = np.eye(dl) * Utp_val
            Ubm = np.eye(dl) * Ubm_val
            if stacking_type == 1:
                H_base_Kp = np.block([
                    [Hintra_Kp + Utp, Hinter_Kp.T.conj()],
                    [Hinter_Kp, Hintra2_Kp + Ubm]
                ])
            else:
                H_base_Kp = np.block([
                    [Hintra_Kp + Utp, Hinter_Kp],
                    [Hinter_Kp.T.conj(), Hintra2_Kp + Ubm]
                ])

    # --- k-mesh ---
    b1 = ktheta * np.array([0, -1])
    b2 = ktheta * np.array([np.sqrt(3) / 2, 1 / 2])

    M_mag = 0.5 * b1 / pp

    Nk_tot = nk1 * nk2
    n1_arr = np.arange(nk1)
    n2_arr = np.arange(nk2)
    n1grid, n2grid = np.meshgrid(n1_arr, n2_arr)
    n11 = n1grid.flatten(order='F')
    n22 = n2grid.flatten(order='F')

    # Minimal sampling zone: all gauge-invariant quantities (spectrum,
    # Berry curvature kernel, |v|^2) are periodic under b1/pp and
    # qfac*b2/pp with qfac = gcd(2*pp, qq).  Sampling the full
    # qq-extended zone (full_zone = 1) gives identical k-averages at
    # qq/qfac times the cost.
    full_zone = int(d.get('full_zone', 0))
    qfac = qq if full_zone else int(np.gcd(2 * pp, qq))
    vb = np.array([b1 / pp, b2 * qfac / pp])
    if qfac != qq:
        print(f"  k-zone: [b1/pp, {qfac}*b2/pp] (minimal; full zone would be {qq}*b2/pp)")

    kpoints = np.zeros((Nk_tot, 2))
    for j in range(Nk_tot):
        frac = np.array([n11[j] / nk1, n22[j] / nk2])
        kpoints[j, :] = vb.T @ frac

    dim_MLG = Hintra_K.shape[0] if 'K' in valley else Hintra_Kp.shape[0]
    dim_total = dim_MLG if nlayers == 1 else 2 * dim_MLG
    print(f"  dim per layer (post-chop) = {dim_MLG}")
    print(f"  dim total = {dim_total}")

    # --- pre-scale and pack shared data for the k-point solver ---
    scale = 1000 / Q_E
    v0_eye_scaled = scale * v0 * np.eye(dim_MLG)
    moire_offset = 0 if nlayers == 1 else dim_MLG

    # =======================================================================
    # Transport calculation (completely separate k-loop)
    # =======================================================================
    if calctype == 'transport':
        mulist_meV = np.asarray(d.get('mulist', np.linspace(-50, 50, 200)))
        Gamma_input = d.get('Gamma', 1.0)
        Gamma_list_meV = np.atleast_1d(np.asarray(Gamma_input, dtype=float))
        transport_buffer_meV = d.get('transport_buffer', None)
        mu_ref_meV = d.get('mu_ref', None)
        kT_meV = float(d.get('kT', 0.0))
        broadening_mode = str(d.get('broadening', 'constant')).strip("'\"")
        scba_mixing = float(d.get('scba_mixing', 0.3))
        scba_tol = float(d.get('scba_tol', 1e-4))
        scba_maxiter = int(d.get('scba_maxiter', 200))
        scba_floor = float(d.get('scba_floor', 0.01))
        scba_anderson = int(d.get('scba_anderson', 5))
        scba_xy_constant = int(d.get('scba_xy_constant', 0))
        use_scba = broadening_mode == 'scba'

        if use_scba and len(Gamma_list_meV) > 1:
            print("  WARNING: Gamma list ignored in SCBA mode; using first value as Gamma_0")
            Gamma_list_meV = Gamma_list_meV[:1]

        n_gamma = len(Gamma_list_meV)
        Gamma_list_eV = Gamma_list_meV / 1000.0
        Gamma_eV = Gamma_list_eV[0]

        mulist_eV = mulist_meV / 1000.0
        kT_eV = kT_meV / 1000.0
        mu_ref_eV = float(mu_ref_meV) / 1000.0 if mu_ref_meV is not None else None
        J_to_eV = 1.0 / Q_E
        m_to_Ang = 1e10
        # Each k-point's spectrum holds exactly one magnetic unit cell's
        # worth of states: 2*pp primitive moire cells (flux qq/(2*pp) per
        # primitive cell -> qq flux quanta, i.e. qq guiding centers).
        # All k-integrated quantities are normalized per this area.
        A_mag_Ang2 = (2 * pp * A_uc) * m_to_Ang ** 2

        # --- Energy-based band selection via k=0 probe ---
        mu_min, mu_max = mulist_meV.min(), mulist_meV.max()
        mu_range = mu_max - mu_min
        if transport_buffer_meV is not None:
            kubo_buffer = float(transport_buffer_meV)
        else:
            kubo_buffer = max(mu_range, 500.0)
        kubo_lo = mu_min - kubo_buffer
        kubo_hi = mu_max + kubo_buffer
        scba_buffer = 5.0 * float(Gamma_list_meV[0])
        scba_lo = mu_min - scba_buffer
        scba_hi = mu_max + scba_buffer

        kpt_probe = np.array([0.0, 0.0])
        kpts_probe = kpt_probe - M_mag
        kx_p, ky_p = kpts_probe
        if 'K' in valley:
            tp1 = np.exp(1j * (pp / qq) * kx_p * Lx)
            tp2 = np.exp(-1j * (pp / qq) * kx_p * Lx / 2) * np.exp(1j * ky_p * Ly * (pp / qq))
            tp3 = np.exp(-1j * (pp / qq) * kx_p * Lx / 2) * np.exp(-1j * ky_p * Ly * (pp / qq))
            V_probe = gamma * tp1 * term1_K + tp2 * term2_K + tp3 * term3_K
            H_probe = H_base_K.copy()
            H_probe[moire_offset:, moire_offset:] += v0 * np.eye(dim_MLG) + V_probe + V_probe.T.conj()
            probe_eigs = np.sort(linalg.eigvalsh(H_probe)) * 1000 / Q_E
        else:
            tp1 = np.exp(-1j * (pp / qq) * kx_p * Lx)
            tp2 = np.exp(1j * (pp / qq) * kx_p * Lx / 2) * np.exp(-1j * ky_p * Ly * (pp / qq))
            tp3 = np.exp(1j * (pp / qq) * kx_p * Lx / 2) * np.exp(1j * ky_p * Ly * (pp / qq))
            V_probe = gamma * tp1 * term1_Kp + tp2 * term2_Kp + tp3 * term3_Kp
            H_probe = H_base_Kp.copy()
            H_probe[moire_offset:, moire_offset:] += v0 * np.eye(dim_MLG) + V_probe + V_probe.T.conj()
            probe_eigs = np.sort(linalg.eigvalsh(H_probe)) * 1000 / Q_E

        kubo_mask = (probe_eigs >= kubo_lo) & (probe_eigs <= kubo_hi)
        band_sel_kubo = np.where(kubo_mask)[0]
        if len(band_sel_kubo) == 0:
            band_sel_kubo = np.arange(dim_total)

        scba_mask = (probe_eigs >= scba_lo) & (probe_eigs <= scba_hi)
        band_sel_scba = np.where(scba_mask)[0]
        if len(band_sel_scba) == 0:
            band_sel_scba = np.arange(dim_total)

        nb_kubo = len(band_sel_kubo)
        nb_scba = len(band_sel_scba)

        if use_scba:
            bmode_str = f"SCBA (Gamma0={Gamma_list_meV[0]} meV)"
        elif n_gamma == 1:
            bmode_str = f"Gamma = {Gamma_list_meV[0]} meV"
        else:
            bmode_str = (f"Gamma = [{Gamma_list_meV[0]}, ..., "
                         f"{Gamma_list_meV[-1]}] meV ({n_gamma} values)")
        print(f"  Transport: {bmode_str}, kT = {kT_meV} meV, {len(mulist_meV)} mu points")
        print(f"  Band selection (k=0 probe): Kubo={nb_kubo}/{dim_total} bands [{kubo_lo:.1f}, {kubo_hi:.1f}] meV")
        if use_scba:
            print(f"                               SCBA={nb_scba}/{dim_total} bands [{scba_lo:.1f}, {scba_hi:.1f}] meV")

        # --- Build velocity operators: v = (i/hbar)[A, H_base] ---
        # Work in eV / Angstrom / eV*s  →  velocity in Ang/s
        if 'K' in valley:
            Ax_K, Ay_K = get_berry_connection_K(N, B, qNslabels_K)
            if nlayers == 1:
                Ax_full_K = Ax_K * m_to_Ang
                Ay_full_K = Ay_K * m_to_Ang
                H_eV_K = H_base_K * J_to_eV
            else:
                zz = np.zeros_like(Ax_K)
                Ax_full_K = np.block([[Ax_K, zz], [zz, Ax_K]]) * m_to_Ang
                Ay_full_K = np.block([[Ay_K, zz], [zz, Ay_K]]) * m_to_Ang
                H_eV_K = H_base_K * J_to_eV
            Vx_K = (1j / HBAR_EV) * (Ax_full_K @ H_eV_K - H_eV_K @ Ax_full_K)
            Vy_K = (1j / HBAR_EV) * (Ay_full_K @ H_eV_K - H_eV_K @ Ay_full_K)

        if 'Kp' in valley:
            Ax_Kp, Ay_Kp = get_berry_connection_Kp(N, B, qNslabels_Kp)
            if nlayers == 1:
                Ax_full_Kp = Ax_Kp * m_to_Ang
                Ay_full_Kp = Ay_Kp * m_to_Ang
                H_eV_Kp = H_base_Kp * J_to_eV
            else:
                zz = np.zeros_like(Ax_Kp)
                Ax_full_Kp = np.block([[Ax_Kp, zz], [zz, Ax_Kp]]) * m_to_Ang
                Ay_full_Kp = np.block([[Ay_Kp, zz], [zz, Ay_Kp]]) * m_to_Ang
                H_eV_Kp = H_base_Kp * J_to_eV
            Vx_Kp = (1j / HBAR_EV) * (Ax_full_Kp @ H_eV_Kp - H_eV_Kp @ Ax_full_Kp)
            Vy_Kp = (1j / HBAR_EV) * (Ay_full_Kp @ H_eV_Kp - H_eV_Kp @ Ay_full_Kp)

        # --- Build mu list for Kubo (including mu_ref if set) ---
        all_mu = list(mulist_eV)
        compute_ref = mu_ref_eV is not None
        if compute_ref:
            all_mu.append(mu_ref_eV)
        all_mu = np.array(all_mu)
        n_all = len(all_mu)

        # --- SCBA: eigenvalue pass + self-consistent broadening solve ---
        scba_E_grid = None
        scba_Gamma_E = None
        scba_niter = 0
        if use_scba:
            eig_shared = {
                'pp': pp, 'qq': qq, 'Lx': Lx, 'Ly': Ly,
                'gamma': gamma,
                'moire_offset': moire_offset,
                'valley': valley, 'M_mag': M_mag,
                'v0_eye': v0_eye_scaled,
                'transport_band_sel': band_sel_scba,
            }
            if 'K' in valley:
                eig_shared.update({
                    'H_base_K': scale * H_base_K,
                    'term1_K': scale * term1_K,
                    'term2_K': scale * term2_K,
                    'term3_K': scale * term3_K,
                })
            if 'Kp' in valley:
                eig_shared.update({
                    'H_base_Kp': scale * H_base_Kp,
                    'term1_Kp': scale * term1_Kp,
                    'term2_Kp': scale * term2_Kp,
                    'term3_Kp': scale * term3_Kp,
                })

            print(" SCBA pass 1: collecting eigenvalues")
            all_eigs_K = np.zeros((Nk_tot, nb_scba)) if 'K' in valley else None
            all_eigs_Kp = np.zeros((Nk_tot, nb_scba)) if 'Kp' in valley else None

            if isparallel:
                default_nw = int(os.environ.get('SLURM_CPUS_PER_TASK',
                                                cpu_count()))
                nworkers = min(int(d.get('nworkers', default_nw)), Nk_tot)
                tasks = [(kc, kpoints[kc, :]) for kc in range(Nk_tot)]
                with Pool(processes=nworkers,
                          initializer=_init_kpoint_worker,
                          initargs=(eig_shared,)) as pool:
                    for kc, eK, eKp in pool.imap_unordered(
                            _solve_kpoint_eigenvalues, tasks):
                        if eK is not None:
                            all_eigs_K[kc, :] = eK
                        if eKp is not None:
                            all_eigs_Kp[kc, :] = eKp
            else:
                for kc in range(Nk_tot):
                    eK, eKp = _solve_kpoint_eigenvalues_core(
                        eig_shared, kpoints[kc, :])
                    if eK is not None:
                        all_eigs_K[kc, :] = eK
                    if eKp is not None:
                        all_eigs_Kp[kc, :] = eKp

            eig_arrays = [a for a in [all_eigs_K, all_eigs_Kp]
                          if a is not None]
            all_eigs_eV = np.concatenate(eig_arrays, axis=0) / 1000.0
            print(f"  Collected {all_eigs_eV.shape[0]} x {all_eigs_eV.shape[1]} eigenvalues")

            scba_E_grid, scba_Gamma_E, scba_niter = _solve_scba(
                all_eigs_eV, Gamma_eV, pp, Nk_tot,
                mixing=scba_mixing, tol=scba_tol,
                maxiter=scba_maxiter, floor_ratio=scba_floor,
                anderson_depth=scba_anderson)

        # --- Pack shared data (meV Hamiltonian, Ang/s velocity, Kubo params) ---
        shared = {
            'pp': pp, 'qq': qq, 'Lx': Lx, 'Ly': Ly,
            'gamma': gamma,
            'moire_offset': moire_offset,
            'valley': valley, 'M_mag': M_mag,
            'v0_eye': v0_eye_scaled,
            'transport_band_sel': band_sel_kubo,
            'all_mu_eV': all_mu,
            'Gamma_list_eV': Gamma_list_eV,
            'kT_eV': kT_eV,
            'scba_E_grid': scba_E_grid,
            'scba_Gamma_E': scba_Gamma_E,
            'scba_xy_constant': scba_xy_constant,
        }
        if 'K' in valley:
            shared.update({
                'H_base_K': scale * H_base_K,
                'term1_K': scale * term1_K,
                'term2_K': scale * term2_K,
                'term3_K': scale * term3_K,
                'Vx_K': Vx_K, 'Vy_K': Vy_K,
            })
        if 'Kp' in valley:
            shared.update({
                'H_base_Kp': scale * H_base_Kp,
                'term1_Kp': scale * term1_Kp,
                'term2_Kp': scale * term2_Kp,
                'term3_Kp': scale * term3_Kp,
                'Vx_Kp': Vx_Kp, 'Vy_Kp': Vy_Kp,
            })

        # --- Accumulators for per-k-point Kubo partial sums ---
        n_mu = len(mulist_eV)
        sxx_K_acc = np.zeros((n_gamma, n_all)) if 'K' in valley else None
        sxy_K_acc = np.zeros((n_gamma, n_all)) if 'K' in valley else None
        l12xx_K_acc = np.zeros((n_gamma, n_all)) if 'K' in valley else None
        l12xy_K_acc = np.zeros((n_gamma, n_all)) if 'K' in valley else None
        sxx_Kp_acc = np.zeros((n_gamma, n_all)) if 'Kp' in valley else None
        sxy_Kp_acc = np.zeros((n_gamma, n_all)) if 'Kp' in valley else None
        l12xx_Kp_acc = np.zeros((n_gamma, n_all)) if 'Kp' in valley else None
        l12xy_Kp_acc = np.zeros((n_gamma, n_all)) if 'Kp' in valley else None

        # Histogram DOS in states per primitive moire cell per bin:
        # each eigenvalue is one state per magnetic cell (2*pp primitive
        # cells) per k-point.
        dos_weight = 1.0 / (Nk_tot * 2 * pp)
        dos_K = np.zeros(n_mu) if 'K' in valley else None
        dos_Kp = np.zeros(n_mu) if 'Kp' in valley else None
        nb_kubo = len(band_sel_kubo)
        eigs_K_all = np.zeros((Nk_tot, nb_kubo)) if 'K' in valley else None
        eigs_Kp_all = np.zeros((Nk_tot, nb_kubo)) if 'Kp' in valley else None

        pass_label = "transport, pass 2" if use_scba else "transport"
        print(f" Entering the k loop ({pass_label})")
        next_pct = 5
        done_count = 0

        def _bin_dos(tek, dos_arr):
            mask = (tek > mulist_meV[0]) & (tek < mulist_meV[-1])
            in_range = tek[mask]
            if len(in_range) > 0:
                bins = np.argmin(np.abs(in_range[:, None] - mulist_meV[None, :]),
                                 axis=1)
                for b in bins:
                    dos_arr[b] += dos_weight

        def _accumulate(kc, sxK, syK, l12xK, l12yK, eK,
                        sxKp, syKp, l12xKp, l12yKp, eKp):
            nonlocal done_count, next_pct
            if sxK is not None:
                sxx_K_acc[:, :] += sxK
                sxy_K_acc[:, :] += syK
                l12xx_K_acc[:, :] += l12xK
                l12xy_K_acc[:, :] += l12yK
                _bin_dos(eK, dos_K)
                eigs_K_all[kc, :] = eK
            if sxKp is not None:
                sxx_Kp_acc[:, :] += sxKp
                sxy_Kp_acc[:, :] += syKp
                l12xx_Kp_acc[:, :] += l12xKp
                l12xy_Kp_acc[:, :] += l12yKp
                _bin_dos(eKp, dos_Kp)
                eigs_Kp_all[kc, :] = eKp
            done_count += 1
            pct = done_count * 100 // Nk_tot
            if pct >= next_pct:
                print(f"  {pct}% ({done_count}/{Nk_tot})", flush=True)
                next_pct = (pct // 5 + 1) * 5

        if isparallel:
            default_nw = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
            nworkers = min(int(d.get('nworkers', default_nw)), Nk_tot)
            print(f"  (parallel: {nworkers} workers)")
            tasks = [(kc, kpoints[kc, :]) for kc in range(Nk_tot)]
            with Pool(processes=nworkers,
                      initializer=_init_kpoint_worker, initargs=(shared,)) as pool:
                for r in pool.imap_unordered(_solve_kpoint_transport, tasks):
                    _accumulate(*r)
        else:
            for kc in range(Nk_tot):
                r = _solve_kpoint_transport_core(shared, kpoints[kc, :])
                _accumulate(kc, *r)

        print(" Done with the k loop")

        # --- Broadened DOS on mulist grid ---
        dos_broad_K = None
        dos_broad_Kp = None
        dos_norm = 1.0 / (np.pi * Nk_tot * 2 * pp)
        for v_label, eigs_all in [('K', eigs_K_all), ('Kp', eigs_Kp_all)]:
            if eigs_all is None:
                continue
            eigs_flat_eV = eigs_all.flatten() / 1000.0
            mu_eV = mulist_eV
            dos_broad = np.zeros((n_gamma, n_mu))
            for ig in range(n_gamma):
                for i, mu in enumerate(mu_eV):
                    if use_scba:
                        G_mu = float(np.interp(mu, scba_E_grid, scba_Gamma_E))
                    else:
                        G_mu = float(Gamma_list_eV[ig])
                    dos_broad[ig, i] = dos_norm * np.sum(
                        G_mu / ((mu - eigs_flat_eV) ** 2 + G_mu ** 2))
            if v_label == 'K':
                dos_broad_K = dos_broad
            else:
                dos_broad_Kp = dos_broad

        # --- Apply prefactors and mu_ref subtraction ---
        # sigma/(e^2/h) = (kernel prefactor) * sum_k / (Nk * A_mag):
        # the k-sum divided by Nk*A_mag is the physical per-area average
        # (one magnetic cell of states per k-point).  No qq dependence.
        # All outputs are per spin, per valley (no spin degeneracy factor).
        pf_xy = -4.0 * np.pi * HBAR_EV ** 2 / (A_mag_Ang2 * Nk_tot)
        if use_scba:
            pf_xx = 2.0 * HBAR_EV ** 2 / (A_mag_Ang2 * Nk_tot)
        else:
            G2_list = Gamma_list_eV ** 2
            pf_xx = 2.0 * HBAR_EV ** 2 * G2_list / (A_mag_Ang2 * Nk_tot)

        result = {'calctype': 'transport', 'params': inp,
                  'mulist': mulist_meV, 'broadening': broadening_mode}
        if n_gamma > 1:
            result['Gamma_list'] = Gamma_list_meV
        if use_scba:
            result.update({
                'Gamma_E_grid': scba_E_grid * 1000.0,
                'Gamma_E': scba_Gamma_E * 1000.0,
                'scba_niter': scba_niter,
            })

        if compute_ref:
            print(f"  Reference mu = {mu_ref_meV} meV (sigma_xy = 0 here)")

        if 'K' in valley:
            sxy_K_acc *= pf_xy
            l12xy_K_acc *= pf_xy
            if use_scba:
                sxx_K_acc *= pf_xx
                l12xx_K_acc *= pf_xx
            else:
                sxx_K_acc *= pf_xx[:, None]
                l12xx_K_acc *= pf_xx[:, None]
            if compute_ref:
                sxy_K_acc[:, :n_mu] -= sxy_K_acc[:, n_mu:n_mu+1]
                l12xy_K_acc[:, :n_mu] -= l12xy_K_acc[:, n_mu:n_mu+1]
            out_K = {
                'sigma_xx_K': sxx_K_acc[:, :n_mu],
                'sigma_xy_K': sxy_K_acc[:, :n_mu],
                'L12_xx_K': l12xx_K_acc[:, :n_mu],
                'L12_xy_K': l12xy_K_acc[:, :n_mu],
                'dos_K': dos_K,
                'dos_broad_K': dos_broad_K,
            }
            if n_gamma == 1:
                for k in ('sigma_xx_K', 'sigma_xy_K', 'L12_xx_K', 'L12_xy_K',
                           'dos_broad_K'):
                    out_K[k] = out_K[k].squeeze(axis=0)
            result.update(out_K)

        if 'Kp' in valley:
            sxy_Kp_acc *= pf_xy
            l12xy_Kp_acc *= pf_xy
            if use_scba:
                sxx_Kp_acc *= pf_xx
                l12xx_Kp_acc *= pf_xx
            else:
                sxx_Kp_acc *= pf_xx[:, None]
                l12xx_Kp_acc *= pf_xx[:, None]
            if compute_ref:
                sxy_Kp_acc[:, :n_mu] -= sxy_Kp_acc[:, n_mu:n_mu+1]
                l12xy_Kp_acc[:, :n_mu] -= l12xy_Kp_acc[:, n_mu:n_mu+1]
            out_Kp = {
                'sigma_xx_Kp': sxx_Kp_acc[:, :n_mu],
                'sigma_xy_Kp': sxy_Kp_acc[:, :n_mu],
                'L12_xx_Kp': l12xx_Kp_acc[:, :n_mu],
                'L12_xy_Kp': l12xy_Kp_acc[:, :n_mu],
                'dos_Kp': dos_Kp,
                'dos_broad_Kp': dos_broad_Kp,
            }
            if n_gamma == 1:
                for k in ('sigma_xx_Kp', 'sigma_xy_Kp', 'L12_xx_Kp',
                           'L12_xy_Kp', 'dos_broad_Kp'):
                    out_Kp[k] = out_Kp[k].squeeze(axis=0)
            result.update(out_Kp)

        return result

    # =======================================================================
    # Standard ek / dos calculation (existing code)
    # =======================================================================

    bands_K = np.zeros((Nk_tot, dim_total))
    bands_Kp = np.zeros((Nk_tot, dim_total))
    if layer_resolved:
        weights_K = np.zeros((Nk_tot, dim_total))
        weights_Kp = np.zeros((Nk_tot, dim_total))

    shared = {
        'pp': pp, 'qq': qq, 'Lx': Lx, 'Ly': Ly,
        'gamma': gamma,
        'moire_offset': moire_offset,
        'valley': valley, 'M_mag': M_mag,
        'v0_eye': v0_eye_scaled,
        'layer_resolved': layer_resolved,
        'dim_MLG': dim_MLG,
    }
    if 'K' in valley:
        shared.update({
            'H_base_K': scale * H_base_K,
            'term1_K': scale * term1_K,
            'term2_K': scale * term2_K,
            'term3_K': scale * term3_K,
        })
    if 'Kp' in valley:
        shared.update({
            'H_base_Kp': scale * H_base_Kp,
            'term1_Kp': scale * term1_Kp,
            'term2_Kp': scale * term2_Kp,
            'term3_Kp': scale * term3_Kp,
        })

    print(" Entering the k loop")
    if isparallel:
        default_nw = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
        nworkers = min(int(d.get('nworkers', default_nw)), Nk_tot)
        print(f"  (parallel: {nworkers} workers)")
        tasks = [(kc, kpoints[kc, :]) for kc in range(Nk_tot)]
        with Pool(processes=nworkers,
                  initializer=_init_kpoint_worker, initargs=(shared,)) as pool:
            results = pool.map(_solve_kpoint, tasks)
        for kc, tek_K_row, tek_Kp_row, wt_K_row, wt_Kp_row in results:
            if tek_K_row is not None:
                bands_K[kc, :] = tek_K_row
            if tek_Kp_row is not None:
                bands_Kp[kc, :] = tek_Kp_row
            if layer_resolved:
                if wt_K_row is not None:
                    weights_K[kc, :] = wt_K_row
                if wt_Kp_row is not None:
                    weights_Kp[kc, :] = wt_Kp_row
    else:
        for kc in range(Nk_tot):
            print(f"  |>>        step {kc + 1} of {Nk_tot}")
            tek_K_row, tek_Kp_row, wt_K_row, wt_Kp_row = _solve_kpoint_core(shared, kpoints[kc, :])
            if tek_K_row is not None:
                bands_K[kc, :] = tek_K_row
            if tek_Kp_row is not None:
                bands_Kp[kc, :] = tek_Kp_row
            if layer_resolved:
                if wt_K_row is not None:
                    weights_K[kc, :] = wt_K_row
                if wt_Kp_row is not None:
                    weights_Kp[kc, :] = wt_Kp_row

    print(" Done with the k loop")

    if calctype == 'ek':
        result = {'calctype': 'ek', 'params': inp,
                  'kpoints': kpoints,
                  'bands_K': bands_K, 'bands_Kp': bands_Kp}
        if layer_resolved:
            result['weights_K'] = weights_K
            result['weights_Kp'] = weights_Kp
        return result

    # --- DOS: bin eigenvalues into energy histogram ---
    # Units: states per primitive moire cell per bin (each eigenvalue is
    # one state per magnetic cell = 2*pp primitive cells, per k-point).
    dos_weight = 1.0 / (Nk_tot * 2 * pp)
    dos_K = np.zeros(len(elist))
    dos_Kp = np.zeros(len(elist))
    if layer_resolved:
        dos_K_top = np.zeros(len(elist))
        dos_K_bottom = np.zeros(len(elist))
        dos_Kp_top = np.zeros(len(elist))
        dos_Kp_bottom = np.zeros(len(elist))

    for kc in range(Nk_tot):
        if 'K' in valley:
            tek = bands_K[kc, :]
            mask = (tek > elist[0]) & (tek < elist[-1])
            in_range = tek[mask]
            bins = np.argmin(np.abs(in_range[:, None] - elist[None, :]), axis=1)
            for b in bins:
                dos_K[b] += dos_weight
            if layer_resolved:
                wt = weights_K[kc, mask]
                for j, b in enumerate(bins):
                    dos_K_top[b] += wt[j] * dos_weight
                    dos_K_bottom[b] += (1 - wt[j]) * dos_weight
        if 'Kp' in valley:
            tek = bands_Kp[kc, :]
            mask = (tek > elist[0]) & (tek < elist[-1])
            in_range = tek[mask]
            bins = np.argmin(np.abs(in_range[:, None] - elist[None, :]), axis=1)
            for b in bins:
                dos_Kp[b] += dos_weight
            if layer_resolved:
                wt = weights_Kp[kc, mask]
                for j, b in enumerate(bins):
                    dos_Kp_top[b] += wt[j] * dos_weight
                    dos_Kp_bottom[b] += (1 - wt[j]) * dos_weight

    result = {'calctype': 'dos', 'params': inp,
              'elist': elist,
              'dos_K': dos_K, 'dos_Kp': dos_Kp}
    if layer_resolved:
        result['dos_K_top'] = dos_K_top
        result['dos_K_bottom'] = dos_K_bottom
        result['dos_Kp_top'] = dos_Kp_top
        result['dos_Kp_bottom'] = dos_Kp_bottom
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _save_result(result, outfile):
    """Save result dict to .npz or .mat depending on file extension."""
    data = {k: v for k, v in result.items() if k not in ('calctype', 'params')}
    params = result.get('params', {})

    if outfile.endswith('.mat'):
        from scipy.io import savemat
        savemat(outfile, {'results': data, 'params': params})
    else:
        for k, v in params.items():
            data[f"input_{k}"] = np.asarray(v)
        np.savez(outfile, **data)

    print(f" Saved to {outfile}")


def main(input_file=None):
    if input_file is None:
        input_file = './input_test.txt'

    result = do_calc(input_file)
    params = result['params']
    pp = int(params['pp'])
    qq = int(params['qq'])

    default_outfile = f"bands_p{pp}_q{qq}.npz"
    outfile = params.get('outputfile', default_outfile)

    _save_result(result, outfile)
    return result


if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else './input_test.txt'
    main(input_file)
