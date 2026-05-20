"""
Semiclassical electronic structure for bilayer graphene on hBN.

Computes moire band structure on a full BZ k-mesh, plus Berry curvature,
orbital magnetic moment, and Fukuyama susceptibility.

Translated from MATLAB code (BLG_hBN_bloch_semiclassical_v3.m).
Uses MATLAB lattice-vector conventions for benchmark compatibility.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import numpy as np
from scipy.linalg import eigh
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from parser import parse_input_file
from hofstadter_system import (build_hofstadter_setup,
                               assemble_H_V_K as hof_assemble_K,
                               assemble_H_V_Kp as hof_assemble_Kp)


# ---------------------------------------------------------------------------
# Moire geometry
# ---------------------------------------------------------------------------

def compute_moire_geometry(theta, a=2.46, a_hBN=2.504):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0, 0, 1]])

    a1 = a * np.array([0.5, -np.sqrt(3)/2, 0])
    a2 = a * np.array([1.0, 0.0, 0.0])
    a3 = np.array([0.0, 0.0, 1.0])

    A = np.eye(3) - (a / a_hBN) * np.linalg.inv(R)
    M1 = np.linalg.solve(A, a1)
    M2 = np.linalg.solve(A, a2)
    M3 = np.linalg.solve(A, a3)

    vol_G = np.dot(M1, np.cross(M2, M3))
    vol_M = vol_G / np.linalg.norm(M3)

    G1 = 2 * np.pi * np.cross(M2, M3) / vol_G
    G2 = 2 * np.pi * np.cross(M3, M1) / vol_G
    G3 = 2 * np.pi * np.cross(M1, M2) / vol_G

    vb = np.array([G1, G2, G3])

    q1 = G1[:2]
    q2 = G2[:2]
    q3 = -q1 - q2

    return q1, q2, q3, vol_M, vb


# ---------------------------------------------------------------------------
# Q-vector lattice
# ---------------------------------------------------------------------------

def build_qvectors(NQ, q1, q2):
    n_lin = np.arange(NQ) - NQ // 2
    n1g, n2g = np.meshgrid(n_lin, n_lin)
    n1 = n1g.ravel()
    n2 = n2g.ravel()

    Q_raw = np.outer(n1, q1) + np.outer(n2, q2)
    norms = np.linalg.norm(Q_raw, axis=1)
    order = np.argsort(norms, kind='stable')

    Q = Q_raw[order]
    NG = len(Q)
    return Q, NG


# ---------------------------------------------------------------------------
# Moire coupling (T-matrices and hopping assembly)
# ---------------------------------------------------------------------------

def _kron_del(v1, v2, tol=1e-6):
    return 1.0 if np.linalg.norm(v1 - v2) < tol else 0.0


def construct_hopping(Q, NG, q1, q2, q3, V0, V1, psi):
    w = np.exp(-1j * 2 * np.pi / 3)

    T0_K  = V0 * np.eye(2, dtype=complex)
    T3_K  = V1 * np.exp(-1j * psi) * np.array([[1, w],    [1, w]])
    T1_K  = V1 * np.exp(-1j * psi) * np.array([[1, w**-1], [w**-1, w]])
    T2_K  = V1 * np.exp(-1j * psi) * np.array([[1, 1],     [w, w]])

    T0_Kp = V0 * np.eye(2, dtype=complex)
    T3_Kp = V1 * np.exp(1j * psi) * np.array([[1, w**-1], [1, w**-1]])
    T1_Kp = V1 * np.exp(1j * psi) * np.array([[1, w],     [w, w**-1]])
    T2_Kp = V1 * np.exp(1j * psi) * np.array([[1, 1],     [w**-1, w**-1]])

    HK  = np.zeros((2*NG, 2*NG), dtype=complex)
    HKp = np.zeros((2*NG, 2*NG), dtype=complex)

    for j in range(NG):
        Qj = Q[j]
        for k in range(NG):
            Qk = Q[k]

            d0  = _kron_del(Qj, Qk)
            d1  = _kron_del(Qj - Qk, q1)
            d1r = _kron_del(Qk - Qj, q1)
            d2  = _kron_del(Qj - Qk, q2)
            d2r = _kron_del(Qk - Qj, q2)
            d3  = _kron_del(Qj - Qk, q3)
            d3r = _kron_del(Qk - Qj, q3)

            HK[2*j:2*j+2, 2*k:2*k+2] = (
                d0 * T0_K
                + d1 * T1_K.conj().T + d1r * T1_K
                + d2 * T2_K.conj().T + d2r * T2_K
                + d3 * T3_K.conj().T + d3r * T3_K)

            d0p  = _kron_del(Qj, Qk)
            d1p  = _kron_del(Qj - Qk, -q1)
            d1rp = _kron_del(Qk - Qj, -q1)
            d2p  = _kron_del(Qj - Qk, -q2)
            d2rp = _kron_del(Qk - Qj, -q2)
            d3p  = _kron_del(Qj - Qk, -q3)
            d3rp = _kron_del(Qk - Qj, -q3)

            HKp[2*j:2*j+2, 2*k:2*k+2] = (
                d0p * T0_Kp
                + d1p * T1_Kp.conj().T + d1rp * T1_Kp
                + d2p * T2_Kp.conj().T + d2rp * T2_Kp
                + d3p * T3_Kp.conj().T + d3rp * T3_Kp)

    return HK, HKp


# ---------------------------------------------------------------------------
# Hamiltonian and velocity assembly (per k-point)
# ---------------------------------------------------------------------------

sigx = np.array([[0, 1],  [1, 0]],  dtype=complex)
sigy = np.array([[0, -1j],[1j, 0]], dtype=complex)
U1   = np.array([[0, 1],  [0, 0]],  dtype=complex)
U2   = np.array([[0, 0],  [1, 0]],  dtype=complex)


def assemble_H_V_K(kpt, Q, NG, vF, gamma1, v3, Utp, Ubm, H_hopp, hbar, nlayers):
    H0_B = np.zeros((2*NG, 2*NG), dtype=complex)
    Vx_B = np.zeros_like(H0_B)
    Vy_B = np.zeros_like(H0_B)

    for j in range(NG):
        kx = kpt[0] - Q[j, 0]
        ky = kpt[1] - Q[j, 1]
        s = slice(2*j, 2*j+2)
        H0_B[s, s] = vF * (kx * sigx + ky * sigy) + Ubm * np.eye(2)
        Vx_B[s, s] = vF * sigx / hbar
        Vy_B[s, s] = vF * sigy / hbar

    if nlayers == 1:
        return H0_B + H_hopp, Vx_B, Vy_B

    H0_T  = np.zeros((2*NG, 2*NG), dtype=complex)
    UBLG  = np.zeros_like(H0_T)
    Vx_T  = np.zeros_like(H0_T)
    Vx_TB = np.zeros_like(H0_T)
    Vy_T  = np.zeros_like(H0_T)
    Vy_TB = np.zeros_like(H0_T)

    for j in range(NG):
        kx = kpt[0] - Q[j, 0]
        ky = kpt[1] - Q[j, 1]
        s = slice(2*j, 2*j+2)
        H0_T[s, s]  = vF * (kx * sigx + ky * sigy) + Utp * np.eye(2)
        UBLG[s, s]  = gamma1 * U1 - v3 * (kx - 1j * ky) * U2
        Vx_T[s, s]  = vF * sigx / hbar
        Vx_TB[s, s] = -v3 * U2 / hbar
        Vy_T[s, s]  = vF * sigy / hbar
        Vy_TB[s, s] = 1j * v3 * U2 / hbar

    H  = np.block([[H0_T, UBLG.conj().T],
                   [UBLG, H0_B + H_hopp]])
    Vx = np.block([[Vx_T, Vx_TB.conj().T],
                   [Vx_TB, Vx_B]])
    Vy = np.block([[Vy_T, Vy_TB.conj().T],
                   [Vy_TB, Vy_B]])
    return H, Vx, Vy


def assemble_H_V_Kp(kpt, Q, NG, vF, gamma1, v3, Utp, Ubm, H_hopp, hbar, nlayers):
    H0_B = np.zeros((2*NG, 2*NG), dtype=complex)
    Vx_B = np.zeros_like(H0_B)
    Vy_B = np.zeros_like(H0_B)

    for j in range(NG):
        kx = kpt[0] - Q[j, 0]
        ky = kpt[1] - Q[j, 1]
        s = slice(2*j, 2*j+2)
        H0_B[s, s] = vF * (-kx * sigx + ky * sigy) + Ubm * np.eye(2)
        Vx_B[s, s] = -vF * sigx / hbar
        Vy_B[s, s] = vF * sigy / hbar

    if nlayers == 1:
        return H0_B + H_hopp, Vx_B, Vy_B

    H0_T  = np.zeros((2*NG, 2*NG), dtype=complex)
    UBLG  = np.zeros_like(H0_T)
    Vx_T  = np.zeros_like(H0_T)
    Vx_TB = np.zeros_like(H0_T)
    Vy_T  = np.zeros_like(H0_T)
    Vy_TB = np.zeros_like(H0_T)

    for j in range(NG):
        kx = kpt[0] - Q[j, 0]
        ky = kpt[1] - Q[j, 1]
        s = slice(2*j, 2*j+2)
        H0_T[s, s]  = vF * (-kx * sigx + ky * sigy) + Utp * np.eye(2)
        UBLG[s, s]  = gamma1 * U1 - v3 * (-kx - 1j * ky) * U2
        Vx_T[s, s]  = -vF * sigx / hbar
        Vx_TB[s, s] = v3 * U2 / hbar
        Vy_T[s, s]  = vF * sigy / hbar
        Vy_TB[s, s] = 1j * v3 * U2 / hbar

    H  = np.block([[H0_T, UBLG.conj().T],
                   [UBLG, H0_B + H_hopp]])
    Vx = np.block([[Vx_T, Vx_TB.conj().T],
                   [Vx_TB, Vx_B]])
    Vy = np.block([[Vy_T, Vy_TB.conj().T],
                   [Vy_TB, Vy_B]])
    return H, Vx, Vy


# ---------------------------------------------------------------------------
# Per-k-point worker
# ---------------------------------------------------------------------------

def _kpoint_worker(args):
    (kpt, Q, NG, vF, gamma1, v3, Utp, Ubm,
     H_hopp_K, H_hopp_Kp, hbar, nlayers,
     target_idx, remote_ind, eta, elist_meV, weight_factor,
     include_chi) = args

    numwann = 2 * NG * nlayers

    # --- K valley ---
    HK, VKx, VKy = assemble_H_V_K(
        kpt, Q, NG, vF, gamma1, v3, Utp, Ubm, H_hopp_K, hbar, nlayers)
    ekK, PsiK = eigh(HK)

    vx_K = PsiK.conj().T @ VKx @ PsiK
    vy_K = PsiK.conj().T @ VKy @ PsiK

    # Berry curvature and orbital moment
    denK = ekK[target_idx, np.newaxis] - ekK[np.newaxis, remote_ind]
    prodK = np.imag(
        vx_K[np.ix_(target_idx, remote_ind)]
        * vy_K[np.ix_(remote_ind, target_idx)].T)
    denomK = denK**2 + eta**2
    Oz_K = -2 * hbar**2 * np.sum(prodK / denomK, axis=1)
    Lz_K = hbar**2 * np.sum(denK * prodK / denomK, axis=1)

    # Susceptibility
    chi_K = np.zeros(len(elist_meV))
    if include_chi:
        ekK_rem = ekK[remote_ind]
        vxeig_K = vx_K[np.ix_(remote_ind, remote_ind)]
        vyeig_K = vy_K[np.ix_(remote_ind, remote_ind)]
        for ec in range(len(elist_meV)):
            xK = 1.0 / (elist_meV[ec] - ekK_rem * 1e3 + 1j * eta)
            A = vxeig_K * xK[np.newaxis, :]
            B = vyeig_K * xK[np.newaxis, :]
            chi_K[ec] = weight_factor * np.imag(np.trace(A @ B @ A @ B))

    # --- K' valley ---
    HKp, VKpx, VKpy = assemble_H_V_Kp(
        kpt, Q, NG, vF, gamma1, v3, Utp, Ubm, H_hopp_Kp, hbar, nlayers)
    ekKp, PsiKp = eigh(HKp)

    vx_Kp = PsiKp.conj().T @ VKpx @ PsiKp
    vy_Kp = PsiKp.conj().T @ VKpy @ PsiKp

    denKp = ekKp[target_idx, np.newaxis] - ekKp[np.newaxis, remote_ind]
    prodKp = np.imag(
        vx_Kp[np.ix_(target_idx, remote_ind)]
        * vy_Kp[np.ix_(remote_ind, target_idx)].T)
    denomKp = denKp**2 + eta**2
    Oz_Kp = -2 * hbar**2 * np.sum(prodKp / denomKp, axis=1)
    Lz_Kp = hbar**2 * np.sum(denKp * prodKp / denomKp, axis=1)

    chi_Kp = np.zeros(len(elist_meV))
    if include_chi:
        ekKp_rem = ekKp[remote_ind]
        vxeig_Kp = vx_Kp[np.ix_(remote_ind, remote_ind)]
        vyeig_Kp = vy_Kp[np.ix_(remote_ind, remote_ind)]
        for ec in range(len(elist_meV)):
            xKp = 1.0 / (elist_meV[ec] - ekKp_rem * 1e3 + 1j * eta)
            A = vxeig_Kp * xKp[np.newaxis, :]
            B = vyeig_Kp * xKp[np.newaxis, :]
            chi_Kp[ec] = weight_factor * np.imag(np.trace(A @ B @ A @ B))

    return (ekK[target_idx], ekKp[target_idx],
            Oz_K, Oz_Kp, Lz_K, Lz_Kp,
            chi_K, chi_Kp)


# ---------------------------------------------------------------------------
# Hofstadter mode: k-point worker and orchestrator
# ---------------------------------------------------------------------------

_hofstadter_shared = {}


def _init_hofstadter_worker(shared):
    global _hofstadter_shared
    _hofstadter_shared = shared


def _kpoint_worker_hofstadter(args):
    kc, kpt, target_idx, remote_ind, eta, hbar = args
    setup = _hofstadter_shared

    HK, VKx, VKy = hof_assemble_K(kpt, setup)
    ekK, PsiK = eigh(HK)

    vx_K = PsiK.conj().T @ VKx @ PsiK
    vy_K = PsiK.conj().T @ VKy @ PsiK

    denK = ekK[target_idx, np.newaxis] - ekK[np.newaxis, remote_ind]
    prodK = np.imag(
        vx_K[np.ix_(target_idx, remote_ind)]
        * vy_K[np.ix_(remote_ind, target_idx)].T)
    denomK = denK**2 + eta**2
    Oz_K = -2 * hbar**2 * np.sum(prodK / denomK, axis=1)
    Lz_K = hbar**2 * np.sum(denK * prodK / denomK, axis=1)

    HKp, VKpx, VKpy = hof_assemble_Kp(kpt, setup)
    ekKp, PsiKp = eigh(HKp)

    vx_Kp = PsiKp.conj().T @ VKpx @ PsiKp
    vy_Kp = PsiKp.conj().T @ VKpy @ PsiKp

    denKp = ekKp[target_idx, np.newaxis] - ekKp[np.newaxis, remote_ind]
    prodKp = np.imag(
        vx_Kp[np.ix_(target_idx, remote_ind)]
        * vy_Kp[np.ix_(remote_ind, target_idx)].T)
    denomKp = denKp**2 + eta**2
    Oz_Kp = -2 * hbar**2 * np.sum(prodKp / denomKp, axis=1)
    Lz_Kp = hbar**2 * np.sum(denKp * prodKp / denomKp, axis=1)

    return (kc, ekK[target_idx], ekKp[target_idx],
            Oz_K, Oz_Kp, Lz_K, Lz_Kp)


def _do_calc_hofstadter(inp):
    setup = build_hofstadter_setup(inp)

    hbar = 6.582119569e-16       # eV * s
    eta = float(inp.get('eta', 1))
    ispar = int(inp.get('isparallel', 0))
    bands_sel = np.atleast_1d(inp['bands']).astype(int)

    target_idx = setup['target_idx']
    remote_ind = setup['remote_ind']
    num_bands = setup['num_bands']
    nk1 = setup['nk1']
    nk2 = setup['nk2']
    Nk_tot = nk1 * nk2
    kpoints = setup['kpoints']

    args_list = [(kc, kpoints[kc], target_idx, remote_ind, eta, hbar)
                 for kc in range(Nk_tot)]

    print(f"  Calculating over {Nk_tot} k-points...")
    if ispar:
        with Pool(initializer=_init_hofstadter_worker,
                  initargs=(setup,)) as pool:
            results = pool.map(_kpoint_worker_hofstadter, args_list)
    else:
        global _hofstadter_shared
        _hofstadter_shared = setup
        results = [_kpoint_worker_hofstadter(a) for a in args_list]

    E_K  = np.zeros((num_bands, Nk_tot))
    E_Kp = np.zeros((num_bands, Nk_tot))
    Oz_K  = np.zeros((num_bands, Nk_tot))
    Oz_Kp = np.zeros((num_bands, Nk_tot))
    Lz_K  = np.zeros((num_bands, Nk_tot))
    Lz_Kp = np.zeros((num_bands, Nk_tot))

    for res in results:
        kc, ek_K, ek_Kp, oz_K, oz_Kp, lz_K, lz_Kp = res
        E_K[:, kc]  = ek_K
        E_Kp[:, kc] = ek_Kp
        Oz_K[:, kc]  = oz_K
        Oz_Kp[:, kc] = oz_Kp
        Lz_K[:, kc]  = lz_K
        Lz_Kp[:, kc] = lz_Kp

    # Band selection within the target window
    dim = num_bands
    bands_idx = dim // 2 - 1 + bands_sel
    E_K  = E_K[bands_idx, :]
    E_Kp = E_Kp[bands_idx, :]
    Oz_K  = Oz_K[bands_idx, :]
    Oz_Kp = Oz_Kp[bands_idx, :]
    Lz_K  = Lz_K[bands_idx, :]
    Lz_Kp = Lz_Kp[bands_idx, :]

    # Unit conversions: internal eV/Ang -> output meV/m
    vol_M_m2 = setup['vol_M']  # already in m^2

    Oz_K  = Oz_K  * 1e-20      # Ang^2 -> m^2
    Oz_Kp = Oz_Kp * 1e-20
    Lz_K  = Lz_K  * 1e-20      # eV*Ang^2 -> eV*m^2
    Lz_Kp = Lz_Kp * 1e-20

    E_K  = E_K  * 1e3           # eV -> meV
    E_Kp = E_Kp * 1e3
    Lz_K  = Lz_K  * 1e3         # eV*m^2 -> meV*m^2
    Lz_Kp = Lz_Kp * 1e3

    print("  Done.")

    return {
        'kpoints': kpoints,
        'E_K': E_K,
        'E_Kp': E_Kp,
        'Oz_K': Oz_K,
        'Oz_Kp': Oz_Kp,
        'Lz_K': Lz_K,
        'Lz_Kp': Lz_Kp,
        'vol_M': vol_M_m2,
    }


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------

def do_calc(filepath):
    inp = parse_input_file(filepath)

    qq = int(inp.get('qq', 0))
    if qq > 0:
        return _do_calc_hofstadter(inp)

    theta    = float(inp.get('theta', 0.0))
    nlayers  = int(inp.get('Nlayers', 2))
    nk1      = int(inp['nk1'])
    nk2      = int(inp['nk2'])
    NQ       = int(inp['NQ'])
    vF       = float(inp['vF'])
    gamma1_ev = float(inp['gamma1'])
    v3       = float(inp.get('v3', 0))
    V0_meV   = float(inp['V0'])
    V1_meV   = float(inp['V1'])
    psi      = float(inp['moire_psi'])
    eta      = float(inp['eta'])
    ispar    = int(inp.get('isparallel', 0))
    U        = np.atleast_1d(inp.get('U', np.array([0, 0])))
    bands_sel = np.atleast_1d(inp['bands']).astype(int)
    elist_meV = np.atleast_1d(inp['elist']).ravel()
    include_chi = int(inp.get('include_chi', 1))

    hbar = 6.582119569e-16       # eV * s

    V0_ev = V0_meV / 1000
    V1_ev = V1_meV / 1000
    Utp   = U[0] / 1000
    Ubm   = U[1] / 1000 if len(U) > 1 else U[0] / 1000

    # --- Moire geometry ---
    q1, q2, q3, vol_M, vb = compute_moire_geometry(theta)
    Q, NG = build_qvectors(NQ, q1, q2)
    numwann = 2 * NG * nlayers

    print(f"  nlayers = {nlayers}, NQ = {NQ}, NG = {NG}")
    print(f"  numwann = {numwann}")
    print(f"  nk1 = {nk1}, nk2 = {nk2}, Nk = {nk1*nk2}")

    # --- Hopping matrices ---
    H_hopp_K, H_hopp_Kp = construct_hopping(
        Q, NG, q1, q2, q3, V0_ev, V1_ev, psi)

    # --- K-mesh ---
    Nk_tot = nk1 * nk2
    n1_mesh = np.arange(nk1)
    n2_mesh = np.arange(nk2)
    n1g, n2g = np.meshgrid(n1_mesh, n2_mesh)
    n11 = n1g.ravel(order='F')
    n22 = n2g.ravel(order='F')

    v1 = n11 / nk1 - 0.5
    v2 = n22 / nk2 - 0.5
    kpoints = np.column_stack([
        v1 * vb[0, 0] + v2 * vb[1, 0],
        v1 * vb[0, 1] + v2 * vb[1, 1]])

    # --- Band indexing ---
    num_bands = numwann
    n_remote  = numwann
    idx_c = numwann // 2
    target_idx = np.arange(idx_c - num_bands//2, idx_c + num_bands//2)
    remote_lo = max(target_idx[0] - n_remote, 0)
    remote_hi = min(target_idx[-1] + n_remote + 1, numwann)
    remote_ind = np.arange(remote_lo, remote_hi)

    NE = len(elist_meV)
    weight_factor = 1.0 / (Nk_tot * vol_M)

    # --- Build worker args ---
    common = (Q, NG, vF, gamma1_ev, v3, Utp, Ubm,
              H_hopp_K, H_hopp_Kp, hbar, nlayers,
              target_idx, remote_ind, eta, elist_meV, weight_factor,
              include_chi)
    args_list = [(kpoints[kc], *common) for kc in range(Nk_tot)]

    # --- Run k-loop ---
    print(f"  Calculating over {Nk_tot} k-points...")
    if ispar:
        with Pool() as pool:
            results = pool.map(_kpoint_worker, args_list)
    else:
        results = [_kpoint_worker(a) for a in args_list]

    # --- Collect results ---
    E_K  = np.zeros((num_bands, Nk_tot))
    E_Kp = np.zeros((num_bands, Nk_tot))
    Oz_K  = np.zeros((num_bands, Nk_tot))
    Oz_Kp = np.zeros((num_bands, Nk_tot))
    Lz_K  = np.zeros((num_bands, Nk_tot))
    Lz_Kp = np.zeros((num_bands, Nk_tot))
    dChi_dE_K  = np.zeros(NE)
    dChi_dE_Kp = np.zeros(NE)

    for kc, res in enumerate(results):
        (ek_K, ek_Kp, oz_K, oz_Kp, lz_K, lz_Kp, chi_K, chi_Kp) = res
        E_K[:, kc]  = ek_K
        E_Kp[:, kc] = ek_Kp
        Oz_K[:, kc]  = oz_K
        Oz_Kp[:, kc] = oz_Kp
        Lz_K[:, kc]  = lz_K
        Lz_Kp[:, kc] = lz_Kp
        dChi_dE_K  += chi_K
        dChi_dE_Kp += chi_Kp

    # --- Band selection ---
    dim = E_K.shape[0]
    bands_idx = dim // 2 - 1 + bands_sel
    E_K  = E_K[bands_idx, :]
    E_Kp = E_Kp[bands_idx, :]

    # --- Unit conversions ---
    vol_M_m2 = vol_M * 1e-20

    Oz_K  = Oz_K[bands_idx, :]  * 1e-20
    Oz_Kp = Oz_Kp[bands_idx, :] * 1e-20
    Lz_K  = Lz_K[bands_idx, :]  * 1e-20
    Lz_Kp = Lz_Kp[bands_idx, :] * 1e-20

    E_K  = E_K  * 1e3
    E_Kp = E_Kp * 1e3

    Lz_K  = Lz_K  * 1e3
    Lz_Kp = Lz_Kp * 1e3

    dChi_dE_K  = dChi_dE_K  * 1e-20 * hbar**4
    dChi_dE_Kp = dChi_dE_Kp * 1e-20 * hbar**4

    E_list = elist_meV / 1000

    print("  Done.")

    return {
        'E_list': E_list,
        'kpoints': kpoints,
        'E_K': E_K,
        'E_Kp': E_Kp,
        'Oz_K': Oz_K,
        'Oz_Kp': Oz_Kp,
        'Lz_K': Lz_K,
        'Lz_Kp': Lz_Kp,
        'dChi_dE_K': dChi_dE_K,
        'dChi_dE_Kp': dChi_dE_Kp,
        'vol_M': vol_M_m2,
    }


def save_result(result, outfile):
    data = {k: v for k, v in result.items()}
    if outfile.endswith('.mat'):
        from scipy.io import savemat
        savemat(outfile, data)
    else:
        np.savez(outfile, **data)
    print(f"  Saved to {outfile}")


if __name__ == '__main__':
    fpath = sys.argv[1] if len(sys.argv) > 1 else 'input_benchmark.txt'
    result = do_calc(fpath)
    inp = parse_input_file(fpath)
    nk1 = int(inp['nk1'])
    nk2 = int(inp['nk2'])
    bands_sel = np.atleast_1d(inp['bands']).astype(int)

    # --- Onsager quantization (if Blist provided) ---
    if 'Blist' in inp:
        from isoenergy import get_energy_resolved_data
        from onsager import onsager_fan

        Blist = np.atleast_1d(inp['Blist']).ravel()
        nmax = int(inp.get('nmax', 50))
        kT = float(inp.get('kT', 3.0))
        E_levels = np.atleast_1d(inp.get('elist_onsager', inp['elist'])).ravel()

        termflags_raw = np.atleast_1d(inp.get('termflags', np.array([1, 1, 1]))).astype(int)
        termflags = tuple(termflags_raw[:3])

        print(f"  Running Onsager quantization: {len(Blist)} B values, nmax={nmax}, termflags={termflags}")

        result['Blist'] = Blist
        result['nmax'] = nmax

        for valley, suffix in [('K', '_K'), ('Kp', '_Kp')]:
            area, enclosedBC, dL_dE = get_energy_resolved_data(
                bands_sel, kT, result[f'E_{valley}'], result[f'Oz_{valley}'],
                result[f'Lz_{valley}'], E_levels, result['vol_M'], nk1, nk2)

            LL_all, band_indices = onsager_fan(
                Blist, nmax, E_levels, area, enclosedBC, dL_dE,
                dChi_dE=result[f'dChi_dE_{valley}'], termflags=termflags)

            for i, (LL, bi) in enumerate(zip(LL_all, band_indices)):
                result[f'LL{suffix}_band{bi}'] = LL

            print(f"  {valley} valley: {len(LL_all)} bands with orbits")

    outfile = inp.get('outputfile', f'electronic_structure_data_{nk1}.mat')
    save_result(result, outfile)
