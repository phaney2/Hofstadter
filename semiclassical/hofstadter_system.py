"""
Hofstadter (magnetic-field) Hamiltonian and velocity operator for the
semiclassical pipeline.

Builds the magnetic Bloch Hamiltonian in a Landau level basis and
constructs the velocity operator.  Two options (controlled by
include_comm input flag):

  include_comm = 0 (default):  v = (1/hbar) dH/dk
    Gauge-dependent — tied to the phase convention used for tphase1/2/3.
    Gives integer Chern per k-mesh cell in the current (square) convention.

  include_comm = 1:  v = (1/hbar)(dH/dk - i[A, H])
    Gauge-invariant — same Berry curvature regardless of unit cell choice
    (square vs triangular).  Chern per mesh cell = 1/pp; integer over
    the full magnetic BZ.

All outputs are in eV / Angstrom units to match the zero-field
semiclassical conventions.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from constants import HBAR, Q_E, A_GRAPHENE, A_HBN
from hamiltonian import (get_interbilayerterms_K, get_interbilayerterms_Kp,
                         get_intermonolayerH_K, get_intermonolayerH_Kp,
                         get_intralayerH_K, get_intralayerH_Kp,
                         get_berry_connection_K, get_berry_connection_Kp)

HBAR_EV = 6.582119569e-16  # eV*s


def build_hofstadter_setup(inp):
    """Build all k-independent matrices for the Hofstadter semiclassical calc.

    Returns a dict with H_base, term1/2/3, Ax/Ay, k-mesh, and indexing
    for both valleys.  All energies in eV, lengths in Angstrom.
    """
    theta = np.radians(float(inp.get('theta', 0.0)))
    qq = int(inp['qq'])
    pp = int(inp['pp'])
    g0 = float(inp['g0'])
    nlayers = int(inp.get('nlayers', 2))
    if nlayers == 2:
        g1 = float(inp['g1'])
        g3 = float(inp['g3'])
        g4 = float(inp.get('g4', 0))
    else:
        g1 = float(inp.get('g1', 0))
        g3 = float(inp.get('g3', 0))
        g4 = float(inp.get('g4', 0))
    delta = 0 if nlayers == 1 else float(inp.get('delta', 0))
    v0_meV = float(inp['v0'])
    v1_meV = float(inp['v1'])
    w_meV = float(inp['w'])
    eta_bc = float(inp.get('eta', 1))
    U = np.atleast_1d(inp.get('U', np.array([0, 0])))
    if len(U) == 1:
        U = np.array([U[0], 0])
    nk1 = int(inp.get('nk1', 25))
    nk2 = int(inp.get('nk2', 40))
    LL_multiplier = float(inp.get('LL_multiplier', 6))
    Nmax = int(inp.get('Nmax', 5000))
    gamma = float(inp.get('gamma', 1))
    vF = float(inp.get('vF', 1e6))
    bands_sel = np.atleast_1d(inp['bands']).astype(int)
    num_bands = 2 * (int(np.max(np.abs(bands_sel))) + 1)
    nremotebands = int(inp.get('nremotebands', 300))

    # --- Derived quantities (SI) ---
    eps = A_HBN / A_GRAPHENE - 1
    L_moire = ((1 + eps) * A_GRAPHENE
               / np.sqrt(eps**2 + 2 * (1 + eps) * (1 - np.cos(theta))))
    ktheta = 4 * np.pi / (3**0.5 * L_moire)
    uc_area = 3**0.5 * L_moire**2 / 2 * 2

    phi_0 = HBAR * 2 * np.pi / Q_E
    B = (qq / pp) * phi_0 / uc_area
    lB = (HBAR / (Q_E * B))**0.5

    eneLL = g0 / 1e3 * Q_E * A_GRAPHENE / lB * 2**0.5
    w_J = w_meV / 1e3 * Q_E
    v0_J = v0_meV / 1e3 * Q_E
    v1_J = v1_meV / 1e3 * Q_E

    N = int(LL_multiplier * round(max(HBAR * vF * ktheta, w_J) / eneLL)**2)
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

    # --- Unit conversion factors: SI → eV/Angstrom ---
    J_to_eV = 1.0 / Q_E
    m_to_Ang = 1e10
    Lx_Ang = Lx * m_to_Ang
    Ly_Ang = Ly * m_to_Ang
    v0_eV = v0_J * J_to_eV

    # --- K valley ---
    print("  Building K-valley Hamiltonian...")
    term1_K, term2_K, term3_K, qNslabels_K = get_interbilayerterms_K(
        N, Nq, ktheta, lB, v0_J, v1_J, eta_bc, qq, pp, theta)
    Hintra_K = get_intralayerH_K(N, 0, B, qNslabels_K, TBGparams, 'A')
    Ax_K, Ay_K = get_berry_connection_K(N, B, qNslabels_K)

    dl = Hintra_K.shape[0]
    if nlayers == 1:
        U_onsite = U[0] / 1e3 * Q_E
        H_base_K = Hintra_K + np.eye(dl) * U_onsite
        Ax_full_K = Ax_K
        Ay_full_K = Ay_K
    else:
        Utp_val = U[0] / 1e3 * Q_E
        Ubm_val = U[1] / 1e3 * Q_E
        Hinter_K = get_intermonolayerH_K(N, 0, B, qNslabels_K, TBGparams)
        Hintra2_K = get_intralayerH_K(N, 0, B, qNslabels_K, TBGparams, 'B')
        H_base_K = np.block([
            [Hintra_K + np.eye(dl) * Utp_val, Hinter_K],
            [Hinter_K.T.conj(), Hintra2_K + np.eye(dl) * Ubm_val]
        ])
        zz = np.zeros_like(Ax_K)
        Ax_full_K = np.block([[Ax_K, zz], [zz, Ax_K]])
        Ay_full_K = np.block([[Ay_K, zz], [zz, Ay_K]])

    # --- K' valley ---
    print("  Building K'-valley Hamiltonian...")
    term1_Kp, term2_Kp, term3_Kp, qNslabels_Kp = get_interbilayerterms_Kp(
        N, Nq, ktheta, lB, v0_J, v1_J, eta_bc, qq, pp, theta)
    Hintra_Kp = get_intralayerH_Kp(N, 0, B, qNslabels_Kp, TBGparams, 'A')
    Ax_Kp, Ay_Kp = get_berry_connection_Kp(N, B, qNslabels_Kp)

    dl = Hintra_Kp.shape[0]
    if nlayers == 1:
        U_onsite = U[0] / 1e3 * Q_E
        H_base_Kp = Hintra_Kp + np.eye(dl) * U_onsite
        Ax_full_Kp = Ax_Kp
        Ay_full_Kp = Ay_Kp
    else:
        Utp_val = U[0] / 1e3 * Q_E
        Ubm_val = U[1] / 1e3 * Q_E
        Hinter_Kp = get_intermonolayerH_Kp(N, 0, B, qNslabels_Kp, TBGparams)
        Hintra2_Kp = get_intralayerH_Kp(N, 0, B, qNslabels_Kp, TBGparams, 'B')
        H_base_Kp = np.block([
            [Hintra_Kp + np.eye(dl) * Utp_val, Hinter_Kp],
            [Hinter_Kp.T.conj(), Hintra2_Kp + np.eye(dl) * Ubm_val]
        ])
        zz = np.zeros_like(Ax_Kp)
        Ax_full_Kp = np.block([[Ax_Kp, zz], [zz, Ax_Kp]])
        Ay_full_Kp = np.block([[Ay_Kp, zz], [zz, Ay_Kp]])

    dim_MLG = Hintra_K.shape[0]
    dim_total = dim_MLG if nlayers == 1 else 2 * dim_MLG
    moire_offset = 0 if nlayers == 1 else dim_MLG

    # --- Scale to eV / Angstrom ---
    H_base_K_eV = H_base_K * J_to_eV
    H_base_Kp_eV = H_base_Kp * J_to_eV
    term1_K_eV = term1_K * J_to_eV
    term2_K_eV = term2_K * J_to_eV
    term3_K_eV = term3_K * J_to_eV
    term1_Kp_eV = term1_Kp * J_to_eV
    term2_Kp_eV = term2_Kp * J_to_eV
    term3_Kp_eV = term3_Kp * J_to_eV
    Ax_full_K_Ang = Ax_full_K * m_to_Ang
    Ay_full_K_Ang = Ay_full_K * m_to_Ang
    Ax_full_Kp_Ang = Ax_full_Kp * m_to_Ang
    Ay_full_Kp_Ang = Ay_full_Kp * m_to_Ang
    v0_eye_eV = v0_eV * np.eye(dim_MLG)

    # --- k-mesh (magnetic BZ, in Angstrom^-1) ---
    b1 = ktheta * np.array([0, -1])
    b2 = ktheta * np.array([np.sqrt(3) / 2, 1 / 2])
    M_mag = 0.5 * b1 / pp

    Nk_tot = nk1 * nk2
    n1_arr = np.arange(nk1)
    n2_arr = np.arange(nk2)
    n1grid, n2grid = np.meshgrid(n1_arr, n2_arr)
    n11 = n1grid.flatten(order='F')
    n22 = n2grid.flatten(order='F')

    vb = np.array([b1 / pp, b2 / pp])
    kpoints = np.zeros((Nk_tot, 2))
    for j in range(Nk_tot):
        frac = np.array([n11[j] / nk1, n22[j] / nk2])
        kpoints[j, :] = vb.T @ frac

    kpoints_Ang = kpoints * (1.0 / m_to_Ang)
    M_mag_Ang = M_mag * (1.0 / m_to_Ang)
    vol_M = pp**2 * uc_area / 2

    # --- Band indexing ---
    idx_c = dim_total // 2
    target_idx = np.arange(idx_c - num_bands // 2, idx_c + num_bands // 2)
    remote_lo = max(target_idx[0] - nremotebands, 0)
    remote_hi = min(target_idx[-1] + nremotebands + 1, dim_total)
    remote_ind = np.arange(remote_lo, remote_hi)

    print(f"  dim per layer (post-chop) = {dim_MLG}")
    print(f"  dim total = {dim_total}")
    print(f"  num_bands = {num_bands}, nremotebands = {nremotebands}")
    print(f"  target_idx range: {target_idx[0]}..{target_idx[-1]}")
    print(f"  remote_ind range: {remote_ind[0]}..{remote_ind[-1]}")

    include_comm = int(inp.get('include_comm', 0))

    setup = {
        'pp': pp, 'qq': qq, 'nlayers': nlayers, 'gamma': gamma,
        'Lx_Ang': Lx_Ang, 'Ly_Ang': Ly_Ang,
        'moire_offset': moire_offset,
        'M_mag_Ang': M_mag_Ang,
        'dim_total': dim_total,
        'dim_MLG': dim_MLG,
        'H_base_K': H_base_K_eV,
        'H_base_Kp': H_base_Kp_eV,
        'term1_K': term1_K_eV, 'term2_K': term2_K_eV, 'term3_K': term3_K_eV,
        'term1_Kp': term1_Kp_eV, 'term2_Kp': term2_Kp_eV, 'term3_Kp': term3_Kp_eV,
        'Ax_K': Ax_full_K_Ang, 'Ay_K': Ay_full_K_Ang,
        'Ax_Kp': Ax_full_Kp_Ang, 'Ay_Kp': Ay_full_Kp_Ang,
        'v0_eye': v0_eye_eV,
        'kpoints': kpoints_Ang,
        'kpoints_raw': kpoints,
        'nk1': nk1, 'nk2': nk2,
        'vol_M': vol_M,
        'target_idx': target_idx,
        'remote_ind': remote_ind,
        'num_bands': num_bands,
        'B': B,
        'include_comm': include_comm,
    }
    return setup


def assemble_H_V_K(kpt, setup):
    """Build H (eV), Vx, Vy (Ang/s) at one k-point for K valley."""
    pp = setup['pp']
    qq = setup['qq']
    Lx = setup['Lx_Ang']
    Ly = setup['Ly_Ang']
    gamma = setup['gamma']
    mo = setup['moire_offset']
    M_mag = setup['M_mag_Ang']
    dim = setup['dim_total']

    kpts = kpt - M_mag
    kx, ky = kpts
    pp_over_qq = pp / qq

    # --- Phase factors (dimensionless) ---
    tphase1 = np.exp(1j * pp_over_qq * kx * Lx)
    tphase2 = (np.exp(-1j * pp_over_qq * kx * Lx / 2)
               * np.exp(1j * ky * Ly * pp_over_qq))
    tphase3 = (np.exp(-1j * pp_over_qq * kx * Lx / 2)
               * np.exp(-1j * ky * Ly * pp_over_qq))

    # --- Hamiltonian (eV) ---
    V_pq = (gamma * tphase1 * setup['term1_K']
            + tphase2 * setup['term2_K']
            + tphase3 * setup['term3_K'])

    H = setup['H_base_K'].copy()
    H[mo:, mo:] += setup['v0_eye'] + V_pq + V_pq.T.conj()

    # --- dH/dk (eV * Ang) ---
    vx_tp1 = (1j * pp_over_qq * Lx) * tphase1
    vx_tp2 = (-1j * pp_over_qq * Lx / 2) * tphase2
    vx_tp3 = (-1j * pp_over_qq * Lx / 2) * tphase3

    vy_tp1 = 0.0
    vy_tp2 = (1j * pp_over_qq * Ly) * tphase2
    vy_tp3 = (-1j * pp_over_qq * Ly) * tphase3

    vx_tmp = (gamma * vx_tp1 * setup['term1_K']
              + vx_tp2 * setup['term2_K']
              + vx_tp3 * setup['term3_K'])
    vy_tmp = (gamma * vy_tp1 * setup['term1_K']
              + vy_tp2 * setup['term2_K']
              + vy_tp3 * setup['term3_K'])

    Hdx = np.zeros((dim, dim), dtype=complex)
    Hdy = np.zeros((dim, dim), dtype=complex)
    Hdx[mo:, mo:] = vx_tmp + vx_tmp.T.conj()
    Hdy[mo:, mo:] = vy_tmp + vy_tmp.T.conj()

    # --- Velocity (Ang/s) ---
    # Two options controlled by setup['include_comm']:
    #   0: v = (1/hbar) dH/dk
    #      Gauge-dependent — gives integer Chern per mesh cell in the
    #      square phase convention.  Matches plaquette Berry curvature
    #      (which is also gauge-dependent in the same way).
    #   1: v = (1/hbar)(dH/dk - i[A, H])
    #      Gauge-invariant — gives the same Berry curvature regardless
    #      of phase convention (square vs triangular unit cell).
    #      Chern per mesh cell = 1/pp; integer over the full MBZ.
    if setup.get('include_comm', 0):
        Ax = setup['Ax_K']
        Ay = setup['Ay_K']
        Vx = (Hdx - 1j * (Ax @ H - H @ Ax)) / HBAR_EV
        Vy = (Hdy - 1j * (Ay @ H - H @ Ay)) / HBAR_EV
    else:
        Vx = Hdx / HBAR_EV
        Vy = Hdy / HBAR_EV

    return H, Vx, Vy


def assemble_H_V_Kp(kpt, setup):
    """Build H (eV), Vx, Vy (Ang/s) at one k-point for K' valley."""
    pp = setup['pp']
    qq = setup['qq']
    Lx = setup['Lx_Ang']
    Ly = setup['Ly_Ang']
    gamma = setup['gamma']
    mo = setup['moire_offset']
    M_mag = setup['M_mag_Ang']
    dim = setup['dim_total']

    kpts = kpt - M_mag
    kx, ky = kpts
    pp_over_qq = pp / qq

    # --- Phase factors (K' signs flipped) ---
    tphase1 = np.exp(-1j * pp_over_qq * kx * Lx)
    tphase2 = (np.exp(1j * pp_over_qq * kx * Lx / 2)
               * np.exp(-1j * ky * Ly * pp_over_qq))
    tphase3 = (np.exp(1j * pp_over_qq * kx * Lx / 2)
               * np.exp(1j * ky * Ly * pp_over_qq))

    # --- Hamiltonian (eV) ---
    V_pq = (gamma * tphase1 * setup['term1_Kp']
            + tphase2 * setup['term2_Kp']
            + tphase3 * setup['term3_Kp'])

    H = setup['H_base_Kp'].copy()
    H[mo:, mo:] += setup['v0_eye'] + V_pq + V_pq.T.conj()

    # --- dH/dk (eV * Ang, K' signs) ---
    vx_tp1 = (-1j * pp_over_qq * Lx) * tphase1
    vx_tp2 = (1j * pp_over_qq * Lx / 2) * tphase2
    vx_tp3 = (1j * pp_over_qq * Lx / 2) * tphase3

    vy_tp1 = 0.0
    vy_tp2 = (-1j * pp_over_qq * Ly) * tphase2
    vy_tp3 = (1j * pp_over_qq * Ly) * tphase3

    vx_tmp = (gamma * vx_tp1 * setup['term1_Kp']
              + vx_tp2 * setup['term2_Kp']
              + vx_tp3 * setup['term3_Kp'])
    vy_tmp = (gamma * vy_tp1 * setup['term1_Kp']
              + vy_tp2 * setup['term2_Kp']
              + vy_tp3 * setup['term3_Kp'])

    Hdx = np.zeros((dim, dim), dtype=complex)
    Hdy = np.zeros((dim, dim), dtype=complex)
    Hdx[mo:, mo:] = vx_tmp + vx_tmp.T.conj()
    Hdy[mo:, mo:] = vy_tmp + vy_tmp.T.conj()

    # --- Velocity (Ang/s) ---
    if setup.get('include_comm', 0):
        Ax = setup['Ax_Kp']
        Ay = setup['Ay_Kp']
        Vx = (Hdx - 1j * (Ax @ H - H @ Ax)) / HBAR_EV
        Vy = (Hdy - 1j * (Ay @ H - H @ Ay)) / HBAR_EV
    else:
        Vx = Hdx / HBAR_EV
        Vy = Hdy / HBAR_EV

    return H, Vx, Vy
