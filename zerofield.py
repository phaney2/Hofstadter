"""
Zero-field moire band structure for mono-, bi-, or ABC-stacked trilayer
graphene on hBN.

Uses a continuum model with plane-wave expansion in the moire reciprocal
lattice.  No magnetic field; the basis states are (sublattice, Q-vector)
rather than Landau levels.

Translated from MATLAB code by Paul M. Haney.
"""

import sys
import numpy as np
from scipy import linalg

from constants import A_GRAPHENE, A_HBN
from parser import parse_input_file

PSI_DEFAULT = -0.29


def _compute_moire_vectors(theta, a, a_hBN):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0, 0, 1]])
    a1 = a * np.array([1.0, 0.0, 0.0])
    a2 = a * np.array([0.5, np.sqrt(3) / 2, 0.0])
    a3 = np.array([0.0, 0.0, 1.0])

    A = np.eye(3) - (a / a_hBN) * np.linalg.inv(R)
    M1 = np.linalg.solve(A, a1)
    M2 = np.linalg.solve(A, a2)
    M3 = np.linalg.solve(A, a3)

    vol = np.dot(M1, np.cross(M2, M3))
    G1 = 2 * np.pi * np.cross(M2, M3) / vol
    G2 = 2 * np.pi * np.cross(M3, M1) / vol

    q1 = G1[:2]
    q2 = G2[:2]
    q3 = -q1 - q2
    return q1, q2, q3


def _build_qvectors(NQ, q1, q2):
    half = NQ // 2
    Q_list = []
    idx_list = []
    for p in range(NQ):
        for r in range(NQ):
            pe = p - half
            re = r - half
            Q_list.append(pe * q1 + re * q2)
            idx_list.append((pe, re))

    Q_arr = np.array(Q_list)
    idx_arr = np.array(idx_list)
    norms = np.linalg.norm(Q_arr, axis=1)
    order = np.argsort(norms, kind='stable')

    return Q_arr[order], idx_arr[order], len(Q_arr)


def _build_coupling_matrices_K(V0, V1, psi=PSI_DEFAULT):
    w = np.exp(1j * 2 * np.pi / 3)
    T0 = V0 * np.eye(2)
    ph = V1 * np.exp(1j * psi)
    T1 = ph * np.array([[1, w**(-1)], [1, w**(-1)]])
    T2 = ph * np.array([[1, w], [w, w**(-1)]])
    T3 = ph * np.array([[1, 1], [w**(-1), w**(-1)]])
    return T0, T1, T2, T3


def _build_coupling_matrices_Kp(V0, V1, psi=PSI_DEFAULT):
    w = np.exp(1j * 2 * np.pi / 3)
    T0 = V0 * np.eye(2)
    ph = V1 * np.exp(-1j * psi)
    T1 = ph * np.array([[1, w], [1, w]])
    T2 = ph * np.array([[1, w**(-1)], [w**(-1), w]])
    T3 = ph * np.array([[1, 1], [w, w]])
    return T0, T1, T2, T3


def _build_moire_hopping(Q_idx, NG, T0, T1, T2, T3, valley):
    H = np.zeros((2 * NG, 2 * NG), dtype=complex)
    s = 1 if valley == 'K' else -1

    for j in range(NG):
        pj, rj = Q_idx[j]
        for k in range(NG):
            pk, rk = Q_idx[k]
            dp = pj - pk
            dr = rj - rk

            block = np.zeros((2, 2), dtype=complex)

            if dp == 0 and dr == 0:
                block += T0
            if dp == s and dr == 0:
                block += T1.conj().T
            if dp == -s and dr == 0:
                block += T1
            if dp == 0 and dr == s:
                block += T2.conj().T
            if dp == 0 and dr == -s:
                block += T2
            if dp == -s and dr == -s:
                block += T3.conj().T
            if dp == s and dr == s:
                block += T3

            H[2 * j:2 * j + 2, 2 * k:2 * k + 2] = block

    return H


def _build_H_singlelayer(Q, NG, dirac_term, U_layers, H_hopp):
    H = np.zeros((2 * NG, 2 * NG), dtype=complex)
    for j in range(NG):
        H[2*j:2*j+2, 2*j:2*j+2] = dirac_term(j) + U_layers[0] * np.eye(2)
    H += H_hopp
    return H


def _build_H_bilayer(Q, NG, dirac_term, interlayer_term, U_layers, H_hopp,
                      stacking_type):
    U_top, U_bot = U_layers
    H0_T = np.zeros((2 * NG, 2 * NG), dtype=complex)
    H0_B = np.zeros((2 * NG, 2 * NG), dtype=complex)
    UBLG = np.zeros((2 * NG, 2 * NG), dtype=complex)
    for j in range(NG):
        H0_T[2*j:2*j+2, 2*j:2*j+2] = dirac_term(j) + U_top * np.eye(2)
        H0_B[2*j:2*j+2, 2*j:2*j+2] = dirac_term(j) + U_bot * np.eye(2)
        UBLG[2*j:2*j+2, 2*j:2*j+2] = interlayer_term(j)
    if stacking_type == 1:
        return np.block([[H0_T, UBLG],
                          [UBLG.conj().T, H0_B + H_hopp]])
    return np.block([[H0_T, UBLG.conj().T],
                      [UBLG, H0_B + H_hopp]])


def _build_H_trilayer(Q, NG, dirac_term, interlayer_term, U_layers, H_hopp,
                       stacking_type):
    """ABC (rhombohedral) stacking: layers 1-2 and 2-3 couple through the
    same gamma1/v3 interlayer operator (identical bond geometry at every
    step of the stack); layers 1-3 do not couple directly.  The hBN moire
    potential acts on layer 3 (the layer nearest the substrate) only."""
    U_top, U_mid, U_bot = U_layers
    H0_T = np.zeros((2 * NG, 2 * NG), dtype=complex)
    H0_M = np.zeros((2 * NG, 2 * NG), dtype=complex)
    H0_B = np.zeros((2 * NG, 2 * NG), dtype=complex)
    UBLG_12 = np.zeros((2 * NG, 2 * NG), dtype=complex)
    UBLG_23 = np.zeros((2 * NG, 2 * NG), dtype=complex)
    Z = np.zeros((2 * NG, 2 * NG), dtype=complex)
    for j in range(NG):
        dirac = dirac_term(j)
        interlayer = interlayer_term(j)
        H0_T[2*j:2*j+2, 2*j:2*j+2] = dirac + U_top * np.eye(2)
        H0_M[2*j:2*j+2, 2*j:2*j+2] = dirac + U_mid * np.eye(2)
        H0_B[2*j:2*j+2, 2*j:2*j+2] = dirac + U_bot * np.eye(2)
        UBLG_12[2*j:2*j+2, 2*j:2*j+2] = interlayer
        UBLG_23[2*j:2*j+2, 2*j:2*j+2] = interlayer
    if stacking_type == 1:
        return np.block([[H0_T, UBLG_12, Z],
                          [UBLG_12.conj().T, H0_M, UBLG_23],
                          [Z, UBLG_23.conj().T, H0_B + H_hopp]])
    return np.block([[H0_T, UBLG_12.conj().T, Z],
                      [UBLG_12, H0_M, UBLG_23.conj().T],
                      [Z, UBLG_23, H0_B + H_hopp]])


def _solve_kpath_K(kpoints, Q, NG, hbar_vF, gamma1, hbar_v3,
                   U_layers, H_hopp, nlayers, stacking_type=2):
    sigx = np.array([[0, 1], [1, 0]], dtype=complex)
    sigy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    U1 = np.array([[0, 1], [0, 0]], dtype=complex)
    U2 = np.array([[0, 0], [1, 0]], dtype=complex)

    dim = nlayers * 2 * NG
    NT = len(kpoints)
    bands = np.zeros((NT, dim))

    for i in range(NT):
        kx, ky = kpoints[i]

        def dirac_term(j, kx=kx, ky=ky):
            qx, qy = Q[j]
            return -hbar_vF * ((kx - qx) * sigx + (ky - qy) * sigy)

        def interlayer_term(j, kx=kx, ky=ky):
            qx, qy = Q[j]
            return gamma1 * U1 - hbar_v3 * ((kx - qx) - 1j * (ky - qy)) * U2

        if nlayers == 1:
            H = _build_H_singlelayer(Q, NG, dirac_term, U_layers, H_hopp)
        elif nlayers == 2:
            H = _build_H_bilayer(Q, NG, dirac_term, interlayer_term,
                                  U_layers, H_hopp, stacking_type)
        else:
            H = _build_H_trilayer(Q, NG, dirac_term, interlayer_term,
                                   U_layers, H_hopp, stacking_type)

        bands[i, :] = np.sort(linalg.eigvalsh(H))

    return bands


def _solve_kpath_Kp(kpoints, Q, NG, hbar_vF, gamma1, hbar_v3,
                    U_layers, H_hopp, nlayers, stacking_type=2):
    sigx = np.array([[0, 1], [1, 0]], dtype=complex)
    sigy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    U1 = np.array([[0, 1], [0, 0]], dtype=complex)
    U2 = np.array([[0, 0], [1, 0]], dtype=complex)

    dim = nlayers * 2 * NG
    NT = len(kpoints)
    bands = np.zeros((NT, dim))

    for i in range(NT):
        kx, ky = kpoints[i]

        def dirac_term(j, kx=kx, ky=ky):
            qx, qy = Q[j]
            return -hbar_vF * (-(kx - qx) * sigx + (ky - qy) * sigy)

        def interlayer_term(j, kx=kx, ky=ky):
            qx, qy = Q[j]
            return gamma1 * U1 - hbar_v3 * (-(kx - qx) - 1j * (ky - qy)) * U2

        if nlayers == 1:
            H = _build_H_singlelayer(Q, NG, dirac_term, U_layers, H_hopp)
        elif nlayers == 2:
            H = _build_H_bilayer(Q, NG, dirac_term, interlayer_term,
                                  U_layers, H_hopp, stacking_type)
        else:
            H = _build_H_trilayer(Q, NG, dirac_term, interlayer_term,
                                   U_layers, H_hopp, stacking_type)

        bands[i, :] = np.sort(linalg.eigvalsh(H))

    return bands


def _make_kpath(q1, q2, dk, valley):
    G = np.array([0.0, 0.0])
    K1 = (1 / 3) * q1 + (2 / 3) * q2
    K2 = (2 / 3) * q1 + (1 / 3) * q2

    if valley == 'K':
        segments = [(K1, G), (G, K2), (K2, K1)]
        labels = ['K1', 'G', 'K2', 'K1']
    else:
        segments = [(K2, K1), (K1, G), (G, K2)]
        labels = ['K2', 'K1', 'G', 'K2']

    kpoints = []
    seg_sizes = []

    for start, end in segments:
        npts = round(np.linalg.norm(end - start) / dk)
        npts = max(npts, 2)
        seg_sizes.append(npts)
        t = np.linspace(0, 1, npts)
        for ti in t:
            kpoints.append(start + ti * (end - start))

    kpoints = np.array(kpoints)
    NT = len(kpoints)
    k_linear = np.linspace(0, 1, NT)

    tick_idx = [0]
    cumul = 0
    for n in seg_sizes:
        cumul += n
        tick_idx.append(cumul - 1)

    tick_positions = k_linear[tick_idx]
    return kpoints, k_linear, tick_positions, labels


def do_calc(filepath):
    inp = parse_input_file(filepath)

    theta = np.radians(inp.get('theta', 0.0))
    nlayers = int(inp.get('nlayers', 2))
    g0 = inp['g0']
    g1 = inp.get('g1', 340)
    g3 = inp.get('g3', 0)
    v0_meV = inp['v0']
    v1_meV = inp['v1']
    U = np.atleast_1d(inp.get('U', np.array([0, 0])))
    NQ = int(inp.get('NQ', 7))
    dk = inp.get('dk', 5e-4)
    valley = inp.get('valley', ['K', 'Kp'])
    stacking_type = int(inp.get('stacking_type', 2))
    psi = -float(inp.get('moire_psi', 0.29))

    a = A_GRAPHENE * 1e10
    a_hBN = A_HBN * 1e10
    # TODO: remove vF/hbar_vF override; always derive from g0 to match Hofstadter conventions
    hbar_vF_override = inp.get('vF', inp.get('hbar_vF', None))
    if hbar_vF_override is not None:
        hbar_vF = float(hbar_vF_override)
    else:
        hbar_vF = np.sqrt(3) / 2 * (g0 / 1000) * a
    gamma1 = g1 / 1000
    hbar_v3 = np.sqrt(3) / 2 * (g3 / 1000) * a
    V0 = v0_meV / 1000
    V1 = v1_meV / 1000

    if nlayers not in (1, 2, 3):
        raise ValueError(f"nlayers = {nlayers} not supported (must be 1, 2, or 3)")

    if nlayers == 1:
        U_layers = (U[0] / 1000,)
    elif nlayers == 2:
        U_top = U[0] / 1000
        U_bot = U[1] / 1000 if len(U) > 1 else U[0] / 1000
        U_layers = (U_top, U_bot)
    else:
        U_top = U[0] / 1000
        U_mid = U[1] / 1000 if len(U) > 1 else U[0] / 1000
        U_bot = U[2] / 1000 if len(U) > 2 else U_mid
        U_layers = (U_top, U_mid, U_bot)

    q1, q2, q3 = _compute_moire_vectors(theta, a, a_hBN)
    Q, Q_idx, NG = _build_qvectors(NQ, q1, q2)

    dim = nlayers * 2 * NG
    print(f"  nlayers = {nlayers}")
    print(f"  NQ = {NQ}, NG = {NG}")
    print(f"  hbar*vF = {hbar_vF:.4f} eV*A")
    print(f"  gamma1 = {gamma1:.4f} eV")
    print(f"  Hamiltonian dim = {dim}")

    result = {'params': inp, 'dim': dim}

    for v in ['K', 'Kp']:
        if v not in valley:
            continue

        print(f"  Building {v} valley...")
        if v == 'K':
            T0, T1, T2, T3 = _build_coupling_matrices_K(V0, V1, psi)
        else:
            T0, T1, T2, T3 = _build_coupling_matrices_Kp(V0, V1, psi)

        H_hopp = _build_moire_hopping(Q_idx, NG, T0, T1, T2, T3, v)

        kpoints, k_linear, tick_pos, tick_labels = _make_kpath(q1, q2, dk, v)
        NT = len(kpoints)
        print(f"    {NT} k-points along path")

        if v == 'K':
            bands = _solve_kpath_K(kpoints, Q, NG, hbar_vF, gamma1,
                                   hbar_v3, U_layers, H_hopp, nlayers,
                                   stacking_type)
        else:
            bands = _solve_kpath_Kp(kpoints, Q, NG, hbar_vF, gamma1,
                                    hbar_v3, U_layers, H_hopp, nlayers,
                                    stacking_type)

        suffix = '_K' if v == 'K' else '_Kp'
        result[f'band{suffix}'] = bands
        result[f'k_region{suffix}'] = k_linear
        result[f'tick_positions{suffix}'] = tick_pos
        result[f'tick_labels{suffix}'] = tick_labels

    return result


def _save_result(result, outfile):
    data = {k: v for k, v in result.items() if k not in ('params',)}
    params = result.get('params', {})

    if outfile.endswith('.mat'):
        from scipy.io import savemat
        savemat(outfile, {'results': data, 'params': params})
    else:
        for k, v in params.items():
            data[f'input_{k}'] = np.asarray(v)
        np.savez(outfile, **data)

    print(f"  Saved to {outfile}")


def main(input_file=None):
    if input_file is None:
        input_file = './input_zerofield.txt'

    result = do_calc(input_file)
    params = result['params']
    outfile = params.get('outputfile', 'bands_zerofield.npz')

    _save_result(result, outfile)
    return result


if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(input_file)
