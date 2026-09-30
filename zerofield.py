"""
Zero-field moire band structure for mono-, bi-, or ABC-stacked trilayer
graphene on hBN.

Uses a continuum model with plane-wave expansion in the moire reciprocal
lattice.  No magnetic field; the basis states are (sublattice, Q-vector)
rather than Landau levels.

Translated from MATLAB code by Paul M. Haney.
"""

import os
import sys
from multiprocessing import cpu_count, get_context

import numpy as np
from scipy import linalg

from constants import A_GRAPHENE, A_HBN
from parser import parse_input_file

PSI_DEFAULT = -0.29

# Layer 0 is the layer furthest from the hBN substrate; the moire potential
# acts on the last layer only.  Names match the bilayer keys in main_v3.py.
_LAYER_NAMES = {1: ('top',), 2: ('top', 'bottom'), 3: ('top', 'mid', 'bottom')}


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


def _layer_weights(evecs, NG, nlayers):
    """Per-eigenstate probability on each layer.  Returns (dim, nlayers);
    rows sum to 1."""
    blk = 2 * NG
    wt = np.empty((evecs.shape[1], nlayers))
    for L in range(nlayers):
        wt[:, L] = np.sum(np.abs(evecs[L * blk:(L + 1) * blk, :])**2, axis=0)
    return wt


def _solve_kpath_K(kpoints, Q, NG, hbar_vF, gamma1, hbar_v3,
                   U_layers, H_hopp, nlayers, stacking_type=2,
                   layer_resolved=0):
    sigx = np.array([[0, 1], [1, 0]], dtype=complex)
    sigy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    U1 = np.array([[0, 1], [0, 0]], dtype=complex)
    U2 = np.array([[0, 0], [1, 0]], dtype=complex)

    dim = nlayers * 2 * NG
    NT = len(kpoints)
    bands = np.zeros((NT, dim))
    weights = np.zeros((NT, dim, nlayers)) if layer_resolved else None

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

        if layer_resolved:
            bands[i, :], evecs = linalg.eigh(H)
            weights[i] = _layer_weights(evecs, NG, nlayers)
        else:
            bands[i, :] = np.sort(linalg.eigvalsh(H))

    return bands, weights


def _solve_kpath_Kp(kpoints, Q, NG, hbar_vF, gamma1, hbar_v3,
                    U_layers, H_hopp, nlayers, stacking_type=2,
                    layer_resolved=0):
    sigx = np.array([[0, 1], [1, 0]], dtype=complex)
    sigy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    U1 = np.array([[0, 1], [0, 0]], dtype=complex)
    U2 = np.array([[0, 0], [1, 0]], dtype=complex)

    dim = nlayers * 2 * NG
    NT = len(kpoints)
    bands = np.zeros((NT, dim))
    weights = np.zeros((NT, dim, nlayers)) if layer_resolved else None

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

        if layer_resolved:
            bands[i, :], evecs = linalg.eigh(H)
            weights[i] = _layer_weights(evecs, NG, nlayers)
        else:
            bands[i, :] = np.sort(linalg.eigvalsh(H))

    return bands, weights


def _solve_chunk(args):
    valley, kchunk, kw, dos_spec = args
    fn = _solve_kpath_K if valley == 'K' else _solve_kpath_Kp
    bands, weights = fn(kchunk, **kw)
    if dos_spec is None:
        return bands, weights
    # Histogram in the worker: returning (nebin,) arrays instead of the
    # (nk, dim, nlayers) eigenvector weights keeps the data sent back to
    # the parent independent of the mesh size.  At nk = 240 that is the
    # difference between ~0.5 GB and ~0.3 MB.
    elist, nk_total, nlayers = dos_spec
    return _accumulate_dos(bands, weights, elist, nlayers, nk_total)


def _solve_parallel(valley, kpoints, kw, nworkers, dos_spec=None):
    """Split the k-points across processes and run the serial solver on each
    chunk.  Chunks are contiguous and `Pool.map` preserves order, so the
    concatenated result is identical to the serial one, not merely
    equivalent.

    With `dos_spec = (elist, nk_total, nlayers)` each worker histograms its
    own chunk and only the DOS arrays come back.  The unweighted DOS is
    again identical to the serial one, since `bincount` counts states as
    integers and `1/Nk` is applied once.  The layer-resolved DOS is a
    weighted `bincount`, so summing partial histograms reassociates a
    float sum and it agrees to ~1e-16 relative rather than exactly."""
    nchunks = min(len(kpoints), max(nworkers * 4, 1))
    tasks = [(valley, kpoints[idx], kw, dos_spec)
             for idx in np.array_split(np.arange(len(kpoints)), nchunks)
             if len(idx)]

    # Set before the pool is created: 'spawn' children read the environment
    # at interpreter startup, so this reaches their BLAS before it loads.
    # One thread per worker -- otherwise nworkers processes each spin up a
    # full BLAS thread pool and oversubscribe the machine.
    for var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'

    with get_context('spawn').Pool(processes=nworkers) as pool:
        out = pool.map(_solve_chunk, tasks, chunksize=1)

    if dos_spec is not None:
        dos = sum(d for d, _ in out)
        dos_layers = (sum(dl for _, dl in out)
                      if out[0][1] is not None else None)
        return dos, dos_layers

    bands = np.concatenate([b for b, _ in out], axis=0)
    weights = (np.concatenate([w for _, w in out], axis=0)
               if out[0][1] is not None else None)
    return bands, weights


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


def _make_kmesh(q1, q2, nk1, nk2):
    """Uniform mesh tiling one moire BZ: k = (i/nk1)*q1 + (j/nk2)*q2."""
    f1 = np.arange(nk1) / nk1
    f2 = np.arange(nk2) / nk2
    F1, F2 = np.meshgrid(f1, f2, indexing='ij')
    return (F1.ravel(order='F')[:, None] * q1[None, :]
            + F2.ravel(order='F')[:, None] * q2[None, :])


def _accumulate_dos(bands, weights, elist, nlayers, nk_total=None):
    """Histogram eigenvalues onto the nearest `elist` grid point.

    Weight 1/Nk per eigenvalue: one state per moire primitive cell per
    k-point, so the output is states per primitive moire cell per bin.
    `nk_total` overrides the normalization when `bands` holds only a chunk
    of the mesh, so per-chunk results sum to the full-mesh DOS.
    """
    nebin = len(elist)
    midpoints = 0.5 * (elist[:-1] + elist[1:])
    wk = 1.0 / (bands.shape[0] if nk_total is None else nk_total)

    e = bands.ravel()
    mask = (e >= elist[0]) & (e <= elist[-1])
    idx = np.searchsorted(midpoints, e[mask])

    dos = np.bincount(idx, minlength=nebin) * wk
    if weights is None:
        return dos, None

    wt = weights.reshape(-1, nlayers)[mask]
    dos_layers = np.empty((nebin, nlayers))
    for L in range(nlayers):
        dos_layers[:, L] = np.bincount(idx, weights=wt[:, L],
                                       minlength=nebin) * wk
    return dos, dos_layers


def _broaden(dos, elist, sigma):
    """Convolve with a normalized Gaussian of width `sigma` (same units as
    `elist`).  Preserves the integral, so the DOS normalization is
    unchanged -- apart from weight pushed past the ends of `elist`, which
    `mode='same'` discards."""
    de = elist[1] - elist[0]
    half = int(np.ceil(4 * sigma / de))
    if half < 1:
        return dos
    x = np.arange(-half, half + 1) * de
    kern = np.exp(-0.5 * (x / sigma)**2)
    kern /= kern.sum()
    return np.convolve(dos, kern, mode='same')


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
    calctype = inp.get('calctype', 'ek')
    layer_resolved = int(inp.get('layer_resolved', 0))
    isparallel = int(inp.get('isparallel', 0))

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
    if calctype not in ('ek', 'dos'):
        raise ValueError(f"calctype = '{calctype}' not supported "
                         f"(must be 'ek' or 'dos')")
    if layer_resolved and nlayers == 1:
        layer_resolved = 0

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
    print(f"  calctype = {calctype}")
    print(f"  nlayers = {nlayers}")
    print(f"  NQ = {NQ}, NG = {NG}")
    print(f"  hbar*vF = {hbar_vF:.4f} eV*A")
    print(f"  gamma1 = {gamma1:.4f} eV")
    print(f"  U = [{', '.join(f'{u*1000:.1f}' for u in U_layers)}] meV")
    print(f"  Hamiltonian dim = {dim}")

    result = {'params': inp, 'dim': dim, 'calctype': calctype,
              'layer_resolved': layer_resolved}

    if calctype == 'dos':
        nk1 = int(inp.get('nk1', 30))
        nk2 = int(inp.get('nk2', nk1))
        nebin = int(inp.get('nebin', 1000))
        elist_meV = np.atleast_1d(inp.get('elist',
                                          np.linspace(-300, 300, nebin)))
        elist = np.asarray(elist_meV, dtype=float) / 1000
        sigma = float(inp.get('dos_broadening', 0.0)) / 1000
        kmesh = _make_kmesh(q1, q2, nk1, nk2)
        result['elist'] = elist
        print(f"  DOS mesh = {nk1} x {nk2} = {len(kmesh)} k-points")
        print(f"  elist = [{elist_meV[0]:.1f}, {elist_meV[-1]:.1f}] meV, "
              f"{len(elist)} bins (output in eV)")
        if sigma > 0:
            print(f"  dos_broadening = {sigma*1000:.2f} meV (Gaussian)")

    for v in ['K', 'Kp']:
        if v not in valley:
            continue

        print(f"  Building {v} valley...")
        if v == 'K':
            T0, T1, T2, T3 = _build_coupling_matrices_K(V0, V1, psi)
        else:
            T0, T1, T2, T3 = _build_coupling_matrices_Kp(V0, V1, psi)

        H_hopp = _build_moire_hopping(Q_idx, NG, T0, T1, T2, T3, v)

        if calctype == 'dos':
            kpoints = kmesh
        else:
            kpoints, k_linear, tick_pos, tick_labels = _make_kpath(q1, q2, dk, v)
            print(f"    {len(kpoints)} k-points along path")

        kw = dict(Q=Q, NG=NG, hbar_vF=hbar_vF, gamma1=gamma1,
                  hbar_v3=hbar_v3, U_layers=U_layers, H_hopp=H_hopp,
                  nlayers=nlayers, stacking_type=stacking_type,
                  layer_resolved=layer_resolved)

        parallel = isparallel and len(kpoints) > 1
        if parallel:
            nworkers = min(int(inp.get('nworkers', cpu_count())), len(kpoints))
            print(f"    (parallel: {nworkers} workers)")

        suffix = '_K' if v == 'K' else '_Kp'

        if calctype == 'dos':
            spec = (elist, len(kpoints), nlayers)
            if parallel:
                dos, dos_layers = _solve_parallel(v, kpoints, kw, nworkers,
                                                  dos_spec=spec)
            else:
                fn = _solve_kpath_K if v == 'K' else _solve_kpath_Kp
                bands, weights = fn(kpoints, **kw)
                dos, dos_layers = _accumulate_dos(bands, weights, elist,
                                                  nlayers, len(kpoints))
            if sigma > 0:
                dos = _broaden(dos, elist, sigma)
            result[f'dos{suffix}'] = dos
            if layer_resolved:
                for L, name in enumerate(_LAYER_NAMES[nlayers]):
                    col = dos_layers[:, L]
                    if sigma > 0:
                        col = _broaden(col, elist, sigma)
                    result[f'dos{suffix}_{name}'] = col
        else:
            if parallel:
                bands, weights = _solve_parallel(v, kpoints, kw, nworkers)
            elif v == 'K':
                bands, weights = _solve_kpath_K(kpoints, **kw)
            else:
                bands, weights = _solve_kpath_Kp(kpoints, **kw)

            result[f'band{suffix}'] = bands
            result[f'k_region{suffix}'] = k_linear
            result[f'tick_positions{suffix}'] = tick_pos
            result[f'tick_labels{suffix}'] = tick_labels
            if layer_resolved:
                result[f'weights{suffix}'] = weights

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
