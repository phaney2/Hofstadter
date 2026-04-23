"""
Magnetic Bloch bands for bilayer graphene on hBN substrate.

Translated from MATLAB code by Paul M. Haney.
Reference: "A Quantum Ruler for Orbital Magnetism in Moire Quantum Matter"
"""

import sys
import numpy as np
from scipy import linalg
from multiprocessing import Pool, cpu_count

from constants import HBAR, Q_E, A_GRAPHENE, A_HBN
from parser import parse_input_file
from basis import getindices
from hamiltonian import (get_interbilayerterms_K, get_interbilayerterms_Kp,
                         get_intermonolayerH_K, get_intermonolayerH_Kp,
                         get_intralayerH_K, get_intralayerH_Kp)


# ---------------------------------------------------------------------------
# Per-k-point solver (serial core + parallel wrapper)
# ---------------------------------------------------------------------------

def _solve_kpoint_core(d, kpt):
    """Solve both valleys at a single k-point. Returns (tek_K, tek_Kp)."""
    kpts = kpt - d['M_mag']
    kx_val, ky_val = kpts
    pp, qq = d['pp'], d['qq']
    Lx, Ly = d['Lx'], d['Ly']
    gamma, dim_MLG = d['gamma'], d['dim_MLG']

    tek_K = None
    tek_Kp = None

    if 'K' in d['valley']:
        tphase1 = np.exp(1j * (pp / qq) * kx_val * Lx)
        tphase2 = np.exp(-1j * (pp / qq) * kx_val * Lx / 2) * np.exp(1j * ky_val * Ly * (pp / qq))
        tphase3 = np.exp(-1j * (pp / qq) * kx_val * Lx / 2) * np.exp(-1j * ky_val * Ly * (pp / qq))

        V_pq = gamma * tphase1 * d['term1_K'] + tphase2 * d['term2_K'] + tphase3 * d['term3_K']

        Htotal_K = d['H_base_K'].copy()
        Htotal_K[dim_MLG:, dim_MLG:] += d['v0_eye'] + V_pq + V_pq.T.conj()
        tek_K = np.sort(linalg.eigvalsh(Htotal_K, overwrite_a=True, check_finite=False))

    if 'Kp' in d['valley']:
        tphase1 = np.exp(-1j * (pp / qq) * kx_val * Lx)
        tphase2 = np.exp(1j * (pp / qq) * kx_val * Lx / 2) * np.exp(-1j * ky_val * Ly * (pp / qq))
        tphase3 = np.exp(1j * (pp / qq) * kx_val * Lx / 2) * np.exp(1j * ky_val * Ly * (pp / qq))

        V_pq = gamma * tphase1 * d['term1_Kp'] + tphase2 * d['term2_Kp'] + tphase3 * d['term3_Kp']

        Htotal_Kp = d['H_base_Kp'].copy()
        Htotal_Kp[dim_MLG:, dim_MLG:] += d['v0_eye'] + V_pq + V_pq.T.conj()
        tek_Kp = np.sort(linalg.eigvalsh(Htotal_Kp, overwrite_a=True, check_finite=False))

    return tek_K, tek_Kp


_worker_shared = {}

def _init_kpoint_worker(shared):
    global _worker_shared
    _worker_shared = shared


def _solve_kpoint(args):
    kc, kpt = args
    tek_K, tek_Kp = _solve_kpoint_core(_worker_shared, kpt)
    return kc, tek_K, tek_Kp


# ---------------------------------------------------------------------------
# Main calculation routine
# ---------------------------------------------------------------------------

def do_calc(filepath):
    """
    Compute magnetic Bloch bands for bilayer graphene on hBN.

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
    readFnmtable = 0

    # --- read input ---
    inp = parse_input_file(filepath)
    locals_dict = dict(inp)

    theta = locals_dict.get('theta', 0.0)
    qq = int(locals_dict['qq'])
    pp = int(locals_dict['pp'])
    g0 = locals_dict['g0']
    g1 = locals_dict['g1']
    g3 = locals_dict['g3']
    g4 = locals_dict['g4']
    delta = locals_dict.get('delta', 0)
    v0_meV = locals_dict['v0']
    v1_meV = locals_dict['v1']
    w_meV = locals_dict['w']
    eta = locals_dict.get('eta', eta)
    U = np.atleast_1d(locals_dict.get('U', np.array([0, 0])))
    nk1 = int(locals_dict.get('nk1', 25))
    nk2 = int(locals_dict.get('nk2', 40))
    LL_multiplier = locals_dict.get('LL_multiplier', 6)
    Nmax = int(locals_dict.get('Nmax', Nmax))
    calctype = locals_dict.get('calctype', calctype)
    valley = locals_dict.get('valley', valley)
    nebin = int(locals_dict.get('nebin', 1000))
    gamma = locals_dict.get('gamma', gamma)
    vF = locals_dict.get('vF', vF)
    isparallel = int(locals_dict.get('isparallel', 0))
    elist = np.asarray(locals_dict.get('elist', np.linspace(-300, 300, nebin)))

    if calctype == 'spectrum':
        calctype = 'dos'

    # --- derived quantities ---
    eps = A_HBN / A_GRAPHENE - 1
    L_moire = (1 + eps) * A_GRAPHENE / np.sqrt(eps ** 2 + 2 * (1 + eps) * (1 - np.cos(theta)))

    kd = 4 * np.pi / (3 * A_GRAPHENE)
    ktheta = 4 * np.pi / (3 ** 0.5 * L_moire)
    uc_area = 3 ** 0.5 * L_moire ** 2 / 2

    phi_0 = HBAR * 2 * np.pi / Q_E
    B = (qq / pp) * phi_0 / uc_area
    lB = (HBAR / (Q_E * B)) ** 0.5

    eneLL = g0 / 1e3 * Q_E * A_GRAPHENE / lB * 2 ** 0.5

    w_J = w_meV / 1e3 * Q_E
    v0 = v0_meV / 1e3 * Q_E
    v1 = v1_meV / 1e3 * Q_E

    Delta_param = 3 ** 0.5 / 2 * ktheta * lB ** 2

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

    Nq = qq
    dim1 = qq * N + qq * (N + 1)

    Utp = np.eye(dim1) * U[0] / 1e3 * Q_E
    Ubm = np.eye(dim1) * U[1] / 1e3 * Q_E

    Lx = L_moire
    Ly = np.sqrt(3) * L_moire / 2

    print(f"  N (Landau levels) = {N}")
    print(f"  dim per layer = {dim1}")
    print(f"  B = {B:.6e} T")

    # --- K valley: k-independent Hamiltonian ---
    if 'K' in valley:
        print("  Building K-valley Hamiltonian...")
        term1_K, term2_K, term3_K, qNslabels_K = get_interbilayerterms_K(
            N, Nq, ktheta, lB, v0, v1, eta, qq, pp)
        Hinter_K = get_intermonolayerH_K(N, 0, B, qNslabels_K, TBGparams)
        Hintra1_K = get_intralayerH_K(N, 0, B, qNslabels_K, TBGparams, 'A')
        Hintra2_K = get_intralayerH_K(N, 0, B, qNslabels_K, TBGparams, 'B')

        H_BLG_K = np.block([
            [Hintra1_K + Utp, Hinter_K],
            [Hinter_K.T.conj(), Hintra2_K + Ubm]
        ])

        chop_K = getindices(qNslabels_K, ['B', f"LL{N}_"])
        qNslabels_K_trimmed = [s for i, s in enumerate(qNslabels_K) if i not in chop_K]

    # --- K' valley: k-independent Hamiltonian ---
    if 'Kp' in valley:
        print("  Building K'-valley Hamiltonian...")
        term1_Kp, term2_Kp, term3_Kp, qNslabels_Kp = get_interbilayerterms_Kp(
            N, Nq, ktheta, lB, v0, v1, eta, qq, pp)
        Hinter_Kp = get_intermonolayerH_Kp(N, 0, B, qNslabels_Kp, TBGparams)
        Hintra1_Kp = get_intralayerH_Kp(N, 0, B, qNslabels_Kp, TBGparams, 'A')
        Hintra2_Kp = get_intralayerH_Kp(N, 0, B, qNslabels_Kp, TBGparams, 'B')

        H_BLG_Kp = np.block([
            [Hintra1_Kp + Utp, Hinter_Kp],
            [Hinter_Kp.T.conj(), Hintra2_Kp + Ubm]
        ])

        chop_Kp = getindices(qNslabels_Kp, ['A', f"LL{N}_"])
        qNslabels_Kp_trimmed = [s for i, s in enumerate(qNslabels_Kp) if i not in chop_Kp]

    # --- k-mesh ---
    b1 = ktheta * np.array([0, -1])
    b2 = ktheta * np.array([np.sqrt(3) / 2, 1 / 2])

    M_pt = 0.5 * b1
    M_mag = M_pt / pp

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

    dim_MLG = Hintra1_K.shape[0] if 'K' in valley else Hintra1_Kp.shape[0]
    bands_K = np.zeros((Nk_tot, 2 * dim_MLG))
    bands_Kp = np.zeros((Nk_tot, 2 * dim_MLG))

    # --- pre-scale and pack shared data for the k-point solver ---
    scale = 1000 / Q_E
    v0_eye_scaled = scale * v0 * np.eye(dim_MLG)

    shared = {
        'pp': pp, 'qq': qq, 'Lx': Lx, 'Ly': Ly,
        'gamma': gamma, 'dim_MLG': dim_MLG,
        'valley': valley, 'M_mag': M_mag,
        'v0_eye': v0_eye_scaled,
    }
    if 'K' in valley:
        shared.update({
            'H_base_K': scale * H_BLG_K,
            'term1_K': scale * term1_K,
            'term2_K': scale * term2_K,
            'term3_K': scale * term3_K,
        })
    if 'Kp' in valley:
        shared.update({
            'H_base_Kp': scale * H_BLG_Kp,
            'term1_Kp': scale * term1_Kp,
            'term2_Kp': scale * term2_Kp,
            'term3_Kp': scale * term3_Kp,
        })

    print(" Entering the k loop")
    if isparallel:
        nworkers = min(cpu_count(), Nk_tot)
        print(f"  (parallel: {nworkers} workers)")
        tasks = [(kc, kpoints[kc, :]) for kc in range(Nk_tot)]
        with Pool(processes=nworkers,
                  initializer=_init_kpoint_worker, initargs=(shared,)) as pool:
            results = pool.map(_solve_kpoint, tasks)
        for kc, tek_K_row, tek_Kp_row in results:
            if tek_K_row is not None:
                bands_K[kc, :] = tek_K_row
            if tek_Kp_row is not None:
                bands_Kp[kc, :] = tek_Kp_row
    else:
        for kc in range(Nk_tot):
            print(f"  |>>        step {kc + 1} of {Nk_tot}")
            tek_K_row, tek_Kp_row = _solve_kpoint_core(shared, kpoints[kc, :])
            if tek_K_row is not None:
                bands_K[kc, :] = tek_K_row
            if tek_Kp_row is not None:
                bands_Kp[kc, :] = tek_Kp_row

    print(" Done with the k loop")

    if calctype == 'ek':
        return {'calctype': 'ek', 'params': inp,
                'kpoints': kpoints,
                'bands_K': bands_K, 'bands_Kp': bands_Kp}

    # --- DOS: bin eigenvalues into energy histogram ---
    dos_K = np.zeros(len(elist))
    dos_Kp = np.zeros(len(elist))
    for kc in range(Nk_tot):
        if 'K' in valley:
            tek = bands_K[kc, :]
            in_range = tek[(tek > elist[0]) & (tek < elist[-1])]
            bins = np.argmin(np.abs(in_range[:, None] - elist[None, :]), axis=1)
            for b in bins:
                dos_K[b] += 1
        if 'Kp' in valley:
            tek = bands_Kp[kc, :]
            in_range = tek[(tek > elist[0]) & (tek < elist[-1])]
            bins = np.argmin(np.abs(in_range[:, None] - elist[None, :]), axis=1)
            for b in bins:
                dos_Kp[b] += 1

    return {'calctype': 'dos', 'params': inp,
            'elist': elist,
            'dos_K': dos_K, 'dos_Kp': dos_Kp}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _save_result(result, outfile):
    """Save result dict to .npz or .mat depending on file extension."""
    data = {k: v for k, v in result.items() if k not in ('calctype', 'params')}
    params = result.get('params', {})

    if outfile.endswith('.mat'):
        from scipy.io import savemat
        data['params'] = params
        savemat(outfile, data)
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
