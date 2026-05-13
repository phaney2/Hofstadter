"""
Hofstadter butterfly with configurable hopping matrix conventions.

Sweeps over rational flux values qq/pp up to ppmax, parallelized
across flux points.  Outputs a .mat file for butterfly plotting.

Usage:
    python hofstadter_testing.py input_testing.txt \
        --order 2 3 1 --sxflag 1 --dagger 1 \
        --conj_flag 0 --psi_conj 0 --ppmax 10
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import argparse
import numpy as np
from scipy import linalg
from scipy.io import savemat, loadmat
from multiprocessing import Pool, cpu_count
from math import gcd

from constants import HBAR, Q_E, A_GRAPHENE, A_HBN
from parser import parse_input_file
from basis import getindices
from hamiltonian import (get_interbilayerterms_K_testing,
                         get_interbilayerterms_Kp_testing,
                         get_intermonolayerH_K, get_intermonolayerH_Kp,
                         get_intralayerH_K, get_intralayerH_Kp)


def rational_fractions(pmax):
    qq_list, pp_list = [], []
    for p in range(1, pmax + 1):
        for q in range(1, p + 1):
            if gcd(q, p) == 1:
                qq_list.append(q)
                pp_list.append(p)
    return qq_list, pp_list


_worker_config = {}


def _init_flux_worker(config):
    global _worker_config
    _worker_config = config


def _compute_single_flux(flux_args):
    pp, qq = flux_args
    cfg = _worker_config

    ktheta = cfg['ktheta']
    uc_area = cfg['uc_area']
    L_moire = cfg['L_moire']
    nlayers = cfg['nlayers']
    valley = cfg['valley']
    nk1 = cfg['nk1']
    nk2 = cfg['nk2']
    gamma = cfg['gamma']

    order = cfg['order']
    sxflag = cfg['sxflag']
    dagger = cfg['dagger']
    conj_flag = cfg['conj_flag']
    psi_conj = cfg['psi_conj']

    phi_0 = HBAR * 2 * np.pi / Q_E
    B = (qq / pp) * phi_0 / uc_area
    lB = (HBAR / (Q_E * B)) ** 0.5
    eneLL = cfg['g0_J'] * A_GRAPHENE / lB * 2 ** 0.5
    w_J = cfg['w_J']

    N = int(cfg['LL_multiplier'] * round(max(HBAR * cfg['vF'] * ktheta, w_J) / eneLL) ** 2)
    if N > cfg['Nmax']:
        N = cfg['Nmax']

    Nq = 2 * qq
    dim1 = 2 * (qq * N + qq * (N + 1))

    TBGparams = {
        'g0': cfg['g0_J'], 'g1': cfg['g1_J'],
        'g3': cfg['g3_J'], 'g4': cfg['g4_J'],
        'delta': cfg['delta_J'],
    }

    v0 = cfg['v0_J']
    v1 = cfg['v1_J']
    Lx = L_moire
    Ly = np.sqrt(3) * L_moire / 2

    U = cfg['U']
    if nlayers == 1:
        U_onsite_val = U[0] / 1e3 * Q_E
    else:
        Utp_val = U[0] / 1e3 * Q_E
        Ubm_val = U[1] / 1e3 * Q_E

    if 'K' in valley:
        term1_K, term2_K, term3_K, qNslabels_K = get_interbilayerterms_K_testing(
            N, Nq, ktheta, lB, v0, v1, cfg['eta'], qq, pp,
            order, sxflag, dagger, conj_flag, psi_conj)
        Hintra_K = get_intralayerH_K(N, 0, B, qNslabels_K, TBGparams, 'A')

        if nlayers == 1:
            H_base_K = Hintra_K + np.eye(Hintra_K.shape[0]) * U_onsite_val
        else:
            Hinter_K = get_intermonolayerH_K(N, 0, B, qNslabels_K, TBGparams)
            Hintra2_K = get_intralayerH_K(N, 0, B, qNslabels_K, TBGparams, 'B')
            dl = Hintra_K.shape[0]
            Utp = np.eye(dl) * Utp_val
            Ubm = np.eye(dl) * Ubm_val
            H_base_K = np.block([
                [Hintra_K + Utp, Hinter_K],
                [Hinter_K.T.conj(), Hintra2_K + Ubm]
            ])

    if 'Kp' in valley:
        term1_Kp, term2_Kp, term3_Kp, qNslabels_Kp = get_interbilayerterms_Kp_testing(
            N, Nq, ktheta, lB, v0, v1, cfg['eta'], qq, pp,
            order, sxflag, dagger, conj_flag, psi_conj)
        Hintra_Kp = get_intralayerH_Kp(N, 0, B, qNslabels_Kp, TBGparams, 'A')

        if nlayers == 1:
            H_base_Kp = Hintra_Kp + np.eye(Hintra_Kp.shape[0]) * U_onsite_val
        else:
            Hinter_Kp = get_intermonolayerH_Kp(N, 0, B, qNslabels_Kp, TBGparams)
            Hintra2_Kp = get_intralayerH_Kp(N, 0, B, qNslabels_Kp, TBGparams, 'B')
            dl = Hintra_Kp.shape[0]
            Utp = np.eye(dl) * Utp_val
            Ubm = np.eye(dl) * Ubm_val
            H_base_Kp = np.block([
                [Hintra_Kp + Utp, Hinter_Kp],
                [Hinter_Kp.T.conj(), Hintra2_Kp + Ubm]
            ])

    b1 = ktheta * np.array([0, -1])
    b2 = ktheta * np.array([np.sqrt(3) / 2, 1 / 2])
    M_mag = 0.5 * b1 / pp

    Nk_tot = nk1 * nk2
    n1_arr = np.arange(nk1)
    n2_arr = np.arange(nk2)
    n1grid, n2grid = np.meshgrid(n1_arr, n2_arr)
    n11 = n1grid.flatten(order='F')
    n22 = n2grid.flatten(order='F')

    vb = np.array([b1 / pp / 2, b2 / pp])
    kpoints = np.zeros((Nk_tot, 2))
    for j in range(Nk_tot):
        frac = np.array([n11[j] / nk1, n22[j] / nk2])
        kpoints[j, :] = vb.T @ frac

    dim_MLG = Hintra_K.shape[0] if 'K' in valley else Hintra_Kp.shape[0]
    dim_total = dim_MLG if nlayers == 1 else 2 * dim_MLG

    scale = 1000 / Q_E
    v0_eye_s = scale * v0 * np.eye(dim_MLG)
    mo = 0 if nlayers == 1 else dim_MLG

    if 'K' in valley:
        H_base_K_s = scale * H_base_K
        t1Ks, t2Ks, t3Ks = scale * term1_K, scale * term2_K, scale * term3_K
    if 'Kp' in valley:
        H_base_Kp_s = scale * H_base_Kp
        t1Kps, t2Kps, t3Kps = scale * term1_Kp, scale * term2_Kp, scale * term3_Kp

    bands_K = np.zeros((Nk_tot, dim_total)) if 'K' in valley else None
    bands_Kp = np.zeros((Nk_tot, dim_total)) if 'Kp' in valley else None

    for kc in range(Nk_tot):
        kpts = kpoints[kc, :] - M_mag
        kx_val, ky_val = kpts

        if 'K' in valley:
            tph1 = np.exp(1j * (pp / qq) * kx_val * Lx)
            tph2 = np.exp(-1j * (pp / qq) * kx_val * Lx / 2) * np.exp(1j * ky_val * Ly * (pp / qq))
            tph3 = np.exp(-1j * (pp / qq) * kx_val * Lx / 2) * np.exp(-1j * ky_val * Ly * (pp / qq))

            V_pq = gamma * tph1 * t1Ks + tph2 * t2Ks + tph3 * t3Ks
            Htotal = H_base_K_s.copy()
            Htotal[mo:, mo:] += v0_eye_s + V_pq + V_pq.T.conj()
            bands_K[kc, :] = np.sort(linalg.eigvalsh(Htotal, overwrite_a=True, check_finite=False))

        if 'Kp' in valley:
            tph1 = np.exp(-1j * (pp / qq) * kx_val * Lx)
            tph2 = np.exp(1j * (pp / qq) * kx_val * Lx / 2) * np.exp(-1j * ky_val * Ly * (pp / qq))
            tph3 = np.exp(1j * (pp / qq) * kx_val * Lx / 2) * np.exp(1j * ky_val * Ly * (pp / qq))

            V_pq = gamma * tph1 * t1Kps + tph2 * t2Kps + tph3 * t3Kps
            Htotal = H_base_Kp_s.copy()
            Htotal[mo:, mo:] += v0_eye_s + V_pq + V_pq.T.conj()
            bands_Kp[kc, :] = np.sort(linalg.eigvalsh(Htotal, overwrite_a=True, check_finite=False))

    eigs_K = np.sort(bands_K.flatten()) if bands_K is not None else np.array([])
    eigs_Kp = np.sort(bands_Kp.flatten()) if bands_Kp is not None else np.array([])

    return pp, qq, eigs_K, eigs_Kp


def do_calc_testing(filepath, order, sxflag, dagger, conj_flag, psi_conj, ppmax,
                    nlayers_arg=None, nworkers_arg=None):
    eta = 1
    vF = 1e6
    gamma = 1
    Nmax = 5000

    inp = parse_input_file(filepath)
    d = dict(inp)

    theta = d.get('theta', 0.0)
    g0 = d['g0']
    nlayers = int(nlayers_arg if nlayers_arg is not None else d.get('nlayers', 2))
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
    nk1 = int(d.get('nk1', 10))
    nk2 = int(d.get('nk2', 10))
    LL_multiplier = d.get('LL_multiplier', 6)
    Nmax = int(d.get('Nmax', Nmax))
    valley = d.get('valley', ['K', 'Kp'])
    gamma = d.get('gamma', gamma)
    vF = d.get('vF', vF)

    eps = A_HBN / A_GRAPHENE - 1
    L_moire = (1 + eps) * A_GRAPHENE / np.sqrt(eps ** 2 + 2 * (1 + eps) * (1 - np.cos(theta)))
    ktheta = 4 * np.pi / (3 ** 0.5 * L_moire)
    uc_area = 3 ** 0.5 * L_moire ** 2 / 2

    qqset, ppset = rational_fractions(ppmax)
    n_flux = len(qqset)

    print(f"  order = ({order[0]},{order[1]},{order[2]})")
    print(f"  sxflag={sxflag}  dagger={dagger}  conj_flag={conj_flag}  psi_conj={psi_conj}")
    print(f"  ppmax = {ppmax}  ->  {n_flux} flux points")
    print(f"  nlayers = {nlayers}")
    print(f"  nk1={nk1}  nk2={nk2}")
    print(f"  valley = {valley}")

    config = {
        'ktheta': ktheta, 'uc_area': uc_area, 'L_moire': L_moire,
        'nlayers': nlayers, 'valley': valley,
        'nk1': nk1, 'nk2': nk2, 'gamma': gamma,
        'g0_J': g0 / 1e3 * Q_E, 'g1_J': g1 / 1e3 * Q_E,
        'g3_J': g3 / 1e3 * Q_E, 'g4_J': g4 / 1e3 * Q_E,
        'delta_J': delta / 1e3 * Q_E,
        'v0_J': v0_meV / 1e3 * Q_E, 'v1_J': v1_meV / 1e3 * Q_E,
        'w_J': w_meV / 1e3 * Q_E,
        'vF': vF, 'LL_multiplier': LL_multiplier, 'Nmax': Nmax,
        'eta': eta, 'U': U,
        'order': order, 'sxflag': sxflag, 'dagger': dagger,
        'conj_flag': conj_flag, 'psi_conj': psi_conj,
    }

    flux_tasks = [(ppset[i], qqset[i]) for i in range(n_flux)]

    outfile = d.get('outputfile',
                    f"butterfly_N{nlayers}_{order[0]}{order[1]}{order[2]}"
                    f"_sx{sxflag}_dag{dagger}_conj{conj_flag}_psiconj{psi_conj}.mat")

    result_map = {}
    if os.path.isfile(outfile):
        prev = loadmat(outfile, squeeze_me=True)
        prev_pp = np.atleast_1d(prev['ppset'])
        prev_qq = np.atleast_1d(prev['qqset'])
        prev_eK = prev['eesetK']
        prev_eKp = prev['eesetKp']
        for j in range(len(prev_pp)):
            eK = np.atleast_1d(prev_eK[j]).flatten()
            eKp = np.atleast_1d(prev_eKp[j]).flatten()
            if len(eK) > 0 or len(eKp) > 0:
                result_map[(int(prev_pp[j]), int(prev_qq[j]))] = (eK, eKp)
        flux_tasks = [t for t in flux_tasks if t not in result_map]
        print(f"  Resumed {len(result_map)} flux points from {outfile}")

    n_todo = len(flux_tasks)
    nworkers = min(nworkers_arg if nworkers_arg else cpu_count(), max(n_todo, 1))
    print(f"  Running {n_todo} new flux points on {nworkers} workers...")

    eesetK = np.empty(n_flux, dtype=object)
    eesetKp = np.empty(n_flux, dtype=object)
    n_done = len(result_map)

    def _save_snapshot():
        flux_K_list, energy_K_list = [], []
        flux_Kp_list, energy_Kp_list = [], []
        for i in range(n_flux):
            key = (ppset[i], qqset[i])
            if key not in result_map:
                eesetK[i] = np.array([])
                eesetKp[i] = np.array([])
                continue
            eK, eKp = result_map[key]
            eesetK[i] = eK
            eesetKp[i] = eKp
            flux_val = qqset[i] / ppset[i]
            if len(eK) > 0:
                flux_K_list.append(np.full(len(eK), flux_val))
                energy_K_list.append(eK)
            if len(eKp) > 0:
                flux_Kp_list.append(np.full(len(eKp), flux_val))
                energy_Kp_list.append(eKp)

        sd = {
            'flux_K': np.concatenate(flux_K_list) if flux_K_list else np.array([]),
            'energy_K': np.concatenate(energy_K_list) if energy_K_list else np.array([]),
            'flux_Kp': np.concatenate(flux_Kp_list) if flux_Kp_list else np.array([]),
            'energy_Kp': np.concatenate(energy_Kp_list) if energy_Kp_list else np.array([]),
            'eesetK': eesetK, 'eesetKp': eesetKp,
            'ppset': np.array(ppset), 'qqset': np.array(qqset),
            'order': np.array(order),
            'sxflag': sxflag, 'dagger': dagger,
            'conj_flag': conj_flag, 'psi_conj': psi_conj,
        }
        savemat(outfile, sd)
        return sd

    if n_todo > 0:
        with Pool(processes=nworkers,
                  initializer=_init_flux_worker, initargs=(config,)) as pool:
            for pp_val, qq_val, eigs_K, eigs_Kp in pool.imap_unordered(
                    _compute_single_flux, flux_tasks):
                result_map[(pp_val, qq_val)] = (eigs_K, eigs_Kp)
                n_done += 1
                print(f"    done qq/pp = {qq_val}/{pp_val}  ({n_done}/{n_flux})")
                save_data = _save_snapshot()
    else:
        save_data = _save_snapshot()

    print(f"  Done. {n_done}/{n_flux} flux points. Saved to {outfile}")

    return save_data


def main():
    parser = argparse.ArgumentParser(
        description='Hofstadter butterfly with configurable hopping matrices')
    parser.add_argument('input_file', nargs='?', default='./input_testing.txt')
    parser.add_argument('--order', type=int, nargs=3, default=[1, 2, 3],
                        metavar=('O1', 'O2', 'O3'),
                        help='Permutation of (1,2,3) mapping tt_i -> t_{order_i}')
    parser.add_argument('--sxflag', type=int, default=0, choices=[0, 1],
                        help='Apply sigma_x conjugation to T matrices')
    parser.add_argument('--dagger', type=int, default=0, choices=[0, 1],
                        help='Conjugate-transpose base T matrices')
    parser.add_argument('--conj_flag', type=int, default=0, choices=[0, 1],
                        help='Use w=exp(-i*2pi/3) instead of exp(+i*2pi/3)')
    parser.add_argument('--psi_conj', type=int, default=0, choices=[0, 1],
                        help='Use psi=+0.29 instead of -0.29')
    parser.add_argument('--ppmax', type=int, default=10,
                        help='Maximum pp for rational fractions qq/pp')
    parser.add_argument('--nlayers', type=int, default=None, choices=[1, 2],
                        help='Number of graphene layers: 1=ML, 2=BL (overrides input file)')
    parser.add_argument('--nworkers', type=int, default=None,
                        help='Number of parallel workers (default: all cores)')

    args = parser.parse_args()

    order = args.order
    if sorted(order) != [1, 2, 3]:
        parser.error('--order must be a permutation of 1 2 3')

    do_calc_testing(args.input_file, order,
                    args.sxflag, args.dagger, args.conj_flag, args.psi_conj,
                    args.ppmax, args.nlayers, args.nworkers)


if __name__ == '__main__':
    main()
