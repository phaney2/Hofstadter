"""
Fukuyama susceptibility for mono/bilayer graphene on hBN.

Standalone executable that computes dChi/dE on a full BZ k-mesh.
Reuses geometry and Hamiltonian assembly from bandstructure.py.

Usage:
    python susceptibility.py input_chi.txt
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import numpy as np
from scipy.linalg import eigh
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from parser import parse_input_file
from bandstructure import (compute_moire_geometry, build_qvectors,
                           construct_hopping,
                           assemble_H_V_K, assemble_H_V_Kp)
from hofstadter_system import (build_hofstadter_setup,
                               assemble_H_V_K as hof_assemble_K,
                               assemble_H_V_Kp as hof_assemble_Kp)


# ---------------------------------------------------------------------------
# Per-k-point worker
# ---------------------------------------------------------------------------

def _chi_worker(args):
    (kpt, Q, NG, vF, gamma1, v3, Utp, Ubm,
     H_hopp_K, H_hopp_Kp, hbar, nlayers,
     remote_ind, eta, elist_K_meV, elist_Kp_meV, weight_factor) = args

    # --- K valley ---
    HK, VKx, VKy = assemble_H_V_K(
        kpt, Q, NG, vF, gamma1, v3, Utp, Ubm, H_hopp_K, hbar, nlayers)
    ekK, PsiK = eigh(HK)

    vx_K = PsiK.conj().T @ VKx @ PsiK
    vy_K = PsiK.conj().T @ VKy @ PsiK

    ekK_rem = ekK[remote_ind]
    vxeig_K = vx_K[np.ix_(remote_ind, remote_ind)]
    vyeig_K = vy_K[np.ix_(remote_ind, remote_ind)]

    chi_K = np.zeros(len(elist_K_meV))
    for ec in range(len(elist_K_meV)):
        xK = 1.0 / (elist_K_meV[ec] - ekK_rem * 1e3 + 1j * eta)
        A = vxeig_K * xK[np.newaxis, :]
        B = vyeig_K * xK[np.newaxis, :]
        AB = A @ B
        chi_K[ec] = weight_factor * np.imag(np.sum(AB * AB.T))

    # --- K' valley ---
    HKp, VKpx, VKpy = assemble_H_V_Kp(
        kpt, Q, NG, vF, gamma1, v3, Utp, Ubm, H_hopp_Kp, hbar, nlayers)
    ekKp, PsiKp = eigh(HKp)

    vx_Kp = PsiKp.conj().T @ VKpx @ PsiKp
    vy_Kp = PsiKp.conj().T @ VKpy @ PsiKp

    ekKp_rem = ekKp[remote_ind]
    vxeig_Kp = vx_Kp[np.ix_(remote_ind, remote_ind)]
    vyeig_Kp = vy_Kp[np.ix_(remote_ind, remote_ind)]

    chi_Kp = np.zeros(len(elist_Kp_meV))
    for ec in range(len(elist_Kp_meV)):
        xKp = 1.0 / (elist_Kp_meV[ec] - ekKp_rem * 1e3 + 1j * eta)
        A = vxeig_Kp * xKp[np.newaxis, :]
        B = vyeig_Kp * xKp[np.newaxis, :]
        AB = A @ B
        chi_Kp[ec] = weight_factor * np.imag(np.sum(AB * AB.T))

    return chi_K, chi_Kp


# ---------------------------------------------------------------------------
# Hofstadter susceptibility (qq > 0)
# ---------------------------------------------------------------------------

_chi_hofstadter_shared = {}


def _init_chi_hofstadter_worker(shared):
    global _chi_hofstadter_shared
    _chi_hofstadter_shared = shared


def _chi_worker_hofstadter(args):
    kc, kpt, remote_ind, eta, elist_K_meV, elist_Kp_meV, weight_factor = args
    setup = _chi_hofstadter_shared

    HK, VKx, VKy = hof_assemble_K(kpt, setup)
    ekK, PsiK = eigh(HK)

    vx_K = PsiK.conj().T @ VKx @ PsiK
    vy_K = PsiK.conj().T @ VKy @ PsiK

    ekK_rem = ekK[remote_ind]
    vxeig_K = vx_K[np.ix_(remote_ind, remote_ind)]
    vyeig_K = vy_K[np.ix_(remote_ind, remote_ind)]

    chi_K = np.zeros(len(elist_K_meV))
    for ec in range(len(elist_K_meV)):
        xK = 1.0 / (elist_K_meV[ec] - ekK_rem * 1e3 + 1j * eta)
        A = vxeig_K * xK[np.newaxis, :]
        B = vyeig_K * xK[np.newaxis, :]
        AB = A @ B
        chi_K[ec] = weight_factor * np.imag(np.sum(AB * AB.T))

    HKp, VKpx, VKpy = hof_assemble_Kp(kpt, setup)
    ekKp, PsiKp = eigh(HKp)

    vx_Kp = PsiKp.conj().T @ VKpx @ PsiKp
    vy_Kp = PsiKp.conj().T @ VKpy @ PsiKp

    ekKp_rem = ekKp[remote_ind]
    vxeig_Kp = vx_Kp[np.ix_(remote_ind, remote_ind)]
    vyeig_Kp = vy_Kp[np.ix_(remote_ind, remote_ind)]

    chi_Kp = np.zeros(len(elist_Kp_meV))
    for ec in range(len(elist_Kp_meV)):
        xKp = 1.0 / (elist_Kp_meV[ec] - ekKp_rem * 1e3 + 1j * eta)
        A = vxeig_Kp * xKp[np.newaxis, :]
        B = vyeig_Kp * xKp[np.newaxis, :]
        AB = A @ B
        chi_Kp[ec] = weight_factor * np.imag(np.sum(AB * AB.T))

    return chi_K, chi_Kp


def _do_calc_chi_hofstadter(inp):
    from semiclassical import load_data

    setup = build_hofstadter_setup(inp)

    hbar = 6.582119569e-16       # eV * s
    eta = float(inp.get('eta', 1))
    ispar = int(inp.get('isparallel', 0))
    nprocs = inp.get('nprocs', os.environ.get('SLURM_CPUS_PER_TASK', None))
    if nprocs is not None:
        nprocs = int(nprocs)
    nE = int(inp['nE'])

    remote_ind = setup['remote_ind']
    nk1 = setup['nk1']
    nk2 = setup['nk2']
    Nk_tot = nk1 * nk2
    kpoints = setup['kpoints']
    vol_M = setup['vol_M']
    weight_factor = 1.0 / (Nk_tot * vol_M)

    bs_data = load_data(inp['inputdata'])
    nbands = bs_data['E_K'].shape[0]

    print(f"  Susceptibility: nE={nE} per band, {nbands} bands")

    result = {'nbands': nbands}

    for n in range(nbands):
        for valley in ('K', 'Kp'):
            emin = bs_data[f'E_{valley}'][n].min()
            emax = bs_data[f'E_{valley}'][n].max()
            elist_meV = np.linspace(emin, emax, nE)
            print(f"    band {n} {valley}: E = [{emin:.2f}, {emax:.2f}] meV")

            if valley == 'K':
                elist_K = elist_meV
            else:
                elist_Kp = elist_meV

        args_list = [(kc, kpoints[kc], remote_ind, eta,
                       elist_K, elist_Kp, weight_factor)
                     for kc in range(Nk_tot)]

        print(f"    band {n}: computing over {Nk_tot} k-points...")
        if ispar:
            results = []
            with Pool(nprocs, initializer=_init_chi_hofstadter_worker,
                      initargs=(setup,)) as pool:
                for i, r in enumerate(pool.imap_unordered(
                        _chi_worker_hofstadter, args_list)):
                    results.append(r)
                    if (i + 1) % max(1, Nk_tot // 20) == 0 or i + 1 == Nk_tot:
                        print(f"\r    {100*(i+1)//Nk_tot}%", end="", flush=True)
            print()
        else:
            global _chi_hofstadter_shared
            _chi_hofstadter_shared = setup
            results = []
            for i, a in enumerate(args_list):
                results.append(_chi_worker_hofstadter(a))
                if (i + 1) % max(1, Nk_tot // 20) == 0 or i + 1 == Nk_tot:
                    print(f"\r    {100*(i+1)//Nk_tot}%", end="", flush=True)
            print()

        dChi_K = np.zeros(len(elist_K))
        dChi_Kp = np.zeros(len(elist_Kp))
        for chi_K, chi_Kp in results:
            dChi_K += chi_K
            dChi_Kp += chi_Kp

        result[f'E_levels_K_band{n}'] = elist_K / 1000
        result[f'E_levels_Kp_band{n}'] = elist_Kp / 1000
        result[f'dChi_dE_K_band{n}'] = dChi_K * hbar**4
        result[f'dChi_dE_Kp_band{n}'] = dChi_Kp * hbar**4

    print("  Done.")
    return result


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------

def do_calc_chi(filepath):
    from semiclassical import load_data

    inp = parse_input_file(filepath)

    qq = int(inp.get('qq', 0))
    if qq > 0:
        return _do_calc_chi_hofstadter(inp)

    theta    = np.radians(float(inp.get('theta', 0.0)))
    nlayers  = int(inp.get('Nlayers', 2))
    nk1      = int(inp['nk1'])
    nk2      = int(inp['nk2'])
    NQ       = int(inp['NQ'])
    if 'g0' in inp:
        vF = float(inp['g0']) * 2.46 / 1000.0
    else:
        vF = float(inp['vF'])
    if 'g1' in inp:
        gamma1_ev = float(inp['g1']) / 1000.0
    else:
        gamma1_ev = float(inp['gamma1'])
    if 'g3' in inp:
        v3 = float(inp['g3']) * 2.46 / 1000.0
    else:
        v3 = float(inp.get('v3', 0))
    V0_meV   = float(inp.get('v0', inp.get('V0')))
    V1_meV   = float(inp.get('v1', inp.get('V1')))
    psi      = float(inp.get('moire_psi', 0.29))
    eta      = float(inp['eta'])
    ispar    = int(inp.get('isparallel', 0))
    nprocs = inp.get('nprocs', os.environ.get('SLURM_CPUS_PER_TASK', None))
    if nprocs is not None:
        nprocs = int(nprocs)
    U        = np.atleast_1d(inp.get('U', np.array([0, 0])))
    nE       = int(inp['nE'])

    hbar = 6.582119569e-16       # eV * s

    V0_ev = V0_meV / 1000
    V1_ev = V1_meV / 1000
    Utp   = U[0] / 1000
    Ubm   = U[1] / 1000 if len(U) > 1 else U[0] / 1000

    q1, q2, q3, vol_M, vb, G1_xy = compute_moire_geometry(theta)
    Q, NG = build_qvectors(NQ, q1, q2)
    numwann = 2 * NG * nlayers

    print(f"  nlayers = {nlayers}, NQ = {NQ}, NG = {NG}")
    print(f"  numwann = {numwann}")
    print(f"  nk1 = {nk1}, nk2 = {nk2}, Nk = {nk1*nk2}")

    H_hopp_K, H_hopp_Kp = construct_hopping(
        Q, NG, q1, q2, q3, V0_ev, V1_ev, psi, G1_xy)

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

    n_remote = numwann
    idx_c = numwann // 2
    target_idx = np.arange(idx_c - numwann//2, idx_c + numwann//2)
    remote_lo = max(target_idx[0] - n_remote, 0)
    remote_hi = min(target_idx[-1] + n_remote + 1, numwann)
    remote_ind = np.arange(remote_lo, remote_hi)

    weight_factor = 1.0 / (Nk_tot * vol_M)

    bs_data = load_data(inp['inputdata'])
    nbands = bs_data['E_K'].shape[0]

    print(f"  Susceptibility: nE={nE} per band, {nbands} bands")

    result = {'nbands': nbands}

    for n in range(nbands):
        for valley in ('K', 'Kp'):
            emin = bs_data[f'E_{valley}'][n].min()
            emax = bs_data[f'E_{valley}'][n].max()
            elist_meV = np.linspace(emin, emax, nE)
            print(f"    band {n} {valley}: E = [{emin:.2f}, {emax:.2f}] meV")

            if valley == 'K':
                elist_K = elist_meV
            else:
                elist_Kp = elist_meV

        common = (Q, NG, vF, gamma1_ev, v3, Utp, Ubm,
                  H_hopp_K, H_hopp_Kp, hbar, nlayers,
                  remote_ind, eta, elist_K, elist_Kp, weight_factor)
        args_list = [(kpoints[kc], *common) for kc in range(Nk_tot)]

        print(f"    band {n}: computing over {Nk_tot} k-points...")
        if ispar:
            results = []
            with Pool(nprocs) as pool:
                for i, r in enumerate(pool.imap_unordered(_chi_worker, args_list)):
                    results.append(r)
                    if (i + 1) % max(1, Nk_tot // 20) == 0 or i + 1 == Nk_tot:
                        print(f"\r    {100*(i+1)//Nk_tot}%", end="", flush=True)
            print()
        else:
            results = []
            for i, a in enumerate(args_list):
                results.append(_chi_worker(a))
                if (i + 1) % max(1, Nk_tot // 20) == 0 or i + 1 == Nk_tot:
                    print(f"\r    {100*(i+1)//Nk_tot}%", end="", flush=True)
            print()

        dChi_K = np.zeros(len(elist_K))
        dChi_Kp = np.zeros(len(elist_Kp))
        for chi_K, chi_Kp in results:
            dChi_K += chi_K
            dChi_Kp += chi_Kp

        result[f'E_levels_K_band{n}'] = elist_K / 1000
        result[f'E_levels_Kp_band{n}'] = elist_Kp / 1000
        result[f'dChi_dE_K_band{n}'] = dChi_K * 1e-20 * hbar**4
        result[f'dChi_dE_Kp_band{n}'] = dChi_Kp * 1e-20 * hbar**4

    print("  Done.")
    return result


# ---------------------------------------------------------------------------
# Standalone driver
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    fpath = sys.argv[1] if len(sys.argv) > 1 else 'input_chi.txt'
    result = do_calc_chi(fpath)

    inp = parse_input_file(fpath)
    outfile = inp.get('outputfile', 'susceptibility_data.mat')

    if outfile.endswith('.mat'):
        from scipy.io import savemat
        savemat(outfile, result)
    else:
        np.savez(outfile, **result)

    print(f"  Saved to {outfile}")
