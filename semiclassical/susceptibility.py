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


# ---------------------------------------------------------------------------
# Per-k-point worker
# ---------------------------------------------------------------------------

def _chi_worker(args):
    (kpt, Q, NG, vF, gamma1, v3, Utp, Ubm,
     H_hopp_K, H_hopp_Kp, hbar, nlayers,
     remote_ind, eta, elist_meV, weight_factor) = args

    # --- K valley ---
    HK, VKx, VKy = assemble_H_V_K(
        kpt, Q, NG, vF, gamma1, v3, Utp, Ubm, H_hopp_K, hbar, nlayers)
    ekK, PsiK = eigh(HK)

    vx_K = PsiK.conj().T @ VKx @ PsiK
    vy_K = PsiK.conj().T @ VKy @ PsiK

    ekK_rem = ekK[remote_ind]
    vxeig_K = vx_K[np.ix_(remote_ind, remote_ind)]
    vyeig_K = vy_K[np.ix_(remote_ind, remote_ind)]

    chi_K = np.zeros(len(elist_meV))
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

    ekKp_rem = ekKp[remote_ind]
    vxeig_Kp = vx_Kp[np.ix_(remote_ind, remote_ind)]
    vyeig_Kp = vy_Kp[np.ix_(remote_ind, remote_ind)]

    chi_Kp = np.zeros(len(elist_meV))
    for ec in range(len(elist_meV)):
        xKp = 1.0 / (elist_meV[ec] - ekKp_rem * 1e3 + 1j * eta)
        A = vxeig_Kp * xKp[np.newaxis, :]
        B = vyeig_Kp * xKp[np.newaxis, :]
        chi_Kp[ec] = weight_factor * np.imag(np.trace(A @ B @ A @ B))

    return chi_K, chi_Kp


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------

def do_calc_chi(filepath):
    inp = parse_input_file(filepath)

    theta    = float(inp.get('theta', 0.0))
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
    psi      = float(inp['moire_psi'])
    eta      = float(inp['eta'])
    ispar    = int(inp.get('isparallel', 0))
    nprocs = inp.get('nprocs', os.environ.get('SLURM_CPUS_PER_TASK', None))
    if nprocs is not None:
        nprocs = int(nprocs)
    U        = np.atleast_1d(inp.get('U', np.array([0, 0])))
    elist_meV = np.atleast_1d(inp['elist']).ravel()

    hbar = 6.582119569e-16       # eV * s

    V0_ev = V0_meV / 1000
    V1_ev = V1_meV / 1000
    Utp   = U[0] / 1000
    Ubm   = U[1] / 1000 if len(U) > 1 else U[0] / 1000

    q1, q2, q3, vol_M, vb = compute_moire_geometry(theta)
    Q, NG = build_qvectors(NQ, q1, q2)
    numwann = 2 * NG * nlayers

    print(f"  nlayers = {nlayers}, NQ = {NQ}, NG = {NG}")
    print(f"  numwann = {numwann}")
    print(f"  nk1 = {nk1}, nk2 = {nk2}, Nk = {nk1*nk2}")

    H_hopp_K, H_hopp_Kp = construct_hopping(
        Q, NG, q1, q2, q3, V0_ev, V1_ev, psi)

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

    NE = len(elist_meV)
    weight_factor = 1.0 / (Nk_tot * vol_M)

    common = (Q, NG, vF, gamma1_ev, v3, Utp, Ubm,
              H_hopp_K, H_hopp_Kp, hbar, nlayers,
              remote_ind, eta, elist_meV, weight_factor)
    args_list = [(kpoints[kc], *common) for kc in range(Nk_tot)]

    print(f"  Calculating susceptibility over {Nk_tot} k-points...")
    if ispar:
        with Pool(nprocs) as pool:
            results = list(pool.imap_unordered(_chi_worker, args_list))
    else:
        results = [_chi_worker(a) for a in args_list]

    dChi_dE_K  = np.zeros(NE)
    dChi_dE_Kp = np.zeros(NE)

    for chi_K, chi_Kp in results:
        dChi_dE_K  += chi_K
        dChi_dE_Kp += chi_Kp

    vol_M_m2 = vol_M * 1e-20
    dChi_dE_K  = dChi_dE_K  * 1e-20 * hbar**4
    dChi_dE_Kp = dChi_dE_Kp * 1e-20 * hbar**4
    E_list = elist_meV / 1000

    print("  Done.")

    return {
        'E_list': E_list,
        'dChi_dE_K': dChi_dE_K,
        'dChi_dE_Kp': dChi_dE_Kp,
    }


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
