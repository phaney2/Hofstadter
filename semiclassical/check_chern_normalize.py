"""
Test Chern normalization: dH/dk only, varying eta, with different BZ area multipliers.
The k-mesh covers area A_mesh = |b1/(2pp) x b2/pp|.
The MBZ might be pp or 2*pp times larger.
"""

import os
import sys
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser import parse_input_file
from hofstadter_system import build_hofstadter_setup, HBAR_EV
from constants import HBAR, Q_E, A_GRAPHENE, A_HBN

input_path = os.path.join(os.path.dirname(__file__), 'input_hofstadter.txt')
inp = parse_input_file(input_path)
setup = build_hofstadter_setup(inp)

target_idx = setup['target_idx']
remote_ind = setup['remote_ind']
num_bands = setup['num_bands']
kpoints = setup['kpoints']
Nk = len(kpoints)
pp = setup['pp']
qq = setup['qq']

Lx = setup['Lx_Ang']
Ly = setup['Ly_Ang']
gamma = setup['gamma']
mo = setup['moire_offset']
M_mag = setup['M_mag_Ang']
dim = setup['dim_total']

bands_sel = np.array([-3, -2, -1, 0, 1, 2, 3])
bands_idx = num_bands // 2 - 1 + bands_sel
pq = pp / qq

# BZ area from k-mesh vectors (the actual k-mesh domain)
eps = A_HBN / A_GRAPHENE - 1
theta = 0.0
L_moire = ((1 + eps) * A_GRAPHENE
           / np.sqrt(eps**2 + 2 * (1 + eps) * (1 - np.cos(theta))))
ktheta = 4 * np.pi / (3**0.5 * L_moire)
b1 = ktheta * np.array([0, -1])
b2 = ktheta * np.array([np.sqrt(3) / 2, 1 / 2])
vb = np.array([b1 / pp / 2, b2 / pp])
A_mesh = abs(vb[0, 0] * vb[1, 1] - vb[0, 1] * vb[1, 0])  # m^-2

print(f"pp = {pp}, qq = {qq}")
print(f"A_mesh (k-mesh area) = {A_mesh:.6e} m^-2")
print(f"A_mesh * pp = {A_mesh * pp:.6e} m^-2")
print(f"A_mesh * 2*pp = {A_mesh * 2 * pp:.6e} m^-2")
print(f"Nk = {Nk}")

for eta in [1e-3, 5e-4, 1e-4]:
    Oz = np.zeros((num_bands, Nk))

    for kc in range(Nk):
        kpt = kpoints[kc]
        kpts = kpt - M_mag
        kx, ky = kpts

        tp1 = np.exp(1j * pq * kx * Lx)
        tp2 = np.exp(-1j * pq * kx * Lx / 2) * np.exp(1j * ky * Ly * pq)
        tp3 = np.exp(-1j * pq * kx * Lx / 2) * np.exp(-1j * ky * Ly * pq)

        V_pq = (gamma * tp1 * setup['term1_K'] + tp2 * setup['term2_K']
                + tp3 * setup['term3_K'])
        H = setup['H_base_K'].copy()
        H[mo:, mo:] += setup['v0_eye'] + V_pq + V_pq.T.conj()

        vx_tp1 = (1j * pq * Lx) * tp1
        vx_tp2 = (-1j * pq * Lx / 2) * tp2
        vx_tp3 = (-1j * pq * Lx / 2) * tp3
        vy_tp2 = (1j * pq * Ly) * tp2
        vy_tp3 = (-1j * pq * Ly) * tp3

        vx_tmp = (gamma * vx_tp1 * setup['term1_K'] + vx_tp2 * setup['term2_K']
                  + vx_tp3 * setup['term3_K'])
        vy_tmp = (gamma * 0 * setup['term1_K'] + vy_tp2 * setup['term2_K']
                  + vy_tp3 * setup['term3_K'])

        Hdx = np.zeros((dim, dim), dtype=complex)
        Hdy = np.zeros((dim, dim), dtype=complex)
        Hdx[mo:, mo:] = vx_tmp + vx_tmp.T.conj()
        Hdy[mo:, mo:] = vy_tmp + vy_tmp.T.conj()

        ek, Psi = eigh(H)

        vx = Psi.conj().T @ Hdx @ Psi
        vy = Psi.conj().T @ Hdy @ Psi

        den = ek[target_idx, np.newaxis] - ek[np.newaxis, remote_ind]
        prod = np.imag(vx[np.ix_(target_idx, remote_ind)]
                       * vy[np.ix_(remote_ind, target_idx)].T)
        denom = den**2 + eta**2
        Oz[:, kc] = -2 * np.sum(prod / denom, axis=1)

    # Oz is in Ang^2, convert to m^2
    Oz_m2 = Oz * 1e-20

    # Chern with different normalizations
    raw_sum = np.sum(Oz_m2[bands_idx, :], axis=1)
    chern_mesh = A_mesh / (2 * np.pi * Nk) * raw_sum
    chern_xp = chern_mesh * pp
    chern_x2p = chern_mesh * 2 * pp

    print(f"\neta = {eta:.0e} eV ({eta*1e3:.2f} meV)")
    print(f"  Band | C(mesh) | C*pp   | C*2pp  | E_avg(meV) | BW(meV)")
    print(f"  -----|---------|--------|--------|------------|--------")

    # Also compute E stats
    E_K = np.zeros((num_bands, Nk))
    for kc in range(Nk):
        kpt = kpoints[kc]
        kpts = kpt - M_mag
        kx, ky = kpts
        tp1 = np.exp(1j * pq * kx * Lx)
        tp2 = np.exp(-1j * pq * kx * Lx / 2) * np.exp(1j * ky * Ly * pq)
        tp3 = np.exp(-1j * pq * kx * Lx / 2) * np.exp(-1j * ky * Ly * pq)
        V_pq = (gamma * tp1 * setup['term1_K'] + tp2 * setup['term2_K']
                + tp3 * setup['term3_K'])
        H = setup['H_base_K'].copy()
        H[mo:, mo:] += setup['v0_eye'] + V_pq + V_pq.T.conj()
        ek = np.linalg.eigvalsh(H)
        E_K[:, kc] = ek[target_idx]

    E_sel = E_K[bands_idx, :] * 1e3
    E_avg = np.mean(E_sel, axis=1)
    E_min = np.min(E_sel, axis=1)
    E_max = np.max(E_sel, axis=1)
    BW = E_max - E_min

    for i in range(len(bands_idx)):
        print(f"    {i:2d} | {chern_mesh[i]:7.3f} | {chern_xp[i]:6.3f} | "
              f"{chern_x2p[i]:6.3f} | {E_avg[i]:10.3f} | {BW[i]:6.3f}")

    # Also print gap below each band
    print(f"\n  Gaps between selected bands (meV):")
    for i in range(1, len(bands_idx)):
        gap = E_min[i] - E_max[i-1]
        print(f"    Band {i-1}-{i}: gap = {gap:.3f} meV")
