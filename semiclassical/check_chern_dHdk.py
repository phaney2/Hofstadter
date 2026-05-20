"""
Chern number check using ONLY dH/dk (no [A,H] commutator).
Tests whether the Berry connection term should be in the Berry curvature formula.
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
hbar = HBAR_EV
eta = 1e-3  # eV

# Compute correct BZ area from vb vectors
eps = A_HBN / A_GRAPHENE - 1
theta = 0.0
L_moire = ((1 + eps) * A_GRAPHENE
           / np.sqrt(eps**2 + 2 * (1 + eps) * (1 - np.cos(theta))))
ktheta = 4 * np.pi / (3**0.5 * L_moire)
b1 = ktheta * np.array([0, -1])
b2 = ktheta * np.array([np.sqrt(3) / 2, 1 / 2])
vb = np.array([b1 / pp / 2, b2 / pp])
BZ_area = abs(vb[0, 0] * vb[1, 1] - vb[0, 1] * vb[1, 0])  # m^-2

Oz_dHdk = np.zeros((num_bands, Nk))
Oz_full = np.zeros((num_bands, Nk))

bands_sel = np.array([-3, -2, -1, 0, 1, 2, 3])
bands_idx = num_bands // 2 - 1 + bands_sel

pq = pp / qq

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

    # Phase derivatives
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

    # Berry connection commutator
    Ax = setup['Ax_K']
    Ay = setup['Ay_K']
    comm_x = Ax @ H - H @ Ax
    comm_y = Ay @ H - H @ Ay

    ek, Psi = eigh(H)

    # dH/dk only velocity (still divide by hbar for the Kubo formula cancellation)
    vx_dHdk = Psi.conj().T @ Hdx @ Psi
    vy_dHdk = Psi.conj().T @ Hdy @ Psi

    # Full velocity
    Vx_full = Hdx - 1j * comm_x
    Vy_full = Hdy - 1j * comm_y
    vx_full = Psi.conj().T @ Vx_full @ Psi
    vy_full = Psi.conj().T @ Vy_full @ Psi

    den = ek[target_idx, np.newaxis] - ek[np.newaxis, remote_ind]
    denom = den**2 + eta**2

    # BC from dH/dk only (note: hbar factors cancel)
    prod_dH = np.imag(vx_dHdk[np.ix_(target_idx, remote_ind)]
                      * vy_dHdk[np.ix_(remote_ind, target_idx)].T)
    Oz_dHdk[:, kc] = -2 * np.sum(prod_dH / denom, axis=1)

    # BC from full velocity (hbar factors cancel)
    prod_full = np.imag(vx_full[np.ix_(target_idx, remote_ind)]
                        * vy_full[np.ix_(remote_ind, target_idx)].T)
    Oz_full[:, kc] = -2 * np.sum(prod_full / denom, axis=1)

# Convert to m^2 and compute Chern
Oz_dHdk_m2 = Oz_dHdk * 1e-20  # Ang^2 -> m^2
Oz_full_m2 = Oz_full * 1e-20

chern_dHdk = BZ_area / (2 * np.pi * Nk) * np.sum(Oz_dHdk_m2[bands_idx, :], axis=1)
chern_full = BZ_area / (2 * np.pi * Nk) * np.sum(Oz_full_m2[bands_idx, :], axis=1)

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

E_sel = E_K[bands_idx, :] * 1e3  # eV -> meV
E_avg = np.mean(E_sel, axis=1)
E_min = np.min(E_sel, axis=1)
E_max = np.max(E_sel, axis=1)

print("\nBand | E_avg (meV) |  BW (meV)  | Chern(dH/dk) | Chern(full)")
print("-" * 72)
for i in range(len(chern_dHdk)):
    bw = E_max[i] - E_min[i]
    print(f"  {i:2d}  | {E_avg[i]:10.3f}  | {bw:9.3f}  | "
          f"{chern_dHdk[i]:12.4f} | {chern_full[i]:12.4f}")
