"""
Compare Chern numbers between gaps: dH/dk only vs dH/dk - i[A,H].
"""

import os
import sys
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser import parse_input_file
from hofstadter_system import build_hofstadter_setup, HBAR_EV
from constants import A_GRAPHENE, A_HBN

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
Ax = setup['Ax_K']
Ay = setup['Ay_K']

pq = pp / qq
eta = 1e-4

# BZ area from k-mesh vectors
eps = A_HBN / A_GRAPHENE - 1
L_moire = ((1 + eps) * A_GRAPHENE
           / np.sqrt(eps**2 + 2 * (1 + eps) * (1 - np.cos(0.0))))
ktheta = 4 * np.pi / (3**0.5 * L_moire)
b1 = ktheta * np.array([0, -1])
b2 = ktheta * np.array([np.sqrt(3) / 2, 1 / 2])
vb = np.array([b1 / pp / 2, b2 / pp])
A_mesh = abs(vb[0, 0] * vb[1, 1] - vb[0, 1] * vb[1, 0])

print(f"pp={pp}, qq={qq}, Nk={Nk}, eta={eta:.0e}")

E_K = np.zeros((num_bands, Nk))
Oz_dHdk = np.zeros((num_bands, Nk))
Oz_full = np.zeros((num_bands, Nk))

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

    # Full velocity: dH/dk - i[A,H]
    comm_x = Ax @ H - H @ Ax
    comm_y = Ay @ H - H @ Ay
    Hdx_full = Hdx - 1j * comm_x
    Hdy_full = Hdy - 1j * comm_y

    ek, Psi = eigh(H)
    E_K[:, kc] = ek[target_idx]

    den = ek[target_idx, np.newaxis] - ek[np.newaxis, remote_ind]
    denom = den**2 + eta**2

    # dH/dk only
    vx = Psi.conj().T @ Hdx @ Psi
    vy = Psi.conj().T @ Hdy @ Psi
    prod = np.imag(vx[np.ix_(target_idx, remote_ind)]
                   * vy[np.ix_(remote_ind, target_idx)].T)
    Oz_dHdk[:, kc] = -2 * np.sum(prod / denom, axis=1)

    # dH/dk - i[A,H]
    vx_f = Psi.conj().T @ Hdx_full @ Psi
    vy_f = Psi.conj().T @ Hdy_full @ Psi
    prod_f = np.imag(vx_f[np.ix_(target_idx, remote_ind)]
                     * vy_f[np.ix_(remote_ind, target_idx)].T)
    Oz_full[:, kc] = -2 * np.sum(prod_f / denom, axis=1)

    if kc % 200 == 0:
        print(f"  k-point {kc}/{Nk}...")

# --- Identify gaps ---
E_meV = E_K * 1e3
E_min = np.min(E_meV, axis=1)
E_max = np.max(E_meV, axis=1)

gap_threshold = 0.5
gap_positions = []
for i in range(num_bands - 1):
    gap = E_min[i+1] - E_max[i]
    if gap > gap_threshold:
        gap_positions.append(i)

# --- Chern per band ---
chern_dHdk = A_mesh / (2 * np.pi * Nk) * np.sum(Oz_dHdk * 1e-20, axis=1)
chern_full = A_mesh / (2 * np.pi * Nk) * np.sum(Oz_full * 1e-20, axis=1)

# --- Group and report ---
boundaries = [-1] + gap_positions + [num_bands - 1]

print(f"\n{'Group':>12} | {'#bands':>6} | {'E range (meV)':>20} | "
      f"{'C(dH/dk)':>10} | {'C(full)':>10}")
print("-" * 75)
for g in range(len(boundaries) - 1):
    i0 = boundaries[g] + 1
    i1 = boundaries[g + 1]
    nb = i1 - i0 + 1
    c_dH = np.sum(chern_dHdk[i0:i1+1])
    c_full = np.sum(chern_full[i0:i1+1])
    e_lo = E_min[i0]
    e_hi = E_max[i1]
    print(f"  Bands {i0:2d}-{i1:2d} | {nb:6d} | [{e_lo:7.1f},{e_hi:7.1f}] | "
          f"{c_dH:10.4f} | {c_full:10.4f}")

print(f"\nTotal (all {num_bands} bands): "
      f"dH/dk = {np.sum(chern_dHdk):8.4f}, "
      f"full = {np.sum(chern_full):8.4f}")
