"""
Compute Berry curvature via the gauge-invariant plaquette method
and compare with Kubo formula (with and without [A,H]).

The plaquette method is the ground truth: it uses only wavefunction
overlaps and doesn't depend on the velocity operator formula.
"""

import os
import sys
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser import parse_input_file
from hofstadter_system import build_hofstadter_setup, assemble_H_V_K, HBAR_EV

input_path = os.path.join(os.path.dirname(__file__), 'input_hofstadter.txt')
inp = parse_input_file(input_path)
setup = build_hofstadter_setup(inp)

target_idx = setup['target_idx']
remote_ind = setup['remote_ind']
num_bands = setup['num_bands']
bands_sel = np.array([-3, -2, -1, 0, 1, 2, 3])
bands_idx = num_bands // 2 - 1 + bands_sel
hbar = HBAR_EV

kpt = setup['kpoints'][450]  # middle of BZ for less singular point


def build_H_at_k(kpt_local, setup):
    pp = setup['pp']
    qq = setup['qq']
    Lx = setup['Lx_Ang']
    Ly = setup['Ly_Ang']
    gamma = setup['gamma']
    mo = setup['moire_offset']
    M_mag = setup['M_mag_Ang']
    pq = pp / qq

    kpts = kpt_local - M_mag
    kx, ky = kpts
    tp1 = np.exp(1j * pq * kx * Lx)
    tp2 = np.exp(-1j * pq * kx * Lx / 2) * np.exp(1j * ky * Ly * pq)
    tp3 = np.exp(-1j * pq * kx * Lx / 2) * np.exp(-1j * ky * Ly * pq)
    V = (gamma * tp1 * setup['term1_K'] + tp2 * setup['term2_K']
         + tp3 * setup['term3_K'])
    H = setup['H_base_K'].copy()
    H[mo:, mo:] += setup['v0_eye'] + V + V.T.conj()
    return H


# ======================================================
# Plaquette Berry curvature (gauge-invariant)
# ======================================================
dk = 1e-6  # Ang^-1

# Four corners of plaquette
k00 = kpt
k10 = kpt + np.array([dk, 0])
k01 = kpt + np.array([0, dk])
k11 = kpt + np.array([dk, dk])

H00 = build_H_at_k(k00, setup)
H10 = build_H_at_k(k10, setup)
H01 = build_H_at_k(k01, setup)
H11 = build_H_at_k(k11, setup)

_, Psi00 = eigh(H00)
_, Psi10 = eigh(H10)
_, Psi01 = eigh(H01)
_, Psi11 = eigh(H11)

# For each target band, compute the link variable overlap
Oz_plaquette = np.zeros(num_bands)

for i in range(num_bands):
    n = target_idx[i]
    # Link overlaps (single band)
    U1 = np.dot(Psi00[:, n].conj(), Psi10[:, n])
    U2 = np.dot(Psi10[:, n].conj(), Psi11[:, n])
    U3 = np.dot(Psi11[:, n].conj(), Psi01[:, n])
    U4 = np.dot(Psi01[:, n].conj(), Psi00[:, n])
    F = np.imag(np.log(U1 * U2 * U3 * U4))
    Oz_plaquette[i] = F / dk**2

# ======================================================
# Kubo formula: dH/dk only
# ======================================================
H = H00
ek, Psi = eigh(H)
eta = 1e-5  # small eta for comparison

# Build Hdx, Hdy analytically
pp = setup['pp']
qq = setup['qq']
Lx = setup['Lx_Ang']
Ly = setup['Ly_Ang']
gamma = setup['gamma']
mo = setup['moire_offset']
M_mag = setup['M_mag_Ang']
dim = setup['dim_total']
pq = pp / qq

kpts = kpt - M_mag
kx, ky = kpts
tp1 = np.exp(1j * pq * kx * Lx)
tp2 = np.exp(-1j * pq * kx * Lx / 2) * np.exp(1j * ky * Ly * pq)
tp3 = np.exp(-1j * pq * kx * Lx / 2) * np.exp(-1j * ky * Ly * pq)

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

Ax = setup['Ax_K']
Ay = setup['Ay_K']
comm_x = Ax @ H - H @ Ax
comm_y = Ay @ H - H @ Ay


def kubo_bc(Mx, My, Psi, ek, target_idx, remote_ind, eta):
    mx = Psi.conj().T @ Mx @ Psi
    my = Psi.conj().T @ My @ Psi
    den = ek[target_idx, np.newaxis] - ek[np.newaxis, remote_ind]
    prod = np.imag(mx[np.ix_(target_idx, remote_ind)]
                   * my[np.ix_(remote_ind, target_idx)].T)
    denom = den**2 + eta**2
    return -2 * np.sum(prod / denom, axis=1)


# Method A: dH/dk only (no commutator, no hbar)
bc_dHdk = kubo_bc(Hdx, Hdy, Psi, ek, target_idx, remote_ind, eta)

# Method B: dH/dk - i[A,H] (Python current, no hbar)
bc_minus = kubo_bc(Hdx - 1j * comm_x, Hdy - 1j * comm_y,
                   Psi, ek, target_idx, remote_ind, eta)

# Method C: dH/dk + i[A,H] (opposite sign)
bc_plus = kubo_bc(Hdx + 1j * comm_x, Hdy + 1j * comm_y,
                  Psi, ek, target_idx, remote_ind, eta)

# Method D: i[A,H] only (MATLAB-like, after scaling)
bc_comm = kubo_bc(-1j * comm_x, -1j * comm_y,
                  Psi, ek, target_idx, remote_ind, eta)

print("=" * 72)
print(f"Berry curvature at k-point 450 (eta={eta:.0e} eV)")
print("=" * 72)
print(f"\n{'Band':>4} | {'Plaquette':>12} | {'dH/dk':>12} | "
      f"{'dH/dk-i[A,H]':>12} | {'dH/dk+i[A,H]':>12} | {'[A,H] only':>12}")
print("-" * 80)
for j, bi in enumerate(bands_idx):
    print(f"  {j:2d}  | {Oz_plaquette[bi]:12.2f} | {bc_dHdk[bi]:12.2f} | "
          f"{bc_minus[bi]:12.2f} | {bc_plus[bi]:12.2f} | {bc_comm[bi]:12.2f}")

# Check which method matches plaquette
print("\nRatios to plaquette:")
print(f"  dH/dk:        {bc_dHdk[bands_idx] / Oz_plaquette[bands_idx]}")
print(f"  dH/dk-i[A,H]: {bc_minus[bands_idx] / Oz_plaquette[bands_idx]}")
print(f"  dH/dk+i[A,H]: {bc_plus[bands_idx] / Oz_plaquette[bands_idx]}")
print(f"  [A,H] only:   {bc_comm[bands_idx] / Oz_plaquette[bands_idx]}")
