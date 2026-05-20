"""
Debug velocity operator: verify dH/dk by finite difference,
compare commutator magnitudes, and test Chern with correct BZ area.
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

kpt = setup['kpoints'][0]
pp = setup['pp']
qq = setup['qq']
Lx = setup['Lx_Ang']
Ly = setup['Ly_Ang']
mo = setup['moire_offset']
M_mag = setup['M_mag_Ang']
dim = setup['dim_total']

# =========================================
# Test 1: dH/dk by finite difference
# =========================================
print("=" * 60)
print("Test 1: Verify dH/dk by finite difference")
print("=" * 60)

def build_H_only(kpt_local, setup):
    """Build just H (no velocity) at a k-point."""
    kpts = kpt_local - setup['M_mag_Ang']
    kx, ky = kpts
    pq = setup['pp'] / setup['qq']
    Lx = setup['Lx_Ang']
    Ly = setup['Ly_Ang']
    gamma = setup['gamma']
    mo = setup['moire_offset']

    tp1 = np.exp(1j * pq * kx * Lx)
    tp2 = np.exp(-1j * pq * kx * Lx / 2) * np.exp(1j * ky * Ly * pq)
    tp3 = np.exp(-1j * pq * kx * Lx / 2) * np.exp(-1j * ky * Ly * pq)

    V = (gamma * tp1 * setup['term1_K'] + tp2 * setup['term2_K']
         + tp3 * setup['term3_K'])
    H = setup['H_base_K'].copy()
    H[mo:, mo:] += setup['v0_eye'] + V + V.T.conj()
    return H

dk = 1e-8  # Ang^-1
H0 = build_H_only(kpt, setup)

# Finite difference dH/dkx
Hpx = build_H_only(kpt + np.array([dk, 0]), setup)
Hmx = build_H_only(kpt - np.array([dk, 0]), setup)
Hdx_fd = (Hpx - Hmx) / (2 * dk)

# Finite difference dH/dky
Hpy = build_H_only(kpt + np.array([0, dk]), setup)
Hmy = build_H_only(kpt - np.array([0, dk]), setup)
Hdy_fd = (Hpy - Hmy) / (2 * dk)

# Analytic dH/dk (extract from assemble_H_V_K internals)
kpts = kpt - M_mag
kx, ky = kpts
pq = pp / qq

tp1 = np.exp(1j * pq * kx * Lx)
tp2 = np.exp(-1j * pq * kx * Lx / 2) * np.exp(1j * ky * Ly * pq)
tp3 = np.exp(-1j * pq * kx * Lx / 2) * np.exp(-1j * ky * Ly * pq)

gamma = setup['gamma']
vx_tp1 = (1j * pq * Lx) * tp1
vx_tp2 = (-1j * pq * Lx / 2) * tp2
vx_tp3 = (-1j * pq * Lx / 2) * tp3

vy_tp1 = 0.0
vy_tp2 = (1j * pq * Ly) * tp2
vy_tp3 = (-1j * pq * Ly) * tp3

vx_tmp = (gamma * vx_tp1 * setup['term1_K'] + vx_tp2 * setup['term2_K']
          + vx_tp3 * setup['term3_K'])
vy_tmp = (gamma * vy_tp1 * setup['term1_K'] + vy_tp2 * setup['term2_K']
          + vy_tp3 * setup['term3_K'])

Hdx_an = np.zeros((dim, dim), dtype=complex)
Hdy_an = np.zeros((dim, dim), dtype=complex)
Hdx_an[mo:, mo:] = vx_tmp + vx_tmp.T.conj()
Hdy_an[mo:, mo:] = vy_tmp + vy_tmp.T.conj()

err_x = np.max(np.abs(Hdx_an - Hdx_fd))
err_y = np.max(np.abs(Hdy_an - Hdy_fd))
norm_x = np.max(np.abs(Hdx_an))
norm_y = np.max(np.abs(Hdy_an))

print(f"  |Hdx_analytic|_max = {norm_x:.6e}")
print(f"  |Hdx_fd - Hdx_an|_max = {err_x:.6e}")
print(f"  Relative error x: {err_x/norm_x:.3e}")
print(f"  |Hdy_analytic|_max = {norm_y:.6e}")
print(f"  |Hdy_fd - Hdy_an|_max = {err_y:.6e}")
print(f"  Relative error y: {err_y/norm_y:.3e}")

# =========================================
# Test 2: Commutator contribution
# =========================================
print("\n" + "=" * 60)
print("Test 2: Relative magnitude of dH/dk vs [A,H]")
print("=" * 60)

Ax = setup['Ax_K']
Ay = setup['Ay_K']
H = H0

comm_x = Ax @ H - H @ Ax
comm_y = Ay @ H - H @ Ay

print(f"  |dH/dkx| Frobenius: {np.linalg.norm(Hdx_an):.6e}")
print(f"  |[Ax, H]| Frobenius: {np.linalg.norm(comm_x):.6e}")
print(f"  Ratio: {np.linalg.norm(comm_x)/np.linalg.norm(Hdx_an):.4f}")
print(f"  |dH/dky| Frobenius: {np.linalg.norm(Hdy_an):.6e}")
print(f"  |[Ay, H]| Frobenius: {np.linalg.norm(comm_y):.6e}")
print(f"  Ratio: {np.linalg.norm(comm_y)/np.linalg.norm(Hdy_an):.4f}")

# =========================================
# Test 3: BC with only dH/dk (no commutator)
# =========================================
print("\n" + "=" * 60)
print("Test 3: BC with only dH/dk (no [A,H] commutator)")
print("=" * 60)

target_idx = setup['target_idx']
remote_ind = setup['remote_ind']
num_bands = setup['num_bands']
eta = 1e-3

# Full velocity
Vx_full = (Hdx_an - 1j * comm_x) / HBAR_EV
Vy_full = (Hdy_an - 1j * comm_y) / HBAR_EV

# dH/dk only
Vx_dHdk = Hdx_an / HBAR_EV
Vy_dHdk = Hdy_an / HBAR_EV

# [A,H] only
Vx_comm = (-1j * comm_x) / HBAR_EV
Vy_comm = (-1j * comm_y) / HBAR_EV

ek, Psi = eigh(H)

def compute_bc(Vx, Vy, Psi, ek, target_idx, remote_ind, eta, hbar):
    vx = Psi.conj().T @ Vx @ Psi
    vy = Psi.conj().T @ Vy @ Psi
    den = ek[target_idx, np.newaxis] - ek[np.newaxis, remote_ind]
    prod = np.imag(vx[np.ix_(target_idx, remote_ind)]
                   * vy[np.ix_(remote_ind, target_idx)].T)
    denom = den**2 + eta**2
    return -2 * hbar**2 * np.sum(prod / denom, axis=1)

bands_sel = np.array([-3, -2, -1, 0, 1, 2, 3])
bands_idx = num_bands // 2 - 1 + bands_sel

bc_full = compute_bc(Vx_full, Vy_full, Psi, ek, target_idx, remote_ind, eta, HBAR_EV)
bc_dHdk = compute_bc(Vx_dHdk, Vy_dHdk, Psi, ek, target_idx, remote_ind, eta, HBAR_EV)
bc_comm = compute_bc(Vx_comm, Vy_comm, Psi, ek, target_idx, remote_ind, eta, HBAR_EV)

print(f"  BC(full)  = {bc_full[bands_idx]}")
print(f"  BC(dH/dk) = {bc_dHdk[bands_idx]}")
print(f"  BC([A,H]) = {bc_comm[bands_idx]}")

# =========================================
# Test 4: Correct BZ area for Chern
# =========================================
print("\n" + "=" * 60)
print("Test 4: BZ area from k-mesh vectors")
print("=" * 60)

from constants import HBAR, Q_E, A_GRAPHENE, A_HBN
eps = A_HBN / A_GRAPHENE - 1
theta = 0.0
L_moire = ((1 + eps) * A_GRAPHENE
           / np.sqrt(eps**2 + 2 * (1 + eps) * (1 - np.cos(theta))))
ktheta = 4 * np.pi / (3**0.5 * L_moire)

b1 = ktheta * np.array([0, -1])
b2 = ktheta * np.array([np.sqrt(3) / 2, 1 / 2])
vb = np.array([b1 / pp / 2, b2 / pp])  # in m^-1

BZ_vb = abs(vb[0, 0] * vb[1, 1] - vb[0, 1] * vb[1, 0])
vol_M = setup['vol_M']
BZ_volM = (2 * np.pi)**2 / vol_M

print(f"  BZ from vb vectors: {BZ_vb:.6e} m^-2")
print(f"  BZ from (2pi)^2/vol_M: {BZ_volM:.6e} m^-2")
print(f"  Ratio (2pi)^2/vol_M / BZ_vb: {BZ_volM/BZ_vb:.4f}")
print(f"  Expected ratio: 2*pp = {2*pp}")
