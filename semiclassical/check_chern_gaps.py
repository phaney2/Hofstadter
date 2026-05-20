"""
Chern number test: compute dH/dk-only Berry curvature for ALL target bands,
identify energy gaps, and report total Chern between gaps.

Tests normalizations: C_mesh, C_mesh*pp, C_mesh*2pp.
Also tests both signs (-2 and +2) in the Kubo formula to resolve the
sign discrepancy seen in the plaquette comparison.
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

pq = pp / qq
eta = 1e-4  # eV

# BZ area from k-mesh vectors
eps = A_HBN / A_GRAPHENE - 1
theta = 0.0
L_moire = ((1 + eps) * A_GRAPHENE
           / np.sqrt(eps**2 + 2 * (1 + eps) * (1 - np.cos(theta))))
ktheta = 4 * np.pi / (3**0.5 * L_moire)
b1 = ktheta * np.array([0, -1])
b2 = ktheta * np.array([np.sqrt(3) / 2, 1 / 2])
vb = np.array([b1 / pp / 2, b2 / pp])
A_mesh = abs(vb[0, 0] * vb[1, 1] - vb[0, 1] * vb[1, 0])

print(f"pp={pp}, qq={qq}, Nk={Nk}, eta={eta:.0e} eV")
print(f"num_bands={num_bands}, target bands: {target_idx[0]}..{target_idx[-1]}")
print(f"A_mesh = {A_mesh:.6e} m^-2")

# --- Compute eigenvalues and Berry curvature for ALL target bands ---
E_K = np.zeros((num_bands, Nk))
Oz_K = np.zeros((num_bands, Nk))

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
    # Store with +2 sign (opposite of standard Kubo) since plaquette
    # comparison showed ratio = -1.  We'll print both signs below.
    Oz_K[:, kc] = -2 * np.sum(prod / denom, axis=1)

    E_K[:, kc] = ek[target_idx]

    if kc % 100 == 0:
        print(f"  k-point {kc}/{Nk}...")

# --- Band energy statistics ---
E_meV = E_K * 1e3
E_min = np.min(E_meV, axis=1)
E_max = np.max(E_meV, axis=1)
E_avg = np.mean(E_meV, axis=1)
BW = E_max - E_min

print(f"\n{'Band':>4} | {'E_min(meV)':>10} | {'E_max(meV)':>10} | {'BW(meV)':>8} | {'Gap above':>10}")
print("-" * 60)
for i in range(num_bands):
    gap = E_min[i+1] - E_max[i] if i < num_bands - 1 else float('nan')
    marker = " ***" if gap > 1.0 else ""
    print(f"  {i:2d}  | {E_min[i]:10.3f} | {E_max[i]:10.3f} | {BW[i]:8.3f} | {gap:10.3f}{marker}")

# --- Identify gaps > threshold ---
gap_threshold = 0.5  # meV
print(f"\nSignificant gaps (> {gap_threshold} meV):")
gap_positions = []
for i in range(num_bands - 1):
    gap = E_min[i+1] - E_max[i]
    if gap > gap_threshold:
        gap_positions.append(i)
        print(f"  Between bands {i} and {i+1}: gap = {gap:.3f} meV")

# --- Per-band Chern (raw, for reference) ---
Oz_m2 = Oz_K * 1e-20
chern_raw = A_mesh / (2 * np.pi * Nk) * np.sum(Oz_m2, axis=1)

print(f"\n--- Per-band Chern (sign=-2, dH/dk only) ---")
print(f"{'Band':>4} | {'C_mesh':>8} | {'C*pp':>8} | {'C*2pp':>8}")
print("-" * 45)
for i in range(num_bands):
    c = chern_raw[i]
    print(f"  {i:2d}  | {c:8.4f} | {c*pp:8.4f} | {c*2*pp:8.4f}")

# --- Group bands between gaps and compute total Chern ---
boundaries = [-1] + gap_positions + [num_bands - 1]

for sign_label, sign_factor in [("sign=-2 (standard)", 1), ("sign=+2 (flipped)", -1)]:
    chern = chern_raw * sign_factor
    print(f"\n{'='*60}")
    print(f"Total Chern between gaps — {sign_label}")
    print(f"{'='*60}")

    for norm_label, norm in [(f"C_mesh", 1), (f"C*{pp} (×pp)", pp), (f"C*{2*pp} (×2pp)", 2*pp)]:
        print(f"\n  Normalization: {norm_label}")
        for g in range(len(boundaries) - 1):
            i_start = boundaries[g] + 1
            i_end = boundaries[g + 1]
            n_in_group = i_end - i_start + 1
            c_total = np.sum(chern[i_start:i_end+1]) * norm
            e_lo = E_min[i_start]
            e_hi = E_max[i_end]
            nearest_int = round(c_total)
            err = abs(c_total - nearest_int)
            quality = "OK" if err < 0.1 else "??" if err < 0.3 else "XX"
            print(f"    Bands {i_start:2d}-{i_end:2d} ({n_in_group:2d} bands, "
                  f"E=[{e_lo:7.1f},{e_hi:7.1f}] meV): "
                  f"C = {c_total:8.4f}  [~{nearest_int:d}] {quality}")
