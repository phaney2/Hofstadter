"""
Debug script: reproduce MATLAB BC calculation exactly, then compare with Python's
consistent-unit approach.

The MATLAB code has a unit inconsistency:
  Htotal = (1000/q) * H_J          -> meV
  Hdx    = (1e9/q)  * dH_J/dk_m    -> eV*nm (= 1000 meV * nm)
  Ax_calc = 1e9 * Ax_m             -> nm (as intended)

So vx = Hdx - i*[Ax_calc, Htotal] mixes eV*nm and meV*nm.

This script:
1. Builds setup from Python hofstadter_system.py (eV/Ang units)
2. Converts back to MATLAB's mixed units at one k-point
3. Computes BC using MATLAB's formula
4. Computes BC using Python's formula (consistent eV/Ang)
5. Compares both to the benchmark values
"""

import os
import sys
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser import parse_input_file
from hofstadter_system import build_hofstadter_setup, assemble_H_V_K, HBAR_EV
from constants import Q_E

from scipy.io import loadmat

bench_path = (r"C:\Users\phaney\OneDrive - NIST\Documents\MATLAB"
              r"\Duartes_code\Hofstadter_codes\benchmark_BC_calc.mat")
input_path = os.path.join(os.path.dirname(__file__), 'input_hofstadter.txt')

inp = parse_input_file(input_path)
setup = build_hofstadter_setup(inp)

bench = loadmat(bench_path)

# Pick a single k-point (kc=0)
kc = 0
kpt = setup['kpoints'][kc]
target_idx = setup['target_idx']
remote_ind = setup['remote_ind']

# ============================================
# Method 1: Python's current approach (eV/Ang)
# ============================================
H_eV, Vx_As, Vy_As = assemble_H_V_K(kpt, setup)
ek_eV, Psi = eigh(H_eV)

# Velocity matrix in eigenbasis
vx_eig = Psi.conj().T @ Vx_As @ Psi
vy_eig = Psi.conj().T @ Vy_As @ Psi

hbar = HBAR_EV  # eV*s
eta_eV = 1e-3   # 1 meV in eV (matching MATLAB's eta=1)

den = ek_eV[target_idx, np.newaxis] - ek_eV[np.newaxis, remote_ind]
prod = np.imag(vx_eig[np.ix_(target_idx, remote_ind)]
               * vy_eig[np.ix_(remote_ind, target_idx)].T)
denom = den**2 + eta_eV**2
Oz_py = -2 * hbar**2 * np.sum(prod / denom, axis=1)

print("=" * 60)
print("Method 1: Python (eV/Ang units, eta=1e-3 eV)")
print("=" * 60)
print(f"  E range (target): {ek_eV[target_idx[0]]:.6f} to {ek_eV[target_idx[-1]]:.6f} eV")
print(f"  Max |Vx|: {np.max(np.abs(Vx_As)):.3e} Ang/s")

# Select same bands as benchmark (middle 7 bands of 26-band window)
num_bands = setup['num_bands']
bands_sel = np.array([-3, -2, -1, 0, 1, 2, 3])
bands_idx = num_bands // 2 - 1 + bands_sel

print(f"  BC (Ang^2): {Oz_py[bands_idx]}")

# ============================================
# Method 2: Reproduce MATLAB exactly
# ============================================
# Reconstruct MATLAB-convention matrices from Python's eV/Ang ones:
#   H_meV = 1000 * H_eV
#   Hdx_ml = 0.1 * Hdx_eVA  (from (1e9/q)*J*m = 0.1*(eV*Ang))
#   Ax_ml = 0.1 * Ax_Ang     (from 1e9*m = 0.1*Ang)

# Build H in eV, compute Hdx and [A,H] separately
pp = setup['pp']
qq = setup['qq']
Lx = setup['Lx_Ang']
Ly = setup['Ly_Ang']
gamma = setup['gamma']
mo = setup['moire_offset']
M_mag = setup['M_mag_Ang']
dim = setup['dim_total']

kpts = kpt - M_mag
kx, ky = kpts
pp_over_qq = pp / qq

tphase1 = np.exp(1j * pp_over_qq * kx * Lx)
tphase2 = (np.exp(-1j * pp_over_qq * kx * Lx / 2)
           * np.exp(1j * ky * Ly * pp_over_qq))
tphase3 = (np.exp(-1j * pp_over_qq * kx * Lx / 2)
           * np.exp(-1j * ky * Ly * pp_over_qq))

V_pq = (gamma * tphase1 * setup['term1_K']
        + tphase2 * setup['term2_K']
        + tphase3 * setup['term3_K'])

H = setup['H_base_K'].copy()
H[mo:, mo:] += setup['v0_eye'] + V_pq + V_pq.T.conj()
# H is in eV

# Phase derivatives
vx_tp1 = (1j * pp_over_qq * Lx) * tphase1
vx_tp2 = (-1j * pp_over_qq * Lx / 2) * tphase2
vx_tp3 = (-1j * pp_over_qq * Lx / 2) * tphase3

vy_tp1 = 0.0
vy_tp2 = (1j * pp_over_qq * Ly) * tphase2
vy_tp3 = (-1j * pp_over_qq * Ly) * tphase3

vx_tmp = (gamma * vx_tp1 * setup['term1_K']
          + vx_tp2 * setup['term2_K']
          + vx_tp3 * setup['term3_K'])
vy_tmp = (gamma * vy_tp1 * setup['term1_K']
          + vy_tp2 * setup['term2_K']
          + vy_tp3 * setup['term3_K'])

Hdx_eVA = np.zeros((dim, dim), dtype=complex)
Hdy_eVA = np.zeros((dim, dim), dtype=complex)
Hdx_eVA[mo:, mo:] = vx_tmp + vx_tmp.T.conj()
Hdy_eVA[mo:, mo:] = vy_tmp + vy_tmp.T.conj()
# Hdx_eVA, Hdy_eVA in eV*Ang

Ax = setup['Ax_K']  # Ang
Ay = setup['Ay_K']  # Ang

# --- MATLAB convention ---
H_meV = 1000 * H  # meV
Hdx_ml = 0.1 * Hdx_eVA  # MATLAB's (1e9/q)*J*m = 0.1*eV*Ang
Hdy_ml = 0.1 * Hdy_eVA
Ax_ml = 0.1 * Ax  # MATLAB's 1e9*m = 0.1*Ang
Ay_ml = 0.1 * Ay

vx_ml = Hdx_ml - 1j * (Ax_ml @ H_meV - H_meV @ Ax_ml)
vy_ml = Hdy_ml - 1j * (Ay_ml @ H_meV - H_meV @ Ay_ml)

ek_ml, Psi_ml = eigh(H_meV)
# MATLAB uses eig (not sorted the same way), but eigh gives sorted eigenvalues
# Let's verify eigenvalues match
ek_matlab = bench['ekset']  # (Nk, 26) in meV
print(f"\n  Eigenvalue check: Python meV = {ek_ml[target_idx[0]]:.6f}, "
      f"MATLAB = {ek_matlab[0, 0]:.6f}")

# Velocity in eigenbasis
vx_ml_eig = Psi_ml.conj().T @ vx_ml @ Psi_ml
vy_ml_eig = Psi_ml.conj().T @ vy_ml @ Psi_ml

eta_ml = 1.0  # meV (MATLAB eta)
den_ml = ek_ml[target_idx, np.newaxis] - ek_ml[np.newaxis, remote_ind]
prod_ml = np.imag(vx_ml_eig[np.ix_(target_idx, remote_ind)]
                  * vy_ml_eig[np.ix_(remote_ind, target_idx)].T)
denom_ml = den_ml**2 + eta_ml**2
BC_ml = -2 * np.sum(prod_ml / denom_ml, axis=1)

print("\n" + "=" * 60)
print("Method 2: MATLAB-convention units")
print("=" * 60)
print(f"  BC (MATLAB units): {BC_ml[bands_idx]}")
print(f"  MATLAB benchmark:  {bench['berrycurv_K'][0, :7]}")

# --- Compare intermediate quantities ---
print("\n" + "=" * 60)
print("Intermediate comparison")
print("=" * 60)

# Norms of the two velocity terms
comm_x = Ax_ml @ H_meV - H_meV @ Ax_ml
comm_y = Ay_ml @ H_meV - H_meV @ Ay_ml
print(f"  |Hdx_ml| Frobenius: {np.linalg.norm(Hdx_ml):.6e}")
print(f"  |[Ax_ml, H_meV]| Frobenius: {np.linalg.norm(comm_x):.6e}")
print(f"  Ratio [A,H]/Hdx: {np.linalg.norm(comm_x)/np.linalg.norm(Hdx_ml):.2f}")

# Same in Python eV/Ang units
comm_x_py = Ax @ H - H @ Ax
comm_y_py = Ay @ H - H @ Ay
print(f"\n  |Hdx_eVA| Frobenius: {np.linalg.norm(Hdx_eVA):.6e}")
print(f"  |[Ax, H_eV]| Frobenius: {np.linalg.norm(comm_x_py):.6e}")
print(f"  Ratio [A,H]/Hdx: {np.linalg.norm(comm_x_py)/np.linalg.norm(Hdx_eVA):.2f}")

# ============================================
# Method 3: Fix MATLAB's scaling to be consistent meV/nm
# ============================================
# If MATLAB intended consistent meV/nm units:
#   H in meV, A in nm, Hdx in meV*nm
# Then: Hdx_consistent = (1e12/q)*J*m = 1000*(1e9/q)*J*m = 1000*Hdx_ml
#   OR equivalently: 100 * Hdx_eVA
Hdx_fix = 100.0 * Hdx_eVA  # = 1000 * Hdx_ml
Hdy_fix = 100.0 * Hdy_eVA

vx_fix = Hdx_fix - 1j * (Ax_ml @ H_meV - H_meV @ Ax_ml)
vy_fix = Hdy_fix - 1j * (Ay_ml @ H_meV - H_meV @ Ay_ml)

vx_fix_eig = Psi_ml.conj().T @ vx_fix @ Psi_ml
vy_fix_eig = Psi_ml.conj().T @ vy_fix @ Psi_ml

prod_fix = np.imag(vx_fix_eig[np.ix_(target_idx, remote_ind)]
                   * vy_fix_eig[np.ix_(remote_ind, target_idx)].T)
BC_fix = -2 * np.sum(prod_fix / denom_ml, axis=1)

print("\n" + "=" * 60)
print("Method 3: Fixed MATLAB units (consistent meV/nm)")
print("=" * 60)
print(f"  BC (fixed MATLAB): {BC_fix[bands_idx]}")

# ============================================
# Method 4: Python consistent units, no hbar division
# ============================================
# BC = -2 * sum(Im[(Hdx - i[A,H]) * (Hdy - i[A,H])] / denom)
# where everything in eV*Ang, denom in eV^2
stuff_x = Hdx_eVA - 1j * comm_x_py
stuff_y = Hdy_eVA - 1j * comm_y_py

stuff_x_eig = Psi.conj().T @ stuff_x @ Psi
stuff_y_eig = Psi.conj().T @ stuff_y @ Psi

prod4 = np.imag(stuff_x_eig[np.ix_(target_idx, remote_ind)]
                * stuff_y_eig[np.ix_(remote_ind, target_idx)].T)
denom4 = den**2 + eta_eV**2
BC4 = -2 * np.sum(prod4 / denom4, axis=1)

print("\n" + "=" * 60)
print("Method 4: Python consistent (eV*Ang), no hbar (should = Method 1)")
print("=" * 60)
print(f"  BC (Ang^2): {BC4[bands_idx]}")

# Relationship between methods
print("\n" + "=" * 60)
print("Ratios")
print("=" * 60)
if np.any(BC_ml[bands_idx] != 0):
    print(f"  BC_py / BC_matlab: {Oz_py[bands_idx] / BC_ml[bands_idx]}")
    print(f"  BC_fix / BC_matlab: {BC_fix[bands_idx] / BC_ml[bands_idx]}")
    print(f"  BC4 / BC_matlab: {BC4[bands_idx] / BC_ml[bands_idx]}")
