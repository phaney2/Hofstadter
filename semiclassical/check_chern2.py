"""
Chern number convergence test: vary eta to see if Chern numbers
converge to integers.
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
kpoints = setup['kpoints']
Nk = len(kpoints)
vol_M = setup['vol_M']

bands_sel = np.array([-3, -2, -1, 0, 1, 2, 3])
bands_idx = num_bands // 2 - 1 + bands_sel

hbar = HBAR_EV

for eta in [1e-3, 1e-4, 1e-5]:
    Oz_K = np.zeros((num_bands, Nk))

    for kc in range(Nk):
        kpt = kpoints[kc]
        H, Vx, Vy = assemble_H_V_K(kpt, setup)
        ek, Psi = eigh(H)

        vx = Psi.conj().T @ Vx @ Psi
        vy = Psi.conj().T @ Vy @ Psi

        den = ek[target_idx, np.newaxis] - ek[np.newaxis, remote_ind]
        prod = np.imag(vx[np.ix_(target_idx, remote_ind)]
                       * vy[np.ix_(remote_ind, target_idx)].T)
        denom = den**2 + eta**2
        Oz_K[:, kc] = -2 * hbar**2 * np.sum(prod / denom, axis=1)

    Oz_sel = Oz_K[bands_idx, :] * 1e-20  # Ang^2 -> m^2
    chern = 2 * np.pi / (vol_M * Nk) * np.sum(Oz_sel, axis=1)

    print(f"\neta = {eta:.0e} eV ({eta*1e3:.2f} meV)")
    print(f"  Chern (K): {np.array2string(chern, precision=4, separator=', ')}")

print("\nDone.")
