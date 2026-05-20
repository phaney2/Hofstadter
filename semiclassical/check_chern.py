"""
Quick sanity check: compute integrated Berry curvature (Chern number)
for the Hofstadter bands. Should be integers for correct physics.

Chern = (1/(2*pi)) * integral(Oz * dk)  over the magnetic BZ
     = (1/(2*pi)) * sum(Oz_k) * (area_BZ / Nk)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser import parse_input_file
from semiclassical import _do_calc_hofstadter

input_path = os.path.join(os.path.dirname(__file__), 'input_hofstadter.txt')
inp = parse_input_file(input_path)

result = _do_calc_hofstadter(inp)

# result['Oz_K'] is (nbands_sel, Nk) in m^2
# result['vol_M'] is the magnetic unit cell area in m^2
Oz_K = result['Oz_K']       # m^2
vol_M = result['vol_M']     # m^2
Nk = Oz_K.shape[1]

# BZ area = (2*pi)^2 / vol_M (in m^-2)
# Chern = (1/(2*pi)) * sum(Oz_k) * BZ_area / Nk
#       = (1/(2*pi)) * sum(Oz_k) * (2*pi)^2 / (vol_M * Nk)
#       = 2*pi / (vol_M * Nk) * sum(Oz_k)

chern_K = 2 * np.pi / (vol_M * Nk) * np.sum(Oz_K, axis=1)

E_K = result['E_K']  # meV
E_avg = np.mean(E_K, axis=1)

print("\nBand | E_avg (meV) | Chern (K valley)")
print("-" * 45)
for i in range(len(chern_K)):
    print(f"  {i:2d}  | {E_avg[i]:10.3f}  | {chern_K[i]:8.4f}")

# Also check K' valley
Oz_Kp = result['Oz_Kp']
chern_Kp = 2 * np.pi / (vol_M * Nk) * np.sum(Oz_Kp, axis=1)
E_Kp = result['E_Kp']
E_avg_Kp = np.mean(E_Kp, axis=1)

print("\nBand | E_avg (meV) | Chern (K' valley)")
print("-" * 45)
for i in range(len(chern_Kp)):
    print(f"  {i:2d}  | {E_avg_Kp[i]:10.3f}  | {chern_Kp[i]:8.4f}")
