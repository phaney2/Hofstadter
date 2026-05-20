"""
Validate Python semiclassical engine against MATLAB benchmark data.

Usage: python validate.py [benchmark.mat] [input.txt]
"""

import sys
import numpy as np
from scipy.io import loadmat
from semiclassical import do_calc


def validate(mat_path, input_path):
    print("Loading MATLAB benchmark...")
    ref = loadmat(mat_path)

    print("Running Python calculation...")
    res = do_calc(input_path)

    checks = [
        ('vol_M',      np.atleast_1d(res['vol_M']),      ref['vol_M'].ravel()),
        ('kpoints',    res['kpoints'],                    ref['kpoints']),
        ('E_K',        res['E_K'],                        ref['E_K']),
        ('E_Kp',       res['E_Kp'],                       ref['E_Kp']),
        ('Oz_K',       res['Oz_K'],                       ref['Oz_K']),
        ('Oz_Kp',      res['Oz_Kp'],                      ref['Oz_Kp']),
        ('Lz_K',       res['Lz_K'],                       ref['Lz_K']),
        ('Lz_Kp',      res['Lz_Kp'],                      ref['Lz_Kp']),
        ('dChi_dE_K',  res['dChi_dE_K'].ravel(),          ref['dChi_dE_K'].ravel()),
        ('dChi_dE_Kp', res['dChi_dE_Kp'].ravel(),         ref['dChi_dE_Kp'].ravel()),
        ('E_list',     res['E_list'].ravel(),              ref['E_list'].ravel()),
    ]

    all_pass = True
    for name, py, mat in checks:
        if py.shape != mat.shape:
            print(f"  {name:15s}  SHAPE MISMATCH  py={py.shape} mat={mat.shape}")
            all_pass = False
            continue

        abs_err = np.max(np.abs(py - mat))
        scale = max(np.max(np.abs(mat)), 1e-30)
        rel_err = abs_err / scale

        ok = rel_err < 1e-8 or abs_err < 1e-15
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {name:15s}  max|err|={abs_err:.3e}  rel={rel_err:.3e}  {status}")

    print()
    if all_pass:
        print("All checks passed.")
    else:
        print("Some checks FAILED — investigate above.")

    return all_pass


if __name__ == '__main__':
    mat_file = (sys.argv[1] if len(sys.argv) > 1
                else r'C:\Users\phaney\OneDrive - NIST\Documents\MATLAB\Duartes_code\Semiclassical_zero_Field\benchmark_data_30.mat')
    inp_file = sys.argv[2] if len(sys.argv) > 2 else 'input_benchmark.txt'
    validate(mat_file, inp_file)
