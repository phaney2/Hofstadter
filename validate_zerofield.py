"""
Validate zero-field Python bands against MATLAB benchmark (bands_BG.mat).

The benchmark stores eigenvalues in eV.
"""

import numpy as np
from scipy.io import loadmat
from zerofield import do_calc


def validate(input_file='./input_zerofield.txt',
             benchmark='./matlab_code/zerofield/bands_BG.mat'):
    print("Running zero-field calculation...")
    result = do_calc(input_file)

    print("Loading MATLAB benchmark...")
    ref = loadmat(benchmark)

    band_K_py = result['band_K']
    band_Kp_py = result['band_Kp']
    band_K_mat = ref['band_K']
    band_Kp_mat = ref['band_Kp']

    print(f"  Python  band_K shape: {band_K_py.shape}")
    print(f"  MATLAB  band_K shape: {band_K_mat.shape}")
    print(f"  Python  band_Kp shape: {band_Kp_py.shape}")
    print(f"  MATLAB  band_Kp shape: {band_Kp_mat.shape}")

    band_K_py_sorted = np.sort(band_K_py, axis=1)
    band_K_mat_sorted = np.sort(band_K_mat, axis=1)
    band_Kp_py_sorted = np.sort(band_Kp_py, axis=1)
    band_Kp_mat_sorted = np.sort(band_Kp_mat, axis=1)

    err_K = np.max(np.abs(band_K_py_sorted - band_K_mat_sorted))
    err_Kp = np.max(np.abs(band_Kp_py_sorted - band_Kp_mat_sorted))

    print(f"\n  Max abs error K:  {err_K:.2e} eV  ({err_K * 1000:.2e} meV)")
    print(f"  Max abs error Kp: {err_Kp:.2e} eV  ({err_Kp * 1000:.2e} meV)")

    # The MATLAB benchmark uses truncated Dq values (6 sig figs) for the
    # Q-vector grid center, while Python uses the exact integer grid.
    # This causes a systematic ~2.4e-6 eV offset (= hbar_vF * |ΔQ|).
    tol = 5e-6
    if err_K < tol and err_Kp < tol:
        print(f"\n  PASS (tolerance {tol} eV)")
    else:
        print(f"\n  FAIL (tolerance {tol} eV)")

    return err_K, err_Kp


if __name__ == '__main__':
    validate()
