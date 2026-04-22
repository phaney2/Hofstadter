"""Compare Python output against MATLAB .mat benchmarks."""
import numpy as np
from scipy.io import loadmat
from main_v2 import BLG_hBN_magnetic_bloch_bands_BZ


def compare(input_file, mat_file, label):
    print(f"\n{'='*60}")
    print(f"  Validating: {label}")
    print(f"  input: {input_file}  |  reference: {mat_file}")
    print(f"{'='*60}")

    result = BLG_hBN_magnetic_bloch_bands_BZ(input_file)
    kpoints_py = result['kpoints']
    bands_K_py = result['bands_K']
    bands_Kp_py = result['bands_Kp']

    mat = loadmat(mat_file)
    bands_K_mat = mat['bands_K']
    bands_Kp_mat = mat['bands_Kp']
    kpoints_mat = mat['kpts']

    bands_K_mat = np.sort(np.real(bands_K_mat), axis=1)
    bands_Kp_mat = np.sort(np.real(bands_Kp_mat), axis=1)
    bands_K_py = np.sort(np.real(bands_K_py), axis=1)
    bands_Kp_py = np.sort(np.real(bands_Kp_py), axis=1)

    print(f"\n  Shape check:")
    print(f"    bands_K:  MATLAB {bands_K_mat.shape}  Python {bands_K_py.shape}")
    print(f"    bands_Kp: MATLAB {bands_Kp_mat.shape}  Python {bands_Kp_py.shape}")

    if bands_K_mat.shape != bands_K_py.shape:
        print("  *** SHAPE MISMATCH for bands_K ***")
        return False
    if bands_Kp_mat.shape != bands_Kp_py.shape:
        print("  *** SHAPE MISMATCH for bands_Kp ***")
        return False

    err_K = np.max(np.abs(bands_K_py - bands_K_mat))
    err_Kp = np.max(np.abs(bands_Kp_py - bands_Kp_mat))
    rel_K = err_K / (np.max(np.abs(bands_K_mat)) + 1e-30)
    rel_Kp = err_Kp / (np.max(np.abs(bands_Kp_mat)) + 1e-30)

    print(f"\n  Max absolute error:")
    print(f"    bands_K:  {err_K:.6e} meV")
    print(f"    bands_Kp: {err_Kp:.6e} meV")
    print(f"  Max relative error:")
    print(f"    bands_K:  {rel_K:.6e}")
    print(f"    bands_Kp: {rel_Kp:.6e}")

    tol = 1e-6
    ok = err_K < tol and err_Kp < tol
    print(f"\n  Result: {'PASS' if ok else 'FAIL'} (tolerance = {tol} meV)")
    return ok


if __name__ == '__main__':
    r1 = compare('./input_test.txt', 'bands_p1_q1.mat', 'p=1, q=1')
    r2 = compare('./input_p3_q1.txt', 'bands_p3_q1.mat', 'p=3, q=1')
    print(f"\n{'='*60}")
    print(f"  Overall: {'ALL PASS' if (r1 and r2) else 'SOME FAILED'}")
    print(f"{'='*60}")
