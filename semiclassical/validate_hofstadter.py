"""
Validate Hofstadter semiclassical calculation against MATLAB benchmark.

Usage:
    python validate_hofstadter.py [benchmark.mat] [input_file]

Compares eigenvalues, Berry curvature, and orbital moment against
benchmark_BC_calc.mat from the MATLAB Hofstadter code.
"""

import sys
import os
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(__file__))
from semiclassical import do_calc


def validate(benchmark_path, input_path):
    print(f"Loading benchmark: {benchmark_path}")
    bench = loadmat(benchmark_path)

    print(f"Running Python calculation: {input_path}")
    result = do_calc(input_path)

    # --- k-points ---
    kpts_matlab = bench['kpoints']       # (Nk, 2) in m^-1
    kpts_python = result['kpoints']      # (Nk, 2) in Ang^-1

    kpts_python_m = kpts_python * 1e10   # convert Ang^-1 -> m^-1
    kpt_err = np.max(np.abs(kpts_python_m - kpts_matlab))
    print(f"\n  k-points max error: {kpt_err:.3e} m^-1")

    # --- Eigenvalues ---
    # MATLAB ekset is (Nk, num_bands) in meV
    # Python E_K is (nbands_sel, Nk) in meV
    # Need to select matching bands from MATLAB's num_bands=26 window
    ek_matlab = bench['ekset']           # (900, 26)
    ekp_matlab = bench['ekpset']         # (900, 26)

    E_K = result['E_K']                  # (nbands_sel, Nk) in meV
    E_Kp = result['E_Kp']

    nbands_sel = E_K.shape[0]
    num_bands_matlab = ek_matlab.shape[1]
    center = num_bands_matlab // 2
    bands_sel_input = np.array([-3, -2, -1, 0, 1, 2, 3])
    matlab_idx = center - 1 + bands_sel_input

    ek_match = ek_matlab[:, matlab_idx].T   # (nbands_sel, Nk)
    ekp_match = ekp_matlab[:, matlab_idx].T

    ek_err = np.max(np.abs(E_K - ek_match))
    ekp_err = np.max(np.abs(E_Kp - ekp_match))
    print(f"  K eigenvalue max error:  {ek_err:.3e} meV")
    print(f"  K' eigenvalue max error: {ekp_err:.3e} meV")

    # --- Berry curvature ---
    # MATLAB berrycurv is in MATLAB velocity units (not m^2)
    # Python Oz is in m^2 (after 1e-20 conversion)
    # Check if ratio is constant (confirms physics match, different units)
    bc_matlab = bench['berrycurv_K'][:, matlab_idx].T  # (nbands_sel, Nk)
    oz_python = result['Oz_K']                          # (nbands_sel, Nk)

    nonzero = np.abs(bc_matlab) > 1e-20
    if np.any(nonzero):
        ratio = oz_python[nonzero] / bc_matlab[nonzero]
        ratio_std = np.std(ratio) / np.abs(np.mean(ratio))
        print(f"\n  BC ratio (Python/MATLAB): mean={np.mean(ratio):.6e}, "
              f"relative std={ratio_std:.3e}")
        if ratio_std < 1e-6:
            print("  -> Constant ratio: physics MATCH (units differ as expected)")
        else:
            print("  -> WARNING: ratio not constant, possible physics mismatch")
    else:
        print("  BC: all zeros in benchmark, skipping ratio check")

    # --- Orbital moment ---
    morb_matlab = bench['morb_K'][:, matlab_idx].T
    lz_python = result['Lz_K']

    nonzero_m = np.abs(morb_matlab) > 1e-20
    if np.any(nonzero_m):
        ratio_m = lz_python[nonzero_m] / morb_matlab[nonzero_m]
        ratio_m_std = np.std(ratio_m) / np.abs(np.mean(ratio_m))
        print(f"  Morb ratio (Python/MATLAB): mean={np.mean(ratio_m):.6e}, "
              f"relative std={ratio_m_std:.3e}")
        if ratio_m_std < 1e-6:
            print("  -> Constant ratio: physics MATCH (units differ as expected)")
        else:
            print("  -> WARNING: ratio not constant, possible physics mismatch")

    # K' valley
    bcp_matlab = bench['berrycurv_Kp'][:, matlab_idx].T
    ozp_python = result['Oz_Kp']
    nonzero_p = np.abs(bcp_matlab) > 1e-20
    if np.any(nonzero_p):
        ratio_p = ozp_python[nonzero_p] / bcp_matlab[nonzero_p]
        ratio_p_std = np.std(ratio_p) / np.abs(np.mean(ratio_p))
        print(f"\n  K' BC ratio: mean={np.mean(ratio_p):.6e}, "
              f"relative std={ratio_p_std:.3e}")

    print("\nValidation complete.")


if __name__ == '__main__':
    bench_default = (r"C:\Users\phaney\OneDrive - NIST\Documents\MATLAB"
                     r"\Duartes_code\Hofstadter_codes\benchmark_BC_calc.mat")
    input_default = os.path.join(os.path.dirname(__file__),
                                 'input_hofstadter.txt')

    bench = sys.argv[1] if len(sys.argv) > 1 else bench_default
    inp = sys.argv[2] if len(sys.argv) > 2 else input_default
    validate(bench, inp)
