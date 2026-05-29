"""
Test script: plot zero-field moire band structure along high-symmetry k-path.

Path:  K1 -> Gamma -> K2 -> M -> K1

High-symmetry points in fractional coordinates (q1, q2 basis):
    Gamma = 0*q1 + 0*q2
    K1    = 1/3*q1 + 2/3*q2
    K2    = 2/3*q1 + 1/3*q2
    M     = 1/2*q1 + 1/2*q2

Usage:
    python test_bandstructure.py input_band.txt
    python test_bandstructure.py input_band.txt --theta 0.5
    python test_bandstructure.py input_band.txt --ylim 100
"""

import sys
import os
import numpy as np
from scipy.linalg import eigvalsh
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from parser import parse_input_file
from bandstructure import (compute_moire_geometry, build_qvectors,
                           construct_hopping, assemble_H_V_K, assemble_H_V_Kp)


def make_kpath(q1, q2, nk_per_seg=100):
    G  = np.array([0.0, 0.0])
    K1 = (1/3) * q1 + (2/3) * q2
    K2 = (2/3) * q1 + (1/3) * q2
    M  = 0.5 * q1 + 0.5 * q2

    segments = [(K1, G), (G, K2), (K2, M), (M, K1)]
    labels = ['K1', r'$\Gamma$', 'K2', 'M', 'K1']

    kpoints = []
    k_linear = []
    ticks = [0.0]
    dist = 0.0

    for start, end in segments:
        seg_len = np.linalg.norm(end - start)
        nk = max(int(round(nk_per_seg * seg_len / np.linalg.norm(q1))), 2)
        for i in range(nk):
            t = i / nk
            kpt = start + t * (end - start)
            kpoints.append(kpt)
            if len(k_linear) > 0:
                dist += np.linalg.norm(kpt - kpoints[-2])
            k_linear.append(dist)
        dist += np.linalg.norm(end - kpoints[-1])
        ticks.append(dist)

    kpoints.append(segments[-1][1])
    k_linear.append(dist)

    return np.array(kpoints), np.array(k_linear), ticks, labels


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_bandstructure.py <input_file> [--theta VALUE]")
        sys.exit(1)

    filepath = sys.argv[1]
    inp = parse_input_file(filepath)

    theta_override = None
    ylim_val = None
    if '--theta' in sys.argv:
        idx = sys.argv.index('--theta')
        theta_override = np.radians(float(sys.argv[idx + 1]))
    if '--ylim' in sys.argv:
        idx = sys.argv.index('--ylim')
        ylim_val = float(sys.argv[idx + 1])

    theta   = theta_override if theta_override is not None else np.radians(float(inp.get('theta', 0.0)))
    nlayers = int(inp.get('Nlayers', inp.get('nlayers', 2)))
    NQ      = int(inp['NQ'])

    if 'g0' in inp:
        vF = float(inp['g0']) * 2.46 / 1000.0
    else:
        vF = float(inp['vF'])
    if 'g1' in inp:
        gamma1_ev = float(inp['g1']) / 1000.0
    else:
        gamma1_ev = float(inp['gamma1'])
    if 'g3' in inp:
        v3 = float(inp['g3']) * 2.46 / 1000.0
    else:
        v3 = float(inp.get('v3', 0))

    V0_meV = float(inp.get('v0', inp.get('V0')))
    V1_meV = float(inp.get('v1', inp.get('V1')))
    psi    = float(inp.get('moire_psi', 0.29))
    U      = np.atleast_1d(inp.get('U', np.array([0, 0])))

    V0_ev = V0_meV / 1000
    V1_ev = V1_meV / 1000
    Utp   = U[0] / 1000
    Ubm   = U[1] / 1000 if len(U) > 1 else U[0] / 1000
    hbar  = 6.582119569e-16  # eV * s

    q1, q2, q3, vol_M, vb, G1_xy = compute_moire_geometry(theta)
    Q, NG = build_qvectors(NQ, q1, q2)
    numwann = 2 * NG * nlayers

    print(f"theta = {theta:.6f} rad")
    print(f"nlayers = {nlayers}, NQ = {NQ}, NG = {NG}, numwann = {numwann}")

    H_hopp_K, H_hopp_Kp = construct_hopping(
        Q, NG, q1, q2, q3, V0_ev, V1_ev, psi, G1_xy)

    kpoints, k_lin, ticks, labels = make_kpath(q1, q2)
    Nk = len(kpoints)
    print(f"k-path: {Nk} points")

    bands_K  = np.zeros((Nk, numwann))
    bands_Kp = np.zeros((Nk, numwann))

    for i, kpt in enumerate(kpoints):
        HK, _, _  = assemble_H_V_K(kpt, Q, NG, vF, gamma1_ev, v3,
                                    Utp, Ubm, H_hopp_K, hbar, nlayers)
        HKp, _, _ = assemble_H_V_Kp(kpt, Q, NG, vF, gamma1_ev, v3,
                                     Utp, Ubm, H_hopp_Kp, hbar, nlayers)
        bands_K[i]  = eigvalsh(HK)
        bands_Kp[i] = eigvalsh(HKp)

    bands_K  *= 1000  # eV -> meV
    bands_Kp *= 1000

    idx_c = numwann // 2
    band_range = slice(idx_c - 10, idx_c + 10)

    fig, ax = plt.subplots(figsize=(8, 6))

    for band in range(band_range.start, band_range.stop):
        ax.plot(k_lin, bands_K[:, band], 'r-', linewidth=0.8)
        #ax.plot(k_lin, bands_Kp[:, band], 'b-', linewidth=0.8)

    for t in ticks:
        ax.axvline(t, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel('E (meV)')
    ax.set_xlim(k_lin[0], k_lin[-1])
    if ylim_val is not None:
        ax.set_ylim(-ylim_val, ylim_val)
    ax.set_title(f'Moire band structure (theta = {np.degrees(theta):.4f} deg)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
