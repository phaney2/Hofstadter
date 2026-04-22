"""
Magnetic Bloch bands for bilayer graphene on hBN substrate.

Translated from MATLAB code by Paul M. Haney.
Reference: "A Quantum Ruler for Orbital Magnetism in Moiré Quantum Matter"
"""

import sys
import os
import re
import numpy as np
from scipy import linalg
from multiprocessing import Pool, cpu_count
from pathlib import Path


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
HBAR = 1.05e-34      # J·s
Q_E = 1.6e-19         # C
A_GRAPHENE = 2.46e-10  # m
A_HBN = 2.504e-10      # m


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def parse_input_file(filepath):
    """Parse a MATLAB-style input file into a dict of parameters."""
    params = {}
    ns = {"np": np}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' not in line or line.startswith('%'):
                continue
            if '%' in line:
                line = line[:line.index('%')]
            varname = line[:line.index('=')].strip()
            valstr = line[line.index('=') + 1:].strip().rstrip(';')
            val = _eval_matlab_value(valstr, ns)
            params[varname] = val
            ns[varname] = val
    return params


def _eval_matlab_value(s, ns=None):
    """Evaluate a simple MATLAB expression to a Python value."""
    s = s.strip()
    if s.startswith("'") and s.endswith("'"):
        return s.strip("'")
    if s.startswith('{') and s.endswith('}'):
        inner = s[1:-1]
        items = [x.strip().strip("'") for x in inner.split(',')]
        return items
    if ns is None:
        ns = {"np": np}
    s_py = s.replace('linspace', 'np.linspace')
    s_py = re.sub(r'\[([^\]]+)\]', _matlab_array_to_python, s_py)
    try:
        return eval(s_py, {"__builtins__": {}}, ns)
    except Exception:
        return s


def _matlab_array_to_python(match):
    """Convert MATLAB array `[1 2 3]` → `np.array([1, 2, 3])`."""
    inner = match.group(1).strip()
    parts = inner.split()
    if len(parts) > 1 and all(_is_number(p) for p in parts):
        return 'np.array([' + ', '.join(parts) + '])'
    return match.group(0)


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def lf_function(m, n, alpha, x):
    """
    Generalized Laguerre function Lf(n, alpha, x) via recurrence.

    Returns array of shape (m, n+1) with columns Lf(0..n, alpha, x).
    m: number of evaluation points (scalar x → m=1)
    n: highest order
    alpha: parameter (> -1)
    x: evaluation point(s), shape (m,) or scalar
    """
    if alpha <= -1.0:
        raise ValueError(f"alpha = {alpha} must be > -1")
    if n < 0:
        return np.array([])

    x = np.atleast_1d(np.asarray(x, dtype=float)).reshape(m)
    v = np.zeros((m, n + 1))
    v[:, 0] = 1.0
    if n == 0:
        return v
    v[:, 1] = 1.0 + alpha - x
    for i in range(2, n + 1):
        v[:, i] = ((2 * i - 1 + alpha - x) * v[:, i - 1]
                   + (-i + 1 - alpha) * v[:, i - 2]) / i
    return v


def fnm5(n, m, q, lB, laguerretable):
    """
    Landau level matrix element F_{nm}(q).

    n >= m assumed on entry.
    """
    qx, qy = q
    zmstar = (-qx * lB + 1j * qy * lB) / np.sqrt(2)
    zsq = (qx * lB) ** 2 / 2 + (qy * lB) ** 2 / 2

    lvalue = laguerretable[m, n - m]

    if np.isinf(lvalue) or np.isnan(lvalue):
        from scipy.special import jv
        f = (1 / np.e) ** ((m - n) / 2) * (m / n) ** (n / 2) * jv(n - m, 2 * (m * zsq) ** 0.5)
        return f

    factorsok = True

    if n > 150:
        nfactor = (1 / (2 * np.pi * n) ** 0.25
                   * 1 / ((n / (zmstar ** 2 * np.e)) ** (n / 2)))
    else:
        from math import factorial
        nfactor = 1 / factorial(n) ** 0.5 * zmstar ** n

    if np.isnan(nfactor) or np.isinf(nfactor):
        factorsok = False

    if m > 150:
        mfactor = ((2 * np.pi * m) ** 0.25
                   * (m / (zmstar ** 2 * np.e)) ** (m / 2))
    else:
        from math import factorial
        mfactor = factorial(m) ** 0.5 * 1 / zmstar ** m

    if np.isnan(mfactor) or np.isinf(mfactor):
        factorsok = False

    if factorsok:
        f = nfactor * mfactor * np.exp(-zsq / 2) * lvalue
        if f == 0:
            tryf = ((m / n) ** 0.25 * (m / n) ** (m / 2)
                    * (np.e * zmstar ** 2 / n) ** ((n - m) / 2)
                    * np.exp(-zsq / 2) * lvalue)
            if not (np.isnan(tryf) or np.isinf(tryf)) and abs(tryf) > 0:
                f = tryf
    else:
        f = ((m / n) ** 0.25 * (m / n) ** (m / 2)
             * (np.e * zmstar ** 2 / n) ** ((n - m) / 2)
             * np.exp(-zsq / 2) * lvalue)
        if np.isnan(f) or np.isinf(f):
            f = 0.0

    return f


def outer_product(mtx1, labels1, mtx2, labels2):
    """
    Kronecker-style outer product of two matrices with label tracking.

    Returns (new_matrix, new_labels) where new_matrix[i,j] blocks are
    mtx1[i1,j1] * mtx2, and labels are concatenated.
    """
    dim1 = mtx1.shape[0]
    dim2 = mtx2.shape[0]
    newmtx = np.zeros((dim1 * dim2, dim1 * dim2), dtype=complex)
    newlabels = []

    for row1 in range(dim1):
        newrows = slice(row1 * dim2, (row1 + 1) * dim2)
        for col1 in range(dim1):
            newcols = slice(col1 * dim2, (col1 + 1) * dim2)
            newmtx[newrows, newcols] = mtx1[row1, col1] * mtx2
        for lab2 in labels2:
            newlabels.append(f"{labels1[row1]}_{lab2}_")
    return newmtx, newlabels


def getindices(labelset, labels):
    """
    Find indices in labelset whose entries contain ALL of the given label substrings.
    """
    result = None
    for lab in labels:
        hits = {i for i, s in enumerate(labelset) if lab in s}
        result = hits if result is None else result & hits
    return sorted(result) if result else []


# ---------------------------------------------------------------------------
# Hamiltonian construction helpers
# ---------------------------------------------------------------------------

def _build_fnm_tables(N, ktheta, lB, q_vectors):
    """Build Laguerre table and F_{nm} tables for a set of q-vectors."""
    zsq = (lB * ktheta) ** 2 / 2
    laguerretable = np.zeros((N + 1, N + 1))
    for m_idx in range(N + 1):
        laguerretable[:, m_idx] = lf_function(1, N, m_idx, zsq).flatten()

    tables = []
    for q in q_vectors:
        tbl = np.zeros((N + 1, N + 1), dtype=complex)
        for n in range(N + 1):
            for m in range(n + 1):
                tbl[n, m] = fnm5(n, m, q, lB, laguerretable)
                if m != n:
                    sign = 1 if (n - m) % 2 == 0 else -1
                    tbl[m, n] = sign * np.conj(tbl[n, m])
        tables.append(tbl)

    LLlabels = [f"LL{n}" for n in range(N + 1)]
    return tables, LLlabels


def _build_chain_matrices_K(Nq, pp, qq):
    """Build guiding-center chain matrices for K valley."""
    chain1 = np.zeros((Nq, Nq), dtype=complex)
    chain2 = np.zeros((Nq, Nq), dtype=complex)
    chain3 = np.zeros((Nq, Nq), dtype=complex)
    chainlabels = []

    for qcounter in range(Nq):
        tqind = qcounter % Nq
        tqindp1 = (qcounter + 1) % Nq
        tqindm1 = (qcounter - 1) % Nq
        tqval = qcounter % Nq

        chain1[tqind, tqind] += np.exp(1j * 2 * np.pi * pp / qq * tqval)
        chain2[tqindm1, tqind] += np.exp(-1j * (np.pi / 2) * pp / qq * (2 * tqval - 1))
        chain3[tqindp1, tqind] += np.exp(-1j * (np.pi / 2) * pp / qq * (2 * tqval + 1))

        chainlabels.append(f"q{qcounter}")
    return chain1, chain2, chain3, chainlabels


def _build_chain_matrices_Kp(Nq, pp, qq):
    """Build guiding-center chain matrices for K' valley."""
    chain1 = np.zeros((Nq, Nq), dtype=complex)
    chain2 = np.zeros((Nq, Nq), dtype=complex)
    chain3 = np.zeros((Nq, Nq), dtype=complex)
    chainlabels = []

    for qcounter in range(Nq):
        tqind = qcounter % Nq
        tqindp1 = (qcounter + 1) % Nq
        tqindm1 = (qcounter - 1) % Nq
        tqval = qcounter % Nq

        chain1[tqind, tqind] += np.exp(-1j * 2 * np.pi * pp / qq * tqval)
        chain2[tqindp1, tqind] += np.exp(1j * (np.pi / 2) * pp / qq * (2 * tqval + 1))
        chain3[tqindm1, tqind] += np.exp(1j * (np.pi / 2) * pp / qq * (2 * tqval - 1))

        chainlabels.append(f"q{qcounter}")
    return chain1, chain2, chain3, chainlabels


def _assemble_interbilayer_terms(N, Nq, chain_matrices, chainlabels,
                                 fnm_tables, LLlabels, t_matrices,
                                 chop_sublattice):
    """
    Assemble moire coupling term matrices from chain, F_nm, and sublattice hopping.

    chop_sublattice: 'B' for K valley (remove B,LL_N), 'A' for K' valley (remove A,LL_N)
    """
    chain1, chain2, chain3 = chain_matrices
    Fnm_q1, Fnm_q2, Fnm_q3 = fnm_tables
    t1, t2, t3 = t_matrices

    term1, qNlabels = outer_product(chain1, chainlabels, Fnm_q1, LLlabels)
    term2, _ = outer_product(chain2, chainlabels, Fnm_q2, LLlabels)
    term3, _ = outer_product(chain3, chainlabels, Fnm_q3, LLlabels)

    sublatticelabels = ['A', 'B']
    term1, qNslabels = outer_product(t1, sublatticelabels, term1, qNlabels)
    term2, _ = outer_product(t2, sublatticelabels, term2, qNlabels)
    term3, _ = outer_product(t3, sublatticelabels, term3, qNlabels)

    lastLLlabel = f"LL{N}"
    chop_idx = getindices(qNslabels, [chop_sublattice, f"{lastLLlabel}_"])

    for term in [term1, term2, term3]:
        pass  # we'll do deletion below

    term1 = np.delete(np.delete(term1, chop_idx, axis=0), chop_idx, axis=1)
    term2 = np.delete(np.delete(term2, chop_idx, axis=0), chop_idx, axis=1)
    term3 = np.delete(np.delete(term3, chop_idx, axis=0), chop_idx, axis=1)

    return term1, term2, term3, qNslabels


def get_interbilayerterms_K(N, Nq, ktheta, lB, v0, v1, eta, qq, pp):
    """Compute inter-bilayer coupling terms for K valley."""
    q1 = ktheta * np.array([0, -1])
    q2 = ktheta * np.array([np.sqrt(3) / 2, 1 / 2])
    q3 = ktheta * np.array([-np.sqrt(3) / 2, 1 / 2])

    psi_angle = -0.29
    w = np.exp(1j * 2 * np.pi / 3)
    t1 = v1 * np.exp(1j * psi_angle) * np.array([[1, w**(-1)], [1, w**(-1)]])
    t2 = v1 * np.exp(1j * psi_angle) * np.array([[1, w], [w, w**(-1)]])
    t3 = v1 * np.exp(1j * psi_angle) * np.array([[1, 1], [w**(-1), w**(-1)]])

    fnm_tables, LLlabels = _build_fnm_tables(N, ktheta, lB, [q1, q2, q3])
    chain1, chain2, chain3, chainlabels = _build_chain_matrices_K(Nq, pp, qq)

    return _assemble_interbilayer_terms(
        N, Nq, (chain1, chain2, chain3), chainlabels,
        fnm_tables, LLlabels, (t1, t2, t3), 'B')


def get_interbilayerterms_Kp(N, Nq, ktheta, lB, v0, v1, eta, qq, pp):
    """Compute inter-bilayer coupling terms for K' valley."""
    q1 = -ktheta * np.array([0, -1])
    q2 = -ktheta * np.array([np.sqrt(3) / 2, 1 / 2])
    q3 = -ktheta * np.array([-np.sqrt(3) / 2, 1 / 2])

    psi_angle = -0.29
    w = np.exp(1j * 2 * np.pi / 3)
    t1 = v1 * np.exp(-1j * psi_angle) * np.array([[1, w], [1, w]])
    t2 = v1 * np.exp(-1j * psi_angle) * np.array([[1, w**(-1)], [w**(-1), w]])
    t3 = v1 * np.exp(-1j * psi_angle) * np.array([[1, 1], [w, w]])

    fnm_tables, LLlabels = _build_fnm_tables(N, ktheta, lB, [q1, q2, q3])
    chain1, chain2, chain3, chainlabels = _build_chain_matrices_Kp(Nq, pp, qq)

    return _assemble_interbilayer_terms(
        N, Nq, (chain1, chain2, chain3), chainlabels,
        fnm_tables, LLlabels, (t1, t2, t3), 'A')


def get_intermonolayerH_K(N, theta, B, qNslabels, params):
    """Inter-monolayer Hamiltonian for K valley (gamma1, gamma3, gamma4 terms)."""
    lB = (HBAR / (Q_E * B)) ** 0.5
    gamma_1 = params['g1']
    eneLL_3 = params['g3'] * A_GRAPHENE / lB * 2 ** 0.5
    eneLL_4 = params['g4'] * A_GRAPHENE / lB * 2 ** 0.5

    dim = len(qNslabels)
    Hinter = np.zeros((dim, dim), dtype=complex)

    for c in range(N + 1):
        LLc = f"LL{c}_"
        LLcp1 = f"LL{c + 1}_"
        LLcm1 = f"LL{c - 1}_"

        A_LLc = getindices(qNslabels, ['A', LLc])
        A_LLcp1 = getindices(qNslabels, ['A', LLcp1])
        A_LLcm1 = getindices(qNslabels, ['A', LLcm1])
        B_LLc = getindices(qNslabels, ['B', LLc])
        B_LLcp1 = getindices(qNslabels, ['B', LLcp1])
        B_LLcm1 = getindices(qNslabels, ['B', LLcm1])

        for rowc in range(len(A_LLc)):
            Hinter[A_LLc[rowc], B_LLc[rowc]] += gamma_1

        for rowc in range(len(A_LLcm1)):
            Hinter[A_LLcm1[rowc], A_LLc[rowc]] += eneLL_4 * np.exp(1j * theta) * c ** 0.5

        for rowc in range(len(B_LLcm1)):
            Hinter[B_LLcm1[rowc], B_LLc[rowc]] += eneLL_4 * np.exp(1j * theta) * c ** 0.5

        for rowc in range(len(B_LLcp1)):
            Hinter[B_LLcp1[rowc], A_LLc[rowc]] += eneLL_3 * np.exp(1j * theta) * (c + 1) ** 0.5

    chop_idx = getindices(qNslabels, ['B', f"LL{N}_"])
    Hinter = np.delete(np.delete(Hinter, chop_idx, axis=0), chop_idx, axis=1)
    return Hinter


def get_intermonolayerH_Kp(N, theta, B, qNslabels, params):
    """Inter-monolayer Hamiltonian for K' valley."""
    lB = (HBAR / (Q_E * B)) ** 0.5
    gamma_1 = params['g1']
    eneLL_3 = params['g3'] * A_GRAPHENE / lB * 2 ** 0.5
    eneLL_4 = params['g4'] * A_GRAPHENE / lB * 2 ** 0.5

    dim = len(qNslabels)
    Hinter = np.zeros((dim, dim), dtype=complex)

    for c in range(N + 1):
        LLc = f"LL{c}_"
        LLcp1 = f"LL{c + 1}_"
        LLcm1 = f"LL{c - 1}_"

        A_LLc = getindices(qNslabels, ['A', LLc])
        A_LLcp1 = getindices(qNslabels, ['A', LLcp1])
        B_LLc = getindices(qNslabels, ['B', LLc])
        B_LLcp1 = getindices(qNslabels, ['B', LLcp1])
        B_LLcm1 = getindices(qNslabels, ['B', LLcm1])

        for rowc in range(len(A_LLc)):
            Hinter[A_LLc[rowc], B_LLc[rowc]] += gamma_1

        for rowc in range(len(A_LLcp1)):
            Hinter[A_LLcp1[rowc], A_LLc[rowc]] += -eneLL_4 * np.exp(1j * theta) * (c + 1) ** 0.5

        for rowc in range(len(B_LLcp1)):
            Hinter[B_LLcp1[rowc], B_LLc[rowc]] += -eneLL_4 * np.exp(1j * theta) * (c + 1) ** 0.5

        for rowc in range(len(B_LLcm1)):
            Hinter[B_LLcm1[rowc], A_LLc[rowc]] += -eneLL_3 * np.exp(1j * theta) * c ** 0.5

    chop_idx = getindices(qNslabels, ['A', f"LL{N}_"])
    Hinter = np.delete(np.delete(Hinter, chop_idx, axis=0), chop_idx, axis=1)
    return Hinter


def get_intralayerH_K(N, theta, B, qNslabels, params, delta_site):
    """Intralayer kinetic Hamiltonian for K valley."""
    lB = (HBAR / (Q_E * B)) ** 0.5
    eneLL_0 = params['g0'] * A_GRAPHENE / lB * 2 ** 0.5
    delta = params['delta']

    dim = len(qNslabels)
    H_intra = np.zeros((dim, dim), dtype=complex)

    for c in range(N + 1):
        LLc = f"LL{c}_"
        LLcp1 = f"LL{c + 1}_"

        A_LLcp1 = getindices(qNslabels, ['A', LLcp1])
        A_LLc = getindices(qNslabels, ['A', LLc])
        B_LLc = getindices(qNslabels, ['B', LLc])

        for rowc in range(len(A_LLcp1)):
            H_intra[A_LLcp1[rowc], B_LLc[rowc]] += -np.exp(1j * theta) * (c + 1) ** 0.5 * eneLL_0

        if delta_site == 'A':
            for rowc in range(len(A_LLc)):
                H_intra[A_LLc[rowc], A_LLc[rowc]] += delta / 2
        elif delta_site == 'B':
            for rowc in range(len(B_LLc)):
                H_intra[B_LLc[rowc], B_LLc[rowc]] += delta / 2

    Hintra = H_intra + H_intra.T.conj()

    chop_idx = getindices(qNslabels, ['B', f"LL{N}_"])
    Hintra = np.delete(np.delete(Hintra, chop_idx, axis=0), chop_idx, axis=1)
    return Hintra


def get_intralayerH_Kp(N, theta, B, qNslabels, params, delta_site):
    """Intralayer kinetic Hamiltonian for K' valley."""
    lB = (HBAR / (Q_E * B)) ** 0.5
    eneLL_0 = params['g0'] * A_GRAPHENE / lB * 2 ** 0.5
    delta = params['delta']

    dim = len(qNslabels)
    H_intra = np.zeros((dim, dim), dtype=complex)

    for c in range(N + 1):
        LLc = f"LL{c}_"
        LLcm1 = f"LL{c - 1}_"

        A_LLcm1 = getindices(qNslabels, ['A', LLcm1])
        A_LLc = getindices(qNslabels, ['A', LLc])
        B_LLc = getindices(qNslabels, ['B', LLc])

        for rowc in range(len(A_LLcm1)):
            H_intra[A_LLcm1[rowc], B_LLc[rowc]] += np.exp(1j * theta) * c ** 0.5 * eneLL_0

        if delta_site == 'A':
            for rowc in range(len(A_LLc)):
                H_intra[A_LLc[rowc], A_LLc[rowc]] += delta / 2
        elif delta_site == 'B':
            for rowc in range(len(B_LLc)):
                H_intra[B_LLc[rowc], B_LLc[rowc]] += delta / 2

    Hintra = H_intra + H_intra.T.conj()

    chop_idx = getindices(qNslabels, ['A', f"LL{N}_"])
    Hintra = np.delete(np.delete(Hintra, chop_idx, axis=0), chop_idx, axis=1)
    return Hintra


# ---------------------------------------------------------------------------
# Per-k-point solver (serial core + parallel wrapper)
# ---------------------------------------------------------------------------

def _solve_kpoint_core(d, kpt):
    """Solve both valleys at a single k-point. Returns (tek_K, tek_Kp)."""
    kpts = kpt - d['M_mag']
    kx_val, ky_val = kpts
    pp, qq = d['pp'], d['qq']
    Lx, Ly = d['Lx'], d['Ly']
    gamma, v0, dim_MLG = d['gamma'], d['v0'], d['dim_MLG']

    tek_K = None
    tek_Kp = None

    if 'K' in d['valley']:
        tphase1 = np.exp(1j * (pp / qq) * kx_val * Lx)
        tphase2 = np.exp(-1j * (pp / qq) * kx_val * Lx / 2) * np.exp(1j * ky_val * Ly * (pp / qq))
        tphase3 = np.exp(-1j * (pp / qq) * kx_val * Lx / 2) * np.exp(-1j * ky_val * Ly * (pp / qq))

        V_hBN_pq_K = gamma * tphase1 * d['term1_K'] + tphase2 * d['term2_K'] + tphase3 * d['term3_K']
        V_hBN_tot_K = v0 * np.eye(dim_MLG) + V_hBN_pq_K + V_hBN_pq_K.T.conj()

        V_hBN_K = np.zeros((2 * dim_MLG, 2 * dim_MLG), dtype=complex)
        V_hBN_K[dim_MLG:, dim_MLG:] = V_hBN_tot_K

        Htotal_K = 1000 / Q_E * (d['H_BLG_K'] + V_hBN_K)
        tek_K = np.sort(linalg.eigvalsh(Htotal_K))

    if 'Kp' in d['valley']:
        tphase1 = np.exp(-1j * (pp / qq) * kx_val * Lx)
        tphase2 = np.exp(1j * (pp / qq) * kx_val * Lx / 2) * np.exp(-1j * ky_val * Ly * (pp / qq))
        tphase3 = np.exp(1j * (pp / qq) * kx_val * Lx / 2) * np.exp(1j * ky_val * Ly * (pp / qq))

        V_hBN_pq_Kp = gamma * tphase1 * d['term1_Kp'] + tphase2 * d['term2_Kp'] + tphase3 * d['term3_Kp']
        V_hBN_tot_Kp = v0 * np.eye(dim_MLG) + V_hBN_pq_Kp + V_hBN_pq_Kp.T.conj()

        V_hBN_Kp = np.zeros((2 * dim_MLG, 2 * dim_MLG), dtype=complex)
        V_hBN_Kp[dim_MLG:, dim_MLG:] = V_hBN_tot_Kp

        Htotal_Kp = 1000 / Q_E * (d['H_BLG_Kp'] + V_hBN_Kp)
        tek_Kp = np.sort(linalg.eigvalsh(Htotal_Kp))

    return tek_K, tek_Kp


_worker_shared = {}

def _init_kpoint_worker(shared):
    global _worker_shared
    _worker_shared = shared


def _solve_kpoint(args):
    kc, kpt = args
    tek_K, tek_Kp = _solve_kpoint_core(_worker_shared, kpt)
    return kc, tek_K, tek_Kp


# ---------------------------------------------------------------------------
# Main band-structure routine
# ---------------------------------------------------------------------------

def BLG_hBN_magnetic_bloch_bands_BZ(filepath):
    """
    Compute magnetic Bloch bands for bilayer graphene on hBN.

    Returns a dict with keys depending on calctype:
      'ek':  kpoints, bands_K, bands_Kp
      'dos': elist, dos_K, dos_Kp
    """
    # --- defaults ---
    eta = 1
    vF = 1e6
    gamma = 1
    Nmax = 5000
    calctype = 'ek'
    valley = ['K', 'Kp']
    readFnmtable = 0

    # --- read input ---
    inp = parse_input_file(filepath)
    locals_dict = dict(inp)

    theta = locals_dict.get('theta', 0.0)
    qq = int(locals_dict['qq'])
    pp = int(locals_dict['pp'])
    g0 = locals_dict['g0']
    g1 = locals_dict['g1']
    g3 = locals_dict['g3']
    g4 = locals_dict['g4']
    delta = locals_dict.get('delta', 0)
    v0_meV = locals_dict['v0']
    v1_meV = locals_dict['v1']
    w_meV = locals_dict['w']
    eta = locals_dict.get('eta', eta)
    U = np.atleast_1d(locals_dict.get('U', np.array([0, 0])))
    nk1 = int(locals_dict.get('nk1', 25))
    nk2 = int(locals_dict.get('nk2', 40))
    LL_multiplier = locals_dict.get('LL_multiplier', 6)
    Nmax = int(locals_dict.get('Nmax', Nmax))
    calctype = locals_dict.get('calctype', calctype)
    valley = locals_dict.get('valley', valley)
    nebin = int(locals_dict.get('nebin', 1000))
    gamma = locals_dict.get('gamma', gamma)
    vF = locals_dict.get('vF', vF)
    isparallel = int(locals_dict.get('isparallel', 0))
    elist = np.asarray(locals_dict.get('elist', np.linspace(-300, 300, nebin)))

    if calctype == 'spectrum':
        calctype = 'dos'

    # --- derived quantities ---
    eps = A_HBN / A_GRAPHENE - 1
    L_moire = (1 + eps) * A_GRAPHENE / np.sqrt(eps ** 2 + 2 * (1 + eps) * (1 - np.cos(theta)))

    kd = 4 * np.pi / (3 * A_GRAPHENE)
    ktheta = 4 * np.pi / (3 ** 0.5 * L_moire)
    uc_area = 3 ** 0.5 * L_moire ** 2 / 2

    phi_0 = HBAR * 2 * np.pi / Q_E
    B = (qq / pp) * phi_0 / uc_area
    lB = (HBAR / (Q_E * B)) ** 0.5

    eneLL = g0 / 1e3 * Q_E * A_GRAPHENE / lB * 2 ** 0.5

    w_J = w_meV / 1e3 * Q_E
    v0 = v0_meV / 1e3 * Q_E
    v1 = v1_meV / 1e3 * Q_E

    Delta_param = 3 ** 0.5 / 2 * ktheta * lB ** 2

    N = int(LL_multiplier * round(max(HBAR * vF * ktheta, w_J) / eneLL) ** 2)
    if N > Nmax:
        N = Nmax

    TBGparams = {
        'g0': g0 / 1e3 * Q_E,
        'g1': g1 / 1e3 * Q_E,
        'g3': g3 / 1e3 * Q_E,
        'g4': g4 / 1e3 * Q_E,
        'delta': delta / 1e3 * Q_E,
    }

    Nq = qq
    dim1 = qq * N + qq * (N + 1)

    Utp = np.eye(dim1) * U[0] / 1e3 * Q_E
    Ubm = np.eye(dim1) * U[1] / 1e3 * Q_E

    Lx = L_moire
    Ly = np.sqrt(3) * L_moire / 2

    print(f"  N (Landau levels) = {N}")
    print(f"  dim per layer = {dim1}")
    print(f"  B = {B:.6e} T")

    # --- K valley: k-independent Hamiltonian ---
    if 'K' in valley:
        print("  Building K-valley Hamiltonian...")
        term1_K, term2_K, term3_K, qNslabels_K = get_interbilayerterms_K(
            N, Nq, ktheta, lB, v0, v1, eta, qq, pp)
        Hinter_K = get_intermonolayerH_K(N, 0, B, qNslabels_K, TBGparams)
        Hintra1_K = get_intralayerH_K(N, 0, B, qNslabels_K, TBGparams, 'A')
        Hintra2_K = get_intralayerH_K(N, 0, B, qNslabels_K, TBGparams, 'B')

        H_BLG_K = np.block([
            [Hintra1_K + Utp, Hinter_K],
            [Hinter_K.T.conj(), Hintra2_K + Ubm]
        ])

        chop_K = getindices(qNslabels_K, ['B', f"LL{N}_"])
        qNslabels_K_trimmed = [s for i, s in enumerate(qNslabels_K) if i not in chop_K]

    # --- K' valley: k-independent Hamiltonian ---
    if 'Kp' in valley:
        print("  Building K'-valley Hamiltonian...")
        term1_Kp, term2_Kp, term3_Kp, qNslabels_Kp = get_interbilayerterms_Kp(
            N, Nq, ktheta, lB, v0, v1, eta, qq, pp)
        Hinter_Kp = get_intermonolayerH_Kp(N, 0, B, qNslabels_Kp, TBGparams)
        Hintra1_Kp = get_intralayerH_Kp(N, 0, B, qNslabels_Kp, TBGparams, 'A')
        Hintra2_Kp = get_intralayerH_Kp(N, 0, B, qNslabels_Kp, TBGparams, 'B')

        H_BLG_Kp = np.block([
            [Hintra1_Kp + Utp, Hinter_Kp],
            [Hinter_Kp.T.conj(), Hintra2_Kp + Ubm]
        ])

        chop_Kp = getindices(qNslabels_Kp, ['A', f"LL{N}_"])
        qNslabels_Kp_trimmed = [s for i, s in enumerate(qNslabels_Kp) if i not in chop_Kp]

    # --- k-mesh ---
    b1 = ktheta * np.array([0, -1])
    b2 = ktheta * np.array([np.sqrt(3) / 2, 1 / 2])

    M_pt = 0.5 * b1
    M_mag = M_pt / pp

    Nk_tot = nk1 * nk2
    n1_arr = np.arange(nk1)
    n2_arr = np.arange(nk2)
    n1grid, n2grid = np.meshgrid(n1_arr, n2_arr)
    n11 = n1grid.flatten(order='F')
    n22 = n2grid.flatten(order='F')

    vb = np.array([b1 / pp, b2 / pp])
    kpoints = np.zeros((Nk_tot, 2))
    for j in range(Nk_tot):
        frac = np.array([n11[j] / nk1, n22[j] / nk2])
        kpoints[j, :] = vb.T @ frac

    dim_MLG = Hintra1_K.shape[0] if 'K' in valley else Hintra1_Kp.shape[0]
    bands_K = np.zeros((Nk_tot, 2 * dim_MLG))
    bands_Kp = np.zeros((Nk_tot, 2 * dim_MLG))

    # --- pack shared data for the k-point solver ---
    shared = {
        'pp': pp, 'qq': qq, 'Lx': Lx, 'Ly': Ly,
        'gamma': gamma, 'v0': v0, 'dim_MLG': dim_MLG,
        'valley': valley, 'M_mag': M_mag,
    }
    if 'K' in valley:
        shared.update({
            'H_BLG_K': H_BLG_K, 'term1_K': term1_K,
            'term2_K': term2_K, 'term3_K': term3_K,
        })
    if 'Kp' in valley:
        shared.update({
            'H_BLG_Kp': H_BLG_Kp, 'term1_Kp': term1_Kp,
            'term2_Kp': term2_Kp, 'term3_Kp': term3_Kp,
        })

    print(" Entering the k loop")
    if isparallel:
        nworkers = min(cpu_count(), Nk_tot)
        print(f"  (parallel: {nworkers} workers)")
        tasks = [(kc, kpoints[kc, :]) for kc in range(Nk_tot)]
        with Pool(processes=nworkers,
                  initializer=_init_kpoint_worker, initargs=(shared,)) as pool:
            results = pool.map(_solve_kpoint, tasks)
        for kc, tek_K_row, tek_Kp_row in results:
            if tek_K_row is not None:
                bands_K[kc, :] = tek_K_row
            if tek_Kp_row is not None:
                bands_Kp[kc, :] = tek_Kp_row
    else:
        for kc in range(Nk_tot):
            print(f"  |>>        step {kc + 1} of {Nk_tot}")
            tek_K_row, tek_Kp_row = _solve_kpoint_core(shared, kpoints[kc, :])
            if tek_K_row is not None:
                bands_K[kc, :] = tek_K_row
            if tek_Kp_row is not None:
                bands_Kp[kc, :] = tek_Kp_row

    print(" Done with the k loop")

    if calctype == 'ek':
        return {'calctype': 'ek', 'kpoints': kpoints,
                'bands_K': bands_K, 'bands_Kp': bands_Kp}

    # --- DOS: bin eigenvalues into energy histogram ---
    dos_K = np.zeros(len(elist))
    dos_Kp = np.zeros(len(elist))
    for kc in range(Nk_tot):
        if 'K' in valley:
            tek = bands_K[kc, :]
            in_range = tek[(tek > elist[0]) & (tek < elist[-1])]
            bins = np.argmin(np.abs(in_range[:, None] - elist[None, :]), axis=1)
            for b in bins:
                dos_K[b] += 1
        if 'Kp' in valley:
            tek = bands_Kp[kc, :]
            in_range = tek[(tek > elist[0]) & (tek < elist[-1])]
            bins = np.argmin(np.abs(in_range[:, None] - elist[None, :]), axis=1)
            for b in bins:
                dos_Kp[b] += 1

    return {'calctype': 'dos', 'elist': elist,
            'dos_K': dos_K, 'dos_Kp': dos_Kp}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(input_file=None):
    if input_file is None:
        input_file = './input_test.txt'

    inp = parse_input_file(input_file)
    pp = int(inp['pp'])
    qq = int(inp['qq'])

    result = BLG_hBN_magnetic_bloch_bands_BZ(input_file)

    outfile = f"bands_p{pp}_q{qq}.npz"
    np.savez(outfile, **{k: v for k, v in result.items() if k != 'calctype'})
    print(f" Saved to {outfile}")
    return result


if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else './input_test.txt'
    main(input_file)
