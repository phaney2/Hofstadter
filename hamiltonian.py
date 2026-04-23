import numpy as np

from constants import HBAR, Q_E, A_GRAPHENE
from numerics import build_fnm_tables
from basis import outer_product, getindices


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

    fnm_tables, LLlabels = build_fnm_tables(N, ktheta, lB, [q1, q2, q3])
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

    fnm_tables, LLlabels = build_fnm_tables(N, ktheta, lB, [q1, q2, q3])
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
