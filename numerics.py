import numpy as np


def lf_function(m, n, alpha, x):
    """
    Generalized Laguerre function Lf(n, alpha, x) via recurrence.

    Returns array of shape (m, n+1) with columns Lf(0..n, alpha, x).
    m: number of evaluation points (scalar x -> m=1)
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


def build_fnm_tables(N, ktheta, lB, q_vectors):
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
