import numpy as np
from scipy.special import gammaln, jv


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
    alphas = np.arange(N + 1, dtype=float)
    laguerretable = np.zeros((N + 1, N + 1))
    laguerretable[0, :] = 1.0
    if N >= 1:
        laguerretable[1, :] = 1.0 + alphas - zsq
    for k in range(2, N + 1):
        laguerretable[k, :] = ((2 * k - 1 + alphas - zsq) * laguerretable[k - 1, :]
                               + (-k + 1 - alphas) * laguerretable[k - 2, :]) / k

    log_fact = gammaln(np.arange(N + 2, dtype=float) + 1)

    n_arr, m_arr = np.tril_indices(N + 1)
    diff = n_arr - m_arr
    laguerre_vals = laguerretable[m_arr, diff]

    bad_L = np.isinf(laguerre_vals) | np.isnan(laguerre_vals)
    zero_L = (laguerre_vals == 0) & ~bad_L
    good = ~bad_L & ~zero_L

    tables = []
    for q in q_vectors:
        qx, qy = q
        zmstar = (-qx * lB + 1j * qy * lB) / np.sqrt(2)
        zsq_q = (qx * lB) ** 2 / 2 + (qy * lB) ** 2 / 2

        tbl = np.zeros((N + 1, N + 1), dtype=complex)
        zmag = abs(zmstar)
        if zmag == 0:
            np.fill_diagonal(tbl, np.exp(-zsq_q / 2))
            tables.append(tbl)
            continue

        log_zmag = np.log(zmag)
        ang_zm = np.angle(zmstar)

        log_mag_base = (0.5 * (log_fact[m_arr] - log_fact[n_arr])
                        + diff * log_zmag
                        - zsq_q / 2)

        f_vals = np.zeros(len(n_arr), dtype=complex)

        if np.any(good):
            log_abs_L = np.log(np.abs(laguerre_vals[good]))
            log_total = log_mag_base[good] + log_abs_L
            phase = diff[good] * ang_zm
            neg_mask = laguerre_vals[good] < 0
            phase[neg_mask] += np.pi
            f_vals[good] = np.exp(log_total + 1j * phase)

        if np.any(bad_L):
            bad_n = n_arr[bad_L].astype(float)
            bad_m = m_arr[bad_L].astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                f_b = ((1 / np.e) ** ((bad_m - bad_n) / 2)
                       * (bad_m / bad_n) ** (bad_n / 2)
                       * jv(bad_n - bad_m, 2 * (bad_m * zsq_q) ** 0.5))
            f_b = np.where(np.isfinite(f_b), f_b, 0.0)
            f_vals[bad_L] = f_b

        tbl[n_arr, m_arr] = f_vals

        off_diag = diff > 0
        signs = np.where(diff[off_diag] % 2 == 0, 1.0, -1.0)
        tbl[m_arr[off_diag], n_arr[off_diag]] = signs * np.conj(f_vals[off_diag])

        tables.append(tbl)

    LLlabels = [f"LL{n}" for n in range(N + 1)]
    return tables, LLlabels
