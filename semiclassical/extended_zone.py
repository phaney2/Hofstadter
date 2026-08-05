"""Extended-zone (moire) unfolding of the zero-field band structure.

The moire potential folds the graphene Dirac cone into the small moire BZ.
For a weak potential that folding is nearly pure bookkeeping, but it wrecks
the semiclassical orbits: once a constant-energy contour grows past the moire
BZ it merges with its own periodic images, and `isoenergy.py` starts tracing
the complementary corner pockets instead.  A hole orbit then appears to turn
electron-like -- area shrinking as the energy drops -- which is an artifact of
the zone choice, not physics.

This module maps each moire eigenstate back to the momentum it actually
carries.  The plane-wave basis makes that direct: block j of the Hamiltonian
lives at k - Q_j, so

    w(n, j) = sum_{layer, sublattice} |Psi(layer, j, sublattice ; n)|^2

is the weight of state n at extended momentum k - Q_j, and the moire BZ mesh
translated by the Q lattice tiles the extended zone exactly, one grid point
per (k, Q) pair.

At V0 = V1 = 0 the weights are 0 or 1 and the unfolding is exact.  At finite V
several states share the weight of one extended momentum -- that is what a
gap at a moire Bragg plane *is* -- and they are combined into one value per
intrinsic branch (see `unfold_kpoint`).

Branches are identified by the bare dispersion: the Q-diagonal block of H is
the moire-free mono/bilayer Hamiltonian at k - Q_j (the only moire term that
survives on the diagonal is the uniform V0), so its eigenvectors label the
2*nlayers branches at that momentum with no extra physics input.

Validity: the unfolded orbits are the semiclassical orbits in the magnetic
breakdown limit, where the cyclotron energy exceeds the moire gaps and the
carrier tunnels straight through the Bragg planes.  The folded orbits are the
opposite limit.  See "Extended-zone unfolding" in `doc_technical.md`.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def qvector_indices(Q, q1, q2):
    """Integer coefficients (m1, m2) of each Q on the (q1, q2) basis."""
    m = np.linalg.solve(np.array([q1, q2]).T, np.asarray(Q).T).T
    mint = np.rint(m).astype(int)
    if np.abs(m - mint).max() > 1e-6:
        raise ValueError("Q-vectors are not integer combinations of q1, q2")
    return mint


def extended_setup(Q, q1, q2, NG, nlayers, NQ, ntile):
    """Index bookkeeping for the extended zone.

    `ntile` must be odd and no larger than NQ so that the kept Q-vectors form
    a complete ntile x ntile block of the moire reciprocal lattice; that block
    is what makes the (k, Q) -> extended-grid map a bijection.
    """
    if ntile % 2 == 0:
        raise ValueError(f"extended_ntile must be odd, got {ntile}")
    off = (ntile - 1) // 2
    if off > NQ // 2:
        raise ValueError(
            f"extended_ntile = {ntile} needs NQ >= {2*off+1}, got NQ = {NQ}")

    mint = qvector_indices(Q, q1, q2)
    keep = np.where((np.abs(mint[:, 0]) <= off)
                    & (np.abs(mint[:, 1]) <= off))[0]
    if len(keep) != ntile**2:
        raise ValueError(f"expected {ntile**2} Q-vectors in the central "
                         f"block, found {len(keep)}")

    # rows of H spanned by Q block j: every layer and sublattice of that block
    rows = np.array([[L * 2 * NG + 2 * j + s
                      for L in range(nlayers) for s in (0, 1)]
                     for j in keep])

    return {'off': off, 'ntile': ntile, 'keep': keep, 'mint': mint[keep],
            'rows': rows, 'nb': 2 * nlayers, 'nlayers': nlayers}


# ---------------------------------------------------------------------------
# Per-k-point reduction
# ---------------------------------------------------------------------------

def unfold_kpoint(ek, Psi, Oz, Lz, H, setup, mode='centroid'):
    """Reduce one k-point's spectrum to one value per (branch, extended point).

    Returns `(E, Oz, Lz, wmax)`, each `(nb, nkeep)`, with nb = 2*nlayers
    intrinsic branches and nkeep = ntile^2 retained Q-vectors.  `wmax` is the
    weight of the largest single state feeding the branch: 1 where one
    eigenstate carries the whole extended momentum, ~1/2 at a two-state Bragg
    anticrossing.

    The branches are the eigenvectors u_b of the Q-diagonal block, not just its
    eigenvalues: state n contributes |u_b^dag Psi_n|^2 to branch b.  That
    partition is exact -- completeness of Psi makes each branch collect total
    weight 1 -- and continuous in k, which a nearest-energy assignment is not.
    Under argmin, a low-weight state drifting across the midpoint between two
    reference energies moves all of its weight from one branch to the other
    and steps the result.

    `mode` selects how the states sharing one extended momentum are combined:

    `centroid` (default)
        weight-weighted mean.  Completeness collapses it to
        u_b^dag H_jj u_b = E_ref,b exactly, so the energies *are* the
        moire-free dispersion at k - Q_j: the magnetic breakdown limit,
        smooth through every Bragg plane, no gap anywhere.  Oz and Lz are not
        eigenvalues of H_jj and do keep the moire mixing; averaging them
        cancels the equal-and-opposite curvature of an anticrossing pair,
        which is what removes the folding-induced Berry curvature.

    `dominant`
        the single largest-weight state.  Keeps the true eigenvalue, so the
        moire gap survives, at the cost of a jump of order the gap across the
        plane itself.  Diagnostic: the spread between the two modes bounds the
        error unfolding introduces.
    """
    rows = setup['rows']

    # Q-diagonal block of H = bare Hamiltonian at k - Q_j (+ uniform V0)
    Eref, U = np.linalg.eigh(H[rows[:, :, None], rows[:, None, :]])

    P = np.abs(U.conj().transpose(0, 2, 1) @ Psi[rows])**2
    W = P.sum(axis=2)

    if mode == 'dominant':
        nstar = P.argmax(axis=2)
        E, O, L = ek[nstar], Oz[nstar], Lz[nstar]
    else:
        E, O, L = P @ ek / W, P @ Oz / W, P @ Lz / W

    return E.T, O.T, L.T, P.max(axis=2).T


# ---------------------------------------------------------------------------
# Assembly onto the extended grid
# ---------------------------------------------------------------------------

def extended_kmesh(setup, nk1, nk2, vb):
    """k-points of the extended grid, in the `order='F'` layout of the code.

    The moire mesh point (n1, n2) shifted by Q = (m1, m2) lands at extended
    index N1 = n1 + (off - m1)*nk1, N2 = n2 + (off - m2)*nk2, which walks over
    the whole ntile*nk1 x ntile*nk2 grid exactly once.
    """
    ntile, off = setup['ntile'], setup['off']
    nk1e, nk2e = ntile * nk1, ntile * nk2
    N1 = np.arange(nk1e)
    N2 = np.arange(nk2e)
    u1 = (N1 / nk1 - 0.5 - off)[None, :]
    u2 = (N2 / nk2 - 0.5 - off)[:, None]
    kx = u1 * vb[0, 0] + u2 * vb[1, 0]
    ky = u1 * vb[0, 1] + u2 * vb[1, 1]
    return np.column_stack([kx.ravel(order='F'), ky.ravel(order='F')])


def scatter_extended(per_k, setup, nk1, nk2):
    """Scatter per-k `(nb, nkeep)` arrays onto the `(nb, Nk_ext)` grid.

    `per_k` is `(Nk, nb, nkeep)`; the k-index is the driver's `kc = n2 + n1*nk2`
    and the output is flattened `order='F'` on `(nk2e, nk1e)` to match the
    reshape convention every downstream stage uses.
    """
    ntile, off, mint = setup['ntile'], setup['off'], setup['mint']
    nb = setup['nb']
    nk1e, nk2e = ntile * nk1, ntile * nk2

    n1 = np.repeat(np.arange(nk1), nk2)          # kc = n2 + n1*nk2
    n2 = np.tile(np.arange(nk2), nk1)

    N1 = n1[:, None] + (off - mint[None, :, 0]) * nk1
    N2 = n2[:, None] + (off - mint[None, :, 1]) * nk2
    flat = (N2 + N1 * nk2e).ravel()              # (Nk*nkeep,)

    out = np.empty((nb, nk1e * nk2e))
    for ib in range(nb):
        out[ib, flat] = per_k[:, ib, :].ravel()
    return out


def assemble_extended(result, unf, setup, nk1, nk2, vb, vol_M, mode):
    """Replace the folded band structure with its extended-zone unfolding.

    The folded arrays are preserved under `*_folded`, exactly as `unfold.py`
    does for the magnetic BZ.  `nk1`/`nk2` grow by `ntile` and `vol_M` shrinks
    by `ntile**2`, so `cell_area = (2*pi)^2/(vol_M*nk1*nk2)` -- and therefore
    every orbit area -- is unchanged.
    """
    ntile = setup['ntile']
    new = dict(result)

    for key in ('E', 'Oz', 'Lz'):
        for v in ('K', 'Kp'):
            new[f'{key}_{v}_folded'] = result[f'{key}_{v}']
    new['kpoints_folded'] = result['kpoints']
    new['nk1_folded'] = nk1
    new['nk2_folded'] = nk2
    new['vol_M_folded'] = result['vol_M']

    for v in ('K', 'Kp'):
        new[f'E_{v}'] = scatter_extended(unf[v]['E'], setup, nk1, nk2) * 1e3
        new[f'Oz_{v}'] = scatter_extended(unf[v]['Oz'], setup, nk1, nk2) * 1e-20
        new[f'Lz_{v}'] = (scatter_extended(unf[v]['Lz'], setup, nk1, nk2)
                          * 1e-20 * 1e3)
        new[f'wt_{v}'] = scatter_extended(unf[v]['wmax'], setup, nk1, nk2)

    new['kpoints'] = extended_kmesh(setup, nk1, nk2, vb)
    new['nk1'] = ntile * nk1
    new['nk2'] = ntile * nk2
    new['vol_M'] = result['vol_M'] / ntile**2

    new['extended_zone'] = 1
    new['extended_ntile'] = ntile
    new['extended_mode'] = mode

    wt = np.concatenate([new[f'wt_{v}'].ravel() for v in ('K', 'Kp')])
    print(f"  Extended zone: {setup['nb']} branches on a "
          f"{ntile*nk1} x {ntile*nk2} grid ({ntile}x{ntile} moire BZs), "
          f"mode={mode}")
    print(f"  Extended zone: dominant-state weight min {wt.min():.4f}, "
          f"median {np.median(wt):.4f} (1 = one state per extended momentum; "
          f"dips at the Bragg planes are the moire gaps)")
    if np.median(wt) < 0.5:
        print("  Extended zone: WARNING - branches are strongly mixed almost "
              "everywhere, not just on the Bragg planes; the moire potential "
              "is too strong for the unfolding to be meaningful.")
    return new
