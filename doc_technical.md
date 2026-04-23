# Technical Documentation

Magnetic Bloch band solver for bilayer graphene (BLG) on hexagonal boron
nitride (hBN) in a perpendicular magnetic field.  The magnetic flux per
moire unit cell is the rational number `qq/pp`.

Reference: P. M. Haney, "A Quantum Ruler for Orbital Magnetism in Moire
Quantum Matter."

---

## 1. Physical setup

The system is a Bernal-stacked bilayer graphene sheet on an hBN substrate.
The lattice mismatch between graphene (a = 2.46 A) and hBN (a = 2.504 A)
produces a moire superlattice with period `L_moire`.  A perpendicular
magnetic field B is applied such that the flux through one moire unit cell
is the rational fraction `Phi/Phi_0 = qq/pp`.

The Hamiltonian is expressed in a Landau-level (LL) basis.  Each basis state
is labeled by:

- **Sublattice** (`A` or `B`)
- **Guiding-center index** (`q0`, `q1`, ..., `q{qq-1}`) -- there are `qq`
  guiding centers per magnetic unit cell
- **Landau-level index** (`LL0`, `LL1`, ..., `LL{N}`)

The total Hamiltonian for each valley is a 2x2 block matrix (two graphene
layers):

```
H_total = H_BLG + V_hBN

H_BLG = [ Hintra1 + U_top    Hinter        ]
        [ Hinter^dagger      Hintra2 + U_bot ]

V_hBN = [ 0           0          ]
        [ 0           V_hBN_tot  ]
```

where `V_hBN_tot` is the moire potential from the hBN substrate acting on
the bottom layer only.  The eigenvalues of `H_total` (in meV) give the
magnetic Bloch band energies.

---

## 2. Module structure

Code is split across six modules:

| Module | Purpose |
|---|---|
| `constants.py` | Physical constants (HBAR, Q_E, A_GRAPHENE, A_HBN) |
| `parser.py` | MATLAB-style input file parser |
| `numerics.py` | Mathematical routines: Laguerre polynomials, F_nm matrix elements |
| `basis.py` | Label-based basis toolkit: outer product and index lookup |
| `hamiltonian.py` | All Hamiltonian construction (intralayer, intermonolayer, interbilayer) |
| `main_v2.py` | Engine (`do_calc`), k-point solver, CLI entry point |

### `parser.py`

| Function | Purpose |
|---|---|
| `parse_input_file(filepath)` | Read MATLAB-style input file into a dict. Evaluates expressions in a shared namespace so later lines can reference earlier variables (e.g., `elist` referencing `nebin`). |
| `_eval_matlab_value(s, ns)` | Evaluate a single value expression. Handles strings, cell arrays `{...}`, numeric arrays `[...]`, and `linspace`. |

### `numerics.py`

| Function | Purpose |
|---|---|
| `lf_function(m, n, alpha, x)` | Generalized Laguerre polynomials via three-term recurrence. Returns shape `(m, n+1)`. |
| `fnm5(n, m, q, lB, laguerretable)` | LL matrix element `F_{nm}(q)` -- the form factor `<n|exp(iq.r)|m>` in the LL basis.  Uses Stirling-like approximations for large n,m to avoid factorial overflow. Requires `n >= m`.  (Legacy scalar interface; no longer called by `build_fnm_tables`.) |
| `build_fnm_tables(N, ktheta, lB, q_vectors)` | Precompute the Laguerre table and all `F_{nm}` tables for a list of momentum-transfer vectors. Returns `([Fnm_q1, Fnm_q2, Fnm_q3], LLlabels)`.  Uses vectorized log-space (`gammaln`) computation over all `(n,m)` pairs simultaneously, with a Bessel-function fallback for entries where the Laguerre recurrence overflows. |

### `basis.py`

| Function | Purpose |
|---|---|
| `outer_product(mtx1, labels1, mtx2, labels2)` | Kronecker-style block outer product with label tracking.  `result[i,j] = mtx1[i1,j1] * mtx2`.  Labels are concatenated as `"{lab1}_{lab2}_"`. |
| `getindices(labelset, labels)` | Find indices in a label list whose entries contain ALL given substrings.  Used to select basis states by sublattice/LL/guiding-center. |

### `hamiltonian.py`

| Function | Purpose |
|---|---|
| `_build_chain_matrices_K(Nq, pp, qq)` | Build guiding-center chain hopping matrices for K valley. Returns `(chain1, chain2, chain3, chainlabels)`. |
| `_build_chain_matrices_Kp(Nq, pp, qq)` | Same for K' valley (phase signs are negated). |
| `_assemble_interbilayer_terms(...)` | Combine chain matrices, F_nm tables, and sublattice hopping matrices via two nested `outer_product` calls.  Chops the spurious LL_N row/column. Returns `(term1, term2, term3, qNslabels)`. |
| `get_interbilayerterms_K(...)` | Top-level: defines q-vectors and sublattice hopping matrices `t1,t2,t3` for the K-valley hBN moire potential, then calls `_assemble_interbilayer_terms`. |
| `get_interbilayerterms_Kp(...)` | Same for K' valley. |
| `get_intermonolayerH_K(N, theta, B, labels, params)` | Inter-monolayer coupling (gamma1 constant, gamma3 raising, gamma4 lowering operators). K valley. |
| `get_intermonolayerH_Kp(N, theta, B, labels, params)` | Same for K' valley (operator directions reversed). |
| `get_intralayerH_K(N, theta, B, labels, params, delta_site)` | Intralayer kinetic Hamiltonian for K valley.  Builds upper-triangular part, then symmetrizes via `H + H^dagger`.  Includes sublattice mass `delta`. |
| `get_intralayerH_Kp(N, theta, B, labels, params, delta_site)` | Same for K' valley. |

### `main_v2.py`

| Function | Purpose |
|---|---|
| `_solve_kpoint_core(shared_dict, kpt)` | Given a k-point (2-vector), compute phase factors, build k-dependent moire potential, form total Hamiltonian, and diagonalize.  Returns `(eigenvalues_K, eigenvalues_Kp)`.  Used by both serial and parallel paths.  Operates on pre-scaled data (already multiplied by `1000/Q_E`) so no per-k-point unit conversion is needed.  Uses `eigvalsh(overwrite_a=True, check_finite=False)` to avoid internal copies. |
| `_init_kpoint_worker(shared)` | Pool initializer: stores shared matrices in module-global `_worker_shared` so they are pickled once per worker, not per task. |
| `_solve_kpoint(args)` | Pool worker entry point.  Unpacks `(kc, kpt)`, calls `_solve_kpoint_core`, returns `(kc, tek_K, tek_Kp)`. |
| `do_calc(filepath)` | Main entry point.  Reads input, computes derived quantities, builds k-independent Hamiltonians, pre-scales them to meV (`1000/Q_E`), runs k-loop (serial or parallel), post-processes into `ek` or `dos` output. |
| `main(input_file)` | CLI wrapper: calls `do_calc`, saves result to `.npz` or `.mat`. |

---

## 3. Basis construction and labeling

The basis labeling system is central to the code.  Each basis state carries a
composite string label built by nested `outer_product` calls:

```
Step 1:  outer_product(chain, chainlabels, Fnm, LLlabels)
         → labels like "q0_LL3_"

Step 2:  outer_product(t_sublattice, ['A','B'], term, qNlabels)
         → labels like "A_q0_LL3__"
```

The `getindices` function finds basis states by substring intersection.
For example, `getindices(labels, ['B', 'LL24_'])` returns all indices whose
label contains both `'B'` and `'LL24_'`.  The trailing `_` is critical to
avoid matching `LL24` against `LL240`.

**Spurious mode removal**: In each valley, one sublattice's highest LL is
"unpaired" (it has no coupling partner in the off-diagonal kinetic term).
This creates a spurious zero mode.  It is removed by deleting rows/columns:

- K valley: remove `(B, LL_N)` states
- K' valley: remove `(A, LL_N)` states

This chopping happens independently in each Hamiltonian builder and in the
`_assemble_interbilayer_terms` function.

---

## 4. Matrix dimensions

For a given `(pp, qq, N)`:

- Guiding centers per magnetic unit cell: `Nq = qq`
- Basis states per layer before chopping: `2 * qq * (N+1)` (two sublattices)
- After chopping LL_N from one sublattice: `qq*N + qq*(N+1) = qq*(2N+1)`
- Total Hamiltonian dimension: `2 * qq*(2N+1)` (two layers)

N is determined automatically:
`N = LL_multiplier * round(max(hbar*vF*ktheta, w) / eneLL)^2`, capped at
`Nmax`.

---

## 5. k-point mesh

The magnetic Brillouin zone (BZ) is `1/pp` the size of the moire BZ.
Reciprocal lattice vectors:

```
b1 = ktheta * [0, -1]
b2 = ktheta * [sqrt(3)/2, 1/2]
```

The magnetic BZ vectors are `b1/pp` and `b2/pp`.  The k-mesh is a uniform
`nk1 x nk2` grid over the magnetic BZ, flattened in column-major (Fortran)
order to match MATLAB conventions.  Each k-point is shifted by `-M_mag`
before entering the Hamiltonian.

---

## 6. Phase factors (k-dependent part)

The moire potential enters through three terms, each with a k-dependent
phase factor:

**K valley (square magnetic unit cell)**:
```
tphase1 = exp(i * (pp/qq) * kx * Lx)
tphase2 = exp(-i * (pp/qq) * kx * Lx/2) * exp(i * ky * Ly * (pp/qq))
tphase3 = exp(-i * (pp/qq) * kx * Lx/2) * exp(-i * ky * Ly * (pp/qq))
```

**K' valley**: all signs in the exponents are negated.

The full moire potential for a single layer is:
```
V_hBN_tot = v0 * I + V_pq + V_pq^dagger

V_pq = gamma * tphase1 * term1 + tphase2 * term2 + tphase3 * term3
```

This is placed in the bottom-right block of the 2-layer Hamiltonian (hBN
substrate acts on the bottom layer only).

---

## 7. Parallelization

The k-loop is embarrassingly parallel.  When `isparallel = 1`:

1. Shared data (k-independent Hamiltonians + moire coupling matrices) is
   packed into a dict.
2. A `multiprocessing.Pool` is created with an initializer that copies the
   shared dict into each worker's global namespace (pickled once per worker,
   not per task).
3. `pool.map` distributes `(kc, kpt)` tuples across workers.
4. Results `(kc, tek_K, tek_Kp)` are collected and assembled into the bands
   arrays.

Worker count defaults to `min(cpu_count(), Nk_tot)`.

---

## 8. Output modes

Controlled by the `calctype` input parameter.  All output dicts include
a `'params'` key containing the full input parameter dict.

### `calctype = 'ek'`

Returns the full band structure: eigenvalues at every k-point.

```python
result = {
    'calctype': 'ek',
    'params':   dict,                       # all input file parameters
    'kpoints':  ndarray (Nk_tot, 2),        # k-points in 1/m
    'bands_K':  ndarray (Nk_tot, 2*dim1),   # sorted eigenvalues in meV
    'bands_Kp': ndarray (Nk_tot, 2*dim1),
}
```

### `calctype = 'dos'`

Bins eigenvalues into an energy histogram (density of states).

```python
result = {
    'calctype': 'dos',
    'params':  dict,                # all input file parameters
    'elist':   ndarray (nebin,),    # energy grid in meV
    'dos_K':   ndarray (nebin,),    # state count per bin, K valley
    'dos_Kp':  ndarray (nebin,),    # state count per bin, K' valley
}
```

The legacy MATLAB value `calctype = 'spectrum'` is mapped to `'dos'`.

### File output

The `main` function saves results to disk.  Output format is determined
by the `outputfile` input parameter extension:

- **`.npz`**: input params stored with `input_` prefix (e.g., `input_pp`)
- **`.mat`**: input params stored as a `params` struct (e.g., `params.pp`)

If `outputfile` is not set, defaults to `bands_p{pp}_q{qq}.npz`.

---

## 9. Physical parameters (all in input file)

| Parameter | Units (input) | Internal conversion | Role |
|---|---|---|---|
| `g0` | meV | `g0/1e3 * Q_E` → J | Intralayer nearest-neighbor hopping |
| `g1` | meV | same | Interlayer A-B coupling (gamma1) |
| `g3` | meV | same | Interlayer trigonal warping (gamma3) |
| `g4` | meV | same | Interlayer electron-hole asymmetry (gamma4) |
| `delta` | meV | same | Sublattice mass term |
| `v0` | meV | same | hBN moire potential: on-site (uniform) |
| `v1` | meV | same | hBN moire potential: spatially modulated |
| `w` | meV | same | Interlayer coupling scale (used for LL cutoff estimate) |
| `U` | meV | same | Layer-dependent on-site energies `[U_top, U_bottom]` |
| `theta` | radians | -- | Twist angle (0 for aligned BLG/hBN) |
| `eta` | dimensionless | -- | AA/AB hopping ratio (legacy; not active for hBN) |

---

## 10. Key physical constants (hardcoded)

```
HBAR       = 1.05e-34    J.s
Q_E        = 1.6e-19     C
A_GRAPHENE = 2.46e-10    m
A_HBN      = 2.504e-10   m
vF         = 1e6         m/s  (Fermi velocity, default)
psi        = -0.29       rad  (hBN moire coupling phase)
```

---

## 11. Files

| File | Role |
|---|---|
| `main_v2.py` | Engine: `do_calc`, k-point solver, CLI entry point |
| `hamiltonian.py` | Hamiltonian construction (intralayer, intermonolayer, interbilayer) |
| `numerics.py` | Laguerre polynomials, F_nm matrix elements, table builder |
| `basis.py` | `outer_product`, `getindices` |
| `parser.py` | MATLAB-style input file parser |
| `constants.py` | Physical constants |
| `input_test.txt` | Example input file (pp=1, qq=1) |
| `input_p3_q1.txt` | Example input file (pp=3, qq=1) |
| `validate.py` | Compares Python output against MATLAB `.mat` benchmarks |
| `bands_p{pp}_q{qq}.mat` | MATLAB benchmark data |

---

## 12. Performance characteristics

The eigenvalue solver (`scipy.linalg.eigvalsh`, backed by LAPACK) dominates
runtime for large matrices.  At the typical scale of pp=15, qq=1 (matrix
size 1178x1178), ~97% of wall time is spent in LAPACK.

### Optimizations in place

- **Vectorized F_nm table construction**: `build_fnm_tables` computes all
  matrix elements in a single vectorized pass using log-space arithmetic
  (`gammaln`) instead of ~N^2/2 scalar Python calls.  The Laguerre
  recurrence is also vectorized over all alpha values simultaneously.
  These two changes reduced Hamiltonian construction from ~1.3s to ~0.01s
  for N=294.

- **Pre-scaled k-loop**: The k-independent Hamiltonian (`H_BLG`) and moire
  coupling terms are pre-multiplied by `1000/Q_E` before the k-loop.
  This eliminates redundant per-k-point scaling and temporary array
  allocations.  The k-point solver operates entirely in meV.

- **Eigenvalue solver flags**: `overwrite_a=True` avoids an internal matrix
  copy in LAPACK; `check_finite=False` skips the NaN/inf scan.

### Scaling bottleneck

The LAPACK eigensolve is O(n^3) and cannot be reduced by computing a
subset of eigenvalues -- `subset_by_index` still performs the full
tridiagonal reduction and provides no speedup at the matrix sizes used
here (tested up to 1178x1178).  For production runs with many k-points,
set `isparallel = 1` to distribute the k-loop across CPU cores; this
provides near-linear speedup since each k-point is independent.

---

## 13. Known subtleties

- **Fortran-order flattening**: The k-mesh grid is flattened with `order='F'`
  to match MATLAB's column-major `reshape`.  Changing this breaks benchmark
  agreement.

- **elist evaluation**: The input parser uses a shared namespace so that
  `elist = linspace(-300,300, nebin)` can reference the `nebin` variable
  defined on a previous line.

- **Eigenvalue solver**: Uses `scipy.linalg.eigvalsh` (Hermitian), not
  `eig`.  The Hamiltonian is Hermitian by construction.

- **Unit conversion**: All tight-binding parameters enter in meV but are
  converted to Joules for Hamiltonian construction.  The k-independent
  Hamiltonian and moire terms are pre-scaled to meV before the k-loop;
  the k-point solver works entirely in meV.
