# Technical Documentation

Moire band structure solvers for mono- or bilayer graphene on hexagonal
boron nitride (hBN).  Two calculation modes:

1. **Hofstadter** (`main_v3.py`): Magnetic Bloch bands in a Landau-level
   basis at rational magnetic flux `qq/pp` per moire unit cell.
   (`main_v2.py` is the legacy driver with old T-matrix conventions.)
2. **Zero-field** (`zerofield.py`): Moire band structure via plane-wave
   expansion (no magnetic field), computed along a k-path through the
   moire Brillouin zone.

Reference: P. M. Haney, "A Quantum Ruler for Orbital Magnetism in Moire
Quantum Matter."

# Part I: Hofstadter (magnetic field)

---

## 1. Physical setup

The system is a graphene sheet (monolayer or Bernal-stacked bilayer) on an
hBN substrate.  The lattice mismatch between graphene (a = 2.46 A) and hBN
(a = 2.504 A) produces a moire superlattice with period `L_moire`.  A
perpendicular magnetic field B is applied such that the flux through one
moire unit cell is the rational fraction `Phi/Phi_0 = qq/pp`.

The number of graphene layers is controlled by the input parameter `nlayers`
(default 2).

The Hamiltonian is expressed in a Landau-level (LL) basis.  Each basis state
is labeled by:

- **Sublattice** (`A` or `B`)
- **Guiding-center index** (`q0`, `q1`, ..., `q{qq-1}`) -- there are
  `qq` guiding centers per magnetic unit cell
- **Landau-level index** (`LL0`, `LL1`, ..., `LL{N}`)

### Bilayer (`nlayers = 2`)

The total Hamiltonian for each valley is a 2x2 block matrix (two graphene
layers).  Two stacking configurations are supported, controlled by the
`stacking_type` input parameter (default 2).  These correspond to the two
AB-stacking geometries in Moon & Koshino, PRB 90, 155406 (2014):

**Type 2 (A1-B2, default)** — Eq. B1:
```
H_BLG = [ Hintra1 + U_top    Hinter        ]
        [ Hinter^dagger      Hintra2 + U_bot ]
```

**Type 1 (B1-A2)** — Eq. 25:
```
H_BLG = [ Hintra1 + U_top    Hinter^dagger ]
        [ Hinter             Hintra2 + U_bot ]
```

The off-diagonal interlayer blocks are swapped between the two types.
The spectra are similar but not identical, with small differences at the
BZ boundary (see Fig. 10 of the reference).

The full Hamiltonian is:
```
H_total = H_BLG + V_hBN

V_hBN = [ 0           0          ]
        [ 0           V_hBN_tot  ]
```

where `V_hBN_tot` is the moire potential from the hBN substrate acting on
the bottom layer only.

### Monolayer (`nlayers = 1`)

The Hamiltonian is a single block:

```
H_total = Hintra + U + V_hBN_tot
```

The intralayer kinetic term and the hBN moire potential act on the same
(only) graphene layer.  Inter-monolayer coupling parameters (`g1`, `g3`,
`g4`) are not used.  The sublattice mass `delta` is forced to zero.

### Eigenvalues

The eigenvalues of `H_total` (in meV) give the magnetic Bloch band
energies.

---

## 2. Module structure (Hofstadter)

Code is split across six modules:

| Module | Purpose |
|---|---|
| `constants.py` | Physical constants (HBAR, Q_E, A_GRAPHENE, A_HBN) |
| `parser.py` | MATLAB-style input file parser |
| `numerics.py` | Mathematical routines: Laguerre polynomials, F_nm matrix elements |
| `basis.py` | Label-based basis toolkit: outer product and index lookup |
| `hamiltonian.py` | All Hamiltonian construction (intralayer, intermonolayer, interbilayer, testing variants) |
| `main_v3.py` | Production engine with doubled unit cell and corrected moire coupling (Nq=qq) |
| `main_v2.py` | Legacy engine (single chain, old moire coupling conventions) |
| `hofstadter_testing.py` | Butterfly sweep engine with configurable T-matrix conventions |

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
| `_compute_RR(theta)` | Compute the 2×2 diagonal unitary rotation matrix `RR` for the T-matrix rotation at finite twist angle.  Derives `thetaT` from the direction of the first moire reciprocal lattice vector G1 at the given `theta`. |
| `get_interbilayerterms_K(...)` | Top-level: defines q-vectors and sublattice hopping matrices `t1,t2,t3` for the K-valley hBN moire potential, then calls `_assemble_interbilayer_terms`.  Uses corrected conventions: `w = exp(-i*2pi/3)`, `psi = +0.29`, order `[3,1,2]`.  When `theta != 0`, applies `inv(RR) @ T @ RR` to each T-matrix to compensate for the rotated moire pattern while keeping q-vectors canonical. |
| `get_interbilayerterms_Kp(...)` | Same for K' valley (negated q-vectors, conjugated T matrices, same RR rotation). |
| `get_interbilayerterms_K_testing(...)` | Parameterized version with configurable `order`, `sxflag`, `dagger`, `conj_flag`, `psi_conj` flags.  Used by `hofstadter_testing.py` for convention exploration. |
| `get_interbilayerterms_Kp_testing(...)` | Same for K' valley. |
| `get_intermonolayerH_K(N, theta, B, labels, params)` | Inter-monolayer coupling (gamma1 constant, gamma3 raising, gamma4 lowering operators). K valley. |
| `get_intermonolayerH_Kp(N, theta, B, labels, params)` | Same for K' valley (operator directions reversed). |
| `get_intralayerH_K(N, theta, B, labels, params, delta_site)` | Intralayer kinetic Hamiltonian for K valley.  Builds upper-triangular part, then symmetrizes via `H + H^dagger`.  Includes sublattice mass `delta`. |
| `get_intralayerH_Kp(N, theta, B, labels, params, delta_site)` | Same for K' valley. |

### `main_v3.py` (production driver)

| Function | Purpose |
|---|---|
| `_solve_kpoint_core(shared_dict, kpt)` | Given a k-point (2-vector), compute phase factors, build k-dependent moire potential, form total Hamiltonian, and diagonalize.  Returns `(eigenvalues_K, eigenvalues_Kp)`.  Used by both serial and parallel paths.  Operates on pre-scaled data (already multiplied by `1000/Q_E`) so no per-k-point unit conversion is needed.  Uses `eigvalsh(overwrite_a=True, check_finite=False)` to avoid internal copies. |
| `_init_kpoint_worker(shared)` | Pool initializer: stores shared matrices in module-global `_worker_shared` so they are pickled once per worker, not per task. |
| `_solve_kpoint(args)` | Pool worker entry point.  Unpacks `(kc, kpt)`, calls `_solve_kpoint_core`, returns `(kc, tek_K, tek_Kp)`. |
| `do_calc(filepath)` | Main entry point.  Reads input, computes derived quantities, builds k-independent Hamiltonians (monolayer or bilayer depending on `nlayers`) using `Nq = qq` with doubled unit cell, pre-scales them to meV (`1000/Q_E`), runs k-loop (serial or parallel), post-processes into `ek` or `dos` output. |
| `main(input_file)` | CLI wrapper: calls `do_calc`, saves result to `.npz` or `.mat`. |

Differences from `main_v2.py` (legacy):

| Feature | main_v2 | main_v3 |
|---|---|---|
| Chain size | `Nq = qq` | `Nq = qq` |
| BZ vectors | `[b1/pp, b2*qq/pp]` | `[b1/pp, b2*qq/pp]` |
| T matrices | Old conventions | Corrected (order=[3,1,2], conj=1, psiconj=1) |
| BLAS threads | `OPENBLAS_NUM_THREADS=1` | `OPENBLAS_NUM_THREADS=1` |
| U identity | `np.eye(dim1)` | `np.eye(Hintra.shape[0])` (post-chop) |

### `hofstadter_testing.py` (convention exploration)

Sweeps over rational flux values `qq/pp` up to `ppmax`, parallelized across
flux points.  Supports configurable T-matrix conventions via CLI flags:

```
python hofstadter_testing.py input_testing.txt \
    --order 3 1 2 --sxflag 0 --dagger 0 \
    --conj_flag 1 --psi_conj 1 --ppmax 10
```

Features incremental `.mat` checkpoint/resume: saves after each flux point
and skips already-computed `qq/pp` pairs on restart.

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

For a given `(pp, qq, N)` with `main_v3.py`:

- Guiding centers per magnetic unit cell: `Nq = qq`
- Basis states per layer before chopping: `2 * qq * (N+1)` (two sublattices)
- After chopping LL_N from one sublattice: `qq*(2N+1)`
- Total Hamiltonian dimension: `nlayers * qq*(2N+1)`

Both `main_v3.py` and `main_v2.py` use `Nq = qq`; they differ in
T-matrix conventions, BZ vectors, and the real-space unit cell definition.

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

Both `main_v3.py` and `main_v2.py` use magnetic BZ vectors `b1/pp` and
`b2*qq/pp`.  The k-mesh is a uniform `nk1 x nk2` grid over this BZ,
flattened in column-major (Fortran) order to match MATLAB conventions.
Each k-point is shifted by `-M_mag` before entering the Hamiltonian.

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

For bilayer, this is placed in the bottom-right block of the 2-layer
Hamiltonian (hBN substrate acts on the bottom layer only).  For monolayer,
it is added directly to the single-layer Hamiltonian.

### T-matrix rotation at finite twist angle

When the twist angle `theta != 0`, the physical moire reciprocal lattice
vectors rotate away from the canonical directions.  However, the
guiding-center chain matrices are always constructed with canonical
(unrotated) q-vectors: `q1 = ktheta * [0, -1]`, etc.  To keep the
T-matrices consistent with these fixed q-vectors, each T-matrix is
conjugated by a sublattice rotation matrix `RR`:

```
T_i → inv(RR) @ T_i @ RR
```

The rotation angle `thetaT` is derived from the actual (rotated) first
moire reciprocal lattice vector G1:

```
thetaT = (-pi/2 - atan2(G1_y, G1_x)) / 2
RR = diag(exp(-i*thetaT), exp(+i*thetaT))
```

At `theta = 0`, G1 points along `[0, -1]`, giving `thetaT = 0` and
`RR = I` (no rotation).  The same rotation is applied in both the
Hofstadter code (`hamiltonian.py`) and the semiclassical code
(`bandstructure.py`).

The input parameter `theta` is specified in **degrees** and converted
to radians internally.

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
| `g1` | meV | same | Interlayer A-B coupling (gamma1). Bilayer only. |
| `g3` | meV | same | Interlayer trigonal warping (gamma3). Bilayer only. |
| `g4` | meV | same | Interlayer electron-hole asymmetry (gamma4). Bilayer only. |
| `delta` | meV | same | Sublattice mass term. Bilayer only (forced to 0 for monolayer). |
| `v0` | meV | same | hBN moire potential: on-site (uniform) |
| `v1` | meV | same | hBN moire potential: spatially modulated |
| `w` | meV | same | Interlayer coupling scale (used for LL cutoff estimate) |
| `nlayers` | int | -- | Number of graphene layers: 1 (monolayer) or 2 (bilayer, default) |
| `U` | meV | same | Layer on-site energies: scalar for monolayer, `[U_top, U_bottom]` for bilayer |
| `theta` | degrees (input) | radians (internal) | Twist angle between graphene and hBN |
| `eta` | dimensionless | -- | AA/AB hopping ratio (legacy; not active for hBN) |

---

## 10. Key physical constants (hardcoded)

```
HBAR       = 1.05e-34    J.s
Q_E        = 1.6e-19     C
A_GRAPHENE = 2.46e-10    m
A_HBN      = 2.504e-10   m
vF         = 1e6         m/s  (Fermi velocity, default)
psi        = +0.29       rad  (hBN moire coupling phase, default)
psi_swap   = +2.386      rad  (hBN B/N swapped, from Eq. 18 with V_N <-> V_B)
w          = exp(-i*2pi/3)    (sublattice phase rotation)
V_N        = -1.40       eV   (nitrogen second-order potential, Moon & Koshino)
V_B        = +3.34       eV   (boron second-order potential, Moon & Koshino)
```

The `hbn_swap` parameter (default 0) recomputes psi from Moon & Koshino
PRB 87, 205404 (2013), Eq. 18 with V_N and V_B interchanged.  This models
a 60°-rotated hBN alignment where boron and nitrogen sublattice positions
are swapped relative to the graphene.  The scalar potential V0 and
modulation amplitude |V1| are unchanged (symmetric under the swap), but
the phase psi shifts from +0.29 to +2.386 rad.  This flips the sign of
the effective sublattice mass M^eff, moving the n=0,1 Landau levels from
one side of the gap to the other.

---

## 11. Files

| File | Role |
|---|---|
| `main_v3.py` | Production Hofstadter engine (doubled unit cell, Nq=qq, corrected coupling) |
| `main_v2.py` | Legacy Hofstadter engine (single chain, old coupling conventions) |
| `hofstadter_testing.py` | Butterfly sweep engine with configurable T-matrix conventions |
| `hamiltonian.py` | Hamiltonian construction (intralayer, intermonolayer, interbilayer, testing variants) |
| `numerics.py` | Laguerre polynomials, F_nm matrix elements, table builder |
| `basis.py` | `outer_product`, `getindices` |
| `parser.py` | MATLAB-style input file parser (shared) |
| `constants.py` | Physical constants (shared) |
| `zerofield.py` | Zero-field engine: moire geometry, plane-wave Hamiltonian, k-path solver |
| `input_test.txt` | Example Hofstadter input (pp=1, qq=1) |
| `input_p3_q1.txt` | Example Hofstadter input (pp=3, qq=1) |
| `input_testing.txt` | Testing input (monolayer, convention exploration) |
| `input_testing_BG.txt` | Testing input (bilayer, convention exploration) |
| `input_zerofield.txt` | Example zero-field input |
| `validate.py` | Legacy Hofstadter benchmark comparison (main_v2) |
| `validate_testing_BG.py` | Bilayer testing benchmark comparison against MATLAB |
| `validate_zerofield.py` | Zero-field benchmark comparison |
| `plot_zerofield.m` | MATLAB plotting script for zero-field bands |
| `bands_p{pp}_q{qq}.mat` | Hofstadter MATLAB benchmark data (legacy) |
| `matlab_code/zerofield/` | Original MATLAB zero-field code and benchmark (`bands_BG.mat`) |
| `matlab_debugging/` | MATLAB debugging/benchmark scripts for convention testing |

---

## 12. Performance characteristics

The eigenvalue solver (`scipy.linalg.eigvalsh`, backed by LAPACK) dominates
runtime for large matrices.  At the typical scale of pp=15, qq=1 (matrix
size 1178x1178), ~97% of wall time is spent in LAPACK.

### Optimizations in place

- **BLAS thread pinning**: `OPENBLAS_NUM_THREADS=1` is set before numpy
  import in both `main_v3.py` and `hofstadter_testing.py`.  OpenBLAS
  eigvalsh scales poorly beyond 4 threads for the matrix sizes used here
  (1960x1960 benchmark: 1 thread = 1.16s, 4 threads = 0.63s, 28 threads
  = 1.45s).  With multiprocessing pools, the default (all cores per BLAS
  call) causes catastrophic thread oversubscription.  Pinning to 1 BLAS
  thread and using pool workers for parallelism gives best throughput.

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

---

## 14. Transport coefficients (`calctype = 'transport'`)

Computes the linear-response Kubo-formula transport coefficients
sigma_xx, sigma_xy, L12_xx, L12_xy as functions of chemical potential mu.

### Velocity operator

The velocity is `v = (i/hbar) [A, H_base]`, where A is the berry
connection (position operator in the LL basis) and H_base is the
k-independent BLG kinetic Hamiltonian.  The moire potential V is local
in position, so `[A, V] = 0` and V does not contribute to the velocity.
The velocity is therefore k-independent and computed once.

This is the same construction used in `semiclassical/hofstadter_system.py`.
The berry connection matrices are from `hamiltonian.get_berry_connection_K/Kp`.
For bilayer, A is block-diagonal: `A_full = diag(A, A)`.

Units: A in Angstrom, H_base in eV, hbar in eV*s → velocity in Ang/s.

### Per-k-point computation

At each k-point, the full Hamiltonian H(k) (in meV) is diagonalized with
`eigh` to obtain eigenvalues E_n and eigenvectors Psi.  Velocity matrix
elements are computed in the eigenbasis:

```
vx_nm = Psi_sel^dag @ Vx @ Psi_sel
vy_nm = Psi_sel^dag @ Vy @ Psi_sel
```

where `Psi_sel` selects `nbands_transport` bands around charge neutrality
(or all bands if `nbands_transport = 0`).  The eigenvalues are histogrammed into a DOS on the mulist energy grid
(same binning as `calctype = 'dos'`) at no extra cost, since the
diagonalization is already required for the transport calculation.

### Kubo formulas

With `D_nm = E_n - E_m`, `G = Gamma` (broadening), `f_n` the occupation
(Fermi-Dirac at temperature kT, or step function at kT=0):

All four transport coefficients (sigma_xx, sigma_xy, L12_xx, L12_xy) are
thermal convolutions of spectral functions Phi(eps):

```
sigma(mu) = pf * int deps (-df/deps) Phi(eps)
L12(mu)   = pf * int deps (-df/deps) (eps - mu) Phi(eps)
```

**Phi_xx** (longitudinal spectral conductivity):

```
Phi_xx(eps) = sum_{k,n,m} |vx_nm|^2 * A_n(eps) * A_m(eps)
```

where `A_n(eps) = 1 / [(eps - E_n)^2 + G^2]` is the spectral weight of
eigenstate n at energy eps.  Includes the n=m intraband (Drude)
contribution.  Manifestly non-negative and vanishes in spectral gaps.

**Phi_xy** (Hall spectral function):

```
Phi_xy(eps) = sum_{k, E_n < eps} K_n
K_n = sum_{m!=n} Im[vx_nm * conj(vy_nm)] / (D_nm^2 + G^2)
```

Cumulative sum of per-eigenstate Berry curvature kernels.  In the
Gamma → 0 limit the broadened Berry curvature per band is
`Omega_n = -2*hbar^2 * sum_{m!=n} Im[vx_nm * conj(vy_nm)] / D_nm^2`,
and the Chern number is `C_n = (1/2pi) * int Omega_n d^2k`.

Both Phi are computed once per k-point on a fine energy grid spanning
all mu values.  The per-mu integrals are cheap 1D quadratures (trapz).
At kT=0, `(-df/deps) = delta(eps-mu)`, so `sigma = Phi(mu)` and
`L12 = 0`.  Fast paths are used in this limit: `L @ vx_sq @ L` for
sigma_xx, `sum K_n[E_n < mu]` for sigma_xy.

### Prefactor derivation

The prefactor for sigma_xy is derived from the Kubo formula and the
relation between the magnetic BZ integral and the k-mesh sum:

```
sigma_xy / (e^2/h) = pp * sum_n f_n * C_n
C_n = (1/2pi) * (A_MBZ / Nk) * sum_k Omega_n(k)
    = (2*pi / (A_m_Ang2 * Nk)) * sum_k Omega_n(k)
```

using `A_MBZ = (2*pi)^2 / A_m_Ang2` (via 1e-20 m^2/Ang^2 conversion).

### BZ folding factor

The magnetic BZ is 1/pp of the full moire BZ.  The k-sum covers only
the magnetic BZ, so the physical conductivity is pp times the BZ-averaged
value.  This factor of pp is included in the prefactor.

### Units

- A_m: magnetic unit cell area = pp^2 * uc_area / (2*qq), in Ang^2
- Eigenvalues from k-loop: meV, converted to eV for the formulas
- Output sigma: dimensionless (units of e^2/h)
- Output L12: units of e^2/h * eV

### Reference subtraction (`mu_ref`)

The truncated LL basis produces unphysical Berry curvature in deep
remote bands.  The absolute cumulative Chern number (summed over all
occupied bands) therefore has a systematic error.  However, per-band
Chern numbers for bands near charge neutrality are individually
quantized to high precision.

When `mu_ref` is set (in meV), the code computes sigma_xy at the
reference chemical potential and subtracts it:

```
sigma_xy_out(mu) = sigma_xy_raw(mu) - sigma_xy_raw(mu_ref)
```

Place `mu_ref` in a spectral gap.  Then sigma_xy in any other gap gives
the integer Chern number measured relative to that reference, with the
unphysical remote-band contribution cancelling exactly.  L12_xy is
similarly referenced.  sigma_xx and L12_xx are not referenced (they are
already well-behaved).

### Gamma → 0 limit

sigma_xy reduces to the TKNN/Chern number formula.  Quantized plateaus
of sigma_xy in spectral gaps are a good numerical check.

---
---

# Part II: Zero-field (no magnetic field)

## 14. Physical setup

The same graphene/hBN system as the Hofstadter calculation, but with no
magnetic field.  The Hamiltonian is expressed in a plane-wave basis: each
state is labeled by (sublattice, Q-vector), where Q runs over moire
reciprocal lattice vectors.

The basis set is a grid of `NQ x NQ` moire reciprocal lattice vectors
`Q = p*q1 + r*q2` with `p, r in {-NQ//2, ..., NQ//2}`, sorted by norm.
The number of retained plane waves is `NG = NQ^2`.

### Bilayer (`nlayers = 2`)

Two stacking configurations, selected by `stacking_type` (default 2):

**Type 2 (default):**
```
H = [ H_top(k)      U_BLG(k)^dag ]
    [ U_BLG(k)       H_bot(k) + H_hopp ]
```

**Type 1:**
```
H = [ H_top(k)      U_BLG(k)     ]
    [ U_BLG(k)^dag   H_bot(k) + H_hopp ]
```

- `H_top/H_bot`: block-diagonal Dirac Hamiltonians, one 2x2 block per
  Q-vector: `-hbar_vF * (k - Q) . sigma + U_{top/bot} * I`
- `U_BLG`: interlayer coupling (gamma1 dimer + v3 trigonal warping)
- `H_hopp`: k-independent hBN moire potential (acts on bottom layer only)
- Dimension: `4 * NG`

### Monolayer (`nlayers = 1`)

```
H = H_intra(k) + H_hopp + U * I
```

Single Dirac cone plus moire potential.  Dimension: `2 * NG`.

### Eigenvalues

The eigenvalues of H (in eV) give the moire band energies at each k-point.

---

## 15. Module structure (zero-field)

The zero-field solver is self-contained in `zerofield.py`, reusing only
`parser.py` and `constants.py` from the Hofstadter code.

| Function | Purpose |
|---|---|
| `_compute_moire_vectors(theta, a, a_hBN)` | Compute moire reciprocal lattice vectors q1, q2, q3 from lattice constants and twist angle.  Uses 3D cross-product formulation matching the MATLAB code. |
| `_build_qvectors(NQ, q1, q2)` | Build integer-centered Q-vector grid `Q = p*q1 + r*q2`, sort by norm.  Returns Q array, integer indices, and NG. |
| `_build_coupling_matrices_K(V0, V1)` | hBN moire coupling T-matrices for K valley: T0 (uniform), T1/T2/T3 (modulated).  Phase convention uses `psi = -0.29`. |
| `_build_coupling_matrices_Kp(V0, V1)` | Same for K' valley (conjugate phases, permuted sublattice factors). |
| `_build_moire_hopping(Q_idx, NG, T0, T1, T2, T3, valley)` | Assemble the k-independent moire hopping matrix `H_hopp` (2*NG x 2*NG) using integer-index Kronecker deltas on Q-vector differences.  Valley-dependent sign convention on the q-vectors. |
| `_solve_kpath_K(...)` | Build and diagonalize the full Hamiltonian at each k-point for K valley.  Handles both monolayer and bilayer. |
| `_solve_kpath_Kp(...)` | Same for K' valley (sign flip on sigma_x in the Dirac term and interlayer coupling). |
| `_make_kpath(q1, q2, dk, valley)` | Build high-symmetry k-path through the moire BZ.  K valley: K1->Gamma->K2->K1.  K' valley: K2->K1->Gamma->K2.  Returns k-points, linearized parameter, tick positions, and tick labels. |
| `do_calc(filepath)` | Main entry point: read input, compute geometry, build coupling matrices, solve along k-path for each valley. |
| `main(input_file)` | CLI wrapper: calls `do_calc`, saves to `.npz` or `.mat`. |

---

## 16. Moire geometry

Moire real-space lattice vectors are computed from the lattice mismatch
and twist angle:

```
M = (I - (a/a_hBN) * R^(-1))^(-1) * a_graphene
```

where `R` is the rotation matrix for twist angle `theta`.  Reciprocal
vectors are obtained via 3D cross products (z-component discarded):

```
q1 = 2*pi * (M2 x M3) / vol     (xy components)
q2 = 2*pi * (M3 x M1) / vol
q3 = -q1 - q2
```

The Q-vector grid uses integer offsets: `Q = p*q1 + r*q2` with
`p, r in {-NQ//2, ..., NQ//2}`.  This is equivalent to the MATLAB code's
half-integer grid with Dq offset, since `Dq = -(q1+q2)/2` exactly cancels
the half-integer shift.

---

## 17. Moire coupling (hBN potential)

The hBN moire potential couples plane-wave states whose Q-vectors differ
by q1, q2, or q3.  For each pair (Q_j, Q_k):

**K valley**:

| Q_j - Q_k | Contribution |
|---|---|
| 0 | T0 (uniform on-site) |
| +q1 | T1^dag |
| -q1 | T1 |
| +q2 | T2^dag |
| -q2 | T2 |
| +q3 | T3^dag |
| -q3 | T3 |

**K' valley**: the q-vector signs are negated (replace +q with -q).

The coupling matrices are:

```
w = exp(i * 2*pi/3)

K valley:
  T0 = V0 * I
  T1 = V1 * exp(i*psi) * [1, w^-1; 1, w^-1]
  T2 = V1 * exp(i*psi) * [1, w; w, w^-1]
  T3 = V1 * exp(i*psi) * [1, 1; w^-1, w^-1]

K' valley:
  T0 = V0 * I
  T1 = V1 * exp(-i*psi) * [1, w; 1, w]
  T2 = V1 * exp(-i*psi) * [1, w^-1; w^-1, w]
  T3 = V1 * exp(-i*psi) * [1, 1; w, w]
```

Using integer indices for Q-vectors, the Kronecker deltas become exact
integer comparisons (no floating-point tolerance needed).

---

## 18. Intralayer and interlayer Hamiltonians

**Intralayer (Dirac)**: block-diagonal, one 2x2 block per Q-vector:

```
K valley:   h_j = -hbar_vF * [(kx - Qx)*sigma_x + (ky - Qy)*sigma_y]
K' valley:  h_j = -hbar_vF * [-(kx - Qx)*sigma_x + (ky - Qy)*sigma_y]
```

**Interlayer (bilayer only)**: block-diagonal, one 2x2 block per Q-vector:

```
K valley:   U_j = gamma1 * [0,1;0,0] - hbar_v3 * [(kx-Qx) - i(ky-Qy)] * [0,0;1,0]
K' valley:  U_j = gamma1 * [0,1;0,0] - hbar_v3 * [-(kx-Qx) - i(ky-Qy)] * [0,0;1,0]
```

Parameters:
- `hbar_vF = sqrt(3)/2 * g0 * a` (eV*A), overridable via `hbar_vF` input
- `gamma1 = g1/1000` (eV)
- `hbar_v3 = sqrt(3)/2 * g3 * a` (eV*A)

---

## 19. k-path

The k-path traces high-symmetry lines through the moire BZ:

```
K1 = (1/3)*q1 + (2/3)*q2
K2 = (2/3)*q1 + (1/3)*q2
G  = (0, 0)

K valley:   K1 -> G -> K2 -> K1
K' valley:  K2 -> K1 -> G -> K2
```

Each segment has `round(|segment_length| / dk)` k-points.  The linearized
`k_region` parameter runs from 0 to 1 over all segments (matching MATLAB's
`linspace(0, 1, NT)` convention).  Segment boundaries produce duplicate
k-points (intentional, matching MATLAB).

---

## 20. Unit conventions (zero-field)

| Stage | Units |
|---|---|
| Input parameters (g0, g1, v0, v1, U) | meV |
| Internal Hamiltonian | eV (and Angstroms for momentum) |
| Output eigenvalues | eV |

The conversion: `hbar_vF (eV*A) = sqrt(3)/2 * g0(meV)/1000 * a(A)`.
Plot in meV by multiplying eigenvalues by 1000.

---

## 21. Files (zero-field)

| File | Role |
|---|---|
| `zerofield.py` | Zero-field engine: geometry, Hamiltonian, k-path solver |
| `input_zerofield.txt` | Default input file |
| `validate_zerofield.py` | Benchmark comparison against MATLAB `bands_BG.mat` |
| `plot_zerofield.m` | MATLAB plotting script |
| `matlab_code/zerofield/` | Original MATLAB code and benchmark data |
