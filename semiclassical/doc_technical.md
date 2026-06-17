# Semiclassical Code — Technical Reference

## Architecture

### Stage dispatcher
```
semiclassical.py          # stage-dispatch driver
  load_data                →  load .mat/.npz with MATLAB dimension handling
  run_bandstructure        →  calls do_calc, augments with nk1/nk2
  run_isoenergy            →  calls get_energy_resolved_data for K/Kp
  run_onsager              →  calls onsager_fan for K/Kp, optionally loads chi
  __main__                 →  calctype dispatch (bandstructure/isoenergy/onsager/all)
```

### Band structure engine (zero-field)
```
bandstructure.py          # moire Hamiltonian, Berry curvature, orbital moment
  compute_moire_geometry   →  q1, q2, q3, vol_M, vb, G1_xy
  build_qvectors           →  Q (NG×2), NG
  construct_hopping         →  H_hopp_K, H_hopp_Kp  (2NG × 2NG)  [applies T-matrix rotation when G1_xy given]
  assemble_H_V_K / _Kp     →  H, Vx, Vy  (numwann × numwann)
  _kpoint_worker            →  per-k eigensolve + Berry curvature + orbital moment
  do_calc                   →  orchestrates k-loop, collects results, unit converts
```

### Hofstadter mode
```
hofstadter_system.py      # Hofstadter H/V setup and per-k-point assembly
  build_hofstadter_setup    →  dict with H_base, term1/2/3, Ax/Ay, k-mesh, indexing
  assemble_H_V_K            →  H (eV), Vx, Vy (Ang/s) at one k-point (K valley)
  assemble_H_V_Kp           →  H (eV), Vx, Vy (Ang/s) at one k-point (K' valley)

bandstructure.py          # mode branching
  _kpoint_worker_hofstadter →  per-k eigensolve + Berry curvature (no susceptibility)
  _do_calc_hofstadter       →  orchestrates Hofstadter k-loop
  do_calc                   →  branches on qq: if qq>0 → Hofstadter, else → zero-field
```

### Susceptibility (standalone)
```
susceptibility.py         # standalone Fukuyama susceptibility executable
  _chi_worker              →  per-k-point chi calculation
  do_calc_chi              →  orchestrates chi k-loop
  __main__                 →  runs do_calc_chi, saves dChi_dE_K/Kp
```

### Shared modules
```
isoenergy.py              # orbit detection (post-processing)
onsager.py                # Onsager quantization
validate.py               # zero-field benchmark comparison
validate_hofstadter.py    # Hofstadter benchmark comparison
```

## Lattice vector conventions

Uses MATLAB semiclassical conventions (NOT the same as `zerofield.py`):
```
a1 = a * (1/2, -sqrt(3)/2, 0)
a2 = a * (1, 0, 0)
```
versus `zerofield.py` which uses `a1 = a*(1,0,0)`, `a2 = a*(1/2, sqrt(3)/2, 0)`.

These differ by a permutation + y-reflection. The physics is identical but
array-level comparisons require matching conventions. The MATLAB conventions
were chosen here so the benchmark `.mat` data matches element-by-element.

## Hamiltonian structure

### Monolayer (nlayers=1)
```
H = H0_B + H_hopp       dim = 2*NG
```

### Bilayer (nlayers=2)

Two stacking configurations, selected by `stacking_type` (default 2).
See Moon & Koshino, PRB 90, 155406 (2014), Eqs. 25 and B1.

**Type 2 (default):**
```
H = [ H0_T    UBLG† ]   dim = 4*NG
    [ UBLG    H0_B + H_hopp ]
```

**Type 1:**
```
H = [ H0_T    UBLG  ]   dim = 4*NG
    [ UBLG†   H0_B + H_hopp ]
```

### Dirac blocks (K valley)
```
H0(j) = vF * (kx*sigx + ky*sigy) + U*I     where kx = kpt_x - Q_j_x
```

### Dirac blocks (K' valley)
```
H0(j) = vF * (-kx*sigx + ky*sigy) + U*I    sign flip on kx term
```

### Interlayer coupling (bilayer)
```
UBLG(j) = gamma1 * U1 - v3 * (kx - i*ky) * U2          K valley
UBLG(j) = gamma1 * U1 - v3 * (-kx - i*ky) * U2         K' valley
```

### Velocity operators
Analytic derivatives dH/d(hbar*k). Block-diagonal, k-independent:
```
K:   Vx = vF*sigx/hbar,   Vy = vF*sigy/hbar
K':  Vx = -vF*sigx/hbar,  Vy = vF*sigy/hbar
```
Bilayer adds off-diagonal v3 terms (Vx_TB, Vy_TB blocks).

## T-matrix conventions

K valley uses `exp(-i*psi)`, K' uses `exp(+i*psi)`.
K' Kronecker deltas use `-q_i` (sign flip on all three moire vectors).

```
K:   T1_K  with ±q1,  T2_K  with ±q2,  T3_K  with ±q3
K':  T1_Kp with ∓q1,  T2_Kp with ∓q2,  T3_Kp with ∓q3
```

Hopping: `H_hopp(j,k) = d0*T0 + (d_fwd*T† + d_rev*T)` for each of the
three moire vectors.

### Canonical q-vectors and T-matrix rotation

The q-vectors used throughout the Hamiltonian (Q-lattice construction,
hopping Kronecker deltas, k-mesh) always use canonical directions
regardless of the twist angle `theta`:
```
q1 = ktheta * [0, -1]
q2 = ktheta * [sqrt(3)/2, 1/2]
q3 = -q1 - q2
```
where `ktheta = |G1|` scales with the moire period (which depends on
theta).  The k-mesh reciprocal lattice vectors `vb` are built from these
same canonical q-vectors so the BZ, Q-lattice, and Hamiltonian periodicity
are all consistent.

When `theta != 0`, the physical moire pattern rotates, and the T-matrices
must be rotated to compensate:
```
T_i → inv(RR) @ T_i @ RR
```
where `RR = diag(exp(-i*thetaT), exp(+i*thetaT))` and
`thetaT = (-pi/2 - atan2(G1_y, G1_x)) / 2`, computed from the actual
(rotated) G1 direction returned by `compute_moire_geometry` as `G1_xy`.

The `T0` (uniform on-site) matrices are diagonal and commute with `RR`,
so only `T1, T2, T3` (and their K' counterparts) are rotated.

At `theta = 0`, `G1` points along `[0, -1]`, giving `thetaT = 0` and
`RR = I`.

The real-space moire cell area `vol_M` is always computed from the full
rotated geometry (it is a scalar invariant under rotation of the q-vectors).

The input parameter `theta` is specified in **degrees** and converted to
radians internally.

## Berry curvature and orbital moment (Kubo formula)

All in the eigenbasis. `v_{x,y}` = Psi† V_{x,y} Psi.

```
den(n,m)  = E_n - E_m
prod(n,m) = Im[ vx(n,m) * vy(m,n) ]
denom     = den^2 + eta^2

Oz(n) = -2 * hbar^2 * sum_m  prod / denom         [Ang^2 internally]
Lz(n) =      hbar^2 * sum_m  den * prod / denom   [eV*Ang^2 internally]
```

Diagonal terms (n=m) contribute zero because vx(n,n) and vy(n,n) are
real for Hermitian V operators.

### Hofstadter mode — velocity operator

The physical velocity in the LL basis is `v = (i/hbar) [R, H]` where
R is the position operator.  In the magnetic Bloch representation,
`R = i d/dk + A` where A is the LL Berry connection, giving:
```
v = (1/hbar) * (dH/dk + i[A, H])
```

The moire potential V_moire(R) is a local function of position, so
`[R, V_moire] = 0`.  This means `dV/dk + i[A, V] = 0` identically —
the moire potential contributes nothing to the velocity.  Since H_base
(the BLG kinetic Hamiltonian) is k-independent, `dH_base/dk = 0`, and
the velocity reduces to:
```
v = (i/hbar) [A, H_base]
```

This is k-independent and precomputed once in `build_hofstadter_setup`.
It avoids LL truncation artifacts that arise when computing `dV/dk` and
`[A, V]` separately in a finite basis (the identity `dV/dk = -i[A,V]`
requires completeness of the LL basis, which fails near the cutoff N).

The velocity is gauge-invariant — same Berry curvature regardless of
unit cell choice (square vs triangular).  Chern per mesh cell = 1/pp;
integer over the full magnetic BZ.

### Hofstadter mode — magnetic flux convention

The magnetic field is defined as `B = (qq/pp) × φ₀ / uc_area` where
`uc_area = √3 L² = 2 A_prim` is the **doubled** (rectangular) unit cell.
This means the flux per **primitive** (triangular) moire cell is
`qq/(2pp) × φ₀`, not `qq/pp × φ₀`. To get one flux quantum per primitive
cell, use `qq/pp = 2` (e.g. qq=2, pp=1).

### Hofstadter mode — BZ normalization for Chern numbers

The k-mesh vectors are `vb = [b1/pp, b2*qq/pp]` where b1, b2 are the
primitive moire reciprocal lattice vectors. The b2 direction is qq times
larger than b1/pp because the phase factors `exp(i*(pp/qq)*k·L)` require
qq periods in b2 for the Hamiltonian to be periodic, while the chain
structure already provides periodicity in b1 at the b1/pp spacing.

The real-space area for BZ normalization is `vol_M = pp² × uc_area / (2*qq)`,
giving `BZ_area = (2π)² / vol_M = qq × (2π)² / (pp² A_prim)`.

For Chern numbers: only the total Chern of a set of bands between two
well-defined energy gaps is quantized. Individual band Chern numbers are
only meaningful when bands are isolated by gaps larger than the Kubo
broadening.  In Hofstadter mode, the Kubo broadening is set by
`eta_kubo` (meV, default 2), separate from the moire coupling `eta`.

## Susceptibility (Fukuyama formula)

Implemented as a standalone executable in `susceptibility.py`, separate
from the band structure pipeline. Imports geometry and H/V assembly
functions from `bandstructure.py`.

```
G_nm(E) = delta_nm / (E - E_n + i*eta)
chi(E) = (1/Nk*vol_M) * Im Tr[ Vx*G * Vy*G * Vx*G * Vy*G ]
```

Implemented as `(vx .* g_row) @ (vy .* g_row) @ ...` where
`g_row = 1/(E - E_m + i*eta)` broadcasts along columns.

Energy argument is in meV (elist), eigenvalues converted from eV via ×1e3.

Output: `dChi_dE_K`, `dChi_dE_Kp`, `E_list`. To include chi in Onsager
quantization, set `susceptibility_datafile` in the onsager input.  The
chi term appears in the `_SBMC` output suffix.

## Unit conversions (post-processing)

### Band structure output (bandstructure.py)

| Quantity | Internal units | Output units | Conversion |
|---|---|---|---|
| E_K, E_Kp | eV | meV | ×1e3 |
| Oz | Ang^2 | m^2 | ×1e-20 |
| Lz | eV·Ang^2 | meV·m^2 | ×1e-20 × 1e3 |
| vol_M | Ang^2 | m^2 | ×1e-20 |

### Susceptibility output (susceptibility.py)

| Quantity | Internal units | Output units | Conversion |
|---|---|---|---|
| dChi_dE | Ang^-2 / ... | m^-2 / ... | ×1e-20 × hbar^4 |
| E_list | meV (input) | eV | ÷1e3 |

## k-mesh ordering

Fractional coordinates `v1 = n1/nk1 - 0.5`, `v2 = n2/nk2 - 0.5`.
Physical: `k = v1*G1 + v2*G2`.

Flattening uses column-major order (Fortran, `order='F'`): n2 varies
fastest, n1 slowest. This matches MATLAB's default reshape and is
required for `isoenergy.py` to correctly reshape back to 2D:
```python
E_2d = E_bands[n, :].reshape(nk2, nk1, order='F')
```

## Isoenergy orbit detection (contour method)

1. Tile energy surface 3×3 for periodic BZ boundaries
2. `skimage.measure.find_contours` (marching squares) at each energy level
3. Filter: closed contours only (first ≈ last vertex)
4. Keep contours with centroid in central tile (deduplication)
5. Compute area via shoelace formula (sub-cell interpolation)
6. Convert pixel² area to k-space: `area_k = area_pixels × cell_area`
7. Find enclosed k-points via `matplotlib.path.Path.contains_points`
8. Map tiled grid indices → original BZ via modulo, F-order linearization

This matches the physics of MATLAB's `contourc` + `polyarea` + `inpolygon`
approach. Orbit areas agree with MATLAB benchmark to machine precision.

## Parallelization

- k-loop: `multiprocessing.Pool` (embarrassingly parallel)
- `onsager_bfield`: `multiprocessing.Pool` over B values
- `OPENBLAS_NUM_THREADS=1` pinned before numpy import
- `isoenergy_areas` is serial over bands (can be parallelized if needed)
- For cluster multi-node: run independent B values as separate jobs

## Band indexing (Python 0-based vs MATLAB 1-based)

MATLAB center index: `round(dim/2)` → 98 for dim=196 (1-indexed)
Python equivalent: `dim // 2 - 1` → 97 (0-indexed)

Band selection: `bands_idx = dim // 2 - 1 + bands_sel`

## Stage-based Onsager pipeline

The calculation is split into three independently runnable stages,
dispatched by `calctype` in the input file:

```
bandstructure  →  isoenergy  →  onsager
    (k-mesh)      (orbit areas)   (LL fan)
```

When `calctype = all` (default) and `Blist` is present, all three stages
run end-to-end. Each stage can also run separately by setting `calctype`
and providing prior results via `inputdata`.

1. `run_bandstructure` — calls `do_calc`, saves E_K, Oz_K, Lz_K, kpoints, vol_M
2. `run_isoenergy` — calls `get_energy_resolved_data` for K/Kp, saves
   orbit areas, enclosed BC, dL/dE
3. `run_onsager` — calls `onsager_fan` for K/Kp, optionally loads
   susceptibility data from `susceptibility_datafile`

The Onsager step uses its own energy grid (`elist_onsager`, defaults to
`elist`) so it can be denser than needed for other purposes.

Output keys: `Blist` (nB,), `nmax` (scalar), and per-band cumulative LL
arrays with suffixes `_S`, `_SB`, `_SBM`, `_SBMC` (e.g.
`LL_K_band{i}_S`, `LL_K_band{i}_SBM`) for each band with orbits.
When Lifshitz transitions split a band into multiple segments, keys are
further suffixed with `_seg0`, `_seg1`, etc. (e.g. `LL_K_band5_SBM_seg1`).

### Root-finding

`_solve_onsager` finds roots of the Onsager condition along the energy
axis using two methods in priority order:

1. **Sign-change interpolation** (primary): detects adjacent energy grid
   points where the Onsager residual changes sign, then linearly
   interpolates to find the sub-grid-cell zero crossing. This produces
   continuous LL dispersion curves even when area(E) changes steeply
   (e.g. near Van Hove singularities), because it only requires the root
   to fall *between* two grid points rather than *near* one.

2. **Argmin with threshold** (fallback): for (B, n) pairs with no sign
   change, falls back to `argmin` of the absolute residual. If the best
   residual exceeds `rtol` (default 5%) times the rhs magnitude
   `B(n+½)/φ₀`, the entry is set to NaN. This suppresses spurious roots
   at saddle-point energies where area reaches a maximum but the Onsager
   condition is never truly satisfied.

### Lifshitz transition segmentation

When a band's orbit area A(E) is non-monotonic — e.g. small orbits
growing into a Lifshitz transition where the orbit topology changes,
then large orbits shrinking — the Onsager condition can have multiple
roots at the same (B, n).  Since `_solve_onsager` returns one root per
(B, n), `onsager_fan_band` splits the area curve at Lifshitz transitions
and solves each monotonic segment independently.

Detection: a Lifshitz transition is identified where `|ΔA|` between
adjacent energy grid points exceeds `lifshitz_threshold` (default 50)
times the median `|ΔA|`.  This is tunable via the `lifshitz_threshold`
input parameter.

## Non-perturbative Onsager (`onsager_bfield`)

An alternative pipeline that includes the orbital moment non-perturbatively.
Branches directly from bandstructure output (not isoenergy):

```
bandstructure  →  onsager_bfield
    (k-mesh)      (B-dependent orbits + LL fan)
```

At each B, forms the modified energy surface:
```
E_mod(k) = E_K(k) + gfactor × B × Lz_K(k)
```

Then computes isoenergy contours on E_mod, finds enclosed Berry curvature,
and solves the Onsager condition. The rhs of the Onsager condition is
`Bmultiplier × B × (n + ½) / φ₀`, where `Bmultiplier` defaults to 1
(`onsager_Bmultiplier` input parameter; diagnostic/testing).

Since the orbital moment is already in the energy surface, output suffixes
are `_SM` (area only, morb in dispersion) and `_SBM` (+ enclosed BC).

Each B value is independent; parallelized over Blist via `multiprocessing.Pool`
when `isparallel=1`. The worker calls `isoenergy_areas` directly (not
`get_energy_resolved_data`) to avoid computing `dL_dE`.

Intermediate data (orbit areas, enclosed BC, energy grids) are saved per B
for debugging.

## Hofstadter mode internals

### Hamiltonian construction

The Hofstadter Hamiltonian is built in a Landau level (LL) basis with
moire coupling. The k-independent parts (`H_base`, `term1/2/3`) are
precomputed in `build_hofstadter_setup` using functions from
`hamiltonian.py`. Per-k-point assembly adds phase factors:
```
tphase1 = exp(i * pp/qq * kx * Lx)
tphase2 = exp(-i * pp/qq * kx * Lx / 2) * exp(i * ky * Ly * pp/qq)
tphase3 = exp(-i * pp/qq * kx * Lx / 2) * exp(-i * ky * Ly * pp/qq)

H = H_base + v0*I + sum(gamma * tphase * term) + h.c.
```
Only the moire coupling block `[mo:, mo:]` depends on k.

### Berry connection (LL ladder operators)

The LL Berry connection connects adjacent LLs via ladder operators:
```
A_x(s,n | s,n+1) = -i * lB/sqrt(2) * sqrt(n+1)   (s = A or B sublattice)
A_y(s,n | s,n+1) =      lB/sqrt(2) * sqrt(n+1)
```
Hermitianized (A + A†), then chopped to remove the highest LL
(B_LL_N for K valley, A_LL_N for K' valley — matching the intralayer
Hamiltonian convention).

Implemented in `hamiltonian.py` as `get_berry_connection_K/Kp`.

### Unit conventions

| Quantity | Units | Notes |
|---|---|---|
| H, H_base, terms | eV | Converted from Joules via /Q_E |
| Ax, Ay | Angstrom | Converted from meters via ×1e10 |
| Velocity (Vx, Vy) | Ang/s | i[A,H_base]/hbar — precomputed, k-independent |
| Berry curvature Oz | Ang^2 | From Kubo formula with hbar^2 prefactor |
| kpoints | Ang^-1 | Converted from m^-1 via /1e10 |
| vol_M | m^2 | Real-space magnetic unit cell area |

Post-processing conversions match zero-field: Oz×1e-20, E×1e3, Lz×1e-20×1e3.

## Known considerations

- `eigh` returns eigenvalues in ascending order (matches MATLAB `eig`
  for Hermitian matrices). Berry curvature is invariant to eigenvector
  phase choice within degenerate subspaces.
- The susceptibility uses mixed units in the Green's function:
  elist is in meV, eigenvalues are converted from eV via ×1e3, eta is
  in eV (0.002 eV = 2 meV for the Berry curvature, but enters the
  susceptibility Green's function as 0.002 in meV context — effectively
  2 μeV broadening). This matches the MATLAB code exactly.
- `construct_hopping` uses a double loop over Q-vectors (O(NG^2)). This
  runs once per calculation and is not a bottleneck, but could be
  vectorized if NG grows large.
