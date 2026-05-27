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
  compute_moire_geometry   →  q1, q2, q3, vol_M, vb
  build_qvectors           →  Q (NG×2), NG
  construct_hopping         →  H_hopp_K, H_hopp_Kp  (2NG × 2NG)
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
```
H = [ H0_T    UBLG† ]   dim = 4*NG
    [ UBLG    H0_B + H_hopp ]
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

### Hofstadter mode — velocity operator for Berry curvature

**IMPORTANT**: In the LL basis, the physical velocity has two terms:
```
v = (1/hbar) * (dH/dk - i[A, H])
```
where A is the LL Berry connection (ladder operator matrix elements).
However, for **Berry curvature**, only the `dH/dk` term should be used.
The `[A, H]` commutator is part of the physical current operator but
does NOT contribute to the gauge-invariant Berry curvature.

This was confirmed by comparing the Kubo formula against gauge-invariant
plaquette Berry curvature (wavefunction overlaps around a small k-space
loop). The dH/dk-only Kubo formula matches the plaquette to ~1e-4
relative, while including the `[A, H]` term destroys the agreement.

The Berry connection matrices (`Ax`, `Ay`) are still computed and stored
in the setup dict because they may be needed for the orbital moment `Lz`
(where the physical velocity enters). This needs verification.

### Hofstadter mode — magnetic flux convention

The magnetic field is defined as `B = (qq/pp) × φ₀ / uc_area` where
`uc_area = √3 L² = 2 A_prim` is the **doubled** (rectangular) unit cell.
This means the flux per **primitive** (triangular) moire cell is
`qq/(2pp) × φ₀`, not `qq/pp × φ₀`. To get one flux quantum per primitive
cell, use `qq/pp = 2` (e.g. qq=2, pp=1).

### Hofstadter mode — BZ normalization for Chern numbers

The k-mesh vectors are `vb = [b1/pp, b2/pp]` where b1, b2 are the
primitive moire reciprocal lattice vectors. This covers a k-space area
that is `1/pp²` of the moire BZ. The half-chain convention (Nq = qq)
uses a doubled real-space unit cell with chain phase factors doubled
to compensate.

The real-space area for BZ normalization is `vol_M = pp² × uc_area / 2
= pp² × A_prim`, giving `BZ_area = (2π)² / vol_M = (2π)² / (pp² A_prim)`.

For Chern numbers: only the total Chern of a set of bands between two
well-defined energy gaps is quantized. Individual band Chern numbers are
only meaningful when bands are isolated by gaps larger than eta.

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
quantization, set `susceptibility_datafile` in the onsager input and use
`termflags = [1 1 1]`.

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

Output keys: `Blist` (nB,), `nmax` (scalar), `LL_K_band{i}` (nB, nmax+1)
and `LL_Kp_band{i}` (nB, nmax+1) for each band with orbits.

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
and solves the Onsager condition. Since the orbital moment is already in
the energy surface, `morbflag` is forced to 0 internally. The `termflags`
input is 2-element `[BCflag, chiflag]`.

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
| dH/dk (Hdx, Hdy) | eV·Ang | Phase derivative × term matrices |
| Velocity (Vx, Vy) | Ang/s | dH/dk / hbar (eV·Ang / eV·s) |
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
