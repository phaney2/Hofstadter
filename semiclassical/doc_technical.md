# Semiclassical Code — Technical Reference

## Architecture

### Stage dispatcher
```
semiclassical.py          # stage-dispatch driver
  load_data                →  load .mat/.npz with MATLAB dimension handling
  _kmesh                   →  (nk1, nk2) from the stored data, falling back to the input file
  run_bandstructure        →  calls do_calc, applies unfolding when unfold=1
  run_isoenergy            →  calls get_energy_resolved_data for K/Kp
  run_onsager              →  calls onsager_fan for K/Kp, optionally loads chi
  __main__                 →  calctype dispatch (bandstructure/isoenergy/onsager/all)
```

`do_calc` returns `nk1`/`nk2` in both the zero-field and Hofstadter
branches, and `run_isoenergy` / `run_onsager_bfield` read the mesh
through `_kmesh` rather than from the input file.  This is what lets an
unfolded band structure carry its doubled `nk1` (and halved `vol_M`)
downstream; files written before these keys existed fall back to the
input file.

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

unfold.py                 # magnetic BZ unfolding (opt-in, qq/pp = 2/odd)
  analyze_pair              →  locate both degeneracy line families of a candidate pair
  branch_label              →  parity label on the (nk2, 2*nk1) doubled grid
  unfold_pair               →  select one branch of E / Oz / Lz
  unfold_bandstructure      →  detect pairs, unfold, double the mesh, keep folded backup
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
Default `psi = 0.29` rad (configurable via `moire_psi` input parameter).
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

**Both outputs are orientation-independent.**  `_shoelace_area` takes
`np.abs` of the signed shoelace sum, and step 7 selects the polygon
interior by point-in-polygon rather than by winding number.  So the
contour orientation returned by `find_contours` — which is mixed CCW/CW
across energies within a single band — never reaches `area` or
`enclosedBC`.  The direction of traversal enters only as a single
field-direction-dependent sign in the Onsager condition (see "Term
signs"); it is not, and must not be, inferred per orbit from the contour.

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

### Term signs

`onsager_fan_band` builds the residual as `rhs + base` and roots it in E,
with `rhs = Bmultiplier·B·(n + ½)/φ₀` and `base` accumulated term by
term.  `base_S` carries an explicit `-Bsign`, so dividing the whole
condition by `sign(B)` puts every term on `|B|` and gives the readable
form (`λ ≡ Bmultiplier`, default 4 — see "The Bmultiplier factor"):

```
S(E)/(2π)² = (|B|/φ₀)·[ λ·(n + ½)
                        − BC_factor  ·sign(B)·Φ_B/(2π)
                        − morb_factor·(dL/dE)/(2π)
                        − chi_factor ·(2π)·(dχ/dE)·B/φ₀ ]
```

Against the textbook `S/(2π)² = (|B|/φ₀)(n + ½ − φ_B/2π)` this is
`φ_B = +sign(B)·BC_factor·Φ_B`.

The `sign(B)` on the Berry curvature term is the one non-obvious factor,
and it is what `np.abs(B2)` in the source achieves — every other term
uses `B2`.  `Φ_B` (`enclosedBC`) is the flux of `Oz` through the orbit
interior with no orientation information attached (see "Isoenergy orbit
detection"), whereas the Onsager phase is `φ_B = ∮ A·dk` taken *along the
direction of motion*.  The equation of motion `ℏk̇ = −e v×B` reverses the
traversal sense under `B → −B`, so `φ_B` is odd in `B` while `Φ_B` is
not.

The **parity** follows from the equation of motion.  The **overall sign**
is fixed by validation (b) below, and the two now agree with the
traversal argument: for an electron-like orbit at `B > 0`, `ṙ ∝ ∇E`
outward gives `k̇ ∝ (−k_y, k_x)` — counterclockwise — hence `φ_B = +Φ_B`
by Stokes, which is what the bracket above yields.  Earlier versions of
this code carried the opposite sign and documented it as empirical and in
conflict with that argument.  The conflict is resolved: the traversal
argument was right and the implementation was wrong.  The candidate
resolutions once enumerated in `notes_onsager.tex` (a convention mismatch
in the textbook form, a Diophantine effective-field sign, a hole-orbit
area convention) are no longer needed.

Two distinct things follow, and conflating them is the trap:

- The sign is **not per-orbit**.  It comes from the carrier charge and
  the field direction, so it does not flip between electron-like and
  hole-like orbits, and it must never be inferred from contour winding.
- The sign **is per-field-direction**.  A `Blist` straddling zero needs
  both branches within one run.  This is the normal case for
  `onsager_bfield`, where `Blist` holds the deviation `δB` from the
  background flux already contained in the band structure and its `Oz`.

Validated three ways:

- **(a)** Semiclassical Chern numbers from `Oz` reproduce the quantum
  σ_xy plateau steps band by band.  This constrains the sign of `Oz`
  itself, *not* the sign of the Berry term in the quantization condition,
  and is unaffected by the correction above.
- **(b)** With the parity switched off — the same sign used on both
  branches, via `recompute_onsager.py --bc-sign-mode fixed` — the exact
  Hofstadter spectrum selects `BC_factor = +1` on `δB < 0` and
  `BC_factor = -1` on `δB > 0`, at all four fluxes tested (`qq/pp` =
  1/2, 1/3, 2/5, 2/3; folded and unfolded).  Both branches correspond to
  the single odd-in-`B` convention above.  The discriminating feature is
  not the residual magnitude but the *crossover*: the two curves swap
  ranking exactly at the background field, which a fitting artifact would
  not do.  Margins are 3.6×/3.9× at 1/2 and 3.8×/4.9× at 2/3, the latter
  with median residuals of 0.099 and 0.083 meV.
- **(c)** Recomputing a fan with the corrected default reproduces one
  built by the old code at `BC_factor = -1` to max |diff| = 0 over all
  bands and both valleys — a self-consistency check on the parity
  plumbing only, not independent physical evidence.

Note that (b) supersedes an earlier, weaker claim that `onsager_bfield`
fans at `qq/pp = 1/2` "track the exact spectrum".  They do, but only
per-branch quantitative scoring distinguishes the two candidate signs;
a qualitative overlay does not.

`φ_B` jumps at `B = 0`; harmless, since the rhs vanishes there and no
level is defined.

The other three terms are all even in `B` in the bracket above.  For the
`chi` term that is expected (it comes from a `B²` energy correction).
For the `dL/dE` term it is **unverified** — if that term descends from an
orbit energy shift `∝ M·B`, it should carry `sign(B)` too.  It is
identically zero in `onsager_bfield` (the orbital moment is already in
the dispersion, correctly odd in `B` there), and the perturbative
`onsager` channel has no negative-`B` validation, so it has been left
alone.  Revisit before trusting perturbative-channel fans at `B < 0`.

### The Bmultiplier factor

`λ ≡ Bmultiplier` (`onsager_Bmultiplier`, `onsager.py`) multiplies `B` in
the rhs.  It **defaults to 4**, and it is not a free knob.

Scanning it against the exact Hofstadter spectrum puts the optimum at 4
at every flux and geometry tested:

| `qq/pp` | `pp,qq` | unfold | `vol_M/A_uc` | `2·pp` | λ optimal | symmetric residual (meV) |
|---|---|---|---|---|---|---|
| 1/2 | 2,1 | no  | 4.00 | 4  | 4   | 0.206 / 0.132 |
| 1/3 | 3,1 | no  | 9.00 | 6  | 3–4 | 0.528 / 0.533 |
| 2/5 | 5,2 | yes | 6.25 | 10 | 4   | 0.446 / 0.289 |
| 2/3 | 3,2 | yes | 2.25 | 6  | 4   | 0.099 / 0.083 |

The two residual columns are the `δB < 0` and `δB > 0` branches.  They
are *not* comparable across rows — each row uses its own energy and `|δB|`
window, and the fluxes differ in how many butterfly columns fall inside
it — but the λ comparison within a row is meaningful.

λ ≈ 4 holds while `vol_M/A_uc` ranges over 2.25–9 and `2·pp` over 4–10,
folded and unfolded, so it is a flux-independent constant, not a geometry
factor.  Both geometric candidates were tested and falsified: λ =
`vol_M/A_uc` predicts 6.25 at 2/5 and 9 at 1/3, and λ = `2·pp` predicts
10 at 2/5 and 6 at 1/3; all four score worse than 4.  That the numbers
happen to coincide at `qq/pp = 1/2` is a coincidence of that flux.

Two independent things happen at λ = 4 and nowhere else:

- The level count comes out right.  `nLL/nEx ≈ 1.05` at 2/3 — about one
  semiclassical level per exact subband.  This is a counting statement,
  independent of any residual metric.
- The Berry-phase parity becomes resolvable.  At λ = 2, 3, 5, 6 the two
  candidate signs score within ~30% of each other on both branches with
  no consistent pattern; at λ = 4 the margin is 3.8× and 4.9×.  A wrong
  effective field smears the parity signature out entirely, so **a null
  result on a sign test at the wrong λ is not evidence against the sign**.

Where the factor actually belongs has not been traced.  Untraced
candidates: the `qq/(2·pp)` primitive-vs-construction-cell flux
convention (a factor of 2 applied twice — the leading suspect, since the
semiclassical `B_bg = b0·(qq/pp)/2` already carries one such factor);
`φ₀ = h/e` vs `h/2e`; and the shoelace/BZ normalization in
`isoenergy.py` that feeds `S(E)`.  Until it is traced, the factor is
carried explicitly on the rhs rather than folded into `S(E)`, so that the
place it is being applied stays visible.  λ does not enter the modified
dispersion in the non-perturbative channel, so it cannot contaminate the
band structure or `Oz`.

**Scoring caveat.**  The obvious metric — distance from each
semiclassical level to the nearest exact subband centre — only penalizes
*extra* levels, so it monotonically rewards a sparser fan and cannot
compare different λ.  Use the symmetric two-way median (LL → exact and
exact → LL) together with the level-count ratio `nLL/nEx`.

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
`Bmultiplier × B × (n + ½) / φ₀`, where `Bmultiplier` defaults to 4
(`onsager_Bmultiplier` input parameter) — see "The Bmultiplier factor".

Since the orbital moment is already in the energy surface, output suffixes
are `_SM` (area only, morb in dispersion) and `_SBM` (+ enclosed BC).
For the same reason `term_factors` is read as 2 elements here —
`[BC_factor chi_factor]`, expanded internally to
`(BC_factor, 0.0, chi_factor)` — since `morb_factor` would multiply an
identically-zero `dL_dE`.

Each B value is independent; parallelized over Blist via `multiprocessing.Pool`
when `isparallel=1`. The worker calls `isoenergy_areas` directly (not
`get_energy_resolved_data`) to avoid computing `dL_dE`.

Intermediate data (orbit areas, enclosed BC, energy grids) are saved per B
to a separate `<outputfile base>_detail.mat`, written flat (no
`results`/`params` wrapper).  The LL fan itself goes to `outputfile`.

### Re-solving from detail data (`recompute_onsager.py`)

`Blist`, `area_{v}_band{n}`, `enclosedBC_{v}_band{n}` and
`E_levels_{v}_band{n}` are the complete input set to `onsager_fan_band`,
so the fan can be rebuilt from the detail file without redoing the
contour work.  `recompute_onsager.py` does exactly that, looping over B
the same way the worker does and applying the same `S` -> `SM`,
`SB` -> `SBM` renaming (with `dL_dE = 0` and `morb_factor = 0`, since the
orbital moment is already in the dispersion).  Given the same prefactors
it reproduces the stored fan bit-for-bit.

Its purpose is cheap prefactor sweeps: `--bc-factor` is `term_factors[0]`
and accepts several values at once, so `--bc-factor 1,-1` puts both signs
of the enclosed Berry curvature term in one output file (keys suffixed
`_bcf0`, `_bcf1`).  A 200-B, 13-band, 2-valley recompute takes ~10 s.

`--bc-sign-mode {odd,fixed}` controls the *parity* of the Berry phase in
`B`, independently of `--bc-factor`.  `odd` (default) is the production
convention, in which the level shift carries `sign(B)`.  `fixed` applies
the same sign on both field branches, so the two can be scored
separately — this is the test that established the sign (validation (b)
in "Term signs").  It needs no change to the solver: since
`B = sign(B)·|B|`, handing the solver a per-field `f·sign(B)` is
algebraically identical to replacing `|B|` with `B` in the `base_SB`
line.  Diagnostic only; production fans should be generated with `odd`.

`--Bmultiplier` defaults to the value stored in `--ref`, or to 4 when no
reference is given, matching `onsager.py`.
This is how the enclosed-BC sign was settled; it is also how to
reproduce a fan file written before that fix (`--bc-factor -1`).

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

## Magnetic BZ unfolding (`unfold.py`)

Enabled by `unfold = 1`; applied in `run_bandstructure` after the k-loop
and before the result is saved.  User-facing behaviour is in
`doc_user_guide.md`; this section covers the mechanism.

### Why the bands fold

The Landau gauge requires a rectangular construction cell, which on a
triangular moire lattice is the centered-rectangular two-lattice-point
cell.  At flux `qq/pp = 2/(odd)` the magnetic translation group that
survives is larger than the one the construction cell exposes: the true
reciprocal lattice is `<2*G1, G2>` while the k-loop samples
`<G1, G2>` with `G1 = b1/pp`, `G2 = qq*b2/pp`.  The computed BZ is
therefore half the physical one along G1, and each physical band appears
twice.

The two copies belong to different eigenvalues of the extra translation
and **never hybridize**, so they cross rather than anticross.  `eigvalsh`
returns eigenvalues sorted ascending, so what is stored as band `n` is
`min` of the two branches and band `n+1` is `max` — each a kinked
composite, not a smooth surface.  The kinks are what corrupt orbit areas
downstream.

### Degeneracy line families

In fractional coordinates `f1 = n1/nk1`, `f2 = n2/nk2` the two branches
are degenerate on two line families:

| Family | Locus | Line direction | Gap at the line |
|---|---|---|---|
| 1 | `f1 - f2 = 1/2`, symmetry-enforced | G1+G2 | 1e-13 … 5e-7 meV |
| 2 | `f2 = h_off`, band- and valley-dependent | G1 | 0 (conical apex, generally off-grid) |

Family 1 sits at exactly `1/2` for every pair tested (12 pairs at
`pp=5,qq=2`; 6 at `pp=3,qq=2`; both valleys).  Family 2 has no fixed
position — for `pp=3` it was 0.128/0.226/0.348 in K and
0.877/0.676/0.775 in Kp.

### Detection

`analyze_pair` takes an adjacent pair as `(nk2, nk1)` arrays and locates
both families in the gap `hi - lo`:

- Family 1 from cuts at fixed `n2`, accepting only rows with **exactly
  one** crossing; the offsets are averaged with circular statistics
  (`_circ_mean`) since they live on `[0,1)`.
- Family 2 from cuts at fixed `n1`, with the family-1 crossing excluded
  by a `max(0.02, 3/nk2)` window in `f1 - f2`.

The crossings are conical, not parabolic: the gap behaves like `|x|`, so
a parabolic sub-grid fit reports a spurious finite gap at the apex.
`_cone_minima` instead extrapolates the two linear flanks and intersects
them, which recovers apexes lying between grid rows.

`_qualifies` accepts a pair when both offsets fit, at least `min_frac`
(default 0.9) of rows contribute a family-1 sample, both circular RMS
values are within `max_rms` (default 0.02), and family 2 was seen on at
least `nk1/4` columns.  The separation is not marginal — scoring every
adjacent pair including deliberately wrong pairings gives
`frac_rows` of exactly 1.000 for real pairs and exactly 0.000 for wrong
ones, with `gap_min` differing by ~13 orders of magnitude (1e-13 meV vs
≥1.8 meV).  A pair is unfolded only when **both valleys** qualify; time
reversal guarantees they agree, and requiring it keeps the two valleys on
a common band indexing.

### Branch labelling

`branch_label` assigns each point of the doubled grid the parity of the
number of degeneracy lines crossed:

```
p_diag = floor(f1 - f2 - d_off + eps) % 2
p_horz = floor(f2 - h_off + eps) % 2
s      = (p_diag + p_horz) % 2
```

Neither parity is separately periodic under `f2 -> f2 + 1`, but their
sum is; and the sum is periodic under `f1 -> f1 + 2` while flipping under
`f1 -> f1 + 1`.  That is exactly the branch structure of a band folded
once along G1, and it is why the label is single-valued only on the
double cover — the algebraic restatement of "the true reciprocal lattice
is `<2*G1, G2>`".  There is no screw boundary condition; the unfolded
data is plainly periodic on the doubled rectangle.

`eps = 0.01/max(nk1, nk2)` breaks a tie that arises when `nk1 == nk2`:
the family-1 line then runs exactly through grid points, and without the
bias their side of the cut is decided by the sign of the ~1e-8 residual
of the offset fit.  The bias is 1% of a grid step — far above the
observed fit error (≤3e-6) and far below a step.  A *half*-step bias
would be wrong: it is right for family 1, whose crossing lands on a grid
point, but family 2's apex genuinely falls between rows (e.g. 71.10) and
a half-step shift would mislabel the adjacent row.

`unfold_pair` then selects `lo` or `hi` by `s` on the `(nk2, 2*nk1)`
grid.  E, Oz and Lz are all carried through the **same** label, so the
three quantities stay mutually consistent.

### Identities the result satisfies

Verified exactly (0.00e+00) for all pairs and both valleys by
`test_unfold_module.py`:

- `{A, B}` is a permutation of `{E_lo, E_hi}` at every k-point — no data
  is created or destroyed, only relabelled.
- `B(k) == A(k + G1)` — the discarded branch is the translate of the kept
  one, which is the defining property of a once-folded band.
- `A` is periodic in `n1` with period `2*nk1`.
- Total Berry curvature is conserved to machine precision.

Roughness (max |second difference| of E along each axis) improves by
30–50× at `pp=5, nk=200` and 18–25× at `pp=3, nk=40`, on both axes, for
every pair.

### Mesh bookkeeping

`nk1 -> 2*nk1`, `nk2` unchanged, `vol_M -> vol_M/2`.  The k-points are
tiled once along G1, with `G1` recovered from the stored mesh as
`nk1 * (kp[0,1] - kp[0,0])`.

`cell_area = (2π)²/(vol_M · nk1 · nk2)` is **invariant** under this
transformation, which is why absolute orbit areas are unaffected.
Confirmed downstream: traced contour area agrees with the exact enclosed
k-point count to a median of 0.07% for the unfolded band, and the
conservation law `S_unf(E) = S_lo(E) + S_hi(E)` holds to a median ratio
of 1.0000 across the energies where the folded contours are still
traceable.

### Pre-existing normalization caveat

The sampled k-zone holds `pp²/qq` primitive cells, which is 12.5 for
`(pp,qq) = (5,2)` — not an integer.  Unfolding halves it to 6.25, still
not the 5 that flux 1/5 would require.  This is a property of the
Hofstadter k-zone convention, not of unfolding, and it does not affect
orbit areas because `cell_area` is unchanged.

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
- The enclosed Berry curvature term in the Onsager condition was wrong
  until corrected in `onsager.py`: it had the wrong overall sign, and it
  was even in `B` where the physics requires odd.  See "Term signs"
  above.  Fan files generated before that change have shifted
  `SB`/`SBM`/`SBMC` levels and should be regenerated; a `--bc-factor -1`
  recompute reproduces the old values only for `B > 0` (at `B < 0` the
  old code happened to agree with the corrected one).  `_S`/`_SM` levels
  are unaffected.
- The `dL/dE` term's parity in `B` is unverified — see the end of "Term
  signs".  Only relevant to the perturbative `onsager` channel at
  negative `B`.
