# CLAUDE.md — Development Mode

This project computes moire band structures for mono- or bilayer graphene
on hBN.  Four calculation modes:

1. **Hofstadter** (`main_v3.py`): Magnetic Bloch bands in a Landau-level
   basis.  The physical flux per **primitive** moire cell is qq/(2*pp);
   the input fraction qq/pp is the flux through the centered-rectangular
   (two-lattice-point) construction cell required by the Landau gauge.
   Uses corrected moire coupling matrices (order=[3,1,2], conj=1,
   psi_conj=1) with Nq=qq.  Supports `calctype = 'ek'` (band structure),
   `'dos'` (density of states), and `'transport'` (Kubo-formula linear
   response transport coefficients: sigma_xx, sigma_xy, L12_xx, L12_xy
   vs mu).
   Supports constant broadening or SCBA (self-consistent Born
   approximation) for energy-dependent broadening that captures
   localization-induced σ_xx suppression in narrow subbands.
   Legacy driver `main_v2.py` is kept for reference.
2. **Zero-field** (`zerofield.py`): Moire band structure via plane-wave
   expansion along a k-path through the moire BZ.
3. **Semiclassical** (`semiclassical/`): Full BZ k-mesh band structure
   plus Berry curvature, orbital moment, Fukuyama susceptibility, and
   Onsager semiclassical quantization (Landau level fan diagrams).
   Includes non-perturbative B-field mode (`onsager_bfield`) that
   recomputes orbits on E(k)+gfactor×B×Lz(k) at each field.
   Also pushed to separate repo: `github.com/phaney2/semiclassical_hofstadter`.

## Code layout

| File | Purpose |
|---|---|
| `main_v3.py` | **Primary** Hofstadter engine: corrected moire coupling, minimal k-zone, physical normalization (Nq=qq) |
| `main_v2.py` | Legacy Hofstadter engine (Nq=qq, old T-matrix conventions) |
| `hofstadter_testing.py` | Convention explorer: sweep order/conj/psi flags to find correct T matrices |
| `hamiltonian.py` | All Hamiltonian construction (intralayer, intermonolayer, interbilayer, testing variants) |
| `numerics.py` | Math routines: Laguerre functions, F_nm matrix elements, table builder |
| `basis.py` | Label-based basis toolkit: `outer_product`, `getindices` |
| `parser.py` | MATLAB-style input file parser (shared) |
| `constants.py` | Physical constants (shared) |
| `zerofield.py` | Zero-field engine: moire geometry, plane-wave Hamiltonian, k-path solver |
| `validate.py` | Hofstadter benchmark against MATLAB `.mat` data (uses legacy conventions) |
| `validate_transport_norm.py` | main_v3 normalization + minimal-zone validation (zone equivalence, same-B invariance, state counting) |
| `validate_transport_kubo.py` | (untracked) Kubo evaluation-knob convergence (`sigma_xx_buffer`, `eps_per_width`, `eps_grid_floor`): reruns a case tightened and reports the difference |
| `validate_zerofield.py` | Zero-field benchmark comparison against `bands_BG.mat` |
| `plot_zerofield.m` | MATLAB plotting script for zero-field band structure |
| `input_test.txt` | Default Hofstadter input (pp=1, qq=1) |
| `input_zerofield.txt` | Default zero-field input |
| `doc_technical.md` | Code structure reference |
| `doc_user_guide.md` | Input/output reference |
| `notes_scba.tex` | Write-up: SCBA formalism and implementation |
| `notes_onsager.tex` | Write-up: Onsager quantization sign conventions and Berry curvature computation (draft source for publication SI) |
| `bands_p*_q*.mat` | Hofstadter MATLAB benchmark data |
| `matlab_code/zerofield/` | Original MATLAB zero-field code and benchmark (`bands_BG.mat`) |
| `matlab_code/` | Original MATLAB Hofstadter code |
| `matlab_debugging/` | MATLAB scripts for debugging/comparison |
| `semiclassical/` | Semiclassical engine (see below) |

### Semiclassical code (`semiclassical/`)

| File | Purpose |
|---|---|
| `semiclassical.py` | Stage-dispatch driver: load/save data, run bandstructure/isoenergy/onsager stages |
| `bandstructure.py` | Band structure engine: moire Hamiltonian, Berry curvature, orbital moment |
| `susceptibility.py` | Standalone Fukuyama susceptibility (dChi/dE) calculation |
| `hofstadter_system.py` | Hofstadter H/V setup and per-k-point assembly |
| `isoenergy.py` | Contour-based isoenergy orbit detection (marching squares + shoelace area); `periodic=False` for the non-periodic extended zone |
| `onsager.py` | Onsager quantization solver: S(E)/(2π)² + corrections = B(n+½)/φ₀ |
| `recompute_onsager.py` | Re-solve the LL fan from a saved `*_detail.mat` with different correction prefactors (e.g. flipped BC sign) |
| `unfold.py` | Magnetic BZ unfolding for folded Hofstadter bands (`unfold = 1`) |
| `extended_zone.py` | Moire BZ unfolding for the zero-field bands (`extended_zone = 1`) |
| `breakdown.py` | Landau-Zener broadening of the extended-zone LLs (`breakdown = 1`) |
| `validate_extended_zone.py` | (untracked) Extended-zone unfolding validation (V=0 exactness, sum rules, bijection, orbit monotonicity) |
| `input.txt` | Example input with Onsager parameters |
| `doc_technical.md` | Technical reference for the semiclassical code |
| `doc_user_guide.md` | Input/output reference for the semiclassical code |

## Before making changes

Read `doc_technical.md`. It documents the full code structure: function
layers, basis labeling system, matrix dimensions, phase conventions,
parallelization, and known subtleties. This is faster and more reliable
than re-parsing the source.

## Code conventions

- **Output format**: All `.mat` output files use a nested structure with
  two top-level structs: `results` (computed data) and `params` (all
  input parameters as parsed).  The semiclassical `load_data()` function
  auto-unwraps this format so downstream stages can access result keys
  directly.
- The function return contract for `do_calc` (in `main_v3.py` / `main_v2.py`)
  is a dict whose keys depend on `calctype`. New calctypes add new key sets.
- **Hofstadter units**: Input parameters are in meV; converted to Joules
  internally. Final eigenvalues are converted back to meV.
- **Hofstadter normalization**: Every k-point's spectrum contains one
  magnetic unit cell (= 2*pp primitive moire cells) of states.  All
  k-integrated outputs are normalized accordingly: DOS histograms use
  weight `1/(Nk*2*pp)` (states per primitive cell), and the transport
  prefactors use `1/(Nk * A_mag)` with `A_mag = 2*pp*A_uc`.  No qq
  appears in any normalization.  The k-mesh samples the minimal zone
  `[b1/pp, gcd(2*pp,qq)*b2/pp]`, on which all gauge-invariant
  quantities are exactly periodic.  Validated by
  `validate_transport_norm.py` — run it after touching the k-mesh,
  DOS binning, or transport prefactors.
- **Zero-field units**: Input parameters are in meV; converted to eV
  internally. Eigenvalues are output in eV.
- **Semiclassical units**: Input in meV; internal calculation in eV;
  output E in meV, Oz/Lz/vol_M in SI (m²), dChi_dE in SI.
  Post-processing conversions: Oz×1e-20, Lz×1e-20×1e3, vol_M×1e-20,
  dChi_dE×1e-20×hbar⁴.
- **Twist angle**: The input parameter `theta` is in **degrees** in all
  input files. It is converted to radians at the point of parsing in each
  driver. When `theta != 0`, the T-matrices are rotated via
  `inv(RR) @ T @ RR` to compensate for the rotated moire pattern while
  keeping q-vectors canonical. The q-vectors always use fixed directions
  `[0,-1]`, `[√3/2, 1/2]`, `[-√3/2, 1/2]` scaled by `ktheta = |G1|`.
- **Bilayer stacking type**: The `stacking_type` parameter (default 2)
  controls the off-diagonal block arrangement in the bilayer Hamiltonian.
  Type 2 (A1-B2) puts `Hinter` in the upper-right; Type 1 (B1-A2) swaps
  the off-diagonal blocks.  See Moon & Koshino, PRB 90, 155406 (2014),
  Eqs. 25 and B1.  This applies to `main_v3.py`, `zerofield.py`, and the
  semiclassical code (`bandstructure.py`, `hofstadter_system.py`).
- **Minimal magnetic k-zone**: `main_v3.py` and the semiclassical
  Hofstadter mode both sample `[b1/pp, qfac*b2/pp]` with
  `qfac = gcd(2*pp, qq)` — the smallest zone on which all
  gauge-invariant quantities are periodic.  `full_zone = 1` restores the
  qq-extended zone (`qq/qfac` identical copies, same k-averages, that
  many times the cost).  In the semiclassical code `vol_M` moves with
  the zone (`pp**2 * uc_area / (2*qfac)`) so that
  `cell_area = (2π)²/(vol_M·nk1·nk2)` — and hence every orbit area — is
  invariant.  `qfac == qq` for all fluxes run before `(7,4)`; run
  `python semiclassical/validate_zone.py` after touching the k-mesh or
  `vol_M`.
- **Magnetic BZ unfolding**: When `gcd(2*pp,qq) == 2` with `pp` odd
  (flux `qq/pp = 2/(odd)`, and also e.g. `(pp,qq) = (7,4)`) the
  Landau-gauge construction cell makes the semiclassical magnetic BZ a
  factor of two too small along G1, so every band appears as two
  non-mixing subbands that swap hi/lo across degeneracy lines.  This is
  a separate effect from the b2 redundancy above, and the minimal zone
  must be in place first — on the qq-extended grid the degeneracy lines
  do not fit the model `analyze_pair` uses.  The optional bandstructure
  parameter `unfold = 1` (default 0, `semiclassical/unfold.py`) detects
  such pairs **from the data** — never from `qq/pp` — and recombines each
  into one smooth band on the doubled zone, doubling `nk1` and halving
  `vol_M`.  The folded arrays are always kept under `*_folded` keys.
  Downstream stages read the mesh from the data via `_kmesh`, so
  `isoenergy` and `onsager_bfield` inherit it automatically.
- **Extended-zone (moire BZ) unfolding**: the *zero-field* counterpart,
  and a completely separate mechanism — `extended_zone = 1` (default 0,
  `semiclassical/extended_zone.py`) is rejected for `qq > 0` and is
  mutually exclusive with `unfold = 1`.  Moire folding makes a
  constant-energy contour merge with its periodic images once it outgrows
  the moire BZ, so the tracer switches to the complementary pockets and a
  hole orbit appears to turn electron-like.  This happens at **zero moire
  potential too** — it is an artifact of the zone.  The unfolding maps
  each state back to the momentum it carries via its plane-wave spectral
  weight, reduces to `2*nlayers` intrinsic branches per extended point,
  and grows the mesh by `ntile` on each axis while shrinking `vol_M` by
  `ntile**2`, so `cell_area` and all absolute orbit areas are invariant.
  Branches are the **eigenvectors** of the Q-diagonal block, so each
  collects total weight exactly 1 and the default `extended_mode =
  centroid` returns `u_b^dag H_jj u_b` — the moire-free dispersion at
  `k - Q_j`, identically.  The extended `E` are therefore smooth through
  every Bragg plane **by construction**; a kink there is a bug, and the
  moire potential survives only in `Oz`/`Lz` (and in `extended_mode =
  dominant`).  Never use a nearest-energy `argmin` for the branch label:
  it steps the surface when a low-weight state crosses a reference
  midpoint.  Folded arrays kept under `*_folded`; `wt_K`/`wt_Kp` record
  the largest single-state weight (1 = one state per extended momentum,
  ~1/2 at a Bragg anticrossing — dips are the gaps and are expected).
  The extended surface is **not periodic**, so
  `isoenergy_areas` must be called with `periodic=False` there — the
  driver does this from the stored `extended_zone` key via `_periodic`.
  Valid in the magnetic breakdown limit, which is the relevant one for
  weak moire potentials but is a physical assumption, not bookkeeping.
  Run `python semiclassical/validate_extended_zone.py` after touching
  `extended_zone.py`, `isoenergy.py`, or the zero-field k-mesh.
- **Magnetic breakdown broadening**: `breakdown = 1` (default 0,
  `semiclassical/breakdown.py`) puts the moire Bragg-plane gaps back into
  the extended-zone Onsager fan.  They do **not** enter as a level shift —
  `extended_mode = dominant` moves the level only on the plane itself and
  leaves the fan essentially unchanged — but as a level *width*:
  `Gamma = (hbar*omega_c/2pi) * sum_i sqrt(1 - exp(-B0_i/B))` over the
  Landau-Zener crossings, with `B0 = pi*Eg^2/(4*hbar*e*v_perp*v_par)` and
  `m_c = (hbar^2/2pi)|dA/dE|` from the orbit areas.  No free parameters;
  validated against `main_v3.py` at `qq = 1` (the one flux where a magnetic
  subband *is* a Landau level), median `w_exact/Gamma = 0.83` over 105
  levels at 1.9–11.6 T.  The width is 60–100% of the level spacing at
  2–12 T.  It is an **envelope**: the exact widths oscillate by ~4x
  level-to-level (coherent interference of the twelve crossings) and an
  incoherent sum of reflection amplitudes cannot reproduce that — that
  needs a Falicov-Stachowiak network, which is not implemented.  `Gamma`
  uses the true `hbar*omega_c`, which is the fan's level spacing only at
  `onsager_Bmultiplier = 1`.  Set the flag at the `isoenergy` **and**
  `onsager` stages; it needs an extended-zone band structure and raises
  without one.  `onsager_bfield` does not support it.  Literature — and
  which parts of it the implementation does *not* follow — under
  "References" in the `breakdown.py` section of
  `semiclassical/doc_technical.md`.
- The basis label system (composite strings with `_` separators, searched
  via substring intersection) is load-bearing. Any change to label
  formatting will silently break `getindices` lookups.
- k-mesh flattening uses `order='F'` (Fortran/column-major) to match
  MATLAB conventions. This is intentional and must not be changed.

## Performance

All Hofstadter drivers (`main_v3.py`, `main_v2.py`, `hofstadter_testing.py`)
pin `OPENBLAS_NUM_THREADS=1` before importing NumPy.  This prevents BLAS
thread oversubscription when using the multiprocessing pool (`isparallel=1`).
Do not remove this setting.

## Validation workflow

MATLAB is on the PATH. After any change to Hamiltonian construction or
the k-loop:

### Hofstadter

**Note:** `validate.py` uses the legacy T-matrix conventions (pre-main_v3)
and will not match MATLAB benchmarks after the hamiltonian.py update.
It is kept for reference only.

For main_v3 validation, compare against `hofstadter_testing.py` with
order=[3,1,2], conj=1, psi_conj=1, sxflag=0, dagger=0 — the spectra
should match to machine precision (~1e-11 meV).

For DOS/transport normalization and k-zone changes, run
`python validate_transport_norm.py` — checks minimal-zone vs full-zone
equivalence (machine precision), same-B invariance ((pp,qq) vs
(2pp,2qq)), and exact state counting.  All tests must PASS.

### Zero-field

1. Run `python validate_zerofield.py` — compares against
   `matlab_code/zerofield/bands_BG.mat`.
2. Max absolute error should be < 5e-6 eV (residual is from truncated
   Dq in the MATLAB benchmark; the Python code is more accurate).

## Testing parameters

### Hofstadter
The default `input_test.txt` uses `pp=1, qq=1` (strong field, small
matrices).  main_v3 uses Nq=qq (same chain size as main_v2) with
corrected T-matrix conventions; it samples the minimal k-zone
`[b1/pp, gcd(2*pp,qq)*b2/pp]` by default (`full_zone = 1` restores the
qq-extended zone).

### Zero-field
The default `input_zerofield.txt` uses `NQ=7` (49 Q-vectors, dim=196 for
bilayer, dim=98 for monolayer). The MATLAB benchmark `bands_BG.mat` uses
`theta=1°`, `nlayers=2`, `hbar_vF=5.2657`.

### Semiclassical
MATLAB benchmarks are at `<OneDrive>/MATLAB/Duartes_code/Semiclassical_zero_Field/`.
`benchmark_data_30.mat` (nk=30) and `benchmark_data_100.mat` (nk=100)
contain E_K, Oz_K, Lz_K, area_K, LLK, etc.

`semiclassical/validate_zone.py` (untracked) checks the minimal k-zone
against `full_zone = 1` at `(pp,qq) = (7,4)`: bit-identical E/Oz/Lz at
the shared k-points, exact `cell_area` invariance.  Run it after any
change to the Hofstadter k-mesh or `vol_M`.

`semiclassical/validate_extended_zone.py` checks the zero-field moire
unfolding: exactness at `V = 0`, the centroid identity `E = E_bare` at
*finite* `V` (the sharp test of the branch partition — the `argmin`
assignment this replaced missed it by 0.14 meV), Berry-curvature and
energy conservation under the centroid reduction, the `(k, Q) → extended
grid` bijection, and orbit-area monotonicity through the moire BZ
boundary.  All checks must PASS.  Run it after any change to
`extended_zone.py`, `isoenergy.py`, or the zero-field k-mesh.

Band structure quantities (E_K, Oz_K, Lz_K, kpoints, vol_M) match MATLAB
to machine precision (~1e-14 relative). Orbit areas match to machine
precision for all resolved orbits. LL fan diagrams have known remaining
differences — see below.

## Semiclassical — known issues and MATLAB differences

1. **MATLAB Onsager bug (valid mask)**: MATLAB's `get_semiclassical_LL.m`
   evaluates the Onsager condition at ALL energies, including those with
   no orbit (area=0). The `dL/dE` term (nonzero everywhere due to Fermi
   derivative broadening) can create spurious minima. Python correctly
   sets the residual to `inf` at zero-area energies. This causes LL
   differences for bands with few orbit energies (e.g., band 6).

2. **MATLAB chi bug**: `get_semiclassical_LL.m` line 31 ends with
   semicolon, making the `chiflag * dChi_dE` term a no-op. When
   comparing against MATLAB, use the `_SBM` output (which excludes chi).

3. **Berry curvature sign (RESOLVED)**: `enclosedBC` is the flux of `Oz`
   through the orbit interior with no orientation attached
   (`_shoelace_area` takes `abs`, and the interior is picked by
   `Path.contains_points`, so contour winding is never consumed).  The
   Onsager phase is the Berry phase along the direction of motion, and
   `ħk̇ = -e v×B` reverses the traversal sense under `B → -B`, so
   `phi_B = +sign(B)*enclosedBC` — **odd in B**, unlike every other term
   in the condition.  In the source the parity is the `np.abs(B2)` in the
   `base_SB` line (all other terms use `B2`) and the overall sign is the
   leading minus on that line.
   The **parity** is forced by the equation of motion.  The **overall
   sign** is fixed by validation (b) below, and now *agrees* with the
   traversal argument: an electron-like orbit at `B > 0` runs
   counterclockwise, giving `phi_B = +Phi_B` by Stokes.  The code
   previously carried the opposite sign and documented the conflict as an
   unresolved open issue; it is resolved — the traversal argument was
   right and the implementation was wrong.
   Two things follow, and conflating them is the trap: the sign is *not*
   per-orbit (it comes from the carrier charge and field direction, so it
   does not flip between electron- and hole-like orbits, and must never
   be read off contour winding), but it *is* per-field-direction.  A
   `Blist` straddling zero needs both branches in one run — the normal
   case for `onsager_bfield`, where `Blist` is the deviation `δB` from
   the background flux already baked into the band structure and `Oz`.
   Validated: (a) semiclassical Chern numbers from `Oz` reproduce the
   `main_v3.py` transport σ_xy plateau steps band by band (bands 5–11,
   `Δσ_xy` matching `C` to 3 decimals) — this constrains the sign of `Oz`
   itself, *not* the Berry term in the quantization condition, so it is
   unaffected by the correction; (b) with the parity switched off
   (`recompute_onsager.py --bc-sign-mode fixed`) the exact Hofstadter
   spectrum picks `BC_factor = +1` on `δB < 0` and `BC_factor = -1` on
   `δB > 0` at all four fluxes tested (`qq/pp` = 1/2, 1/3, 2/5, 2/3;
   folded and unfolded) — both are the single odd-in-`B` convention
   above.  The discriminator is the *crossover* at the background field,
   not the residual magnitude; margins are 3.6×/3.9× at 1/2 and 3.8×/4.9×
   at 2/3.  Only quantitative per-branch scoring separates the two
   candidate signs — the earlier, weaker claim that fans at 1/2
   qualitatively "track the exact spectrum" does not.
   **Fan files generated before 2026-07-30 are stale**: they predate both
   this sign flip and the `onsager_Bmultiplier` default change, so `_SB`,
   `_SBM`, `_SBMC` will not reproduce (`_S`/`_SM` are unaffected by the
   sign but do move with `Bmultiplier`).  Regenerate them.
   The `dL/dE` term's parity in `B` is unverified and may have the same
   problem; it is identically zero in `onsager_bfield`, so this only
   concerns the perturbative `onsager` channel at negative `B`.

   **`onsager_Bmultiplier` defaults to 4**, not 1, and is not a free
   knob: scanned against the exact Hofstadter spectrum the optimum is 4
   at every flux tested, while the two geometric candidates
   (`vol_M/A_uc`, ranging 2.25–9 over those runs, and `2*pp`, 4–10) track
   neither each other nor the optimum.  A flux-independent constant means
   a missing factor in the rhs or in the orbit-area normalization; where
   it belongs is untraced, so it rides on the rhs rather than being
   folded into `S(E)`.  At the correct value the fan gives ~one level per
   exact subband (`nLL/nEx ≈ 1.05` at 2/3); at the wrong value the Berry
   parity signature above is smeared out entirely, so **a null result on
   a sign test at the wrong λ is not evidence against the sign**.  See
   "The Bmultiplier factor" in `semiclassical/doc_technical.md`.

4. **Energy grid resolution**: The Onsager solver uses `argmin` over a
   discrete energy grid. With `kT=3 meV` broadening, energy grids coarser
   than ~2 meV cause 1-bin LL shifts. Use `elist_onsager` for a denser
   grid independent of the susceptibility grid.

5. **`include_chi` flag**: Set `include_chi = 0` in the input file to
   skip the expensive Fukuyama susceptibility loop. Defaults to 1.

6. **Lifshitz segment detection heuristic**: The Onsager solver splits
   non-monotonic area(E) at Lifshitz transitions using a magnitude-based
   heuristic (`lifshitz_threshold` × median |ΔA|, default 50). This
   works but is somewhat arbitrary. A potentially better approach:
   split at sign changes of dA/dE (local extrema), which is
   parameter-free and physically motivated. Risk: numerical noise in
   area(E) could create spurious extrema. Worth revisiting if the
   threshold needs tuning for different parameter regimes.

## Documentation requirements

**Any change to `.py` source files MUST include corresponding updates to
the documentation.** This is mandatory, not optional.

- Changes to Hamiltonian construction, basis, dimensions, or physics:
  update `doc_technical.md`
- Changes to input parameters, output format, or usage: update
  `doc_user_guide.md`
- Changes to file roles, code layout, or conventions: update this
  `CLAUDE.md`
- Changes to semiclassical code: update `semiclassical/doc_technical.md`
  and/or `semiclassical/doc_user_guide.md`

If a code change affects numerical values cited in docs (matrix
dimensions, BZ vectors, parameter defaults, etc.), grep the docs for the
old values and fix every occurrence. A past failure to do this caused
`Nq = 2*qq` to persist across four doc files when the code actually uses
`Nq = qq`.

## Transport: band selection buffer

The `transport_buffer` parameter controls how many bands beyond the
`mulist` range are included in the velocity matrix element sum.  The
Berry curvature kernel `K_n = Σ_{m≠n} Im[vx·vy*] / D_nm²` decays as
1/D² with inter-band spacing, so remote bands contribute non-negligibly.
With too few bands, σ_xy plateaus are not quantized to integers.

The default is `max(mulist_range, 500)` meV on each side.  The 500 meV
floor ensures enough remote bands are included for the Chern number sum
to converge (tested to ~10⁻⁶ accuracy for BLG LLs).  The user can
override via the `transport_buffer` input parameter, but setting it
below ~500 meV will degrade σ_xy quantization.

σ_xx uses a **separate, narrower window** (`sigma_xx_buffer`, default
`max(250Γ, 100)` meV): its summand carries `A_n A_m` and decays as 1/D⁴,
so it converges long before the Berry-curvature tail does.  Two more
knobs size the σ_xx energy grid, `eps_per_width` and `eps_grid_floor`.
σ_xy and L12_xy are evaluated in **closed form** (Φ_xy is a step
function, so the thermal integrals collapse to
`Σ_n K_n f(E_n−μ)`) and are independent of all three.  Run
`python validate_transport_kubo.py` after touching any of them — it
reruns a case with all three tightened and reports the difference.

## What not to do

- Don't add type hints, docstring expansions, or comments that restate
  what the code does. The code is already well-structured.
- Don't refactor the K/Kp valley duplication into a generic "valley" loop
  unless asked. The two valleys have different operator directions, phase
  signs, and chopping rules; keeping them explicit prevents subtle bugs.
- Don't change the `eig` -> `eigvalsh` choice. The Hamiltonian is Hermitian
  by construction and `eigvalsh` is both faster and more numerically stable.
