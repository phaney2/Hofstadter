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
| `validate_zerofield.py` | Zero-field benchmark comparison against `bands_BG.mat` |
| `plot_zerofield.m` | MATLAB plotting script for zero-field band structure |
| `input_test.txt` | Default Hofstadter input (pp=1, qq=1) |
| `input_zerofield.txt` | Default zero-field input |
| `doc_technical.md` | Code structure reference |
| `doc_user_guide.md` | Input/output reference |
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
| `isoenergy.py` | Contour-based isoenergy orbit detection (marching squares + shoelace area) |
| `onsager.py` | Onsager quantization solver: S(E)/(2π)² + corrections = B(n+½)/φ₀ |
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

3. **Enclosed Berry curvature sign**: There is an unresolved systematic
   shift in LL positions for bands in the +40 to +60 meV range (K valley).
   Flipping the sign of the enclosed BC term partially corrects it.
   Likely a contour winding / "inside vs outside" convention difference
   between Python's `find_contours` + `Path.contains_points` and MATLAB's
   `contourc` + `inpolygon`. Needs investigation.

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

## What not to do

- Don't add type hints, docstring expansions, or comments that restate
  what the code does. The code is already well-structured.
- Don't refactor the K/Kp valley duplication into a generic "valley" loop
  unless asked. The two valleys have different operator directions, phase
  signs, and chopping rules; keeping them explicit prevents subtle bugs.
- Don't change the `eig` -> `eigvalsh` choice. The Hamiltonian is Hermitian
  by construction and `eigvalsh` is both faster and more numerically stable.
