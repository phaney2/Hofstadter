# Semiclassical Electronic Structure — User Guide

## Quick start

```bash
python semiclassical.py input_benchmark.txt
```

This computes the moire band structure, Berry curvature, and orbital
moment, then saves everything to a `.mat` file.

## Stage pipeline

The calculation is split into three independent stages.  By default all
stages run end-to-end (`calctype = all`), but each stage can be run
separately by setting `calctype` and providing prior results via
`inputdata`:

```
bandstructure  →  isoenergy  →  onsager
    (k-mesh)      (orbit areas)   (LL fan)

bandstructure  →  onsager_bfield
    (k-mesh)      (B-dependent orbits + LL fan)
```

| calctype          | Reads from             | Produces                                           |
|-------------------|------------------------|----------------------------------------------------|
| `bandstructure`   | physics params         | E, Berry curvature, orbital moment, k-mesh          |
| `isoenergy`       | `inputdata` (bs file)  | orbit areas, enclosed BC, dL/dE                    |
| `onsager`         | `inputdata` (iso file) | Landau level fan diagrams                          |
| `onsager_bfield`  | `inputdata` (bs file)  | B-dependent orbits + LL fan (non-perturbative Lz)  |
| `all` (default)   | physics params         | everything merged into one file                    |

Running stages separately lets you iterate on downstream parameters
(e.g. different B-field ranges or term_factors) without re-running the
expensive k-mesh calculation.

## Input file

MATLAB-style key = value format.  Lines starting with `%` are comments.

### Required parameters

| Parameter    | Units | Description |
|---|---|---|
| `nk1`        | —     | k-mesh points along G1 |
| `nk2`        | —     | k-mesh points along G2 (can set `nk2 = nk1`) |
| `NQ`         | —     | Moire Q-vector grid: NQ x NQ, centered at Q=0 |
| `Nlayers`    | —     | 1 = monolayer, 2 = bilayer |
| `g0` or `vF` | meV or eV·A | Dirac velocity: `g0` in meV (converted via `vF = g0 * 2.46 / 1000`), or `vF` directly in eV·A |
| `g1` or `gamma1` | meV or eV | Interlayer coupling (bilayer only): `g1` in meV, or `gamma1` in eV |
| `v0` (or `V0`) | meV   | Moire scalar potential |
| `v1` (or `V1`) | meV   | Moire vector potential |
| `moire_psi`  | rad   | Moire coupling phase |
| `eta`        | eV    | Broadening for Berry curvature |
| `bands`      | —     | Band offsets from center, e.g. `[-3 -2 -1 0 1 2 3]` |
| `nE`         | —     | Number of energy points per band for isoenergy orbit detection |

### Optional parameters

| Parameter       | Default | Description |
|---|---|---|
| `calctype`      | `all`   | Stage to run: `bandstructure`, `isoenergy`, `onsager`, `onsager_bfield`, or `all` |
| `inputdata`     | —       | Path to prior stage output file (required for `isoenergy`, `onsager`, `onsager_bfield`) |
| `theta`         | 0       | Twist angle between graphene and hBN (degrees; converted to radians internally) |
| `U`             | [0 0]   | Layer potentials [U_top U_bottom] (meV) |
| `g3` or `v3`    | 0       | Trigonal warping: `g3` in meV (converted via `v3 = g3 * 2.46 / 1000`), or `v3` in eV·A |
| `isparallel`    | 0       | 1 = use multiprocessing for k-loop |
| `stacking_type` | 2       | Bilayer stacking: 1 = B1-A2 (Type 1), 2 = A1-B2 (Type 2). See Moon & Koshino, PRB 90, 155406 (2014). Ignored for monolayer. |
| `moire_psi`     | 0.29    | Moire coupling phase psi (rad). |
| `extended_zone` | 0       | 1 = unfold the moire folding into the extended zone (zero-field only).  See below. |
| `extended_ntile`| 3       | Odd, `≤ NQ`.  Side of the extended zone in moire BZs. |
| `extended_mode` | `centroid` | `centroid` (breakdown-limit dispersion) or `dominant` (largest-weight state). |
| `kT`            | 3       | Thermal broadening for dL/dE (meV) |
| `iso_Erange`    | —       | `[emin emax]` (meV): restrict the isoenergy / `onsager_bfield` energy grids to this window instead of each band's full range. |
| `Blist`         | —       | Magnetic field values (T) for Onsager quantization, e.g. `linspace(0,12,100)` |
| `nmax`          | 50      | Maximum Landau level index (used with `Blist`) |
| `term_factors`  | [1 1 1] | Multiplicative factors for Onsager correction terms: `[BC_factor morb_factor chi_factor]` (see below) |
| `susceptibility_datafile` | — | Path to susceptibility `.mat` file (for `onsager` stage with `chiflag=1`) |
| `gfactor`       | 1       | Orbital moment prefactor for `onsager_bfield`: E_mod = E_K + gfactor×B×Lz_K |
| `onsager_Bmultiplier` | 4 | Multiplicative factor `λ` on B in the Onsager rhs. Not a free knob — 4 is the value set by comparison against the exact Hofstadter spectrum (see below). |
| `lifshitz_threshold` | 50 | Lifshitz transition detection: a jump in orbit area is flagged when `|ΔA|` exceeds this factor × median `|ΔA|`. Each monotonic segment is solved independently. |
| `outputfile`    | auto    | Output filename; defaults to `electronic_structure_data_{nk1}.mat` |

### Onsager quantization terms (`term_factors`)

The Onsager quantization condition solved by the code is:

```
S(E)/(2π)² = (|B|/φ₀) × [ λ × (n + ½)
                          − BC_factor   × sign(B)·Φ_B/(2π)
                          − morb_factor × (dL/dE)/(2π)
                          − chi_factor  × (2π)·(dχ/dE)·B/φ₀ ]
```

where `S(E)` is the orbit area in k-space, `Φ_B` is the enclosed Berry
curvature, `dL/dE` is the energy derivative of the orbital moment, and
`dχ/dE` is the Fukuyama susceptibility derivative. `φ₀ = 2πℏ/e` is the
flux quantum, and `λ ≡ onsager_Bmultiplier` (default 4, see below).
Comparing with the textbook form
`S/(2π)² = (|B|/φ₀)(n + ½ − φ_B/2π)` identifies the Berry phase as
`φ_B = +sign(B)·BC_factor·Φ_B`.

**Sign of the Berry curvature term.** `Φ_B` is computed as the flux of
`Oz` through the orbit interior, which is orientation-independent (the
interior is selected by point-in-polygon, not by contour winding).  The
Onsager phase, however, is the Berry phase accumulated *along the
direction of motion*, and `ℏk̇ = −e v×B` reverses the traversal sense
under `B → −B`.  So `φ_B` is **odd in B**: `φ_B = +sign(B)·Φ_B`.  Every
other term in the bracket above is even in `B`; only this one carries
`sign(B)`.  The parity is forced by the equation of motion.  The overall
sign is fixed by validation against the exact Hofstadter spectrum, and
agrees with the traversal argument (an electron-like orbit at `B > 0`
runs counterclockwise, giving `φ_B = +Φ_B` by Stokes).  Code before
2026-07-30 carried the opposite sign; fans generated with it need
regenerating, or recomputing with `BC_factor = -1`.

This matters whenever `Blist` contains both signs.  In `onsager_bfield`
the field in `Blist` is the deviation `δB` from the background flux
already built into the band structure, so `Blist` routinely straddles
zero and both branches are exercised in a single run.  The sign was
established by holding it fixed across both branches
(`recompute_onsager.py --bc-sign-mode fixed`) and scoring each branch
against the exact spectrum, at `qq/pp` = 1/2, 1/3, 2/5 and 2/3 — see
"Berry curvature sign" in the top-level `CLAUDE.md` and "Term signs" in
`doc_technical.md`.

`φ_B` is discontinuous at `B = 0`, which is harmless: the rhs vanishes
there and no Landau level is defined.

The code **always** computes four cumulative sets of Landau levels:

| Output suffix | Terms included |
|---|---|
| `_S`   | S(E) only (bare Onsager) |
| `_SB`  | S + BC_factor × enclosed Berry curvature |
| `_SBM` | S + BC + morb_factor × dL/dE |
| `_SBMC`| S + BC + morb + chi_factor × chi' (only if susceptibility data provided) |

The `term_factors` parameter is an optional 3-element array
`[BC_factor morb_factor chi_factor]` of multiplicative prefactors on each
correction term.  Default is `[1 1 1]`.  Use e.g. `[1 -1 1]` to flip the
sign of the orbital moment term.

**`onsager_bfield` reads only 2 elements.**  In the non-perturbative
channel the orbital moment is already folded into the dispersion
(`E_mod = E + gfactor·B·Lz`), so the `dL/dE` term is identically zero and
`morb_factor` is meaningless.  There, `term_factors` is
`[BC_factor chi_factor]` — the second element is the chi prefactor, not
the morb prefactor.  The stage internally expands it to
`(BC_factor, 0, chi_factor)`.  Passing 3 elements silently ignores the
third.

| Factor         | Term | Physical origin |
|---|---|---|
| `BC_factor`    | `−sign(B)·Φ_B/(2π)` | Enclosed Berry curvature — shifts the Maslov phase.  `BC_factor = 1` is the physically correct value (see sign note above); `BC_factor = -1` flips it |
| `morb_factor`  | `−(dL/dE)/(2π)` | Orbital magnetic moment — energy shift of orbits in B |
| `chi_factor`   | `−(2π)·(dχ/dE)·B/φ₀` | Fukuyama susceptibility — second-order B² correction |

### The B multiplier (`onsager_Bmultiplier`)

`λ` multiplies `B` in the right-hand side of the condition above.  It
**defaults to 4**, and despite looking like a fudge factor it is not a
free parameter: scanning it against the exact Hofstadter spectrum puts
the optimum at 4 at every flux tested (`qq/pp` = 1/2, 1/3, 2/5, 2/3;
folded and unfolded), while `vol_M/A_uc` ranges over 2.25–9 and `2·pp`
over 4–10 across those same runs.  Both of those geometric candidates
were tried and score worse.  A flux-independent constant means a missing
factor in the rhs (equivalently, in the orbit-area normalization); where
it belongs has not been traced, so it is carried explicitly here.

At `λ = 4` the fan produces about one level per exact subband
(`nLL/nEx ≈ 1.05`), and the Berry-phase sign test above discriminates by
a factor of ~4; at other `λ` the sign signature is smeared out entirely.
Change it only for testing.  It applies to both the `onsager` and
`onsager_bfield` stages, and it does not enter the modified dispersion,
so it cannot affect the band structure or `Oz`.

### Example input file

```
isparallel = 1
theta = 0
U = [0 0]
nk1 = 200
nk2 = nk1
NQ = 7
Nlayers = 2
g0 = 2140.5
g1 = 340
g3 = 0
v0 = 30
v1 = 21
moire_psi = 0.29
eta = 2e-3
bands = [-3 -2 -1 0 1 2 3]
kT = 3
outputfile = 'results_200.mat'

% Onsager quantization (optional — omit Blist to skip)
nE = 500
Blist = linspace(0,12,100)
nmax = 50
```

### Staged input files

When running stages separately, each stage only needs its own parameters
plus `inputdata` pointing to the prior stage's output.

**Stage 1 — bandstructure:**
```
calctype = 'bandstructure'
outputfile = 'bs_nk100.mat'
isparallel = 1
theta = 0
nk1 = 100
nk2 = nk1
NQ = 7
Nlayers = 2
g0 = 2140.5
g1 = 340
g3 = 0
v0 = 30
v1 = 21
moire_psi = 0.29
eta = 2e-3
bands = [-3 -2 -1 0 1 2 3]
```

**Stage 2 — isoenergy:**
```
calctype = 'isoenergy'
inputdata = 'bs_nk100.mat'
outputfile = 'iso_nk100.mat'
kT = 3
nE = 500
```

**Stage 3 — onsager:**
```
calctype = 'onsager'
inputdata = 'iso_nk100.mat'
outputfile = 'onsager_result.mat'
Blist = linspace(0,12,100)
nmax = 30
```

**Alternative Stage 3 — onsager_bfield (non-perturbative Lz):**
```
calctype = 'onsager_bfield'
inputdata = 'bs_nk100.mat'
outputfile = 'onsager_bfield_result.mat'
isparallel = 1
Blist = linspace(0,12,50)
nmax = 30
nE = 200
gfactor = 1
onsager_Bmultiplier = 4
```

This mode branches directly from bandstructure output (not isoenergy).
At each B, it forms E_mod(k) = E_K(k) + gfactor×B×Lz_K(k) and
recomputes isoenergy contours on the modified energy surface. The
orbital moment is included non-perturbatively in the dispersion, so
all output already contains the M correction.  Output keys use suffixes
`_SM` (area only) and `_SBM` (area + enclosed Berry curvature).
Intermediate orbit data (areas, enclosed BC) are saved per B value for
debugging.

## Output file

Determined by the `outputfile` parameter in the input file.
If not specified, defaults to `electronic_structure_data_{nk1}.mat`.
Use `.mat` extension for MATLAB-compatible output, `.npz` for numpy.

### Output structure (`.mat` files)

Output is nested into two top-level structs:

- `results` — all computed data (eigenvalues, Berry curvature, etc.)
- `params` — all input parameters as parsed from the input file

Access in MATLAB: `d = load('file.mat'); d.results.E_K`,
`d.params.nk1`.  Access in Python:
`d = loadmat('file.mat'); d['results']['E_K'][0,0]`.

When loading prior-stage output for downstream stages (e.g., isoenergy
reading bandstructure output), the code auto-unwraps the nested format.

### Saved variables (inside `results`)

| Variable      | Shape               | Units   | Description |
|---|---|---|---|
| `E_K`         | (nbands, Nk)        | meV     | K valley band energies |
| `E_Kp`        | (nbands, Nk)        | meV     | K' valley band energies |
| `Oz_K`        | (nbands, Nk)        | m^2     | K valley Berry curvature |
| `Oz_Kp`       | (nbands, Nk)        | m^2     | K' valley Berry curvature |
| `Lz_K`        | (nbands, Nk)        | m^2·meV | K valley orbital moment |
| `Lz_Kp`       | (nbands, Nk)        | m^2·meV | K' valley orbital moment |
| `kpoints`     | (Nk, 2)             | Ang^-1  | k-point coordinates (kx, ky) |
| `vol_M`       | scalar               | m^2     | Moire unit cell area |
| `nk1`         | scalar               | —       | k-mesh dimension (metadata for downstream stages) |
| `nk2`         | scalar               | —       | k-mesh dimension (metadata for downstream stages) |

Where `nbands = len(bands)`, `Nk = nk1 * nk2`.

Downstream stages read `nk1`, `nk2` and `vol_M` from the data rather than
from the input file, so a band structure written with `unfold = 1`
carries its doubled mesh through the pipeline automatically.  See
[Magnetic BZ unfolding](#magnetic-bz-unfolding-unfold--1).  The same holds
for `extended_zone = 1`, which additionally sets `nbands = 2*Nlayers` and
adds `wt_K` / `wt_Kp`; see
[Extended-zone unfolding](#extended-zone-unfolding-extended_zone--1).

### Isoenergy output (present when `nE` is in input, or `calctype = isoenergy`)

Per-band arrays are stored with suffix `_band{n}` where `n` is the 0-based
band index.  Each band has its own energy grid, auto-determined from the
band's energy range.

| Variable                | Shape              | Units   | Description |
|---|---|---|---|
| `nbands`                | scalar             | —       | Number of bands |
| `E_levels_K_band{n}`    | (nE,)              | meV     | Energy grid for band n, K valley |
| `E_levels_Kp_band{n}`   | (nE,)              | meV     | Energy grid for band n, K' valley |
| `area_K_band{n}`        | (nE, npockets)     | m^-2    | K valley orbit areas for band n |
| `area_Kp_band{n}`       | (nE, npockets)     | m^-2    | K' valley orbit areas for band n |
| `enclosedBC_K_band{n}`  | (nE, npockets)     | —       | K valley enclosed Berry curvature |
| `enclosedBC_Kp_band{n}` | (nE, npockets)     | —       | K' valley enclosed Berry curvature |
| `dL_dE_K_band{n}`       | (nE,)              | —       | K valley orbital moment derivative |
| `dL_dE_Kp_band{n}`      | (nE,)              | —       | K' valley orbital moment derivative |

Energy grids are per-valley: each valley uses `linspace(Emin, Emax, nE)`
from its own bandwidth, so the `nE` energy points are concentrated
within the actual band range rather than spanning the union of both valleys.

### Onsager output (present when `Blist` is in input, or `calctype = onsager`)

| Variable                   | Shape               | Units   | Description |
|---|---|---|---|
| `Blist`                    | (nB,)               | T       | Magnetic field values |
| `nmax`                     | scalar               | —       | Maximum Landau level index |
| `LL_{v}_band{i}_S`         | (nB, nmax+1)        | meV     | LL from isoenergy area only |
| `LL_{v}_band{i}_SB`        | (nB, nmax+1)        | meV     | + enclosed Berry curvature |
| `LL_{v}_band{i}_SBM`       | (nB, nmax+1)        | meV     | + dL/dE orbital moment |
| `LL_{v}_band{i}_SBMC`      | (nB, nmax+1)        | meV     | + chi' susceptibility (if data provided) |

where `{v}` is `K` or `Kp` and `{i}` is the 0-based band index.
One set of suffixed matrices is saved per band that has closed orbits.
Entries where the Onsager condition has no valid root (e.g., above the
band edge) are NaN.  When Lifshitz transitions split a band's area
curve into multiple monotonic segments, keys are further suffixed
`_seg0`, `_seg1`, etc. (e.g. `LL_K_band5_SBM_seg0`, `LL_K_band5_SBM_seg1`).

### Onsager_bfield output (`calctype = onsager_bfield`)

This stage writes **two** files.  The LL fan goes to `outputfile`; the
bulky per-B intermediates go to `<outputfile base>_detail.mat`.

Fan file (`outputfile`), nested under `results` as usual:

| Variable                  | Shape                 | Units   | Description |
|---|---|---|---|
| `Blist`                   | (nB,)                | T       | Magnetic field values |
| `nmax`                    | scalar                | —       | Maximum Landau level index |
| `nE`                      | scalar                | —       | Energy points per band |
| `nbands`                  | scalar                | —       | Number of bands |
| `gfactor`                 | scalar                | —       | Orbital moment prefactor |
| `onsager_Bmultiplier`     | scalar                | —       | B multiplier `λ` in Onsager rhs |
| `LL_{v}_band{n}_SM`       | (nB, nmax+1)          | meV     | LL from area (morb in dispersion) |
| `LL_{v}_band{n}_SBM`      | (nB, nmax+1)          | meV     | + enclosed BC |

Detail file (`<base>_detail.mat`), saved **flat** — no `results`/`params`
wrapper, since it holds no parameters:

| Variable                  | Shape                 | Units   | Description |
|---|---|---|---|
| `Blist`                   | (nB,)                | T       | Magnetic field values |
| `area_K_band{n}`          | (nB, nE, npockets)    | m^-2    | K valley orbit areas per B |
| `area_Kp_band{n}`         | (nB, nE, npockets)    | m^-2    | K' valley orbit areas per B |
| `enclosedBC_K_band{n}`    | (nB, nE, npockets)    | —       | K valley enclosed BC per B |
| `enclosedBC_Kp_band{n}`   | (nB, nE, npockets)    | —       | K' valley enclosed BC per B |
| `E_levels_K_band{n}`      | (nB, nE)              | meV     | Energy grid per B (K valley) |
| `E_levels_Kp_band{n}`     | (nB, nE)              | meV     | Energy grid per B (K' valley) |

These four arrays (`Blist`, `area`, `enclosedBC`, `E_levels`) are the
complete input to the quantization step, so the fan can be rebuilt from
the detail file alone — see `recompute_onsager.py` below.

### Re-solving the fan from detail data (`recompute_onsager.py`)

Standalone utility.  Re-runs only the Onsager root-finding, skipping the
expensive contour work, so a full 13-band / 2-valley / 200-B file takes
seconds instead of hours.  Intended for sweeping the correction
prefactors — this is the tool that resolved the enclosed Berry curvature
sign (see "Berry curvature sign" in the project `CLAUDE.md`).

```bash
# rebuild the fan from detail data
python recompute_onsager.py onsager_12_2_detail.mat \
       --ref onsager_12_2.mat --out none

# both BC signs side by side in one file
python recompute_onsager.py onsager_12_2_detail.mat \
       --ref onsager_12_2.mat --bc-factor 1,-1 --out onsager_12_2_bcsign.mat

# same, but with the B-parity switched off, so each field branch can be
# scored against the exact spectrum independently
python recompute_onsager.py onsager_25_10_unfold_detail.mat \
       --bc-sign-mode fixed --bc-factor 1,-1 --out bcfixed.mat
```

| Option | Default | Description |
|---|---|---|
| `detail` (positional) | — | path to `<name>_detail.mat` |
| `--ref FILE` | none | original fan `.mat`; supplies `nmax` and `onsager_Bmultiplier`, and every recomputed band is diffed against it |
| `--out FILE` | `<base>_recomp.mat` | output file; `none` to compare without writing |
| `--bc-factor` | `1` | comma-separated Berry curvature prefactors (= `term_factors[0]`) |
| `--bc-sign-mode` | `odd` | parity of the Berry phase in `B`.  `odd` is the production convention (phase shift carries `sign(B)`); `fixed` uses the same sign on both field branches |
| `--nmax` | from `--ref`, else 50 | maximum LL index |
| `--Bmultiplier` | from `--ref`, else 4 | B multiplier `λ` in the rhs |
| `--lifshitz-threshold` | 50 | segment-splitting threshold |
| `--bands`, `--valleys` | all | restrict the recompute |
| `--in-memory` | off | load the whole detail file at once (faster, needs several GB) |

With a single `--bc-factor` the output keys match the original file
exactly (`LL_{v}_band{n}_SM`, `..._SBM`, plus `_seg{i}` variants).  With
several, `_bcf0`, `_bcf1`, ... are appended in the order given and the
values are recorded in `bc_factors`.  Note that the `SM` (area-only)
levels are identical across BC factors by construction — only `SBM`
responds.  `bc_sign_mode` is recorded in both `results` and `params`.

`--bc-sign-mode fixed` needs no change to the solver: since
`B = sign(B)·|B|`, passing a per-field `f·sign(B)` is algebraically the
same as replacing `|B|` with `B` in the Berry term.  It is a diagnostic
for the sign question above; production fans use `odd`.

The round trip is bit-exact: rebuilding a fan from its own detail file
reproduces it to max |diff| = 0 with no NaN-pattern mismatches (verified
on `onsager_12_2.mat`, 13 bands × 2 valleys × 200 B, and on
`onsager_12_4.mat`, 90 B).  Note that fan files written **before
2026-07-30** predate the Berry curvature sign flip and the `λ = 4`
default, so their `SB`/`SBM`/`SBMC` levels will not reproduce under the
current defaults — recompute with `--bc-factor -1 --Bmultiplier <old λ>`
to match them, or regenerate.

## Post-processing pipeline

### 1. Isoenergy orbit areas (single band)

```python
from isoenergy import get_energy_resolved_data
import numpy as np

E_levels = np.linspace(-50, 50, 200)   # meV, per-band grid
area, enclosedBC, dL_dE = get_energy_resolved_data(
    kT, E_band, Oz_band, Lz_band, E_levels, vol_M, nk1, nk2)
```

Returns:
- `area[i, p]` — orbit area at energy i, pocket p (m^-2)
- `enclosedBC[i, p]` — enclosed Berry curvature (dimensionless)
- `dL_dE[i]` — Fermi-weighted orbital moment derivative

### 2. Onsager quantization (single band)

```python
from onsager import onsager_fan_band

Blist = np.linspace(0, 12, 100)   # Tesla
ll_dict = onsager_fan_band(Blist, nmax=30, E_levels=E_levels,
                           area=area, enclosedBC=enclosedBC,
                           dL_dE=dL_dE)
```

Returns a dict `{'S': LL_S, 'SB': LL_SB, 'SBM': LL_SBM}` where each
value is (nB, nmax+1).  If `dChi_dE` is provided, adds `'SBMC'`.
Returns `None` if the band has no closed orbits.

## Progress output

The band structure k-loop prints progress at every 5% completion
(both serial and parallel modes).  The percentage is updated in-place
on a single line via carriage return.

## Running on a cluster

```bash
sbatch run.slurm
```

The SLURM script sets `OPENBLAS_NUM_THREADS=1` to prevent thread
oversubscription.  `multiprocessing.Pool` automatically uses all cores
allocated by `--cpus-per-task`.

## Loading output in MATLAB

```matlab
d = load('electronic_structure_data_200.mat');
E_K = d.E_K;          % (nbands x Nk), meV
Oz_K = d.Oz_K;        % (nbands x Nk), m^2
kpoints = d.kpoints;  % (Nk x 2), Ang^-1
```

## Fukuyama susceptibility

The susceptibility calculation is a separate standalone program:

```bash
python susceptibility.py input_chi.txt
```

### Susceptibility input parameters

Same physics parameters as the bandstructure stage (`g0` or `vF`, `g1` or
`gamma1`, `g3` or `v3`, `v0`, `v1`, `moire_psi`, `NQ`, `Nlayers`,
`U`, `bands`), plus `eta` (eV) for the Green's function broadening
(converted to meV internally).

There are two modes for specifying the energy grid:

**Mode 1 — Band-adaptive (recommended):** Provide `inputdata` (a band
structure output file) and `nE`.  The code reads band energies from the
file, merges overlapping band intervals per valley, and distributes `nE`
points across only those intervals — no points are wasted in band gaps.

| Parameter    | Description |
|---|---|
| `inputdata`  | Path to band structure output file |
| `nE`         | Total number of energy points (distributed across occupied intervals) |

**Mode 2 — Explicit:** Provide `elist` directly.  The energy grid is
used as-is for both valleys.  This may include energies with no states.

| Parameter    | Description |
|---|---|
| `elist`      | Energy grid in meV, e.g. `linspace(-100,100,500)` |

### Susceptibility output

Per-valley arrays (not band-resolved — the susceptibility is a property
of the full spectrum at each energy):

| Variable      | Shape   | Units | Description |
|---|---|---|---|
| `E_list_K`    | (NE,)   | eV    | Energy grid, K valley |
| `E_list_Kp`   | (NE,)   | eV    | Energy grid, K' valley |
| `dChi_dE_K`   | (NE,)   | —     | K valley dChi/dE (×hbar^4, in m^-2 units) |
| `dChi_dE_Kp`  | (NE,)   | —     | K' valley dChi/dE |

In Mode 1 the K and K' grids may differ (each covers its own band
intervals).  In Mode 2 both valleys use the same `elist`.

To include the susceptibility correction in Onsager quantization, set
`susceptibility_datafile` in the onsager input file:

```
calctype = 'onsager'
inputdata = 'iso_nk100.mat'
susceptibility_datafile = 'chi_data.mat'
Blist = linspace(0,12,100)
nmax = 30
```

This adds a fourth cumulative level (`_SBMC`) to the output.

## Extended-zone unfolding (`extended_zone = 1`)

**Zero-field only.**  Rejected for `qq > 0`, and mutually exclusive with
`unfold = 1` — that flag unfolds the *magnetic* BZ of a Hofstadter run,
this one unfolds the *moire* BZ of a zero-field run.

### The problem it solves

The moire potential folds the graphene Dirac cone into the small moire BZ.
Once a constant-energy contour outgrows that BZ it merges with its own
periodic images, and the orbit tracer switches to the complementary corner
pockets.  A hole orbit then looks electron-like: its area *shrinks* as the
energy drops.  This happens at **zero moire potential too**, which is the
proof that it is an artifact of the zone and not of the physics.

At `theta = 0.965°`, `V0 = -6.5`, `V1 = 9.0` meV the primary valence orbit
grows to `0.82 A_BZ` and then collapses to `0.10` over the next 5 meV.
With `extended_zone = 1` the same orbit runs monotonically past `A_BZ`:

| E (meV) | −130 | −115 | −100 | −85 | −70 | −55 | −40 |
|---|---|---|---|---|---|---|---|
| folded, `A/A_BZ`   | 0.002 | 0.027 | 0.088 | 0.720 | 0.564 | 0.420 | 0.286 |
| extended, `A/A_BZ` | 1.210 | 1.032 | 0.863 | 0.704 | 0.554 | 0.412 | 0.280 |

### What it does

Each moire eigenstate is mapped back to the momentum it actually carries,
using its spectral weight on each plane-wave block, and the states sharing
one extended momentum are combined into one value per intrinsic branch.
Exact at zero moire potential; controlled and measurable at weak potential.
Full description in `doc_technical.md`.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `extended_zone`  | 0 | 1 = unfold |
| `extended_ntile` | 3 | Odd, `≤ NQ`.  Extended zone is `ntile × ntile` moire BZs.  Raise it if your orbits run off the edge. |
| `extended_mode`  | `centroid` | `centroid`: weight-weighted mean — the smooth magnetic-breakdown dispersion, and the mode that cancels the folding-induced Berry curvature.  `dominant`: largest-weight state — keeps the true eigenvalue and the O(V²) level repulsion, but jumps by the gap at each Bragg plane.  Diagnostic; the spread between the two bounds the unfolding error. |

### Effect on the output

| Key | Change |
|---|---|
| `E_K`, `E_Kp`, `Oz_*`, `Lz_*` | shape `(2*Nlayers, ntile²*nk1*nk2)` — one row per **intrinsic branch**, ascending (bilayer: 0,1 = valence, 2,3 = conduction).  `bands` is ignored. |
| `wt_K`, `wt_Kp` | new: spectral weight each branch collected.  1 = clean unfolding. |
| `kpoints` | `(ntile²*nk1*nk2, 2)`, covering the extended zone |
| `nk1`, `nk2` | each × `ntile` |
| `vol_M` | ÷ `ntile²` |

`nk1 * nk2 * cell_area` is unchanged, so absolute orbit areas stay on the
same scale.  The folded arrays are kept under `E_K_folded`, `Oz_Kp_folded`,
`Lz_K_folded`, `kpoints_folded`, `nk1_folded`, `nk2_folded`,
`vol_M_folded`.  Three bookkeeping keys are added: `extended_zone`,
`extended_ntile`, `extended_mode`.  Downstream stages pick all of this up
from the data automatically, including the fact that the extended surface
must not be tiled when tracing contours.

### Two things to check

- **Weight.** `assemble_extended` prints the branch-weight range and warns
  outside `[0.5, 1.5]`.  A clean unfolding sits at 1 (the weak-potential
  case above gives `[0.993, 1.007]`).  Far from 1 means the moire potential
  is too strong for the branches to be separable and the result is not
  meaningful.
- **Grid size.** A `ntile = 3` zone resolves orbits up to roughly
  `9 A_BZ`; beyond that the contour reaches the border and is discarded
  (silently — the level simply has no orbit).  If the largest orbits you
  need are missing, raise `extended_ntile` (cost scales as `ntile²` in
  memory, not in k-loop time).

### Restricting the energy grid

A branch on the extended zone can span 900 meV, and `linspace(Emin, Emax,
nE)` then wastes most of the grid.  `iso_Erange = [emin emax]` (meV)
clips the per-band grids for both `isoenergy` and `onsager_bfield`; bands
falling entirely outside the window are skipped.

### Validity

The unfolded orbits are the semiclassical orbits in the **magnetic
breakdown** limit, where the cyclotron energy exceeds the moire gaps and
the carrier tunnels through the Bragg planes.  The folded orbits are the
opposite limit.  Neither is right in between — that needs a coupled-orbit
network, which is not implemented.  For weak moire potentials breakdown is
the relevant limit, but this is a physical assumption, not a bookkeeping
fix, and `wt_*` is what tells you whether it holds.

Run `python semiclassical/validate_extended_zone.py` after any change to
`extended_zone.py` or the zero-field k-mesh.

## Hofstadter mode

When `qq > 0` in the input file, the code switches to Hofstadter mode:
magnetic Bloch bands in a Landau level basis at rational flux qq/pp.

### Hofstadter input parameters

| Parameter        | Units | Description |
|---|---|---|
| `qq`             | —     | Numerator of flux ratio qq/pp (flux quanta per **doubled** moire cell; flux per primitive cell is qq/(2pp)) |
| `pp`             | —     | Denominator of flux ratio qq/pp |
| `g0`             | meV   | Dirac velocity parameter |
| `g1`             | meV   | Interlayer coupling |
| `g3`             | meV   | Trigonal warping |
| `g4`             | meV   | Electron-hole asymmetry |
| `delta`          | meV   | Sublattice splitting |
| `v0`             | meV   | Moire scalar potential |
| `v1`             | meV   | Moire vector potential |
| `w`              | meV   | TBG interlayer coupling |
| `nremotebands`   | —     | Remote bands for Kubo sum (default: 300) |
| `LL_multiplier`  | —     | Controls LL basis truncation (default: 6) |
| `Nmax`           | —     | Maximum LL cutoff (default: 5000) |
| `gamma`          | —     | Reduction factor for moire coupling (default: 1) |
| `vF`             | m/s   | Fermi velocity (default: 1e6) |
| `nlayers`        | —     | 1 = monolayer, 2 = bilayer |
| `stacking_type`  | —     | Bilayer stacking: 1 = B1-A2 (Type 1), 2 = A1-B2 (Type 2, default). Ignored for monolayer. |
| `moire_psi`      | 0.29  | Moire coupling phase psi (rad). |
| `eta`            | —     | Moire coupling parameter (passed to Hamiltonian construction) |
| `eta_kubo`       | meV   | Broadening for Berry curvature Kubo sum (default: 2) |
| `unfold`         | —     | 1 = unfold the doubled magnetic BZ after the k-loop (default: 0).  See below. |
| `full_zone`      | —     | 1 = sample the qq-extended k-zone `[b1/pp, qq*b2/pp]` instead of the minimal zone `[b1/pp, gcd(2*pp,qq)*b2/pp]` (default: 0).  Identical k-averages at `qq/gcd(2*pp,qq)` times the cost; for regression tests. |

Shared parameters (`nk1`, `nk2`, `bands`, `isparallel`, `outputfile`,
`U`) work identically to zero-field mode.

### Minimal k-zone

The k-mesh spans `[b1/pp, qfac*b2/pp]` with `qfac = gcd(2*pp, qq)`, which
is the smallest zone on which the spectrum, Berry curvature and velocity
are periodic.  `nk2` therefore counts points across `qfac*b2/pp`, not
`qq*b2/pp`.

For most fluxes `qfac == qq` and this makes no difference — `(pp,qq)` =
(1,1), (2,1), (3,1), (3,2), (5,2), (7,2), (9,6) all have `qfac == qq`.
Where they differ, the qq-extended zone contains `qq/qfac` identical
copies of the data along b2 and costs that factor more to compute.  The
first such case is `(7,4)`, where `qfac = 2`: an output generated before
this convention was adopted will show every band repeating exactly under
`n2 -> n2 + nk2/2`.  Re-run it to get the correct mesh — the values
themselves are right, there are just twice as many of them as needed.
`vol_M` moves with the zone (`pp²·uc_area/(2·qfac)`), so `cell_area` and
all orbit areas are unchanged.

### Hofstadter example input

```
isparallel = 0
theta = 0.0
qq = 1
pp = 3
g0 = 2134
g1 = 340
g3 = 136
g4 = 0
delta = 0
v0 = 28.9
v1 = 21
w = 110
eta = 1e-3
U = [0 0]
nk1 = 30
nk2 = nk1
LL_multiplier = 6
gamma = 1
vF = 1e6
nremotebands = 300
nlayers = 2
bands = [-3 -2 -1 0 1 2 3]
outputfile = 'hofstadter_benchmark.mat'
```

### Hofstadter output

Same format as zero-field output: E_K, E_Kp, Oz_K, Oz_Kp, Lz_K, Lz_Kp,
kpoints, vol_M.

### Magnetic BZ unfolding (`unfold = 1`)

When `gcd(2*pp, qq) == 2` with `pp` odd — flux `qq/pp = 2/(odd)`, and
also cases like `(pp,qq) = (7,4)` — the Landau gauge forces a rectangular
construction cell on a triangular moire lattice, and the magnetic BZ
produced by the k-loop is a factor of two too small along G1.  Every
physical band then appears as **two folded subbands** that overlap in
energy, never mix, and swap hi/lo ordering across lines of exact
degeneracy.  Because the solver returns eigenvalues sorted by energy,
each subband is a kinked composite of both branches rather than a smooth
surface — orbit areas, and hence Onsager LL fans, come out wrong.

Setting `unfold = 1` runs the detection and unfolding described in
`doc_technical.md` immediately after the k-loop, before anything is
saved.  Adjacent band pairs are tested for the two degeneracy-line
families; a pair is unfolded only if **both valleys** qualify.  Detection
is from the data, never from `qq/pp` alone, so the flag is safe to leave
on for fluxes that are not folded (nothing is detected and the band
structure is returned unchanged, with a warning).

Effect on the output:

| Key | Change |
|---|---|
| `E_K`, `E_Kp`, `Oz_*`, `Lz_*` | one unfolded band per detected pair, shape `(npairs, 2*nk1*nk2)` |
| `kpoints`  | `(2*nk1*nk2, 2)`, tiled once along G1 |
| `nk1`      | doubled |
| `nk2`      | unchanged |
| `vol_M`    | halved |

The original arrays are always kept alongside, under `E_K_folded`,
`Oz_Kp_folded`, `Lz_K_folded`, `kpoints_folded`, `nk1_folded`,
`nk2_folded` and `vol_M_folded`, so the folded result remains available
for inspection without re-running the k-loop.  Three bookkeeping keys are
added: `unfold = 1`, `unfold_pairs` (shape `(npairs, 2)`, the folded band
indices making up each unfolded band) and `unfold_dropped` (folded bands
with no partner inside the `bands` window — these are **discarded**, and
a warning is printed).

Because `nk1` and `vol_M` are stored in the output, the `isoenergy`,
`onsager` and `onsager_bfield` stages pick the doubled mesh up
automatically; the `nk1`/`nk2` entries in the input file are ignored when
the data carries its own.  Note that `nk1 * nk2 * cell_area` is
unchanged by unfolding — `vol_M` halves exactly as `nk1` doubles — so
orbit areas remain on the same absolute scale.

Two consequences worth knowing:

- **Band count halves.** Request an even number of bands spanning
  complete pairs.  A `bands` window ending mid-pair loses its odd band
  to `unfold_dropped`.
- **Detection needs a reasonably fine mesh.** The degeneracy-line fit
  requires at least `min_frac` (0.9) of rows to contribute a clean
  crossing.  At `(pp,qq) = (7,4)`, `nk1 = nk2 = 48` detects every pair
  with `frac_rows = 1.000`; `nk1 = nk2 = 24` detects none.  If unfolding
  reports zero pairs on a flux you expect to be folded, try a finer mesh
  before concluding it is unfolded.
- **Berry curvature on a degeneracy line is basis-dependent.** Where the
  two subbands are exactly degenerate, `Oz_lo` and `Oz_hi` are
  individually arbitrary (LAPACK returns some combination) and only their
  sum is well defined.  Unfolding inherits this; it affects a measure-zero
  set of k-points and the total curvature is conserved exactly.

### Hofstadter susceptibility

The susceptibility calculator (`susceptibility.py`) also supports
Hofstadter mode.  When `qq > 0` is present in the input file, it uses
the Hofstadter Hamiltonian and velocity operators from
`hofstadter_system.py` instead of the zero-field plane-wave construction.
The input file needs the same Hofstadter parameters as the bandstructure
input (see above).  Both energy grid modes are supported:

**Band-adaptive (with `inputdata` + `nE`):**
```
qq = 1
pp = 4
g0 = 2134
g1 = 340
g3 = 136
g4 = 0
delta = 0
v0 = 29
v1 = 21
w = 110
eta = 1e-3
U = [0 0]
nk1 = 200
nk2 = nk1
LL_multiplier = 6
gamma = 1
vF = 1e6
nremotebands = 300
nlayers = 2
bands = [-3 -2 -1 0 1 2 3]
isparallel = 1
nE = 500
inputdata = 'bs_1_4.mat'
outputfile = 'chi_hofstadter.mat'
```

**Explicit (with `elist`):**
```
% same Hofstadter params as above, but replace nE/inputdata with:
elist = linspace(35,55,500)
```

## Code files

| File | Purpose |
|---|---|
| `semiclassical.py` | Stage-dispatch driver: load/save, run_bandstructure/isoenergy/onsager |
| `bandstructure.py` | Band structure engine: moire Hamiltonian, Berry curvature, orbital moment |
| `susceptibility.py` | Standalone Fukuyama susceptibility (dChi/dE) calculation |
| `hofstadter_system.py` | Hofstadter H/V setup and per-k-point assembly |
| `isoenergy.py`      | Grid-based orbit area detection (scipy.ndimage.label) |
| `onsager.py`        | Onsager quantization: E(B) fan diagram |
| `recompute_onsager.py` | Re-solve the fan from `*_detail.mat` with different correction prefactors |
| `unfold.py`         | Magnetic BZ unfolding for folded Hofstadter bands (`unfold = 1`) |
| `validate.py`       | Zero-field benchmark comparison against MATLAB `.mat` data |
| `validate_hofstadter.py` | Hofstadter benchmark comparison |
| `run.slurm`         | SLURM batch script |

## Validation

```bash
python validate.py benchmark_data_30.mat input_benchmark.txt
```

Compares all output quantities against the MATLAB benchmark.
Expected: all checks pass with relative errors < 1e-8.
