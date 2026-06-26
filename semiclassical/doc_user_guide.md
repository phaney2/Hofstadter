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
| `kT`            | 3       | Thermal broadening for dL/dE (meV) |
| `Blist`         | —       | Magnetic field values (T) for Onsager quantization, e.g. `linspace(0,12,100)` |
| `nmax`          | 50      | Maximum Landau level index (used with `Blist`) |
| `term_factors`  | [1 1 1] | Multiplicative factors for Onsager correction terms: `[BC_factor morb_factor chi_factor]` (see below) |
| `susceptibility_datafile` | — | Path to susceptibility `.mat` file (for `onsager` stage with `chiflag=1`) |
| `gfactor`       | 1       | Orbital moment prefactor for `onsager_bfield`: E_mod = E_K + gfactor×B×Lz_K |
| `onsager_Bmultiplier` | 1 | Multiplicative factor on B in the Onsager rhs (`onsager_bfield` only). Diagnostic/testing parameter. |
| `lifshitz_threshold` | 50 | Lifshitz transition detection: a jump in orbit area is flagged when `|ΔA|` exceeds this factor × median `|ΔA|`. Each monotonic segment is solved independently. |
| `outputfile`    | auto    | Output filename; defaults to `electronic_structure_data_{nk1}.mat` |

### Onsager quantization terms (`term_factors`)

The Onsager quantization condition solved by the code is:

```
S(E)/(2π)² + BC_factor × Φ_B·B/(2π·φ₀)
           + morb_factor × (dL/dE)·B/(2π·φ₀)
           + chi_factor × (2π)·(dχ/dE)·B²/φ₀²
           = B·(n + ½)/φ₀
```

where `S(E)` is the orbit area in k-space, `Φ_B` is the enclosed Berry
curvature, `dL/dE` is the energy derivative of the orbital moment, and
`dχ/dE` is the Fukuyama susceptibility derivative. `φ₀ = 2πℏ/e` is the
flux quantum.

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

| Factor         | Term | Physical origin |
|---|---|---|
| `BC_factor`    | `Φ_B·B/(2π·φ₀)` | Enclosed Berry curvature — shifts the Maslov phase |
| `morb_factor`  | `(dL/dE)·B/(2π·φ₀)` | Orbital magnetic moment — energy shift of orbits in B |
| `chi_factor`   | `(2π)·(dχ/dE)·B²/φ₀²` | Fukuyama susceptibility — second-order B² correction |

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
onsager_Bmultiplier = 1
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

| Variable                  | Shape                 | Units   | Description |
|---|---|---|---|
| `Blist`                   | (nB,)                | T       | Magnetic field values |
| `nmax`                    | scalar                | —       | Maximum Landau level index |
| `nE`                      | scalar                | —       | Energy points per band |
| `nbands`                  | scalar                | —       | Number of bands |
| `gfactor`                 | scalar                | —       | Orbital moment prefactor |
| `onsager_Bmultiplier`     | scalar                | —       | B multiplier in Onsager rhs (diagnostic) |
| `LL_{v}_band{n}_SM`       | (nB, nmax+1)          | meV     | LL from area (morb in dispersion) |
| `LL_{v}_band{n}_SBM`      | (nB, nmax+1)          | meV     | + enclosed BC |
| `area_K_band{n}`          | (nB, nE, npockets)    | m^-2    | K valley orbit areas per B |
| `area_Kp_band{n}`         | (nB, nE, npockets)    | m^-2    | K' valley orbit areas per B |
| `enclosedBC_K_band{n}`    | (nB, nE, npockets)    | —       | K valley enclosed BC per B |
| `enclosedBC_Kp_band{n}`   | (nB, nE, npockets)    | —       | K' valley enclosed BC per B |
| `E_levels_K_band{n}`      | (nB, nE)              | meV     | Energy grid per B (K valley) |
| `E_levels_Kp_band{n}`     | (nB, nE)              | meV     | Energy grid per B (K' valley) |

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

Shared parameters (`nk1`, `nk2`, `bands`, `isparallel`, `outputfile`,
`U`) work identically to zero-field mode.

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
| `validate.py`       | Zero-field benchmark comparison against MATLAB `.mat` data |
| `validate_hofstadter.py` | Hofstadter benchmark comparison |
| `run.slurm`         | SLURM batch script |

## Validation

```bash
python validate.py benchmark_data_30.mat input_benchmark.txt
```

Compares all output quantities against the MATLAB benchmark.
Expected: all checks pass with relative errors < 1e-8.
