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
(e.g. different B-field ranges or termflags) without re-running the
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
| `kT`            | 3       | Thermal broadening for dL/dE (meV) |
| `Blist`         | —       | Magnetic field values (T) for Onsager quantization, e.g. `linspace(0,12,100)` |
| `nmax`          | 50      | Maximum Landau level index (used with `Blist`) |
| `termflags`     | [1 1 1] | Onsager correction terms to include: `[BCflag morbflag chiflag]` (see below) |
| `susceptibility_datafile` | — | Path to susceptibility `.mat` file (for `onsager` stage with `chiflag=1`) |
| `gfactor`       | 1       | Orbital moment prefactor for `onsager_bfield`: E_mod = E_K + gfactor×B×Lz_K |
| `outputfile`    | auto    | Output filename; defaults to `electronic_structure_data_{nk1}.mat` |

### Onsager quantization terms (`termflags`)

The Onsager quantization condition solved by the code is:

```
S(E)/(2π)² + BCflag × Φ_B·B/(2π·φ₀)
           + morbflag × (dL/dE)·B/(2π·φ₀)
           + chiflag × (2π)·(dχ/dE)·B²/φ₀²
           = B·(n + ½)/φ₀
```

where `S(E)` is the orbit area in k-space, `Φ_B` is the enclosed Berry
curvature, `dL/dE` is the energy derivative of the orbital moment, and
`dχ/dE` is the Fukuyama susceptibility derivative. `φ₀ = 2πℏ/e` is the
flux quantum.

The `termflags` parameter is a 3-element array `[BCflag morbflag chiflag]`
that toggles each correction term (1 = include, 0 = exclude):

| Flag       | Term | Physical origin |
|---|---|---|
| `BCflag`   | `Φ_B·B/(2π·φ₀)` | Enclosed Berry curvature — shifts the Maslov phase |
| `morbflag` | `(dL/dE)·B/(2π·φ₀)` | Orbital magnetic moment — energy shift of orbits in B |
| `chiflag`  | `(2π)·(dχ/dE)·B²/φ₀²` | Fukuyama susceptibility — second-order B² correction |

Setting `termflags = [1 1 1]` (default) includes all three corrections.
Setting `termflags = [1 1 0]` matches the MATLAB benchmark (which has an
inactive `chiflag` term due to a code bug). Setting `termflags = [0 0 0]`
gives bare Onsager quantization with no corrections.

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
termflags = [1 1 0]
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
termflags = [0 1 0]
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
termflags = [1 0]
```

This mode branches directly from bandstructure output (not isoenergy).
At each B, it forms E_mod(k) = E_K(k) + gfactor×B×Lz_K(k) and
recomputes isoenergy contours on the modified energy surface. The
orbital moment is included non-perturbatively, so `morbflag` is forced
to 0 internally. The `termflags` array is 2-element: `[BCflag chiflag]`.
Intermediate orbit data (areas, enclosed BC) are saved per B value for
debugging.

## Output file

Determined by the `outputfile` parameter in the input file.
If not specified, defaults to `electronic_structure_data_{nk1}.mat`.
Use `.mat` extension for MATLAB-compatible output, `.npz` for numpy.

### Saved variables

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

| Variable      | Shape               | Units   | Description |
|---|---|---|---|
| `Blist`       | (nB,)               | T       | Magnetic field values |
| `nmax`        | scalar               | —       | Maximum Landau level index |
| `LL_band{i}`  | (nB, nmax+1)        | meV     | Landau level energies for band i (K valley) |

One `LL_band{i}` matrix is saved per band that has closed orbits.
The index `i` is the 0-based band index within the selected bands array.

### Onsager_bfield output (`calctype = onsager_bfield`)

| Variable                  | Shape                 | Units   | Description |
|---|---|---|---|
| `Blist`                   | (nB,)                | T       | Magnetic field values |
| `nmax`                    | scalar                | —       | Maximum Landau level index |
| `nE`                      | scalar                | —       | Energy points per band |
| `nbands`                  | scalar                | —       | Number of bands |
| `gfactor`                 | scalar                | —       | Orbital moment prefactor |
| `LL_K_band{n}`            | (nB, nmax+1)          | meV     | K valley LL energies |
| `LL_Kp_band{n}`           | (nB, nmax+1)          | meV     | K' valley LL energies |
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
LL = onsager_fan_band(Blist, nmax=30, E_levels=E_levels,
                      area=area, enclosedBC=enclosedBC,
                      dL_dE=dL_dE)
```

Returns `LL` as (nB, nmax+1) array of Landau level energies, or
`None` if the band has no closed orbits.

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

| Parameter    | Units | Description |
|---|---|---|
| `nk1`, `nk2` | —    | k-mesh dimensions (same as bandstructure) |
| `NQ`         | —     | Moire Q-vector grid |
| `Nlayers`    | —     | 1 = monolayer, 2 = bilayer |
| `vF`         | eV·A  | hbar * v_F |
| `gamma1`     | eV    | Interlayer coupling |
| `V0`         | meV   | Moire scalar potential |
| `V1`         | meV   | Moire vector potential |
| `moire_psi`  | rad   | Moire coupling phase |
| `eta`        | eV    | Broadening |
| `nebin`      | —     | Number of energy bins |
| `elist`      | meV   | Energy grid, e.g. `linspace(-100,100,nebin)` |

### Susceptibility output

| Variable      | Shape   | Units | Description |
|---|---|---|---|
| `dChi_dE_K`   | (NE,)   | —     | K valley susceptibility derivative (×hbar^4, Ang^-2 → m^-2) |
| `dChi_dE_Kp`  | (NE,)   | —     | K' valley susceptibility derivative |
| `E_list`      | (NE,)   | eV    | Energy grid |

To include the susceptibility correction in Onsager quantization, set
`susceptibility_datafile` in the onsager input file and use
`termflags = [1 1 1]`:

```
calctype = 'onsager'
inputdata = 'iso_nk100.mat'
susceptibility_datafile = 'chi_data.mat'
termflags = [1 1 1]
Blist = linspace(0,12,100)
nmax = 30
```

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
| `eta`            | eV    | Broadening for Berry curvature |

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
input (see above), plus the `elist` energy grid:

```
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
isparallel = 1
elist = linspace(-100,100,200)
outputfile = 'chi_hofstadter.mat'
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
