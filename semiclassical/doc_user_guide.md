# Semiclassical Electronic Structure — User Guide

## Quick start

```bash
python semiclassical.py input_benchmark.txt
```

This computes the moire band structure, Berry curvature, orbital moment,
and Fukuyama susceptibility, then saves everything to a `.mat` file.

## Input file

MATLAB-style key = value format.  Lines starting with `%` are comments.

### Required parameters

| Parameter    | Units | Description |
|---|---|---|
| `nk1`        | —     | k-mesh points along G1 |
| `nk2`        | —     | k-mesh points along G2 (can set `nk2 = nk1`) |
| `NQ`         | —     | Moire Q-vector grid: NQ x NQ, centered at Q=0 |
| `Nlayers`    | —     | 1 = monolayer, 2 = bilayer |
| `vF`         | eV·A  | hbar * v_F (Dirac velocity parameter) |
| `gamma1`     | eV    | Interlayer coupling (bilayer only) |
| `V0`         | meV   | Moire scalar potential |
| `V1`         | meV   | Moire vector potential |
| `moire_psi`  | rad   | Moire coupling phase |
| `eta`        | eV    | Broadening for Berry curvature and susceptibility |
| `bands`      | —     | Band offsets from center, e.g. `[-3 -2 -1 0 1 2 3]` |
| `nebin`      | —     | Number of energy bins for susceptibility |
| `elist`      | meV   | Energy list for susceptibility, e.g. `linspace(-100,100,nebin)` |

### Optional parameters

| Parameter       | Default | Description |
|---|---|---|
| `theta`         | 0       | Twist angle (radians) |
| `U`             | [0 0]   | Layer potentials [U_top U_bottom] (meV) |
| `v3`            | 0       | Trigonal warping parameter (eV·A) |
| `isparallel`    | 0       | 1 = use multiprocessing for k-loop |
| `kT`            | 3       | Thermal broadening for dL/dE (meV) |
| `Blist`         | —       | Magnetic field values (T) for Onsager quantization, e.g. `linspace(0,12,100)` |
| `nmax`          | 50      | Maximum Landau level index (used with `Blist`) |
| `elist_onsager` | `elist` | Separate energy grid (meV) for Onsager step; defaults to `elist` if omitted |
| `termflags`     | [1 1 1] | Onsager correction terms to include: `[BCflag morbflag chiflag]` (see below) |
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
nebin = 100
elist = linspace(-100,100,nebin)
NQ = 7
Nlayers = 2
vF = 5.2657
gamma1 = 0.34
v3 = 0
V0 = 30
V1 = 21
moire_psi = 0.29
eta = 2e-3
bands = [-3 -2 -1 0 1 2 3]
kT = 3
outputfile = 'results_200.mat'

% Onsager quantization (optional — omit Blist to skip)
Blist = linspace(0,12,100)
nmax = 50
elist_onsager = linspace(-100,100,500)
termflags = [1 1 0]
```

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
| `dChi_dE_K`   | (NE,)               | —       | K valley susceptibility (×hbar^4, Ang^-2 → m^-2) |
| `dChi_dE_Kp`  | (NE,)               | —       | K' valley susceptibility |
| `kpoints`     | (Nk, 2)             | Ang^-1  | k-point coordinates (kx, ky) |
| `E_list`      | (NE,)               | eV      | Energy grid for susceptibility |
| `vol_M`       | scalar               | m^2     | Moire unit cell area |

Where `nbands = len(bands)`, `Nk = nk1 * nk2`, `NE = nebin`.

### Onsager output (present when `Blist` is in the input file)

| Variable      | Shape               | Units   | Description |
|---|---|---|---|
| `Blist`       | (nB,)               | T       | Magnetic field values |
| `nmax`        | scalar               | —       | Maximum Landau level index |
| `LL_band{i}`  | (nB, nmax+1)        | meV     | Landau level energies for band i (K valley) |

One `LL_band{i}` matrix is saved per band that has closed orbits.
The index `i` is the 0-based band index within the selected bands array.

## Post-processing pipeline

### 1. Isoenergy orbit areas

```python
from isoenergy import get_energy_resolved_data
import numpy as np

E_levels = np.linspace(-100, 100, 500)   # meV
area, enclosedBC, dL_dE = get_energy_resolved_data(
    bands, kT, E_K, Oz_K, Lz_K, E_levels, vol_M, nk1, nk2)
```

Returns:
- `area[i, n, p]` — orbit area at energy i, band n, pocket p (m^-2)
- `enclosedBC[i, n, p]` — enclosed Berry curvature (dimensionless)
- `dL_dE[i, n]` — Fermi-weighted orbital moment derivative

### 2. Onsager quantization (Landau level fan)

```python
from onsager import onsager_fan

Blist = np.linspace(0, 12, 100)   # Tesla
LL, band_indices = onsager_fan(Blist, nmax=30, E_levels=E_levels,
                                area=area, enclosedBC=enclosedBC,
                                dL_dE=dL_dE)
```

Returns `LL[j]` as (nB, nmax+1) array of Landau level energies for
each band with nonzero orbits.

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

## Hofstadter mode

When `qq > 0` in the input file, the code switches to Hofstadter mode:
magnetic Bloch bands in a Landau level basis at rational flux qq/pp.

### Hofstadter input parameters

| Parameter        | Units | Description |
|---|---|---|
| `qq`             | —     | Numerator of flux ratio qq/pp (flux quanta per moire cell) |
| `pp`             | —     | Denominator of flux ratio qq/pp |
| `g0`             | meV   | Dirac velocity parameter |
| `g1`             | meV   | Interlayer coupling |
| `g3`             | meV   | Trigonal warping |
| `g4`             | meV   | Electron-hole asymmetry |
| `delta`          | meV   | Sublattice splitting |
| `v0`             | meV   | Moire scalar potential |
| `v1`             | meV   | Moire vector potential |
| `w`              | meV   | TBG interlayer coupling |
| `num_bands`      | —     | Number of bands to analyze (centered window) |
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
num_bands = 26
nremotebands = 300
nlayers = 2
bands = [-3 -2 -1 0 1 2 3]
outputfile = 'hofstadter_benchmark.mat'
```

### Hofstadter output

Same format as zero-field output: E_K, E_Kp, Oz_K, Oz_Kp, Lz_K, Lz_Kp,
kpoints, vol_M. No susceptibility (dChi_dE) in Hofstadter mode.

## Code files

| File | Purpose |
|---|---|
| `semiclassical.py` | Core engine: moire Hamiltonian, Berry curvature, orbital moment, susceptibility |
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
