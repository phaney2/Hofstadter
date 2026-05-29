# User Guide

Moire band structure solvers for mono- or bilayer graphene on hBN.

Two calculation modes:

- **Hofstadter** (`main_v2.py`): Magnetic Bloch bands at rational flux
  qq/pp, using a Landau-level basis.
- **Zero-field** (`zerofield.py`): Moire band structure along a k-path
  through the moire BZ, using a plane-wave expansion.

---

## Quick start

### Hofstadter (magnetic field)

```bash
python main_v2.py                   # default input_test.txt
python main_v2.py my_params.txt     # custom input
```

Output defaults to `bands_p{pp}_q{qq}.npz`.

### Zero-field (no magnetic field)

```bash
python zerofield.py                     # default input_zerofield.txt
python zerofield.py my_params.txt       # custom input
```

Output defaults to `bands_zerofield.npz`.  Plot in MATLAB:

```matlab
plot_zerofield    % loads bands_zerofield.mat
```

Output file format (`.npz` or `.mat`) is detected from the extension.

---

## Input file format

Plain text, one parameter per line, MATLAB-style assignment:

```
variable = value;
```

Lines starting with `%` are comments.  Later lines can reference earlier
variables (e.g., `elist` can use `nebin`).

### Hofstadter parameter reference

#### Required physical parameters

| Parameter | Type | Example | Description |
|---|---|---|---|
| `pp` | int | `3` | Denominator of flux fraction qq/pp |
| `qq` | int | `1` | Numerator of flux fraction qq/pp |
| `g0` | float (meV) | `2796` | Graphene intralayer hopping |
| `g1` | float (meV) | `340` | BLG interlayer coupling (gamma1). Not required for `nlayers=1`. |
| `g3` | float (meV) | `0` | Trigonal warping (gamma3). Not required for `nlayers=1`. |
| `g4` | float (meV) | `0` | Electron-hole asymmetry (gamma4). Not required for `nlayers=1`. |
| `delta` | float (meV) | `0` | Sublattice mass (bilayer only; ignored for `nlayers=1`) |
| `v0` | float (meV) | `30` | hBN uniform moire potential |
| `v1` | float (meV) | `21` | hBN modulated moire potential |
| `w` | float (meV) | `110` | Interlayer coupling scale (used for LL cutoff estimate) |

#### Computation control

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nlayers` | int | `2` | Number of graphene layers: 1 = monolayer, 2 = bilayer |
| `theta` | float (deg) | `0.0` | Twist angle between graphene and hBN |
| `eta` | float | `2` | AA/AB ratio (legacy) |
| `U` | array (meV) | `0*[1 1]` | Layer on-site energies: scalar for monolayer, `[top, bottom]` for bilayer |
| `nk1` | int | `10` | k-mesh points along b1 |
| `nk2` | int | `10` | k-mesh points along b2 |
| `LL_multiplier` | int | `6` | Controls Landau level cutoff N |
| `Nmax` | int | `1000` | Hard cap on number of Landau levels |
| `isparallel` | int | `1` | 0 = serial, 1 = parallel k-loop |

#### Output control

| Parameter | Type | Default | Description |
|---|---|---|---|
| `calctype` | string | `'ek'` | `'ek'` = band structure, `'dos'` = density of states |
| `valley` | cell | `{'K', 'Kp'}` | Which valleys to compute |
| `nebin` | int | `1000` | Number of energy bins (for dos mode) |
| `elist` | array | `linspace(-300,300,nebin)` | Energy grid in meV (for dos mode) |
| `outputfile` | string | `bands_p{pp}_q{qq}.npz` | Output filename; use `.mat` extension for MATLAB format |

### Zero-field parameter reference

#### Physical parameters

| Parameter | Type | Example | Description |
|---|---|---|---|
| `g0` | float (meV) | `2472` | Graphene intralayer hopping (determines hbar*vF) |
| `g1` | float (meV) | `340` | BLG interlayer coupling (gamma1). Not used for `nlayers=1`. |
| `g3` | float (meV) | `0` | Trigonal warping (gamma3). Not used for `nlayers=1`. |
| `v0` | float (meV) | `29.8` | hBN uniform moire potential |
| `v1` | float (meV) | `21` | hBN modulated moire potential |
| `vF` (or `hbar_vF`) | float (eV*A) | `5.2657` | Dirac velocity (optional; overrides g0-derived value) |

#### Computation control

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nlayers` | int | `2` | 1 = monolayer, 2 = bilayer |
| `theta` | float (deg) | `0.0` | Twist angle between graphene and hBN |
| `U` | array (meV) | `[0 0]` | Layer on-site energies: scalar for monolayer, `[top, bottom]` for bilayer |
| `NQ` | int | `7` | Q-vector grid size per direction (total: NQ^2 plane waves) |
| `dk` | float (1/A) | `5e-4` | k-point spacing along the path |
| `valley` | cell | `{'K', 'Kp'}` | Which valleys to compute |
| `outputfile` | string | `bands_zerofield.npz` | Output filename |

---

## Output format

Two output formats are supported, selected by the `outputfile` extension:

- **`.npz`** (NumPy): load with `np.load('file.npz')`
- **`.mat`** (MATLAB): load with `scipy.io.loadmat` or MATLAB's `load`

All input parameters are stored in the output file alongside the results.

### `.npz` format

Results are top-level keys.  Input parameters are stored with an `input_`
prefix (e.g., `input_pp`, `input_g0`).

### `.mat` format

Results are top-level variables.  Input parameters are stored in a
`params` struct (e.g., `params.pp`, `params.g0`).

### Result keys by calctype

#### `calctype = 'ek'`

| Key | Shape | Units | Description |
|---|---|---|---|
| `kpoints` | (Nk, 2) | 1/m | k-point coordinates |
| `bands_K` | (Nk, Nbands) | meV | Sorted eigenvalues, K valley |
| `bands_Kp` | (Nk, Nbands) | meV | Sorted eigenvalues, K' valley |

where `Nk = nk1 * nk2` and `Nbands = nlayers * qq * (2*N + 1)`.

#### `calctype = 'dos'`

| Key | Shape | Units | Description |
|---|---|---|---|
| `elist` | (nebin,) | meV | Energy grid |
| `dos_K` | (nebin,) | counts | States per bin, K valley |
| `dos_Kp` | (nebin,) | counts | States per bin, K' valley |

### Zero-field output

| Key | Shape | Units | Description |
|---|---|---|---|
| `band_K` | (NT, dim) | eV | Sorted eigenvalues along k-path, K valley |
| `band_Kp` | (NT, dim) | eV | Sorted eigenvalues along k-path, K' valley |
| `k_region_K` | (NT,) | -- | Linearized k-path parameter [0, 1], K valley |
| `k_region_Kp` | (NT,) | -- | Linearized k-path parameter [0, 1], K' valley |
| `tick_positions_K` | (4,) | -- | High-symmetry point positions on k_region |
| `tick_labels_K` | (4,) | -- | Labels: K1, G, K2, K1 |
| `tick_positions_Kp` | (4,) | -- | High-symmetry point positions on k_region |
| `tick_labels_Kp` | (4,) | -- | Labels: K2, K1, G, K2 |
| `dim` | int | -- | Hamiltonian dimension |

where `NT` is the total number of k-points and `dim = 2*NQ^2` (monolayer)
or `4*NQ^2` (bilayer).  Multiply eigenvalues by 1000 for meV.

---

## Programmatic usage

```python
from main_v2 import do_calc

result = do_calc('input_test.txt')

# Input parameters are always available
params = result['params']        # dict of all input file parameters
print(params['pp'], params['qq'])

if result['calctype'] == 'ek':
    kpoints = result['kpoints']
    bands_K = result['bands_K']       # shape (Nk, Nbands), meV
    bands_Kp = result['bands_Kp']

elif result['calctype'] == 'dos':
    elist = result['elist']           # energy grid, meV
    dos_K = result['dos_K']           # histogram counts
    dos_Kp = result['dos_Kp']
```

### Zero-field programmatic usage

```python
from zerofield import do_calc

result = do_calc('input_zerofield.txt')

band_K = result['band_K']           # shape (NT, dim), eV
band_Kp = result['band_Kp']
k_region = result['k_region_K']     # [0, 1] parameter
ticks = result['tick_positions_K']  # high-symmetry points
labels = result['tick_labels_K']    # ['K1', 'G', 'K2', 'K1']
```

---

## Example input files

### Band structure at flux 1/1

```
isparallel = 1;
theta = 0.0;
qq = 1;
pp = 1;
g0 = 2796;
g1 = 340;
g3 = 0;
g4 = 0;
delta = 0;
v0 = 30;
v1 = 21;
w = 110;
eta = 2;
U = 0*[1 1];
nk1 = 10;
nk2 = 10;
LL_multiplier = 6;
Nmax = 1000;
calctype = 'ek';
valley = {'K', 'Kp'};
```

### Monolayer graphene on hBN at flux 1/1

```
nlayers = 1;
isparallel = 1;
theta = 0.0;
qq = 1;
pp = 1;
g0 = 2796;
v0 = 30;
v1 = 21;
w = 110;
eta = 2;
nk1 = 10;
nk2 = 10;
LL_multiplier = 6;
Nmax = 1000;
calctype = 'ek';
valley = {'K', 'Kp'};
```

### DOS at flux 1/5

```
isparallel = 1;
theta = 0.0;
qq = 1;
pp = 5;
g0 = 2796;
g1 = 340;
g3 = 0;
g4 = 0;
delta = 0;
v0 = 30;
v1 = 21;
w = 110;
eta = 2;
U = 0*[1 1];
nk1 = 10;
nk2 = 10;
LL_multiplier = 6;
Nmax = 1000;
calctype = 'dos';
valley = {'K', 'Kp'};
nebin = 1000;
elist = linspace(-300, 300, nebin);
```

### Zero-field bilayer (matching MATLAB benchmark)

```
theta = 1.0;
nlayers = 2;
g0 = 2472;
hbar_vF = 5.2657;
g1 = 340;
g3 = 0;
v0 = 29.8;
v1 = 21;
U = [0 0];
NQ = 7;
dk = 5e-4;
valley = {'K', 'Kp'};
outputfile = 'bands_zerofield.mat';
```

### Zero-field monolayer

```
theta = 0.0;
nlayers = 1;
g0 = 2134;
v0 = 29.8;
v1 = 21;
U = [0 0];
NQ = 7;
dk = 5e-4;
valley = {'K', 'Kp'};
outputfile = 'bands_zerofield.mat';
```

---

## Magnetic field

The magnetic field B is not set directly.  It is determined by `pp`, `qq`,
and the moire lattice:

```
B = (qq/pp) * Phi_0 / A_moire
```

where `Phi_0 = h/e` is the flux quantum and `A_moire` is the moire unit
cell area.  Larger `pp` means smaller B.

---

## Convergence parameters

- **`LL_multiplier`**: Controls how many Landau levels to include. Higher
  values give more accurate high-energy states but increase matrix size and
  runtime quadratically. Value of 6 is usually sufficient.

- **`Nmax`**: Hard cap on N. For large pp (weak field), N grows and this
  cap prevents excessive matrix sizes. Matrix dimension scales as
  `nlayers * qq * (2*N + 1)`.

- **`nk1`, `nk2`**: k-mesh density. For DOS calculations, 10x10 is a
  starting point; increase for smoother histograms.

- **`nebin`**: Energy resolution for DOS output. 1000 bins over a 600 meV
  window gives 0.6 meV resolution.

---

## Runtime scaling

The dominant costs are:

1. **Hamiltonian setup** (F_nm tables): O(N^2) per valley. Done once.
2. **k-loop**: `nk1 * nk2` eigenvalue problems, each O(dim^3) where
   `dim = 2 * qq * (2*N + 1)`.

Typical runtimes (parallel on ~28 cores):

| pp | qq | N | dim | Setup | k-loop (100 pts) |
|---|---|---|---|---|---|
| 1 | 1 | 24 | 98 | ~1s | ~1s |
| 3 | 1 | 54 | 218 | ~5s | ~3s |
| 10 | 1 | ~200 | ~802 | ~60s | ~30s |

Large pp and/or large qq will increase both N and the matrix dimension.
