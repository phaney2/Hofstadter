# User Guide

Compute magnetic Bloch bands or density of states for bilayer graphene on
hBN at rational magnetic flux `qq/pp` per moire unit cell.

---

## Quick start

```bash
# Run with default input file
python main_v2.py

# Run with custom input file
python main_v2.py my_params.txt
```

Output is saved to `bands_p{pp}_q{qq}.npz`.

---

## Input file format

Plain text, one parameter per line, MATLAB-style assignment:

```
variable = value;
```

Lines starting with `%` are comments.  Later lines can reference earlier
variables (e.g., `elist` can use `nebin`).

### Complete parameter reference

#### Required physical parameters

| Parameter | Type | Example | Description |
|---|---|---|---|
| `pp` | int | `3` | Denominator of flux fraction qq/pp |
| `qq` | int | `1` | Numerator of flux fraction qq/pp |
| `g0` | float (meV) | `2796` | Graphene intralayer hopping |
| `g1` | float (meV) | `340` | BLG interlayer coupling (gamma1) |
| `g3` | float (meV) | `0` | Trigonal warping (gamma3) |
| `g4` | float (meV) | `0` | Electron-hole asymmetry (gamma4) |
| `delta` | float (meV) | `0` | Sublattice mass |
| `v0` | float (meV) | `30` | hBN uniform moire potential |
| `v1` | float (meV) | `21` | hBN modulated moire potential |
| `w` | float (meV) | `110` | Interlayer coupling scale |

#### Computation control

| Parameter | Type | Default | Description |
|---|---|---|---|
| `theta` | float (rad) | `0.0` | Twist angle |
| `eta` | float | `2` | AA/AB ratio (legacy) |
| `U` | array (meV) | `0*[1 1]` | Layer on-site energies [top, bottom] |
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

---

## Output format

Output is a `.npz` file (load with `np.load`).  Contents depend on
`calctype`:

### `calctype = 'ek'`

| Key | Shape | Units | Description |
|---|---|---|---|
| `kpoints` | (Nk, 2) | 1/m | k-point coordinates |
| `bands_K` | (Nk, Nbands) | meV | Sorted eigenvalues, K valley |
| `bands_Kp` | (Nk, Nbands) | meV | Sorted eigenvalues, K' valley |

where `Nk = nk1 * nk2` and `Nbands = 2 * qq * (2*N + 1)`.

### `calctype = 'dos'`

| Key | Shape | Units | Description |
|---|---|---|---|
| `elist` | (nebin,) | meV | Energy grid |
| `dos_K` | (nebin,) | counts | States per bin, K valley |
| `dos_Kp` | (nebin,) | counts | States per bin, K' valley |

---

## Programmatic usage

```python
from main_v2 import do_calc

result = do_calc('input_test.txt')

if result['calctype'] == 'ek':
    kpoints = result['kpoints']
    bands_K = result['bands_K']       # shape (Nk, Nbands), meV
    bands_Kp = result['bands_Kp']

elif result['calctype'] == 'dos':
    elist = result['elist']           # energy grid, meV
    dos_K = result['dos_K']           # histogram counts
    dos_Kp = result['dos_Kp']
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
  `2 * qq * (2*N + 1)`.

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
