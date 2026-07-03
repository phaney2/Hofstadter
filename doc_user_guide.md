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
| `layer_resolved` | int | `0` | 1 = compute per-eigenstate layer weights (bilayer only; uses `eigh` instead of `eigvalsh`) |
| `stacking_type` | int | `2` | Bilayer stacking: 1 = B1-A2 (Type 1), 2 = A1-B2 (Type 2). See Moon & Koshino, PRB 90, 155406 (2014), Eqs. 25 and B1. Ignored for monolayer. |
| `moire_psi` | float (rad) | `0.29` | Moire coupling phase psi. |

#### Output control

| Parameter | Type | Default | Description |
|---|---|---|---|
| `calctype` | string | `'ek'` | `'ek'` = band structure, `'dos'` = density of states, `'transport'` = Kubo transport coefficients |
| `valley` | cell | `{'K', 'Kp'}` | Which valleys to compute |
| `nebin` | int | `1000` | Number of energy bins (for dos mode) |
| `elist` | array | `linspace(-300,300,nebin)` | Energy grid in meV (for dos mode) |
| `outputfile` | string | `bands_p{pp}_q{qq}.npz` | Output filename; use `.mat` extension for MATLAB format |
| `mulist` | array (meV) | `linspace(-50,50,200)` | Chemical potential grid (transport mode only) |
| `Gamma` | float or array (meV) | `1.0` | Broadening parameter (transport mode only). Scalar or list of values. In constant mode: Lorentzian half-width(s); when a list is given, transport coefficients are computed for each Γ value at negligible extra cost (the expensive diagonalization runs once). In SCBA mode: disorder strength Γ₀ (must be scalar). |
| `transport_buffer` | float (meV) | max(mulist range, 500) | Energy buffer beyond mulist range for Kubo band selection. Must be large enough to include remote bands that contribute to the Berry curvature sum. Default: max of mulist width and 500 meV. |
| `kT` | float (meV) | `0.0` | Thermal energy for Fermi-Dirac occupation (transport mode only). 0 = zero temperature (step function). |
| `mu_ref` | float (meV) | (none) | Reference chemical potential for sigma_xy (transport mode only). When set, sigma_xy is computed relative to this value: sigma_xy(mu_ref) = 0. Place in a spectral gap to get integer-quantized Hall conductivity in neighboring gaps. |
| `broadening` | string | `'constant'` | Broadening model: `'constant'` = fixed Lorentzian width, `'scba'` = self-consistent Born approximation (energy-dependent Γ(E)). |
| `scba_mixing` | float | `0.3` | SCBA linear mixing parameter α (0 < α ≤ 1). Used for the first iteration and as fallback if Anderson mixing is singular. |
| `scba_tol` | float | `1e-4` | SCBA convergence tolerance (relative max change in Γ). |
| `scba_maxiter` | int | `200` | Maximum SCBA iterations. |
| `scba_floor` | float | `0.01` | Minimum Γ(E)/Γ₀ ratio; prevents Γ from vanishing in spectral gaps. |
| `scba_anderson` | int | `5` | Anderson/Pulay mixing depth (number of prior iterations retained). Set to 0 for pure linear mixing. |
| `scba_xy_constant` | int | `0` | If 1, use constant Γ₀ (instead of Γ(E_n)) in the σ_xy Berry curvature kernel. Default 0 = use SCBA broadening for σ_xy. |

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
| `stacking_type` | int | `2` | Bilayer stacking: 1 = B1-A2 (Type 1), 2 = A1-B2 (Type 2). Ignored for monolayer. |
| `moire_psi` | float (rad) | `0.29` | Moire coupling phase psi. |
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

Output is nested into two top-level structs:

- `results` — all computed data (eigenvalues, k-points, etc.)
- `params` — all input parameters as parsed from the input file

Access in MATLAB: `d = load('file.mat'); d.results.bands_K`,
`d.params.pp`.  Access in Python:
`d = loadmat('file.mat'); d['results']['bands_K'][0,0]`.

### Result keys by calctype

#### `calctype = 'ek'`

| Key | Shape | Units | Description |
|---|---|---|---|
| `kpoints` | (Nk, 2) | 1/m | k-point coordinates |
| `bands_K` | (Nk, Nbands) | meV | Sorted eigenvalues, K valley |
| `bands_Kp` | (Nk, Nbands) | meV | Sorted eigenvalues, K' valley |

where `Nk = nk1 * nk2` and `Nbands = nlayers * qq * (2*N + 1)`.

With `layer_resolved = 1` (bilayer only), the output additionally includes:

| Key | Shape | Units | Description |
|---|---|---|---|
| `weights_K` | (Nk, Nbands) | -- | Top-layer weight per eigenstate, K valley |
| `weights_Kp` | (Nk, Nbands) | -- | Top-layer weight per eigenstate, K' valley |

Bottom-layer weight is `1 - weights_K`.

#### `calctype = 'dos'`

| Key | Shape | Units | Description |
|---|---|---|---|
| `elist` | (nebin,) | meV | Energy grid |
| `dos_K` | (nebin,) | counts | States per bin, K valley |
| `dos_Kp` | (nebin,) | counts | States per bin, K' valley |

With `layer_resolved = 1` (bilayer only), the output additionally includes:

| Key | Shape | Units | Description |
|---|---|---|---|
| `dos_K_top` | (nebin,) | counts | Top-layer-weighted DOS, K valley |
| `dos_K_bottom` | (nebin,) | counts | Bottom-layer-weighted DOS, K valley |
| `dos_Kp_top` | (nebin,) | counts | Top-layer-weighted DOS, K' valley |
| `dos_Kp_bottom` | (nebin,) | counts | Bottom-layer-weighted DOS, K' valley |

`dos_K_top + dos_K_bottom = dos_K` (exact to machine precision).

**DOS normalization**: The DOS arrays are normalized so that each
eigenvalue contributes `1/Nk_tot` to its bin (i.e., the sum over all bins
and bands equals the number of bands). To convert to a physical density
of states per unit area, multiply by `1 / (A_uc * pp)`, where `A_uc` is
the moire unit cell area and `pp` is the flux denominator. This accounts
for the magnetic zone folding that maps the moire BZ onto the physical
magnetic BZ.

#### `calctype = 'transport'`

| Key | Shape | Units | Description |
|---|---|---|---|
| `mulist` | (n_mu,) | meV | Chemical potential grid |
| `Gamma_list` | (n_gamma,) | meV | Gamma values used (only present when Gamma is a list with n_gamma > 1) |
| `dos_K` | (n_mu,) | counts | Crude histogram DOS (states per bin), K valley |
| `dos_Kp` | (n_mu,) | counts | Crude histogram DOS (states per bin), K' valley |
| `dos_broad_K` | (n_mu,) or (n_gamma, n_mu) | states/eV/cell | Lorentzian-broadened DOS, K valley |
| `dos_broad_Kp` | (n_mu,) or (n_gamma, n_mu) | states/eV/cell | Lorentzian-broadened DOS, K' valley |
| `sigma_xx_K` | (n_mu,) or (n_gamma, n_mu) | e²/h | Longitudinal conductivity, K valley |
| `sigma_xy_K` | (n_mu,) or (n_gamma, n_mu) | e²/h | Hall conductivity, K valley |
| `L12_xx_K` | (n_mu,) or (n_gamma, n_mu) | e²/h × eV | Longitudinal thermoelectric (L12), K valley |
| `L12_xy_K` | (n_mu,) or (n_gamma, n_mu) | e²/h × eV | Transverse thermoelectric (L12), K valley |
| `sigma_xx_Kp` | (n_mu,) or (n_gamma, n_mu) | e²/h | Longitudinal conductivity, K' valley |
| `sigma_xy_Kp` | (n_mu,) or (n_gamma, n_mu) | e²/h | Hall conductivity, K' valley |
| `L12_xx_Kp` | (n_mu,) or (n_gamma, n_mu) | e²/h × eV | Longitudinal thermoelectric (L12), K' valley |
| `L12_xy_Kp` | (n_mu,) or (n_gamma, n_mu) | e²/h × eV | Transverse thermoelectric (L12), K' valley |
| `broadening` | string | -- | Broadening model used: `'constant'` or `'scba'` |
| `Gamma_E_grid` | (n_E,) | meV | Energy grid for SCBA Γ(E) (SCBA mode only) |
| `Gamma_E` | (n_E,) | meV | Self-consistent broadening Γ(E) (SCBA mode only) |
| `scba_niter` | int | -- | Number of SCBA iterations to convergence (SCBA mode only) |

When `Gamma` is a scalar (default), all transport arrays are 1D with shape
`(n_mu,)` — fully backward compatible.  When `Gamma` is a list of n_gamma
values, transport arrays become 2D with shape `(n_gamma, n_mu)`.  The
crude histogram DOS (`dos_K`, `dos_Kp`) is always 1D since it is
Gamma-independent.  SCBA mode requires scalar Gamma (a list triggers a
warning and only the first value is used as Γ₀).

Two DOS outputs are included: `dos_K`/`dos_Kp` is a crude eigenvalue
histogram (states per mulist bin, weight 1/Nk per eigenvalue), and
`dos_broad_K`/`dos_broad_Kp` is a Lorentzian-broadened DOS on the mulist
grid in units of states/eV/cell.  The broadened DOS uses constant Γ₀
(CBA) or the SCBA Γ(E) evaluated at each probe energy, with
normalization `1/(π Nk 2pp)` consistent with the SCBA self-consistency
equation.

Uses the standard interband Kubo formula for sigma_xy (broadened Berry
curvature) and Kubo-Greenwood formula for sigma_xx (two spectral
functions).

With `broadening = 'scba'`, the longitudinal conductivity (sigma_xx,
L12_xx) uses an energy-dependent broadening Γ(E) determined by the
self-consistent Born approximation, which captures the suppression of
σ_xx in narrow subbands due to localization.  The Hall conductivity
(sigma_xy, L12_xy) also uses Γ(E_n) by default, sharpening features at
low flux where subbands are narrower than Γ₀.  Set
`scba_xy_constant = 1` to revert to constant Γ₀ in σ_xy.

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

elif result['calctype'] == 'transport':
    mulist = result['mulist']             # meV
    sigma_xy = result['sigma_xy_K']       # e^2/h
    sigma_xx = result['sigma_xx_K']       # e^2/h
    L12_xy = result['L12_xy_K']           # e^2/h * eV
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

### Transport coefficients at flux 1/3

```
isparallel = 1;
theta = 0.0;
qq = 1;
pp = 3;
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
calctype = 'transport';
valley = {'K'};
mulist = linspace(-100, 100, 400);
Gamma = 2.0;
kT = 1.0;
mu_ref = 16.0;
outputfile = 'transport_p3_q1.mat';
```

### Transport with multiple Gamma values

```
isparallel = 1;
theta = 0.0;
qq = 1;
pp = 3;
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
calctype = 'transport';
valley = {'K'};
mulist = linspace(-100, 100, 400);
Gamma = [0.5, 1.0, 2.0, 4.0];
kT = 1.0;
mu_ref = 16.0;
outputfile = 'transport_multiG_p3_q1.mat';
```

### Transport with SCBA broadening

```
isparallel = 1;
theta = 0.0;
qq = 1;
pp = 3;
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
calctype = 'transport';
valley = {'K'};
mulist = linspace(-100, 100, 400);
Gamma = 2.0;
broadening = 'scba';
kT = 1.0;
mu_ref = 16.0;
outputfile = 'transport_scba_p3_q1.mat';
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
