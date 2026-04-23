# CLAUDE.md — Calculation Mode

This project computes magnetic Bloch bands and density of states for
bilayer graphene on hBN at rational magnetic flux qq/pp.

## Source code location

The Hofstadter code lives at: `C:\Users\phaney\wrk\hofstadter\source_code`
Full parameter reference: `source_code\doc_user_guide.md`

To import from this directory in a Python script:

```python
import sys
sys.path.insert(0, r'C:\Users\phaney\wrk\hofstadter\source_code')
from main_v2 import main, do_calc
```

## How to run calculations

1. Write an input file (MATLAB-style `key = value;` lines).
2. Run: `python main_v2.py input_file.txt`
3. Load results from the `outputfile` path (defaults to `bands_p{pp}_q{qq}.npz`).
   Use `.mat` extension for MATLAB format, `.npz` for NumPy.

Or call directly from Python:

```python
result = do_calc('input_file.txt')
```

## Windows multiprocessing

The code uses `multiprocessing.Pool`. On Windows, any script that calls
`main()` or `do_calc()` **must** guard the entry point:

```python
if __name__ == '__main__':
    main(input_file)
```

Without this, child processes re-execute module-level code and crash with
`RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.`

### Pickle size limit for large matrices

Windows pipes have a size limit for pickling data to Pool workers. When
`dim_per_layer` exceeds ~1300 (e.g., large qq with moderate N), the
serialization fails with `OSError: [Errno 22] Invalid argument`.

dim_per_layer = qq * (2*N + 1). Large qq and/or large N triggers this.
Lower g0 increases N (smaller Landau level spacing), making this more
likely.

**Fix: use subprocess.run() per calculation instead of calling main() in
a loop.** Create a small worker script (`_run_single.py`) that imports
and runs `main()`, then call it via `subprocess.run()` from the sweep
script. See `MK_params/run_dos_sweep.py` and `MK_params/_run_single.py`
for the working pattern. This ensures each calculation gets a fresh
process and avoids memory/pipe accumulation.

## P-core affinity (i7-14700K)

This machine has 8 P-cores (logical CPUs 0-15, hyperthreaded) and
12 E-cores (CPUs 16-27). Pin computation to P-cores at script startup:

```python
import ctypes
kernel32 = ctypes.windll.kernel32
handle = kernel32.OpenProcess(0x0200, False, kernel32.GetCurrentProcessId())
kernel32.SetProcessAffinityMask(handle, 0xFFFF)
kernel32.CloseHandle(handle)
```

Place this before importing main_v2 so child processes inherit the mask.

## Generating input files programmatically

The input format is simple `key = value;` lines. Example helper:

```python
def write_input(filepath, pp, qq, calctype='dos', **overrides):
    defaults = dict(
        isparallel=1, theta=0.0, qq=qq, pp=pp,
        g0=2796, g1=340, g3=0, g4=0, delta=0,
        v0=30, v1=21, w=110, eta=2,
        nk1=10, nk2=10, LL_multiplier=6, Nmax=1000,
        nebin=1000,
    )
    defaults.update(overrides)
    with open(filepath, 'w') as f:
        for k, v in defaults.items():
            f.write(f"{k} = {v};\n")
        f.write(f"calctype = '{calctype}';\n")
        f.write("valley = {'K', 'Kp'};\n")
        f.write("U = 0*[1 1];\n")
        if calctype == 'dos':
            f.write(f"elist = linspace(-300, 300, {defaults['nebin']});\n")
```

## Sweeping over magnetic flux (Hofstadter butterfly)

Flux per unit cell = qq/pp. To sweep B-field, enumerate reduced fractions
qq/pp with pp up to some pmax. Use `math.gcd` to reduce fractions and
avoid duplicates. See `run_dos_sweep.py` for a working example that:
- Takes a target number of flux values and pmax
- Finds the closest reduced fraction for each target
- Skips already-computed output files
- Runs calculations sequentially (each one parallelizes internally)

Typical naming: `dos_p{pp}_q{qq}.mat` for output, `input_p{pp}_q{qq}.txt`
for input files.

## Output format quick reference

All input parameters are stored in the output file.

### calctype = 'ek'
- `kpoints`: (Nk, 2) array, k-coordinates in 1/m
- `bands_K`, `bands_Kp`: (Nk, Nbands) arrays, eigenvalues in meV

### calctype = 'dos'
- `elist`: (nebin,) array, energy grid in meV
- `dos_K`, `dos_Kp`: (nebin,) arrays, state counts per energy bin

### File formats
- `.npz`: params stored with `input_` prefix (e.g., `input_pp`, `input_g0`)
- `.mat`: params stored in a `params` struct (e.g., `params.pp`, `params.g0`)

## Magnetic field

B is not set directly. It follows from pp, qq, and the moire lattice:

```
B = (qq/pp) * Phi_0 / A_moire
```

For the default parameters (theta=0, graphene/hBN):
- pp=1, qq=1: B ~ 24.3 T
- pp=3, qq=1: B ~ 8.1 T
- pp=10, qq=1: B ~ 2.4 T

Larger pp = weaker field = more Landau levels = bigger matrices = slower.

## Practical limits and runtime

- Matrix dimension = 2 * qq * (2N + 1). N grows with pp.
- Nmax caps N to prevent runaway. Default 1000 -> max dim ~ 4002.
- Always use `isparallel = 1` unless debugging.
- pmax=10: all flux values complete in ~1 minute total.
- pmax=20: ~104 flux values, ~30-40 minutes total. Individual p=19/20
  cases with large qq can take several minutes each.
- For pmax > 20, expect long runtimes. Consider reducing nk or Nmax.
- nk1=nk2=5 is sufficient for butterfly plots; increase for smoother DOS.

## Plotting

`plot_butterfly.m` (MATLAB) and `plot_butterfly.py` (Python) both
auto-discover all `dos_p*_q*.mat` files in the calculations directory,
compute B from stored pp/qq, and plot occupied energy bins vs B.

B is computed from pp/qq using:
```
HBAR=1.05e-34; Q_E=1.6e-19; A_GRAPHENE=2.46e-10; A_HBN=2.504e-10;
eps = A_HBN/A_GRAPHENE - 1;
L_moire = (1+eps)*A_GRAPHENE / sqrt(eps^2 + 2*(1+eps)*(1-cos(theta)));
uc_area = sqrt(3)/2 * L_moire^2;
phi_0 = HBAR*2*pi/Q_E;
B = (qq/pp) * phi_0 / uc_area;
```

## Physics context

### System
- Bernal-stacked bilayer graphene on hBN substrate
- Moire pattern from graphene/hBN lattice mismatch (eps ~ 1.8%)
- Perpendicular magnetic field -> Landau levels -> Hofstadter butterfly

### Key energy scales (default parameters)
- g0 = 2796 meV: intralayer hopping (sets overall bandwidth)
- g1 = 340 meV: interlayer coupling (sets BLG gap-related features)
- v0 = 30 meV, v1 = 21 meV: moire potential (sets miniband structure)
- Landau level spacing ~ g0 * a / (lB * sqrt(2)), depends on B

### What to look for in results
- Hofstadter butterfly: plot DOS vs qq/pp to see fractal spectrum
- Flat bands near zero energy: signature of moire + magnetic field interplay
- Valley splitting: compare K and K' spectra for broken symmetries
