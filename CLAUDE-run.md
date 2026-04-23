# CLAUDE.md — Calculation Mode

This project computes magnetic Bloch bands and density of states for
bilayer graphene on hBN at rational magnetic flux qq/pp.

## How to run calculations

Read `doc_user_guide.md` for the complete parameter reference and output
format. The essential workflow is:

1. Write an input file (or modify `input_test.txt`).
2. Run: `python main_v2.py input_file.txt`
3. Load results: `np.load('bands_p{pp}_q{qq}.npz')`

Or call directly from Python:

```python
from main_v2 import do_calc
result = do_calc('input_file.txt')
```

## Generating input files programmatically

To sweep over parameters (e.g., varying pp/qq for a Hofstadter scan),
write input files from Python. The format is simple `key = value;` lines.
Example:

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

## Output format quick reference

### calctype = 'ek'
- `kpoints`: (Nk, 2) array, k-coordinates in 1/m
- `bands_K`, `bands_Kp`: (Nk, Nbands) arrays, eigenvalues in meV

### calctype = 'dos'
- `elist`: (nebin,) array, energy grid in meV
- `dos_K`, `dos_Kp`: (nebin,) arrays, state counts per energy bin

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

## Practical limits

- Matrix dimension = 2 * qq * (2N + 1). N grows with pp.
- Nmax caps N to prevent runaway. Default 1000 → max dim ~ 4002.
- For pp > ~20 with the default LL_multiplier, expect runtimes of minutes
  to tens of minutes even with parallelization.
- Always use `isparallel = 1` unless debugging.

## Physics context

This section will grow as we learn from calculations. Check the memory
files for accumulated physics insights and observations.

### System
- Bernal-stacked bilayer graphene on hBN substrate
- Moire pattern from graphene/hBN lattice mismatch (eps ~ 1.8%)
- Perpendicular magnetic field → Landau levels → Hofstadter butterfly

### Key energy scales (default parameters)
- g0 = 2796 meV: intralayer hopping (sets overall bandwidth)
- g1 = 340 meV: interlayer coupling (sets BLG gap-related features)
- v0 = 30 meV, v1 = 21 meV: moire potential (sets miniband structure)
- Landau level spacing ~ g0 * a / (lB * sqrt(2)), depends on B

### What to look for in results
- Hofstadter butterfly: plot DOS vs qq/pp to see fractal spectrum
- Flat bands near zero energy: signature of moire + magnetic field interplay
- Valley splitting: compare K and K' spectra for broken symmetries
