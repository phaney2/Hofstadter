# CLAUDE.md — Development Mode

This project computes moire band structures for mono- or bilayer graphene
on hBN.  Two calculation modes:

1. **Hofstadter** (`main_v2.py`): Magnetic Bloch bands in a Landau-level
   basis at rational flux qq/pp.
2. **Zero-field** (`zerofield.py`): Moire band structure via plane-wave
   expansion along a k-path through the moire BZ.

## Code layout

| File | Purpose |
|---|---|
| `main_v2.py` | Hofstadter engine: `do_calc`, k-point solver, `main` entry point |
| `hamiltonian.py` | All Hamiltonian construction (intralayer, intermonolayer, interbilayer) |
| `numerics.py` | Math routines: Laguerre functions, F_nm matrix elements, table builder |
| `basis.py` | Label-based basis toolkit: `outer_product`, `getindices` |
| `parser.py` | MATLAB-style input file parser (shared) |
| `constants.py` | Physical constants (shared) |
| `zerofield.py` | Zero-field engine: moire geometry, plane-wave Hamiltonian, k-path solver |
| `validate.py` | Hofstadter benchmark comparison against MATLAB `.mat` data |
| `validate_zerofield.py` | Zero-field benchmark comparison against `bands_BG.mat` |
| `plot_zerofield.m` | MATLAB plotting script for zero-field band structure |
| `input_test.txt` | Default Hofstadter input (pp=1, qq=1) |
| `input_zerofield.txt` | Default zero-field input |
| `doc_technical.md` | Code structure reference |
| `doc_user_guide.md` | Input/output reference |
| `bands_p*_q*.mat` | Hofstadter MATLAB benchmark data |
| `matlab_code/zerofield/` | Original MATLAB zero-field code and benchmark (`bands_BG.mat`) |

## Before making changes

Read `doc_technical.md`. It documents the full code structure: function
layers, basis labeling system, matrix dimensions, phase conventions,
parallelization, and known subtleties. This is faster and more reliable
than re-parsing the source.

## Code conventions

- The function return contract for `do_calc` (in `main_v2.py`) is a dict
  whose keys depend on `calctype`. New calctypes add new key sets.
- **Hofstadter units**: Input parameters are in meV; converted to Joules
  internally. Final eigenvalues are converted back to meV.
- **Zero-field units**: Input parameters are in meV; converted to eV
  internally. Eigenvalues are output in eV.
- The basis label system (composite strings with `_` separators, searched
  via substring intersection) is load-bearing. Any change to label
  formatting will silently break `getindices` lookups.
- k-mesh flattening uses `order='F'` (Fortran/column-major) to match
  MATLAB conventions. This is intentional and must not be changed.

## Validation workflow

MATLAB is on the PATH. After any change to Hamiltonian construction or
the k-loop:

### Hofstadter

1. Run the MATLAB code to generate a `.mat` reference:
   ```
   matlab -batch "file='./input_test.txt'; ... save('bands_p1_q1.mat', ...)"
   ```
2. Run `python validate.py` — it forces `calctype='ek'` and compares
   eigenvalues against the `.mat` benchmarks.
3. Max absolute error should be < 1e-6 meV.

### Zero-field

1. Run `python validate_zerofield.py` — compares against
   `matlab_code/zerofield/bands_BG.mat`.
2. Max absolute error should be < 5e-6 eV (residual is from truncated
   Dq in the MATLAB benchmark; the Python code is more accurate).

## Testing parameters

### Hofstadter
The default `input_test.txt` uses `pp=1, qq=1` (strong field, small
matrices: dim=98). For a more demanding test, use `pp=3, qq=1` (dim=218).
Both have MATLAB `.mat` benchmarks.

### Zero-field
The default `input_zerofield.txt` uses `NQ=7` (49 Q-vectors, dim=196 for
bilayer, dim=98 for monolayer). The MATLAB benchmark `bands_BG.mat` uses
`theta=1°`, `nlayers=2`, `hbar_vF=5.2657`.

## What not to do

- Don't add type hints, docstring expansions, or comments that restate
  what the code does. The code is already well-structured.
- Don't refactor the K/Kp valley duplication into a generic "valley" loop
  unless asked. The two valleys have different operator directions, phase
  signs, and chopping rules; keeping them explicit prevents subtle bugs.
- Don't change the `eig` -> `eigvalsh` choice. The Hamiltonian is Hermitian
  by construction and `eigvalsh` is both faster and more numerically stable.
