# CLAUDE.md — Development Mode

This project computes magnetic Bloch bands for bilayer graphene on hBN.

## Code layout

| File | Purpose |
|---|---|
| `main_v2.py` | Engine: `do_calc`, k-point solver, `main` entry point |
| `hamiltonian.py` | All Hamiltonian construction (intralayer, intermonolayer, interbilayer) |
| `numerics.py` | Math routines: Laguerre functions, F_nm matrix elements, table builder |
| `basis.py` | Label-based basis toolkit: `outer_product`, `getindices` |
| `parser.py` | MATLAB-style input file parser |
| `constants.py` | Physical constants (HBAR, Q_E, A_GRAPHENE, A_HBN) |
| `validate.py` | Benchmark comparison against MATLAB `.mat` data |
| `input_test.txt` | Default input (pp=1, qq=1) |
| `doc_technical.md` | Code structure reference |
| `doc_user_guide.md` | Input/output reference |
| `bands_p*_q*.mat` | MATLAB benchmark data |

## Before making changes

Read `doc_technical.md`. It documents the full code structure: function
layers, basis labeling system, matrix dimensions, phase conventions,
parallelization, and known subtleties. This is faster and more reliable
than re-parsing the source.

## Code conventions

- The function return contract for `do_calc` (in `main_v2.py`) is a dict
  whose keys depend on `calctype`. New calctypes add new key sets.
- Input parameters are in meV; they are converted to Joules internally.
  Final eigenvalues are converted back to meV. Don't mix unit systems.
- The basis label system (composite strings with `_` separators, searched
  via substring intersection) is load-bearing. Any change to label
  formatting will silently break `getindices` lookups.
- k-mesh flattening uses `order='F'` (Fortran/column-major) to match
  MATLAB conventions. This is intentional and must not be changed.

## Validation workflow

MATLAB is on the PATH. After any change to Hamiltonian construction or
the k-loop:

1. Run the MATLAB code to generate a `.mat` reference:
   ```
   matlab -batch "file='./input_test.txt'; ... save('bands_p1_q1.mat', ...)"
   ```
2. Run `python validate.py` — it forces `calctype='ek'` and compares
   eigenvalues against the `.mat` benchmarks.
3. Max absolute error should be < 1e-6 meV.

For changes that only affect post-processing (e.g., new calctypes), the
k-loop eigenvalues are the invariant — validate that they haven't changed.

## Testing parameters

The default `input_test.txt` uses `pp=1, qq=1` (strong field, small
matrices: dim=98). For a more demanding test, use `pp=3, qq=1` (dim=218).
Both have MATLAB `.mat` benchmarks.

## What not to do

- Don't add type hints, docstring expansions, or comments that restate
  what the code does. The code is already well-structured.
- Don't refactor the K/Kp valley duplication into a generic "valley" loop
  unless asked. The two valleys have different operator directions, phase
  signs, and chopping rules; keeping them explicit prevents subtle bugs.
- Don't change the `eig` -> `eigvalsh` choice. The Hamiltonian is Hermitian
  by construction and `eigvalsh` is both faster and more numerically stable.
