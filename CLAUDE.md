# CLAUDE.md — Development Mode

This project computes moire band structures for mono- or bilayer graphene
on hBN.  Two calculation modes:

1. **Hofstadter** (`main_v3.py`): Magnetic Bloch bands in a Landau-level
   basis at rational flux qq/pp.  Uses corrected moire coupling matrices
   (order=[3,1,2], conj=1, psi_conj=1) and doubled guiding-center chain
   (Nq=2*qq).  Legacy driver `main_v2.py` is kept for reference.
2. **Zero-field** (`zerofield.py`): Moire band structure via plane-wave
   expansion along a k-path through the moire BZ.

## Code layout

| File | Purpose |
|---|---|
| `main_v3.py` | **Primary** Hofstadter engine: corrected moire coupling, chain doubling (Nq=2*qq) |
| `main_v2.py` | Legacy Hofstadter engine (Nq=qq, old T-matrix conventions) |
| `hofstadter_testing.py` | Convention explorer: sweep order/conj/psi flags to find correct T matrices |
| `hamiltonian.py` | All Hamiltonian construction (intralayer, intermonolayer, interbilayer, testing variants) |
| `numerics.py` | Math routines: Laguerre functions, F_nm matrix elements, table builder |
| `basis.py` | Label-based basis toolkit: `outer_product`, `getindices` |
| `parser.py` | MATLAB-style input file parser (shared) |
| `constants.py` | Physical constants (shared) |
| `zerofield.py` | Zero-field engine: moire geometry, plane-wave Hamiltonian, k-path solver |
| `validate.py` | Hofstadter benchmark against MATLAB `.mat` data (uses legacy conventions) |
| `validate_zerofield.py` | Zero-field benchmark comparison against `bands_BG.mat` |
| `plot_zerofield.m` | MATLAB plotting script for zero-field band structure |
| `input_test.txt` | Default Hofstadter input (pp=1, qq=1) |
| `input_zerofield.txt` | Default zero-field input |
| `doc_technical.md` | Code structure reference |
| `doc_user_guide.md` | Input/output reference |
| `bands_p*_q*.mat` | Hofstadter MATLAB benchmark data |
| `matlab_code/zerofield/` | Original MATLAB zero-field code and benchmark (`bands_BG.mat`) |
| `matlab_code/` | Original MATLAB Hofstadter code |
| `matlab_debugging/` | MATLAB scripts for debugging/comparison |

## Before making changes

Read `doc_technical.md`. It documents the full code structure: function
layers, basis labeling system, matrix dimensions, phase conventions,
parallelization, and known subtleties. This is faster and more reliable
than re-parsing the source.

## Code conventions

- The function return contract for `do_calc` (in `main_v3.py` / `main_v2.py`)
  is a dict whose keys depend on `calctype`. New calctypes add new key sets.
- **Hofstadter units**: Input parameters are in meV; converted to Joules
  internally. Final eigenvalues are converted back to meV.
- **Zero-field units**: Input parameters are in meV; converted to eV
  internally. Eigenvalues are output in eV.
- The basis label system (composite strings with `_` separators, searched
  via substring intersection) is load-bearing. Any change to label
  formatting will silently break `getindices` lookups.
- k-mesh flattening uses `order='F'` (Fortran/column-major) to match
  MATLAB conventions. This is intentional and must not be changed.

## Performance

All Hofstadter drivers (`main_v3.py`, `main_v2.py`, `hofstadter_testing.py`)
pin `OPENBLAS_NUM_THREADS=1` before importing NumPy.  This prevents BLAS
thread oversubscription when using the multiprocessing pool (`isparallel=1`).
Do not remove this setting.

## Validation workflow

MATLAB is on the PATH. After any change to Hamiltonian construction or
the k-loop:

### Hofstadter

**Note:** `validate.py` uses the legacy T-matrix conventions (pre-main_v3)
and will not match MATLAB benchmarks after the hamiltonian.py update.
It is kept for reference only.

For main_v3 validation, compare against `hofstadter_testing.py` with
order=[3,1,2], conj=1, psi_conj=1, sxflag=0, dagger=0 — the spectra
should match to machine precision (~1e-11 meV).

### Zero-field

1. Run `python validate_zerofield.py` — compares against
   `matlab_code/zerofield/bands_BG.mat`.
2. Max absolute error should be < 5e-6 eV (residual is from truncated
   Dq in the MATLAB benchmark; the Python code is more accurate).

## Testing parameters

### Hofstadter
The default `input_test.txt` uses `pp=1, qq=1` (strong field, small
matrices).  With main_v3 (Nq=2*qq), matrix dimensions are doubled
relative to main_v2 for the same flux.

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
