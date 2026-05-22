"""
Semiclassical electronic structure driver.

Stage-based pipeline for bilayer graphene on hBN:
  bandstructure → isoenergy → onsager

Each stage can be run independently via the 'calctype' input parameter.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from parser import parse_input_file
from bandstructure import do_calc


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def load_data(filepath):
    if filepath.endswith('.mat'):
        from scipy.io import loadmat
        raw = loadmat(filepath)
        data = {}
        for k, v in raw.items():
            if k.startswith('__'):
                continue
            if isinstance(v, np.ndarray):
                if v.ndim == 2 and v.shape == (1, 1):
                    data[k] = v.item()
                elif v.ndim == 2 and v.shape[0] == 1:
                    data[k] = v.ravel()
                else:
                    data[k] = v
            else:
                data[k] = v
        return data
    else:
        return dict(np.load(filepath))


def save_result(result, outfile):
    data = {k: v for k, v in result.items()}
    if outfile.endswith('.mat'):
        from scipy.io import savemat
        savemat(outfile, data)
    else:
        np.savez(outfile, **data)
    print(f"  Saved to {outfile}")


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def run_bandstructure(inp, fpath):
    result = do_calc(fpath)
    result['nk1'] = int(inp['nk1'])
    result['nk2'] = int(inp['nk2'])
    return result


def run_isoenergy(inp, bs_data):
    from isoenergy import get_energy_resolved_data

    nk1 = int(bs_data['nk1'])
    nk2 = int(bs_data['nk2'])
    nbands = bs_data['E_K'].shape[0]
    kT = float(inp.get('kT', 3.0))
    nE = int(inp['nE'])

    print(f"  Isoenergy: nE={nE} per band, {nbands} bands, kT={kT} meV")

    result = {'nbands': nbands}

    for n in range(nbands):
        emin = min(bs_data['E_K'][n].min(), bs_data['E_Kp'][n].min())
        emax = max(bs_data['E_K'][n].max(), bs_data['E_Kp'][n].max())
        E_levels_n = np.linspace(emin, emax, nE)
        result[f'E_levels_band{n}'] = E_levels_n

        print(f"    band {n}: E = [{emin:.2f}, {emax:.2f}] meV")

        for valley in ('K', 'Kp'):
            area, enclosedBC, dL_dE = get_energy_resolved_data(
                kT,
                bs_data[f'E_{valley}'][n],
                bs_data[f'Oz_{valley}'][n],
                bs_data[f'Lz_{valley}'][n],
                E_levels_n, bs_data['vol_M'], nk1, nk2)

            result[f'area_{valley}_band{n}'] = area
            result[f'enclosedBC_{valley}_band{n}'] = enclosedBC
            result[f'dL_dE_{valley}_band{n}'] = dL_dE

    print("  Done.")
    return result


def run_onsager(inp, iso_data):
    from onsager import onsager_fan_band

    Blist = np.atleast_1d(inp['Blist']).ravel()
    nmax = int(inp.get('nmax', 50))
    nbands = int(iso_data['nbands'])

    termflags_raw = np.atleast_1d(
        inp.get('termflags', np.array([1, 1, 1]))).astype(int)
    termflags = tuple(termflags_raw[:3])

    chi_data = None
    chi_E_meV = None
    if 'susceptibility_datafile' in inp:
        chi_data = load_data(inp['susceptibility_datafile'])
        chi_E_meV = np.atleast_1d(chi_data['E_list']).ravel() * 1000
        print(f"  Loaded susceptibility from {inp['susceptibility_datafile']}")

    print(f"  Onsager: {len(Blist)} B values, nmax={nmax}, termflags={termflags}")

    result = {'Blist': Blist, 'nmax': nmax}

    for valley in ('K', 'Kp'):
        n_with_orbits = 0
        for n in range(nbands):
            key = f'area_{valley}_band{n}'
            if key not in iso_data:
                continue

            E_levels_n = np.atleast_1d(iso_data[f'E_levels_band{n}']).ravel()

            dChi = None
            if chi_data is not None:
                dChi_raw = np.atleast_1d(
                    chi_data[f'dChi_dE_{valley}']).ravel()
                dChi = np.interp(E_levels_n, chi_E_meV, dChi_raw)

            LL = onsager_fan_band(
                Blist, nmax, E_levels_n,
                iso_data[key],
                iso_data[f'enclosedBC_{valley}_band{n}'],
                np.atleast_1d(iso_data[f'dL_dE_{valley}_band{n}']).ravel(),
                dChi_dE=dChi, termflags=termflags)

            if LL is not None:
                result[f'LL_{valley}_band{n}'] = LL
                n_with_orbits += 1

        print(f"  {valley} valley: {n_with_orbits} bands with orbits")

    print("  Done.")
    return result


def _onsager_bfield_worker(args):
    from isoenergy import isoenergy_areas
    from onsager import onsager_fan_band

    (B, E_bands, Lz_bands, Oz_bands,
     vol_M, nk1, nk2, nE, nmax, gfactor, termflags) = args

    nbands = E_bands.shape[0]
    Nk = nk1 * nk2
    kweight = (2 * np.pi)**2 / (Nk * vol_M)
    termflags_3 = (int(termflags[0]), 0, int(termflags[1]))

    hbar = 1.054571817e-34       # J * s
    e_charge = 1.602176634e-19   # C
    Lz_prefactor = e_charge / (2 * hbar)  # 1/m^2 when multiplied by B [T]

    band_results = {}
    for n in range(nbands):
        E_mod = E_bands[n] + gfactor * B * Lz_prefactor * Lz_bands[n]
        E_levels = np.linspace(E_mod.min(), E_mod.max(), nE)

        A, K = isoenergy_areas(E_mod, E_levels, vol_M, nk1, nk2)

        max_pockets = max((len(a) for a in A if a), default=1)
        area = np.zeros((nE, max_pockets))
        enclosedBC = np.zeros((nE, max_pockets))

        for i in range(nE):
            if not A[i]:
                continue
            for p, (a_val, k_idx) in enumerate(zip(A[i], K[i])):
                area[i, p] = a_val
                enclosedBC[i, p] = np.sum(Oz_bands[n][k_idx]) * kweight

        LL = onsager_fan_band(
            [B], nmax, E_levels, area, enclosedBC,
            np.zeros(nE), termflags=termflags_3)

        band_results[n] = {
            'area': area,
            'enclosedBC': enclosedBC,
            'E_levels': E_levels,
            'LL': LL[0] if LL is not None else None,
        }

    return band_results


def run_onsager_bfield(inp, bs_data):
    Blist = np.atleast_1d(inp['Blist']).ravel()
    nmax = int(inp.get('nmax', 50))
    nE = int(inp['nE'])
    gfactor = float(inp.get('gfactor', 1.0))
    isparallel = int(inp.get('isparallel', 0))
    nprocs = inp.get('nprocs', os.environ.get('SLURM_CPUS_PER_TASK', None))
    if nprocs is not None:
        nprocs = int(nprocs)

    termflags_raw = np.atleast_1d(
        inp.get('termflags', np.array([1, 0]))).astype(int)
    termflags = tuple(termflags_raw[:2])

    nk1 = int(bs_data['nk1'])
    nk2 = int(bs_data['nk2'])
    vol_M = float(bs_data['vol_M'])
    nbands = bs_data['E_K'].shape[0]
    nB = len(Blist)

    print(f"  onsager_bfield: {nB} B values, nmax={nmax}, nE={nE}, "
          f"gfactor={gfactor}, termflags={termflags}")

    result = {'Blist': Blist, 'nmax': nmax, 'nE': nE,
              'nbands': nbands, 'gfactor': gfactor}

    for valley in ('K', 'Kp'):
        E_v = bs_data[f'E_{valley}']
        Lz_v = bs_data[f'Lz_{valley}']
        Oz_v = bs_data[f'Oz_{valley}']

        args_list = [
            (B, E_v, Lz_v, Oz_v, vol_M, nk1, nk2,
             nE, nmax, gfactor, termflags)
            for B in Blist
        ]

        if isparallel:
            import multiprocessing
            all_results = []
            with multiprocessing.Pool(nprocs) as pool:
                for i, r in enumerate(pool.imap(_onsager_bfield_worker, args_list)):
                    all_results.append(r)
                    print(f"\r  {valley}: {100*(i+1)//nB}%", end="", flush=True)
            print()
        else:
            all_results = []
            for i, a in enumerate(args_list):
                all_results.append(_onsager_bfield_worker(a))
                print(f"\r  {valley}: {100*(i+1)//nB}%", end="", flush=True)
            print()

        for n in range(nbands):
            max_pock = max(all_results[iB][n]['area'].shape[1]
                          for iB in range(nB))

            LL_arr = np.full((nB, nmax + 1), np.nan)
            area_arr = np.zeros((nB, nE, max_pock))
            bc_arr = np.zeros((nB, nE, max_pock))
            elev_arr = np.zeros((nB, nE))

            has_orbits = False
            for iB in range(nB):
                br = all_results[iB][n]
                np_b = br['area'].shape[1]
                area_arr[iB, :, :np_b] = br['area']
                bc_arr[iB, :, :np_b] = br['enclosedBC']
                elev_arr[iB, :] = br['E_levels']
                if br['LL'] is not None:
                    LL_arr[iB, :] = br['LL']
                    has_orbits = True

            result[f'area_{valley}_band{n}'] = area_arr
            result[f'enclosedBC_{valley}_band{n}'] = bc_arr
            result[f'E_levels_{valley}_band{n}'] = elev_arr
            if has_orbits:
                result[f'LL_{valley}_band{n}'] = LL_arr

        n_with_orbits = sum(
            1 for n in range(nbands)
            if f'LL_{valley}_band{n}' in result)
        print(f"  {valley} valley: {n_with_orbits} bands with orbits")

    print("  Done.")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    fpath = sys.argv[1] if len(sys.argv) > 1 else 'input_benchmark.txt'
    inp = parse_input_file(fpath)

    calctype = str(inp.get('calctype', 'all')).lower()

    if calctype == 'bandstructure':
        print("=== Stage: bandstructure ===")
        result = run_bandstructure(inp, fpath)
        outfile = inp.get('outputfile',
                          f'electronic_structure_data_{int(inp["nk1"])}.mat')
        save_result(result, outfile)

    elif calctype == 'isoenergy':
        print("=== Stage: isoenergy ===")
        bs_data = load_data(inp['inputdata'])
        result = run_isoenergy(inp, bs_data)
        outfile = inp.get('outputfile', 'isoenergy_data.mat')
        save_result(result, outfile)

    elif calctype == 'onsager':
        print("=== Stage: onsager ===")
        iso_data = load_data(inp['inputdata'])
        result = run_onsager(inp, iso_data)
        outfile = inp.get('outputfile', 'onsager_data.mat')
        save_result(result, outfile)

    elif calctype == 'onsager_bfield':
        print("=== Stage: onsager_bfield ===")
        bs_data = load_data(inp['inputdata'])
        result = run_onsager_bfield(inp, bs_data)
        outfile = inp.get('outputfile', 'onsager_bfield_data.mat')
        save_result(result, outfile)

    elif calctype == 'all':
        print("=== Running all stages ===")
        bs_result = run_bandstructure(inp, fpath)
        result = bs_result

        if 'nE' in inp:
            iso_result = run_isoenergy(inp, bs_result)
            result = {**result, **iso_result}

            if 'Blist' in inp:
                ons_result = run_onsager(inp, iso_result)
                result = {**result, **ons_result}

        outfile = inp.get('outputfile',
                          f'electronic_structure_data_{int(inp["nk1"])}.mat')
        save_result(result, outfile)

    else:
        raise ValueError(
            f"Unknown calctype: {calctype!r}. "
            f"Choose from: bandstructure, isoenergy, onsager, onsager_bfield, all")
