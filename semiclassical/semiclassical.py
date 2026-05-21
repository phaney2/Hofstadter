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
    bands_sel = np.arange(nbands)
    kT = float(inp.get('kT', 3.0))

    if 'elist_onsager' in inp:
        E_levels = np.atleast_1d(inp['elist_onsager']).ravel()
    elif 'elist' in inp:
        E_levels = np.atleast_1d(inp['elist']).ravel()
    else:
        raise ValueError("isoenergy stage requires 'elist_onsager' or 'elist'")

    print(f"  Isoenergy: {len(E_levels)} energy levels, kT={kT} meV")

    result = {'E_levels': E_levels}

    for valley in ('K', 'Kp'):
        area, enclosedBC, dL_dE = get_energy_resolved_data(
            bands_sel, kT,
            bs_data[f'E_{valley}'], bs_data[f'Oz_{valley}'],
            bs_data[f'Lz_{valley}'],
            E_levels, bs_data['vol_M'], nk1, nk2)

        result[f'area_{valley}'] = area
        result[f'enclosedBC_{valley}'] = enclosedBC
        result[f'dL_dE_{valley}'] = dL_dE

    print("  Done.")
    return result


def run_onsager(inp, iso_data):
    from onsager import onsager_fan

    Blist = np.atleast_1d(inp['Blist']).ravel()
    nmax = int(inp.get('nmax', 50))
    E_levels = np.atleast_1d(iso_data['E_levels']).ravel()

    termflags_raw = np.atleast_1d(
        inp.get('termflags', np.array([1, 1, 1]))).astype(int)
    termflags = tuple(termflags_raw[:3])

    chi_data = None
    if 'susceptibility_datafile' in inp:
        chi_data = load_data(inp['susceptibility_datafile'])
        print(f"  Loaded susceptibility from {inp['susceptibility_datafile']}")

    print(f"  Onsager: {len(Blist)} B values, nmax={nmax}, termflags={termflags}")

    result = {'Blist': Blist, 'nmax': nmax}

    for valley in ('K', 'Kp'):
        dChi = None
        if chi_data is not None:
            dChi = np.atleast_1d(chi_data.get(f'dChi_dE_{valley}')).ravel()

        LL_all, band_indices = onsager_fan(
            Blist, nmax, E_levels,
            iso_data[f'area_{valley}'],
            iso_data[f'enclosedBC_{valley}'],
            iso_data[f'dL_dE_{valley}'],
            dChi_dE=dChi, termflags=termflags)

        for LL, bi in zip(LL_all, band_indices):
            result[f'LL_{valley}_band{bi}'] = LL

        print(f"  {valley} valley: {len(LL_all)} bands with orbits")

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

    elif calctype == 'all':
        print("=== Running all stages ===")
        bs_result = run_bandstructure(inp, fpath)

        if 'Blist' in inp:
            iso_result = run_isoenergy(inp, bs_result)
            ons_result = run_onsager(inp, iso_result)
            result = {**bs_result, **iso_result, **ons_result}
        else:
            result = bs_result

        outfile = inp.get('outputfile',
                          f'electronic_structure_data_{int(inp["nk1"])}.mat')
        save_result(result, outfile)

    else:
        raise ValueError(
            f"Unknown calctype: {calctype!r}. "
            f"Choose from: bandstructure, isoenergy, onsager, all")
