"""End-to-end check that the isoenergy stage inherits the unfolded mesh.

Builds a folded and an unfolded bandstructure file from the verified 25flux
data, runs the isoenergy stage on each through the real driver, and compares
the orbit areas.  The unfolded band must give a single smooth orbit branch
where the two sorted subbands give a fragmented pair.
"""
import os
import sys
import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'semiclassical'))
sys.path.insert(0, HERE)
from unfold import unfold_bandstructure                      # noqa: E402
from semiclassical import run_isoenergy, save_result         # noqa: E402

OUT = r'C:/Users/phaney/wrk/hofstadter/_unfold_e2e'
os.makedirs(OUT, exist_ok=True)
F = (r'C:/Users/phaney/wrk/hofstadter/calculations/semiclassical/'
     r'blackbird/moire_0.5_Um20_Em0.5/25flux.mat')
raw = loadmat(F)['results'][0, 0]

nk1 = nk2 = 200
PAIR = (7, 8)          # the reference pair; keep only it to stay fast

folded = {}
for q in ('E', 'Oz', 'Lz'):
    for v in ('K', 'Kp'):
        folded[f'{q}_{v}'] = raw[f'{q}_{v}'][list(PAIR)]
folded['kpoints'] = raw['kpoints']
folded['vol_M'] = float(raw['vol_M'].ravel()[0])
folded['nk1'], folded['nk2'] = nk1, nk2

unfolded = unfold_bandstructure(folded)

inp = {'nk1': 999, 'nk2': 999, 'nE': 60, 'kT': 3.0}   # deliberately wrong:
#   if _kmesh ever falls back to the input file instead of the data, the
#   isoenergy stage will produce garbage and this test will fail loudly.

print("\n=== isoenergy on the FOLDED pair ===")
iso_f = run_isoenergy(inp, folded)
print("\n=== isoenergy on the UNFOLDED band ===")
iso_u = run_isoenergy(inp, unfolded)

save_result(folded, os.path.join(OUT, 'bs_folded_25.mat'))
save_result(unfolded, os.path.join(OUT, 'bs_unfold_25.mat'))

fails = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


print("\n=== results ===")
check("unfolded run produced 1 band, folded produced 2",
      int(iso_u['nbands']) == 1 and int(iso_f['nbands']) == 2)

A_u = iso_u['area_K_band0']
E_u = iso_u['E_levels_K_band0']
check("unfolded orbit areas are non-degenerate", A_u.max() > 0,
      f"max area {A_u.max():.4e} m^-2, {A_u.shape[1]} pocket column(s)")

# The two sorted subbands span the same energy window as the unfolded band,
# because {A,B} is a permutation of {E_lo,E_hi}.
span_f = (min(iso_f['E_levels_K_band0'][0], iso_f['E_levels_K_band1'][0]),
          max(iso_f['E_levels_K_band0'][-1], iso_f['E_levels_K_band1'][-1]))
check("energy window preserved",
      np.isclose(E_u[0], span_f[0]) and np.isclose(E_u[-1], span_f[1]),
      f"unfolded [{E_u[0]:.4f}, {E_u[-1]:.4f}]  folded [{span_f[0]:.4f}, {span_f[1]:.4f}]")

# Physical area scale: the doubled BZ.
BZ_u = (2 * np.pi)**2 / unfolded['vol_M']
BZ_f = (2 * np.pi)**2 / folded['vol_M']
check("unfolded BZ area is twice the folded one",
      np.isclose(BZ_u, 2 * BZ_f), f"{BZ_u:.4e} vs {BZ_f:.4e} m^-2")
check("largest unfolded orbit fits inside the doubled BZ",
      A_u.max() < BZ_u, f"{A_u.max()/BZ_u:.3f} of the doubled BZ")

# Fragmentation: count energies where the leading orbit exists.
lead_u = A_u[:, 0]
n_u = int((lead_u > 0).sum())
n_f = [int((iso_f[f'area_K_band{n}'][:, 0] > 0).sum()) for n in (0, 1)]
print(f"  energies with a resolved leading orbit: unfolded {n_u}/60, "
      f"folded subbands {n_f[0]}/60 and {n_f[1]}/60")
print(f"  pocket columns: unfolded {A_u.shape[1]}, "
      f"folded {iso_f['area_K_band0'].shape[1]} / {iso_f['area_K_band1'].shape[1]}")

# Contour area vs the exact enclosed k-point count.  cell_area is invariant
# under unfolding, so both sides are directly comparable.  Near the band bottom
# the pocket is closed and simply connected, so the two must agree; the kinks in
# the sorted subbands are exactly what breaks that agreement.
cell_area = BZ_u / (unfolded['nk1'] * unfolded['nk2'])


def area_vs_count(levels, contour_area, bands):
    dev = []
    for i, lvl in enumerate(levels):
        n_below = sum(int((b < lvl).sum()) for b in bands)
        if n_below < 20:
            continue
        counted = n_below * cell_area
        traced = contour_area[i].sum()
        if traced <= 0:
            continue
        dev.append(abs(traced - counted) / counted)
    return np.array(dev)


lo_third = slice(0, 20)
dev_u = area_vs_count(E_u[lo_third], A_u[lo_third], [unfolded['E_K'][0]])

# folded: sum the contour areas of BOTH subbands at the same energies
lv_f = iso_f['E_levels_K_band0'][lo_third]
tot_f = np.zeros((len(lv_f),))
for n in (0, 1):
    lev_n = iso_f[f'E_levels_K_band{n}']
    ar_n = iso_f[f'area_K_band{n}'].sum(axis=1)
    tot_f += np.interp(lv_f, lev_n, ar_n, left=0.0, right=ar_n[-1])
dev_f = []
for i, lvl in enumerate(lv_f):
    n_below = sum(int((folded['E_K'][n] < lvl).sum()) for n in (0, 1))
    if n_below < 20 or tot_f[i] <= 0:
        continue
    dev_f.append(abs(tot_f[i] - n_below * cell_area) / (n_below * cell_area))
dev_f = np.array(dev_f)

print(f"  |traced area - counted area| / counted, lower third of the band:")
print(f"    unfolded        median {np.median(dev_u):.4f}  "
      f"max {dev_u.max():.4f}  ({len(dev_u)} levels)")
print(f"    folded subbands median {np.median(dev_f):.4f}  "
      f"max {dev_f.max():.4f}  ({len(dev_f)} levels)")
check("unfolded orbits reproduce the enclosed k-point count to <5%",
      np.median(dev_u) < 0.05, f"median {np.median(dev_u):.4f}")
check("unfolding improves agreement with the k-point count",
      np.median(dev_u) < np.median(dev_f),
      f"{np.median(dev_u):.4f} vs {np.median(dev_f):.4f}")

print("\n" + ("PIPELINE CHECKS PASSED" if not fails
              else f"{len(fails)} FAILURE(S): {fails}"))
sys.exit(1 if fails else 0)
