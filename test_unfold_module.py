"""Validate semiclassical/unfold.py against the verified reference result.

Checks the auto-detected pairing, the exact algebraic identities the unfolding
must satisfy, conservation of Berry curvature, the smoothness improvement, and
the mesh/vol_M bookkeeping handed downstream.
"""
import os
import sys
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'semiclassical'))
from unfold import unfold_bandstructure, branch_label, _cone_minima  # noqa: E402

F = (r'C:/Users/phaney/wrk/hofstadter/calculations/semiclassical/'
     r'blackbird/moire_0.5_Um20_Em0.5/25flux.mat')
raw = loadmat(F)['results'][0, 0]

nk1 = nk2 = 200
res = {k: raw[k] for k in ('E_K', 'E_Kp', 'Oz_K', 'Oz_Kp', 'Lz_K', 'Lz_Kp',
                           'kpoints')}
res['vol_M'] = float(raw['vol_M'].ravel()[0])
res['nk1'], res['nk2'] = nk1, nk2

print("=== unfold_bandstructure ===")
new = unfold_bandstructure(res)

fails = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


print("\n=== detection ===")
check("pairs are (1,2),(3,4),...,(11,12)",
      new['unfold_pairs'].tolist() == [[n, n + 1] for n in range(1, 12, 2)],
      str(new['unfold_pairs'].tolist()))
check("band 0 dropped as unpaired", new['unfold_dropped'].tolist() == [0])

print("\n=== mesh bookkeeping ===")
check("nk1 doubled", new['nk1'] == 2 * nk1)
check("nk2 unchanged", new['nk2'] == nk2)
check("vol_M halved", np.isclose(new['vol_M'], res['vol_M'] / 2))
check("cell_area unchanged (what isoenergy actually uses)",
      np.isclose((2 * np.pi)**2 / new['vol_M'] / (new['nk1'] * new['nk2']),
                 (2 * np.pi)**2 / res['vol_M'] / (nk1 * nk2)))
check("kpoints shape", new['kpoints'].shape == (2 * nk1 * nk2, 2))
check("E_K shape", new['E_K'].shape == (6, 2 * nk1 * nk2))
check("folded backup preserved bit-for-bit",
      np.array_equal(new['E_K_folded'], res['E_K'])
      and np.array_equal(new['Oz_Kp_folded'], res['Oz_Kp'])
      and new['nk1_folded'] == nk1
      and np.isclose(new['vol_M_folded'], res['vol_M']))

# k-mesh must remain the affine map (n1/nk1)*G1 + (n2/nk2)*G2 on the new grid
kp = new['kpoints'].reshape(nk2, 2 * nk1, 2, order='F')
G1 = nk1 * (kp[0, 1] - kp[0, 0])
G2 = nk2 * (kp[1, 0] - kp[0, 0])
f1 = (np.arange(2 * nk1) / nk1)[None, :, None]
f2 = (np.arange(nk2) / nk2)[:, None, None]
check("kpoints affine on the doubled grid",
      np.abs(kp - (f1 * G1 + f2 * G2)).max() < 1e-12,
      f"max dev {np.abs(kp - (f1 * G1 + f2 * G2)).max():.2e}")
check("doubled G1 equals the folded zone width",
      np.allclose(G1, nk1 * (res['kpoints'][nk2] - res['kpoints'][0])))

print("\n=== exact identities, every pair, both valleys ===")
w = np.arange(2 * nk1) % nk1
worst = {k: 0.0 for k in ('sum', 'shift', 'per1', 'oz', 'lz')}
for j, (n, m) in enumerate(new['unfold_pairs']):
    for v in ('K', 'Kp'):
        lo = res[f'E_{v}'][n].reshape(nk2, nk1, order='F')
        hi = res[f'E_{v}'][m].reshape(nk2, nk1, order='F')
        A = new[f'E_{v}'][j].reshape(nk2, 2 * nk1, order='F')
        # complementary branch by label selection, not float subtraction
        sel = (A != lo[:, w]).astype(int)
        B = np.where(sel == 0, hi[:, w], lo[:, w])
        Olo = res[f'Oz_{v}'][n].reshape(nk2, nk1, order='F')
        Ohi = res[f'Oz_{v}'][m].reshape(nk2, nk1, order='F')
        AO = new[f'Oz_{v}'][j].reshape(nk2, 2 * nk1, order='F')
        Llo = res[f'Lz_{v}'][n].reshape(nk2, nk1, order='F')
        Lhi = res[f'Lz_{v}'][m].reshape(nk2, nk1, order='F')
        AL = new[f'Lz_{v}'][j].reshape(nk2, 2 * nk1, order='F')

        worst['sum'] = max(worst['sum'],
                           np.abs(np.sort(np.stack([A, B]), axis=0)
                                  - np.stack([lo, hi])[:, :, w]).max())
        worst['shift'] = max(worst['shift'],
                             np.abs(B - np.roll(A, -nk1, axis=1)).max())
        worst['per1'] = max(worst['per1'],
                            np.abs(A - np.roll(A, 2 * nk1, axis=1)).max())
        worst['oz'] = max(worst['oz'],
                          np.abs(AO + (Olo + Ohi)[:, w] - AO
                                 - np.where(AO == Olo[:, w], Ohi[:, w],
                                            Olo[:, w])).max())
        # E, Oz and Lz must be carried by the SAME branch label
        worst['lz'] = max(
            worst['lz'],
            np.abs(AL - np.where(sel == 0, Llo[:, w], Lhi[:, w])).max()
            + np.abs(AO - np.where(sel == 0, Olo[:, w], Ohi[:, w])).max())

check("{A,B} is a permutation of {E_lo,E_hi}", worst['sum'] == 0.0,
      f"max dev {worst['sum']:.2e}")
check("B(k) == A(k + G1)", worst['shift'] == 0.0,
      f"max dev {worst['shift']:.2e}")
check("A periodic in n1 with period 2*nk1", worst['per1'] == 0.0,
      f"max dev {worst['per1']:.2e}")
check("Oz, Lz carried by the same branch label as E", worst['lz'] == 0.0,
      f"max dev {worst['lz']:.2e}")

tot_f = sum(res[f'Oz_{v}'][1:13].sum() for v in ('K', 'Kp'))
tot_u = sum(new[f'Oz_{v}'].sum() for v in ('K', 'Kp'))
check("total Berry curvature conserved",
      np.isclose(tot_f, tot_u, rtol=1e-12, atol=0),
      f"folded {tot_f:.9e}  unfolded {tot_u:.9e}")

print("\n=== smoothness: max |2nd difference| in meV (n1 / n2) ===")
d2 = lambda Z, ax: np.abs(np.roll(Z, -1, ax) - 2 * Z + np.roll(Z, 1, ax)).max()
for j, (n, m) in enumerate(new['unfold_pairs']):
    lo = res['E_K'][n].reshape(nk2, nk1, order='F')
    A = new['E_K'][j].reshape(nk2, 2 * nk1, order='F')
    r_s, r_u = (d2(lo, 1), d2(lo, 0)), (d2(A, 1), d2(A, 0))
    better = r_u[0] < r_s[0] and r_u[1] < r_s[1]
    print(f"  K pair ({n:2d},{m:2d}): sorted {r_s[0]:8.5f} / {r_s[1]:8.5f}"
          f"   ->  unfolded {r_u[0]:8.5f} / {r_u[1]:8.5f}"
          f"   {'ok' if better else '<-- NOT SMOOTHER'}")
    if not better:
        fails.append(f"smoothness pair ({n},{m})")

# The offset below was derived by hand from the raw 25flux data, independently
# of analyze_pair; the auto-detected labelling must agree with it.
print("\n=== reference: pair 7/8 must reproduce the hand-verified labelling ===")
j = new['unfold_pairs'].tolist().index([7, 8])
A = new['E_K'][j].reshape(nk2, 2 * nk1, order='F')
lo = res['E_K'][7].reshape(nk2, nk1, order='F')
hi = res['E_K'][8].reshape(nk2, nk1, order='F')
s_ref = branch_label(nk1, nk2, 0.5, 0.355479)
A_ref = np.where(s_ref == 0, lo[:, w], hi[:, w])
check("matches the reference labelling exactly",
      np.abs(A - A_ref).max() == 0.0 or np.abs(A - np.roll(A_ref, nk1, 1)).max() == 0.0,
      "(up to the harmless A<->B gauge choice)")
check("n1 roughness 0.00043, n2 roughness 0.00171",
      np.isclose(d2(A, 1), 0.00043, atol=5e-6)
      and np.isclose(d2(A, 0), 0.00171, atol=5e-6),
      f"{d2(A,1):.5f} / {d2(A,0):.5f}")

print("\n=== negative control: unfold must decline an unfolded band ===")
res2 = {k: new[k] for k in ('E_K', 'E_Kp', 'Oz_K', 'Oz_Kp', 'Lz_K', 'Lz_Kp',
                            'kpoints', 'vol_M', 'nk1', 'nk2')}
again = unfold_bandstructure(res2, verbose=False)
check("no pairs found in already-unfolded data",
      'unfold_pairs' not in again and again['nk1'] == 2 * nk1)

print("\n" + ("ALL CHECKS PASSED" if not fails
              else f"{len(fails)} FAILURE(S): {fails}"))
sys.exit(1 if fails else 0)
