"""Compare the magnetic-breakdown broadening estimate against the exact
qq = 1 Landau-level widths."""

import json
import numpy as np

bd = json.load(open('_breakdown.json'))
ex = json.load(open('_exactwidth.json'))

Eg = np.array([r['E'] for r in bd['rows']])
Bg = np.array(bd['Blist'])
Gg = np.array([r['G'] for r in bd['rows']])          # (nE, nB)
hw1 = np.array([r['hw1'] for r in bd['rows']])
ok = Gg[:, 0] > 0

print(f"{'B(T)':>6} {'E':>6} {'w_exact':>8} {'Gamma':>7} {'ratio':>7} "
      f"{'hwc_ex':>7} {'hwc_sc':>7}")
allr, allB = [], []
for r in sorted(ex, key=lambda r: r['B']):
    B = r['B']
    ib = int(np.argmin(np.abs(Bg - B)))
    c, w = np.array(r['ctr']), np.array(r['w'])
    hwc_ex = np.median(np.diff(c))
    m = (c >= Eg[ok].min()) & (c <= Eg[ok].max())
    if not m.any():
        continue
    G = np.interp(c[m], Eg[ok], Gg[ok, ib])
    hwsc = np.interp(c[m], Eg[ok], hw1[ok]) * B
    for cc, ww, gg, hh in zip(c[m], w[m], G, hwsc):
        print(f"{B:6.2f} {cc:6.0f} {ww:8.2f} {gg:7.2f} {ww/gg:7.3f} "
              f"{hwc_ex:7.2f} {hh:7.2f}")
        allr.append(ww / gg)
        allB.append(B)
    print()

allr, allB = np.array(allr), np.array(allB)
print(f"n = {len(allr)}   ratio w_exact/Gamma: "
      f"median {np.median(allr):.3f}, mean {allr.mean():.3f}, "
      f"geo-mean {np.exp(np.log(allr).mean()):.3f}, "
      f"spread [{allr.min():.2f}, {allr.max():.2f}]")

print("\n  per-field summary (is the ratio B-independent?)")
print(f"{'B(T)':>6} {'n':>3} {'med ratio':>10} {'w_med':>7} {'G_med':>7}")
for B in sorted(set(allB)):
    s = allB == B
    print(f"{B:6.2f} {s.sum():3d} {np.median(allr[s]):10.3f}")

lo, hi = allB < 4, allB > 4
print(f"\n  B < 4 T : median ratio {np.median(allr[lo]):.3f}  (n={lo.sum()})")
print(f"  B > 4 T : median ratio {np.median(allr[hi]):.3f}  (n={hi.sum()})")
