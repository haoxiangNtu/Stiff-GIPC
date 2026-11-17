#!/usr/bin/env python3
"""Sponge-transfer metrics: water in the cup (ml), water inside the sponge (ml), sponge volume ratio, per segment."""
import sys
if __name__ == "__main__": import os as _os; _os.environ.setdefault("LIQ_UG", "0")   # main-path example, when run as a script: pinned to the main liquid path (the engine default is the UG mode since 2026-11-18)
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import sponge_transfer as S

p = sys.argv[1] if len(sys.argv) > 1 else "out/robotgym/sponge/recording.npz"
d = np.load(p, allow_pickle=True)
fem = [str(x) for x in d["fem_names"]] if "fem_names" in d else ["sponge"]
t, seg = d["t"], d["segment"]; X = d[f"fem_verts_{fem.index('sponge') if 'sponge' in fem else 0}"]; meta = d["meta"].item()   # tactile runs: pads come first
import json
if isinstance(meta, (str, bytes)): meta = json.loads(meta)
liq = [d[f"liquid_pts_{i}"] for i in range(int(d["liquid_pts__n"]))] if "liquid_pts__n" in d else None
n_sub = max(1, int(round(8 ** (1 / 3)))); hs = meta.get("liquid_dx", 0.006) / n_sub; vp = hs ** 3 * 1e6
# older recordings do not store sponge_cell: pick the cell size whose mesh has the recorded vertex count
cell = meta.get("sponge_cell") or next(c for c in (0.0125, 0.00625, 0.025) if len(S.box_tets(S.SPONGE, c)[0]) == X.shape[1])
v0, tets = S.box_tets(S.SPONGE, cell)
def vol(V):
    a, b, c, dd = (V[tets[:, k]] for k in range(4)); return np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), dd - a)).sum() / 6
V0 = vol(v0)
print(f"frames {len(t)} t_end {t[-1]:.2f} sponge verts {X.shape[1]} rest volume {V0*1e6:.0f} ml, particle {vp:.3f} ml")
last = None
for i in range(len(t)):
    if seg[i] != last or i == len(t) - 1:
        s = f"t={t[i]:6.2f} {seg[i][:24]:24s} sponge V/V0 {vol(X[i])/V0:.3f} centre y {X[i][:,1].mean():.3f}"
        if liq:
            P = liq[i]
            cup = (np.abs(P[:, 0] - S.CUP_C[0]) < S.CUP_IN[0] / 2) & (np.abs(P[:, 2] - S.CUP_C[2]) < S.CUP_IN[2] / 2) & (P[:, 1] < S.CUP_C[1] + S.CUP_IN[1])
            lo, hi = X[i].min(0), X[i].max(0); insp = np.all((P > lo) & (P < hi), 1)
            s += f" | cup {cup.sum()*vp:6.1f} ml  in-sponge-box {insp.sum()*vp:6.1f} ml  water ymax {P[:,1].max():.3f}"
        print(s); last = seg[i]
