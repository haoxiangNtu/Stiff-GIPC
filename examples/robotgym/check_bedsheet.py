#!/usr/bin/env python3
"""Numbers for a bed-sheet recording: did the pinch hold (cloth point nearest each TCP follows it), how high the
near edge got, and how much of the near half ended up on top of the far half.
    python examples/robotgym/check_bedsheet.py out/robotgym/bedsheet/recording.npz"""
import sys, os.path as osp, numpy as np
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "lab_tactile"))
from lab_rec import load_recording
z = load_recording(sys.argv[1]); t = np.asarray(z["t"], float); seg = [str(s) for s in z["segment"]]
V = z["fem_verts"][list(z["fem_names"]).index("sheet")]          # (frames, n, 3)
meta = z["meta"]; arms = meta.get("arms", [])
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__))))
from common import Arm
kins = [Arm(a["name"], np.asarray(a["base_T"], float)).kin for a in arms] if arms else []
q_arms = z["q_arms"] if "q_arms" in z else None
q_meas = z["q_meas"] if "q_meas" in z else None
V0 = V[0]; near = V0[:, 2] < V0[:, 2].mean(); far = ~near
print(f"{len(t)} records, sheet {V.shape[1]} verts; near-half mean z0 {V0[near,2].mean():.3f}, far-half {V0[far,2].mean():.3f}")
last = None
for k in range(len(t)):
    if seg[k] != last or k == len(t) - 1:
        line = f"t {t[k]:6.2f}s {seg[k]:26s} near-edge y max {V[k][near,1].max():.3f} mean {V[k][near,1].mean():.3f} z mean {V[k][near,2].mean():.3f}"
        if kins and q_arms is not None:
            for a, kin in enumerate(kins):
                tcp = kin.fk(q_arms[k][a])[:3, 3]; d = np.linalg.norm(V[k] - tcp, axis=1).min()
                dq = np.abs(q_meas[k][a] - q_arms[k][a]).max() if q_meas is not None and q_meas.shape[1] > a else float("nan")
                line += f" | {arms[a]['name']}: cloth-TCP {d*1e3:5.1f} mm, |q_meas-q_cmd| max {np.degrees(dq):5.1f} deg"
        print(line); last = seg[k]
# fold quality at the end: near-half vertices that lie above a far-half vertex (xz within 2 cm) and higher than it
Vn, Vf = V[-1][near], V[-1][far]
from scipy.spatial import cKDTree
d, j = cKDTree(Vf[:, [0, 2]]).query(Vn[:, [0, 2]])
on_top = (d < 0.02) & (Vn[:, 1] > Vf[j, 1] + 0.5e-3)
print(f"final: {on_top.mean()*100:.0f}% of the near half lies on top of the far half; near-half z range {Vn[:,2].min():.2f}..{Vn[:,2].max():.2f}, far {Vf[:,2].min():.2f}..{Vf[:,2].max():.2f}")
