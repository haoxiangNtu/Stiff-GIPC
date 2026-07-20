"""Geometry-inferred loop-closure joints for the FD dexterous hand.

URDF is a tree, so each finger's four-bar linkage lost its loop-closing pin on
export (the `connecting` / `slider_abpart` bars end up with a FREE tip that
floats away when driven). This re-adds those pins by inference: each dangling
finger link is pinned (revolute, about the finger's flexion axis) to the nearest
same-finger link at their closest-point at rest. The engine is maximal-coordinate
+ penalty joints, so a closed loop just works (validated by test_fourbar_loop).

Call add_loop_closures(eng, urdf_path, mesh_root, recs) AFTER load_urdf and
BEFORE finalize.  Returns the number of pins added.
"""
import os, numpy as np, trimesh
import xml.etree.ElementTree as ET

FINGERS = ("if", "mf", "rf", "lf", "th")

def _closest_midpoint(A, B, rng=np.random):
    a = A[np.random.RandomState(0).choice(len(A), min(len(A), 800), replace=False)]
    b = B[np.random.RandomState(1).choice(len(B), min(len(B), 800), replace=False)]
    d = np.linalg.norm(a[:, None] - b[None], axis=2)
    i, j = np.unravel_index(d.argmin(), d.shape)
    return (a[i] + b[j]) / 2.0, float(d.min())

def add_loop_closures(eng, urdf_path, mesh_root, recs, strength=2000.0, verbose=True):
    root = ET.parse(urdf_path).getroot()
    meshf, jparent, jaxis = {}, {}, {}
    for l in root.findall("link"):
        m = l.find("collision/geometry/mesh")
        if m is not None:
            meshf[l.get("name")] = m.get("filename")
    children = {}
    for j in root.findall("joint"):
        c = j.find("child").get("link")
        jparent[c] = j.find("parent").get("link")
        children.setdefault(j.find("parent").get("link"), []).append(c)
        ax = j.find("axis")
        jaxis[c] = (np.array([float(x) for x in ax.get("xyz").split()])
                    if ax is not None else np.array([0.0, 0.0, 1.0]))

    def linkT(nm): return np.array(eng.native.get_urdf_link_transform(nm))
    def wverts(nm):
        m = trimesh.load(os.path.join(mesh_root, meshf[nm]), process=True)
        T = linkT(nm); v = np.asarray(m.vertices)
        return (T[:3, :3] @ v.T).T + T[:3, 3]
    def waxis(nm):
        T = linkT(nm); a = T[:3, :3] @ jaxis[nm]; return a / (np.linalg.norm(a) + 1e-12)

    n = 0
    for side in ("_r", "_l"):
        for f in FINGERS:
            links = [nm for nm in meshf if nm.startswith(f + "_") and nm.endswith(side)]
            for dl in [l for l in links if l not in children and "distal" not in l]:
                Wd = wverts(dl)
                cands = [l for l in links if l != dl and l != jparent.get(dl)]
                if not cands:
                    continue
                best = None
                for tl in cands:
                    pt, dist = _closest_midpoint(Wd, wverts(tl))
                    if best is None or dist < best[2]:
                        best = (tl, pt, dist)
                tl, pt, _ = best
                if dl not in recs or tl not in recs:
                    continue
                eng.native.add_revolute_joint(recs[dl].body_offset, recs[tl].body_offset,
                                              waxis(dl).astype(float), pt.astype(float),
                                              -3.14, 3.14, float(strength), f"loop_{dl}")
                n += 1
    if verbose:
        print(f"[hand-loops] added {n} inferred loop-closure pins (both hands)")
    return n
