"""Full-resolution VISUAL skin over a COARSE simulation proxy.

Why: a full-precision robot mesh (~1.1M verts) costs ~34 ms/step in the ABD
system even with ALL collision excluded — the rigid-body bookkeeping recomputes
every vertex every step, a cost proportional to mesh resolution, not to whether
the body collides. A coarse box/tip proxy costs ~8 ms.  So we SIMULATE the coarse
proxy and RENDER the full-res mesh: each rigid body's world transform is recovered
by Kabsch from a handful of its (cheap) proxy vertices every frame and pushed to a
pre-registered full-res polyscope mesh via set_transform.  For rigid ABD bodies
this reproduces the true full-res motion to ~0.002 mm at ~zero per-frame cost.

Usage (after ps.init() and after building the sim "scene" mesh):
    skin = FullResSkin(ps, eng, recs, full_urdf, mesh_root, V_rest)
    ... per frame:  skin.update(Vnow)
Non-robot bodies (toys/table/drawer: names absent from `full_urdf`) are left for
the normal dynamic scene mesh to render.
"""
import os
import numpy as np
import trimesh
import xml.etree.ElementTree as ET


def _kabsch(P, Q):
    """Rigid R,t (proper rotation) best-fitting Q ~ R@P + t."""
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cQ - R @ cP


class FullResSkin:
    def __init__(self, ps, eng, recs, full_urdf, mesh_root, V_rest,
                 max_proxy=48, color=(0.72, 0.77, 0.86)):
        self.ps = ps
        root = ET.parse(full_urdf).getroot()
        vis = {}
        for l in root.findall("link"):
            m = l.find("visual/geometry/mesh")
            if m is not None:
                vis[l.get("name")] = m.get("filename")

        rng = np.random.RandomState(0)
        self.items = []            # (ps_mesh, abs_idx, proxy_rest_world)
        self.robot_ranges = []     # vertex ranges of skinned (robot) bodies
        n_v = 0
        for r in recs:
            fn = vis.get(r.label)
            if fn is None:
                continue           # not a robot link (toy/table/drawer) -> skip
            path = os.path.join(mesh_root, fn)
            if not os.path.exists(path):
                continue
            m = trimesh.load(path, process=True)
            T = np.array(eng.native.get_urdf_link_transform(r.label))
            Wv = (T[:3, :3] @ np.asarray(m.vertices).T).T + T[:3, 3]
            pm = ps.register_surface_mesh("robot::" + r.label, Wv,
                                          np.asarray(m.faces), color=color,
                                          smooth_shade=True)
            lo, hi = r.vertex_offset, r.vertex_offset + r.vertex_count
            seg = V_rest[lo:hi]
            k = min(max_proxy, len(seg))
            sub = rng.choice(len(seg), k, replace=False) if len(seg) > k \
                else np.arange(len(seg))
            abs_idx = lo + sub
            self.items.append((pm, abs_idx, V_rest[abs_idx].copy()))
            self.robot_ranges.append((lo, hi))
            n_v += len(m.vertices)
        print(f"[skin] full-res visual overlay: {len(self.items)} robot links, "
              f"{n_v} render verts (sim proxy stays coarse)")

    def robot_vertex_mask(self, n_total):
        """Boolean mask over the SIM vertex buffer: True where a robot proxy
        vertex lives (so the caller can drop those faces from the scene mesh)."""
        mask = np.zeros(n_total, bool)
        for lo, hi in self.robot_ranges:
            mask[lo:hi] = True
        return mask

    def update(self, Vnow):
        for pm, abs_idx, prest in self.items:
            R, t = _kabsch(prest, Vnow[abs_idx])
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = t
            pm.set_transform(M)
