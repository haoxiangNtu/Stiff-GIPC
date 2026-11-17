"""Lab-bench props for the tactile chemistry demo (metres, Y-up).

Sources
-------
* ``Assets/lab_tactile/src/glass_beaker.glb`` -- the user's beaker (Glass_Beaker
  node only; the engraved scale/labels are dropped for physics).
* ``Assets/lab_tactile/src/straw.glb`` -- the user's rubber-bulb dropper
  ("Glass_tube_hollow_open_tip" + "Blue_bulb_hollow_elastomer").  The bulb's
  outer profile is sampled from the GLB and revolved into a SOLID of
  revolution together with the glass tube, so the whole dropper is ONE tet
  mesh: tube tets get a glass Young's modulus, bulb tets the rubber one
  (``Engine.native.set_per_tet_young_for_body``).
* everything else (Erlenmeyer flask, test tube, rack, drip tray, glass rod)
  is procedural: revolved closed profiles and merged boxes, all watertight.

``build_all()`` returns a dict of {name: Prop} and caches the result in
``Assets/lab_tactile/lab_assets.npz`` (delete the file to rebuild).
"""

from __future__ import annotations

import os
import os.path as osp
from dataclasses import dataclass, field

import numpy as np
import trimesh

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", ".."))
ASSET_DIR = osp.join(REPO, "Assets", "lab_tactile")
SRC_DIR = osp.join(ASSET_DIR, "src")
CACHE = osp.join(ASSET_DIR, "lab_assets.npz")

GLASS_RHO = 2500.0       # kg/m^3
RUBBER_RHO = 1100.0
E_GLASS_TET = 2e8        # per-tet Young for the dropper's glass tube
E_BULB = 3e5             # per-tet Young for the rubber bulb (ground truth)

# dropper geometry (after the 0.5 scale used in the user's scene file)
TUBE_R = 2.5e-3
TUBE_LEN = 0.100
BULB_SCALE = 0.5


@dataclass
class Prop:
    name: str
    verts: np.ndarray                     # (V,3) float64, local frame
    faces: np.ndarray                     # (F,3) int32 surface triangles
    kind: str = "ABD"                     # "ABD" | "FEM"
    density: float = GLASS_RHO
    tets: np.ndarray | None = None        # FEM only (T,4)
    tet_young: np.ndarray | None = None   # FEM only (T,)
    color: tuple = (0.75, 0.85, 0.95)
    meta: dict = field(default_factory=dict)

    @property
    def volume(self) -> float:
        return float(trimesh.Trimesh(self.verts, self.faces, process=False).volume)

    @property
    def mass(self) -> float:
        return self.volume * self.density

    def bounds(self):
        return self.verts.min(0), self.verts.max(0)


# ----------------------------------------------------------------- helpers

def _fix(m: trimesh.Trimesh) -> trimesh.Trimesh:
    m = trimesh.Trimesh(np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64),
                        process=True)
    m.merge_vertices()
    m.remove_unreferenced_vertices()
    if m.volume < 0:
        m.invert()
    trimesh.repair.fix_normals(m)
    return m


def revolve(profile_ry, sections=48) -> trimesh.Trimesh:
    """Revolve a closed (r, y) polygon (r>=0) about the Y axis -> watertight."""
    p = np.asarray(profile_ry, np.float64)
    if not np.allclose(p[0], p[-1]):
        p = np.vstack([p, p[:1]])
    m = trimesh.creation.revolve(p, sections=sections)
    # trimesh revolves about +Z; our world is Y-up -> map +z to +y
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return _fix(m)


def box(size, centre) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=np.asarray(size, float))
    m.apply_translation(np.asarray(centre, float))
    return m


def merge(meshes) -> trimesh.Trimesh:
    """Concatenate closed shells WITHOUT merging vertices across shells (each
    box stays watertight on its own; abutting faces are intentional)."""
    vs, fs, off = [], [], 0
    for m in meshes:
        vs.append(np.asarray(m.vertices, np.float64))
        fs.append(np.asarray(m.faces, np.int64) + off)
        off += len(m.vertices)
    return trimesh.Trimesh(np.vstack(vs), np.vstack(fs), process=False)


def decimate(m: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(m.faces) <= target_faces:
        return m
    import pyvista as pv
    faces = np.hstack([np.full((len(m.faces), 1), 3), m.faces]).ravel()
    pd = pv.PolyData(np.asarray(m.vertices, np.float64), faces)
    red = 1.0 - target_faces / len(m.faces)
    out = pd.decimate_pro(red, preserve_topology=True, boundary_vertex_deletion=False)
    tri = out.faces.reshape(-1, 4)[:, 1:]
    d = _fix(trimesh.Trimesh(np.asarray(out.points, np.float64), tri))
    if not d.is_watertight:                       # fall back to quadric decimation
        out = pd.decimate(red)
        tri = out.faces.reshape(-1, 4)[:, 1:]
        d = _fix(trimesh.Trimesh(np.asarray(out.points, np.float64), tri))
    return d


def _tetrahedralise(m: trimesh.Trimesh, max_volume: float):
    import tetgen
    tg = tetgen.TetGen(np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64))
    tg.tetrahedralize(order=1, quality=True, mindihedral=10.0, minratio=2.0,
                      fixedvolume=True, maxvolume=max_volume, verbose=0)
    return np.asarray(tg.node, np.float64), np.asarray(tg.elem, np.int32)


def _boundary_faces(tets: np.ndarray) -> np.ndarray:
    f = np.vstack([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [0, 3, 2]], tets[:, [1, 2, 3]]])
    key = np.sort(f, axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return f[idx[cnt == 1]].astype(np.int32)


# ------------------------------------------------------------------ props

def build_beaker(scale: float, name: str, target_faces=2600) -> Prop:
    sc = trimesh.load(osp.join(SRC_DIR, "glass_beaker.glb"))
    g = sc.geometry["Glass_Beaker"].copy()
    g = _fix(g)
    g.apply_scale(scale)
    g = decimate(g, target_faces)
    # GLB is Y-up with the bottom at y=0 already
    g.apply_translation([0.0, -g.bounds[0][1], 0.0])
    outer_r = float(np.sqrt(g.vertices[:, 0] ** 2 + g.vertices[:, 2] ** 2).max())
    return Prop(name, np.asarray(g.vertices), np.asarray(g.faces, np.int32), "ABD", GLASS_RHO,
                color=(0.80, 0.88, 0.95),
                meta=dict(height=float(g.bounds[1][1]), outer_r=outer_r, scale=scale))


def build_flask() -> Prop:
    t = 2.5e-3
    P = [(0, 0), (0.042, 0), (0.042, 0.012), (0.013, 0.105), (0.013, 0.140),
         (0.013 - t, 0.140), (0.013 - t, 0.105), (0.042 - t, 0.012), (0.042 - t, 0.003),
         (0, 0.003)]
    m = revolve(P, 56)
    return Prop("flask", np.asarray(m.vertices), np.asarray(m.faces, np.int32), "ABD", GLASS_RHO,
                color=(0.80, 0.88, 0.95),
                meta=dict(height=0.140, neck_r_in=0.013 - t, neck_r_out=0.013, base_r=0.042))


def build_test_tube(name="test_tube") -> Prop:
    ro, ri, L = 8e-3, 6.8e-3, 0.100
    n = 10
    ang = np.linspace(-np.pi / 2, 0.0, n)
    outer_arc = [(ro * np.cos(a), ro + ro * np.sin(a)) for a in ang]       # tip -> side
    inner_arc = [(ri * np.cos(a), ro + ri * np.sin(a)) for a in ang[::-1]]  # side -> tip
    P = [(0, 0)] + outer_arc[1:] + [(ro, L), (ri, L)] + inner_arc[:-1] + [(0, ro - ri)]
    m = revolve(P, 40)
    return Prop(name, np.asarray(m.vertices), np.asarray(m.faces, np.int32), "ABD", GLASS_RHO,
                color=(0.80, 0.88, 0.95), meta=dict(length=L, r_out=ro, r_in=ri))


def build_rack() -> Prop:
    """Test-tube rack: base plate + two end walls + a top frame with three
    24 x 24 mm square openings (centres x = -32, 0, +32 mm) at y = 64..70 mm."""
    parts = [box((0.100, 0.006, 0.036), (0, 0.003, 0)),            # base plate
             box((0.006, 0.064, 0.036), (-0.047, 0.038, 0)),        # end walls on the plate
             box((0.006, 0.064, 0.036), (+0.047, 0.038, 0)),
             box((0.088, 0.006, 0.006), (0, 0.067, -0.015)),        # long strips between walls
             box((0.088, 0.006, 0.006), (0, 0.067, +0.015)),
             box((0.008, 0.006, 0.024), (-0.016, 0.067, 0)),        # cross strips between long strips
             box((0.008, 0.006, 0.024), (+0.016, 0.067, 0))]
    m = merge(parts)
    return Prop("rack", np.asarray(m.vertices), np.asarray(m.faces, np.int32), "ABD", 700.0,
                color=(0.72, 0.56, 0.38),
                meta=dict(slot_x=[-0.032, 0.0, 0.032], base_top=0.006, frame_top=0.070))


def build_tray() -> Prop:
    parts = [box((0.160, 0.004, 0.110), (0, 0.002, 0)),            # floor
             box((0.160, 0.010, 0.004), (0, 0.009, -0.053)),        # rims standing on the floor
             box((0.160, 0.010, 0.004), (0, 0.009, +0.053)),
             box((0.004, 0.010, 0.102), (-0.078, 0.009, 0)),
             box((0.004, 0.010, 0.102), (+0.078, 0.009, 0))]
    m = merge(parts)
    return Prop("tray", np.asarray(m.vertices), np.asarray(m.faces, np.int32), "ABD", 1200.0,
                color=(0.85, 0.85, 0.87), meta=dict(floor_top=0.004))


def build_rod() -> Prop:
    r, L = 3e-3, 0.150
    m = trimesh.creation.capsule(height=L - 2 * r, radius=r, count=[10, 10])
    m = _fix(m)                                   # axis along +z, centred
    # lie along +x
    R = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    m.apply_transform(R)
    m.apply_translation(-m.centroid)
    return Prop("rod", np.asarray(m.vertices), np.asarray(m.faces, np.int32), "ABD", GLASS_RHO,
                color=(0.85, 0.92, 0.98), meta=dict(length=L, radius=r))


def _bulb_profile() -> np.ndarray:
    """(r, y) outer profile of the GLB bulb, scaled, y measured from the bulb's
    lowest point (where it meets the tube)."""
    sc = trimesh.load(osp.join(SRC_DIR, "straw.glb"))
    b = sc.geometry["Blue_bulb_hollow_elastomer"]
    v = np.asarray(b.vertices, np.float64) * BULB_SCALE
    y = v[:, 1] - v[:, 1].min()
    r = np.sqrt(v[:, 0] ** 2 + v[:, 2] ** 2)
    nb = 34
    edges = np.linspace(0.0, y.max() + 1e-9, nb + 1)
    prof = []
    for i in range(nb):
        sel = (y >= edges[i]) & (y < edges[i + 1])
        if sel.sum() < 5:
            continue
        prof.append((float(np.percentile(r[sel], 99.0)), float(0.5 * (edges[i] + edges[i + 1]))))
    prof = np.array(prof)
    return prof


TUBE_R_IN = 3.0e-3       # hollow dropper: bore radius (one 2.5 mm liquid particle passes with the 0.5 mm barrier)
TUBE_R_OUT = 4.5e-3      # hollow dropper: outer radius (1.5 mm glass wall)


def build_dropper(max_volume=8.0e-9, hollow=True) -> Prop:
    """One FEM solid: glass tube (y in [0, TUBE_LEN]) + rubber bulb above it.
    hollow=True: the tube has an open bore (radius TUBE_R_IN) from the tip up to
    3 mm below the bulb, so liquid can be drawn in / dripped out (engine pump);
    the bulb stays solid rubber (it is what the gripper squeezes)."""
    prof = _bulb_profile()
    r0 = float(prof[0, 0])
    y_b0 = TUBE_LEN
    if hollow:
        r_in, r_out = TUBE_R_IN, max(TUBE_R_OUT, TUBE_R)
        y_cav = y_b0 - 3e-3
        ys = np.linspace(0.0, y_cav, 11)                 # axial subdivision -> well-shaped wall triangles
        pts = [(0.0, y_cav)] + [(r_in, float(y)) for y in ys[::-1]]      # cavity top (axis) -> inner wall down to the tip
        pts += [(r_out, float(y)) for y in np.linspace(0.0, y_b0 - 1e-3, 11)]   # outer wall up
        pts.append((r0, y_b0))
    else:
        r_in, r_out, y_cav = 0.0, TUBE_R, 0.0
        pts = [(0.0, 0.0), (TUBE_R, 0.0), (TUBE_R, y_b0 - 1e-3), (r0, y_b0)]
    for r, y in prof[1:-1]:
        pts.append((float(r), float(y_b0 + y)))
    y_top = y_b0 + float(prof[-1, 1]) + 1.5e-3
    pts.append((float(prof[-1, 0]) * 0.6, y_top - 0.5e-3))
    pts.append((0.0, y_top))
    m = revolve(pts, 16)
    verts, tets = _tetrahedralise(m, max_volume)
    faces = _boundary_faces(tets)
    cy = verts[tets].mean(1)[:, 1]
    young = np.where(cy < y_b0 + 1.0e-3, E_GLASS_TET, E_BULB)
    bulb_r = float(prof[:, 0].max())
    bulb_cy = float(y_b0 + prof[np.argmax(prof[:, 0]), 1])
    rr = np.hypot(verts[:, 0], verts[:, 2])
    tip_idx = np.where(verts[:, 1] < 0.5e-3)[0]                          # tip ring (both radii)
    top_idx = np.where((np.abs(verts[:, 1] - y_cav) < 0.5e-3) & (rr < r_in + 0.5e-3))[0] if hollow else np.array([], int)
    return Prop("dropper", verts, faces, "FEM", RUBBER_RHO, tets=tets, tet_young=young,
                color=(0.25, 0.45, 0.85),
                meta=dict(tube_len=TUBE_LEN, tube_r=r_out, tube_r_in=r_in, cavity_top=y_cav, bulb_r=bulb_r, bulb_cy=bulb_cy,
                          y_top=y_top, E_bulb=E_BULB, E_tube=E_GLASS_TET, hollow=hollow,
                          tip_idx=tip_idx.astype(np.int32), top_idx=top_idx.astype(np.int32)))


def build_all(force=False, verbose=True) -> dict:
    if osp.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=True)
        props = {}
        for k, d in z["props"].item().items():
            props[k] = Prop(k, d["verts"], d["faces"], d["kind"], float(d["density"]),
                            tets=d.get("tets"), tet_young=d.get("tet_young"),
                            color=tuple(d["color"]), meta=dict(d["meta"]))
        if verbose:
            print(f"[lab_assets] loaded {len(props)} props from cache")
        return props
    props = {}
    props["beaker_mix"] = build_beaker(0.85, "beaker_mix")
    props["beaker_waste"] = build_beaker(1.00, "beaker_waste")
    props["flask"] = build_flask()
    props["test_tube"] = build_test_tube("test_tube")
    props["test_tube2"] = build_test_tube("test_tube2")
    props["rack"] = build_rack()
    props["tray"] = build_tray()
    props["rod"] = build_rod()
    props["dropper"] = build_dropper()
    os.makedirs(ASSET_DIR, exist_ok=True)
    plain = {k: dict(verts=p.verts, faces=p.faces, kind=p.kind, density=p.density, tets=p.tets,
                     tet_young=p.tet_young, color=p.color, meta=p.meta) for k, p in props.items()}
    np.savez_compressed(CACHE, props=np.array(plain, dtype=object))
    if verbose:
        for p in props.values():
            lo, hi = p.bounds()
            tm = trimesh.Trimesh(p.verts, p.faces, process=False)
            wt = all(c.is_watertight for c in tm.split(only_watertight=False))
            extra = f" tets={len(p.tets)}" if p.tets is not None else ""
            print(f"[lab_assets] {p.name:12s} {p.kind} V={len(p.verts):5d} F={len(p.faces):5d}"
                  f"{extra} watertight={wt} mass={p.mass*1e3:6.1f} g "
                  f"size={np.round((hi - lo) * 1e3, 1).tolist()} mm")
    return props


if __name__ == "__main__":
    build_all(force=True)


# ------------------------------------------------------------------ dropper with a HOLLOW bulb (2026-10-19 re-audit, step 4)

def build_dropper_shell(max_volume=8.0e-9, t_bulb=1.5e-3) -> Prop:
    """One FEM solid like build_dropper(hollow=True) but the rubber bulb is a SHELL (wall t_bulb) whose cavity
    connects to the bore: squeezing the bulb changes the enclosed air volume, which is what a physical pump
    model needs (the solid bulb of build_dropper cannot pump: rubber is incompressible).
    meta['cavity_faces'] are the boundary triangles of the cavity (bore wall + bulb inside + top), oriented
    with the solid's outward normal (i.e. pointing INTO the cavity); the open tip disc (y = 0) closes the
    cavity for the volume: V_cav = -sum_f (x_a . (x_b x x_c)) / 6 with the origin on the tip plane."""
    prof = _bulb_profile()
    y_b0 = TUBE_LEN
    r_in, r_out = TUBE_R_IN, max(TUBE_R_OUT, TUBE_R)
    outer = [(r_out, 0.0)] + [(r_out, float(y)) for y in np.linspace(0.0, y_b0 - 1e-3, 11)[1:]]
    outer += [(float(r), float(y_b0 + y)) for r, y in prof[1:-1]]
    y_top = y_b0 + float(prof[-1, 1]) + 1.5e-3
    outer += [(float(prof[-1, 0]) * 0.6, y_top - 0.5e-3), (0.0, y_top)]
    inner = [(0.0, y_top - t_bulb), (float(prof[-1, 0]) * 0.6 - t_bulb, y_top - 0.5e-3 - t_bulb)]
    for r, y in prof[1:-1][::-1]:
        ri = float(r) - t_bulb
        if ri <= r_in + 0.3e-3:
            continue                                   # neck: the cavity narrows into the bore
        inner.append((ri, float(y_b0 + y)))
    inner += [(r_in, float(y)) for y in np.linspace(y_b0 - 1e-3, 0.0, 11)]
    m = revolve(np.array(outer + inner), 16)
    verts, tets = _tetrahedralise(m, max_volume)
    faces = _boundary_faces(tets)
    cy = verts[tets].mean(1)[:, 1]
    young = np.where(cy < y_b0 + 1.0e-3, E_GLASS_TET, E_BULB)
    # cavity faces: boundary triangles whose centroid radius is below the outer profile radius at that height
    # minus half the wall (the outer polyline is monotone in y: tube wall -> bulb -> apex)
    op = np.array(outer)
    cen = verts[faces].mean(1)
    r_c = np.hypot(cen[:, 0], cen[:, 2])
    r_outer = np.interp(cen[:, 1], op[:, 1], op[:, 0])
    cavity_faces = faces[r_c < r_outer - 0.5 * min(t_bulb, r_out - r_in)]
    rr = np.hypot(verts[:, 0], verts[:, 2])
    tip_idx = np.where(verts[:, 1] < 0.5e-3)[0]
    y_cav = y_b0 - 3e-3
    top_idx = np.where((np.abs(verts[:, 1] - y_cav) < 1.5e-3) & (rr < r_in + 0.5e-3))[0]
    # enclosed cavity volume with the origin on the tip plane (open tip disc contributes nothing)
    a, b, c = verts[cavity_faces[:, 0]], verts[cavity_faces[:, 1]], verts[cavity_faces[:, 2]]
    v_cav = -float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    bulb_r = float(prof[:, 0].max())
    return Prop("dropper", verts, faces, "FEM", RUBBER_RHO, tets=tets, tet_young=young, color=(0.25, 0.45, 0.85),
                meta=dict(tube_len=TUBE_LEN, tube_r=r_out, tube_r_in=r_in, cavity_top=y_cav, bulb_r=bulb_r, y_top=y_top,
                          E_bulb=E_BULB, E_tube=E_GLASS_TET, hollow=True, hollow_bulb=True, t_bulb=t_bulb,
                          tip_idx=tip_idx.astype(np.int32), top_idx=top_idx.astype(np.int32),
                          cavity_faces=cavity_faces.astype(np.int32), cavity_volume=v_cav))
