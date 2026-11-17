"""LinkerHand L20 V10.1: hand config + fingertip sensor pads.

The L20 is a 5-finger, ~20-DoF dexterous hand (thumb: 5 DoF incl. an
independent CMC yaw — richer opposition than the Allegro's 4-DoF thumb).
Every finger ends in a `*_distal` link plus a bare `*_tip` frame, so the
generic URDF FK/IK in allegro.py drives it unchanged; this module only
supplies the L20-specific configuration:

  * finger chains (which joints move which distal), and
  * per-finger PAD MOUNTS.

Pad mounting, and why it is computed rather than measured off a face: the L20
distal pulp is ROUNDED (no single flat facet — normal-bucket clustering finds
nothing coplanar bigger than ~100 mm², and the SolidWorks meshes carry interior
geometry that poisons plane fits). So the mount is a DESIGN choice, the same
one an integrator gluing a flat pad onto a curved fingertip makes: pick the
press direction d in the distal frame (palmar +x with a mild tip-ward tilt —
the fingers flex toward +x, verified by FK), take the SUPPORT PLANE of the
distal surface along d over the pulp window, and glue the pad's stick face
onto that tangent plane. The pad is then proud of the shell by construction —
the mistake that made one Allegro pad look sunken cannot happen here.

The pad itself is fabricated (fabricate.py) at LinkerHand scale: the L20
distal is 15.4 mm wide, so the 25 x 18.4 mm Allegro digit gel physically does
not fit. Default: 12 x 10 x 2.5 mm, camera 160x160 px at the standard
0.079375 mm pitch (12.7 mm FOV).

The real L20 ships a tactile variant with a pressure-taxel matrix on the
fingertips, so the piezoresistive readout (piezo.py) on this pad mirrors the
actual product; the VBTS camera is the upgrade path.
"""

from __future__ import annotations

import os.path as osp
import struct

import numpy as np

from .allegro import AllegroURDF

L20_CHAINS = {
    "thumb": ("thumb_distal", ["thumb_cmc_roll", "thumb_cmc_yaw",
                               "thumb_cmc_pitch", "thumb_mcp", "thumb_dip"]),
    "index": ("index_distal", ["index_mcp_roll", "index_mcp_pitch",
                               "index_pip", "index_dip"]),
    "middle": ("middle_distal", ["middle_mcp_roll", "middle_mcp_pitch",
                                 "middle_pip", "middle_dip"]),
    "ring": ("ring_distal", ["ring_mcp_roll", "ring_mcp_pitch",
                             "ring_pip", "ring_dip"]),
    "pinky": ("pinky_distal", ["pinky_mcp_roll", "pinky_mcp_pitch",
                               "pinky_pip", "pinky_dip"]),
}

# Press direction per finger in its distal frame: palmar (+x) with a tip-ward
# tilt. The tilt is a CO-DESIGNED parameter, not anatomy: sweeping it against
# the oppose() solver (thumb tilt x finger tilt grid, 30 restarts each) moved
# the best reachable index/thumb pad opposition from 24.7 deg of residual tilt
# (both pads at 0.25) to 6.3 deg (thumb 0.65, fingers 0.40) at the same
# 12.1 mm pinch -- better than the Allegro's hard 15 deg limit, thanks to the
# L20's 5-DoF thumb. Angling a fingertip pad mount is exactly what a real
# integrator does with a wedge shim.
PRESS_DIRS = {"thumb": np.array([1.0, 0.0, 0.65]) / np.linalg.norm([1.0, 0.0, 0.65]),
              "default": np.array([1.0, 0.0, 0.40]) / np.linalg.norm([1.0, 0.0, 0.40])}
# pulp window along the distal's length axis (z), metres
PULP_Z_WIN = {"thumb": (8e-3, 27e-3), "default": (6e-3, 21e-3)}
PAD_SIZE = (12e-3, 10e-3, 2.5e-3)      # pad-local x (length), y (width), thickness


def _read_stl_verts(path: str) -> np.ndarray:
    d = open(path, "rb").read()
    n = struct.unpack("<I", d[80:84])[0]
    a = np.frombuffer(d[84:84 + n * 50], dtype=np.uint8).reshape(n, 50)
    return a[:, 12:48].copy().view("<f4").reshape(-1, 3).astype(np.float64)


def pad_mount(urdf_dir: str, finger: str, thickness: float | None = None) -> np.ndarray:
    """4x4 pad-local -> distal-frame transform for this finger's mount.

    Pad-local convention is the fabricated flat pad's: coat (sensing face) at
    z=0 facing -z, body extending to +z, stick face at z=thickness glued to
    the finger. So pad +z maps to -press and the coat plane sits `thickness`
    proud of the support plane.

    thickness=0 gives the NATIVE frame: the sensing plane IS the fingertip's
    own support plane — used when the hand's original contact surface is the
    sensor (the real L20 tactile variant: a taxel film in the fingertip skin,
    no added gel), with per-vertex contact forces read straight off the
    distal mesh and binned in this frame.
    """
    if thickness is None:
        thickness = PAD_SIZE[2]
    link = L20_CHAINS[finger][0]
    press = PRESS_DIRS.get(finger, PRESS_DIRS["default"])
    v = _read_stl_verts(osp.join(urdf_dir, "meshes", f"{link}.STL"))
    lo, hi = PULP_Z_WIN.get(finger, PULP_Z_WIN["default"])
    win = (v[:, 2] > lo) & (v[:, 2] < hi) & (np.abs(v[:, 1]) < 7e-3)
    if not win.any():
        raise ValueError(f"{finger}: empty pulp window on {link}")
    s_max = float((v[win] @ press).max())            # support plane offset

    z_pad = -press                                   # into the finger
    x_pad = np.array([0.0, 0.0, 1.0])
    x_pad = x_pad - (x_pad @ z_pad) * z_pad          # finger-length, in-plane
    x_pad /= np.linalg.norm(x_pad)
    y_pad = np.cross(z_pad, x_pad)

    # Centre the frame on the actual TANGENT VERTEX, not the window midpoint:
    # the rounded pulp's support point against a parallel plane sits where the
    # surface bulges most, which on the L20 index is ~8 mm base-ward of the
    # window middle -- a taxel grid centred on the midpoint missed the whole
    # contact patch (index read 0.0 N while carrying 120 N).
    i_sup = int(np.argmax(v[win] @ press))
    v_sup = v[win][i_sup]
    centre_on_plane = v_sup + (s_max - v_sup @ press) * press
    coat_centre = centre_on_plane + press * thickness

    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = x_pad, y_pad, z_pad
    M[:3, 3] = coat_centre
    return M


def load_l20(urdf_path: str, native: bool = False) -> AllegroURDF:
    """AllegroURDF configured for the LinkerHand L20 with per-finger pad mounts.

    native=True: sensing planes ON the bare fingertip surface (thickness 0),
    for the built-in-taxel configuration with no added gel.

    NOTE: use the package-root URDF (next to meshes/), not the copy in urdf/ --
    its `meshes/...` paths resolve relative to the file.
    """
    d = osp.dirname(urdf_path)
    t = 0.0 if native else None
    pad_c, pad_n = {}, {}
    for f in L20_CHAINS:
        M = pad_mount(d, f, thickness=t)
        pad_c[f] = M[:3, 3]
        pad_n[f] = PRESS_DIRS.get(f, PRESS_DIRS["default"]).copy()
    hand = AllegroURDF(urdf_path, chains=L20_CHAINS,
                       pad_centre=pad_c, pad_normal=pad_n)
    hand.pad_mounts = {f: pad_mount(d, f, thickness=t) for f in L20_CHAINS}
    return hand


def distal_surface(urdf_dir: str, finger: str):
    """(verts (N,3), faces (F,3)) of the finger's OWN distal mesh, deduplicated
    -- the body the native taxel film lives on."""
    link = L20_CHAINS[finger][0]
    raw = _read_stl_verts(osp.join(urdf_dir, "meshes", f"{link}.STL"))
    verts, inv = np.unique(np.round(raw, 9), axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3).astype(np.int32)
    ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])         & (faces[:, 0] != faces[:, 2])
    return verts, faces[ok]


def pad_box_stl(out_path: str, size=PAD_SIZE, seg_mm: float = 1.0) -> str:
    """Write a subdivided box STL for the pad (pad-local frame: coat at z=0).

    Subdivided (not 8-corner) so tetgen's surface conformity gives the coat an
    even vertex distribution before refinement.
    """
    sx, sy, sz = size
    nx = max(2, int(round(sx * 1e3 / seg_mm)) + 1)
    ny = max(2, int(round(sy * 1e3 / seg_mm)) + 1)
    nz = max(2, int(round(sz * 1e3 / seg_mm)) + 1)
    xs = np.linspace(-sx / 2, sx / 2, nx)
    ys = np.linspace(-sy / 2, sy / 2, ny)
    zs = np.linspace(0.0, sz, nz)

    quads = []

    def face(u, vv, fixed_axis, fixed_val, flip):
        for i in range(len(u) - 1):
            for j in range(len(vv) - 1):
                c = []
                for (a, b) in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                    p = [0.0, 0.0, 0.0]
                    ax = [k for k in range(3) if k != fixed_axis]
                    p[ax[0]], p[ax[1]] = u[a], vv[b]
                    p[fixed_axis] = fixed_val
                    c.append(p)
                if flip:
                    c = c[::-1]
                quads.append(c)

    face(xs, ys, 2, 0.0, True)      # coat (z=0), normal -z
    face(xs, ys, 2, sz, False)      # stick (z=t), normal +z
    face(xs, zs, 1, -sy / 2, False)
    face(xs, zs, 1, +sy / 2, True)
    face(ys, zs, 0, -sx / 2, True)
    face(ys, zs, 0, +sx / 2, False)

    tris = []
    for c in quads:
        tris.append([c[0], c[1], c[2]])
        tris.append([c[0], c[2], c[3]])
    tris = np.asarray(tris, np.float64)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)

    with open(out_path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for i in range(len(tris)):
            f.write(struct.pack("<3f", *n[i]))
            for v in tris[i]:
                f.write(struct.pack("<3f", *v))
            f.write(b"\0\0")
    return out_path
