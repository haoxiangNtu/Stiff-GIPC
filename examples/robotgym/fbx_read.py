"""Minimal binary FBX (7.x) reader: node tree -> models / geometries / materials with world transforms.

    python -I fbxread.py <file.fbx> [--obj out_dir]       # summary (+ one OBJ per mesh model, metres, Y up)

Pure python + numpy; no FBX SDK. Handles the 7.4 (32-bit offsets) and 7.5 (64-bit) layouts, zlib arrays, the
Properties70 transform stack (pre/post rotation, pivots, geometric transform) and the GlobalSettings unit/axes.
"""
from __future__ import annotations

import os
import struct
import sys
import zlib

import numpy as np


class Node:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, props, children):
        self.name, self.props, self.children = name, props, children

    def find(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def findall(self, name):
        return [c for c in self.children if c.name == name]

    def __repr__(self):
        return f"Node({self.name!r}, {len(self.props)} props, {len(self.children)} children)"


_ARR = {"f": "<f4", "d": "<f8", "l": "<i8", "i": "<i4", "b": "u1"}


def read_tree(path):
    data = open(path, "rb").read()
    if not data.startswith(b"Kaydara FBX Binary  \x00"):
        raise ValueError("not a binary FBX")
    ver = struct.unpack_from("<I", data, 23)[0]
    wide = ver >= 7500
    hdr = struct.Struct("<QQQ" if wide else "<III")

    def node(off):
        end, nprops, _plen = hdr.unpack_from(data, off)
        off += hdr.size
        nlen = data[off]
        off += 1
        if end == 0:
            return None, off
        name = data[off:off + nlen].decode("latin1")
        off += nlen
        props = []
        for _ in range(nprops):
            t = chr(data[off]); off += 1
            if t == "Y":
                v = struct.unpack_from("<h", data, off)[0]; off += 2
            elif t == "C":
                v = data[off] != 0; off += 1
            elif t == "I":
                v = struct.unpack_from("<i", data, off)[0]; off += 4
            elif t == "F":
                v = struct.unpack_from("<f", data, off)[0]; off += 4
            elif t == "D":
                v = struct.unpack_from("<d", data, off)[0]; off += 8
            elif t == "L":
                v = struct.unpack_from("<q", data, off)[0]; off += 8
            elif t in _ARR:
                alen, enc, clen = struct.unpack_from("<III", data, off); off += 12
                raw = data[off:off + clen]; off += clen
                if enc == 1:
                    raw = zlib.decompress(raw)
                v = np.frombuffer(raw, _ARR[t], count=alen).copy()
            elif t in "SR":
                ln = struct.unpack_from("<I", data, off)[0]; off += 4
                v = data[off:off + ln]; off += ln
                if t == "S":
                    v = v.decode("utf-8", "replace")
            else:
                raise ValueError(f"unknown property type {t!r} at {off}")
            props.append(v)
        children = []
        while off < end:
            ch, off = node(off)
            if ch is None:
                break
            children.append(ch)
        return Node(name, props, children), end

    top, off = [], 27
    while off < len(data) - hdr.size:
        n, off = node(off)
        if n is None:
            break
        top.append(n)
    return ver, Node("<root>", [], top)


def props70(n):
    out = {}
    p = n.find("Properties70") if n is not None else None
    if p is None:
        return out
    for c in p.findall("P"):
        key = c.props[0]
        vals = c.props[4:]
        out[key] = vals[0] if len(vals) == 1 else (tuple(vals) if vals else None)
    return out


def clean(s):
    return s.split("\x00\x01")[0] if isinstance(s, str) else s


def euler(rot_deg, order=0):
    a, b, c = np.radians(np.asarray(rot_deg, float))
    def R(ax, t):
        cs, sn = np.cos(t), np.sin(t)
        M = np.eye(3)
        i, j = [(1, 2), (0, 2), (0, 1)][ax]
        M[i, i], M[i, j], M[j, i], M[j, j] = cs, -sn, sn, cs
        if ax == 1:
            M[0, 2], M[2, 0] = sn, -sn
        return M
    X, Y, Z = R(0, a), R(1, b), R(2, c)
    # FBX enum: 0 XYZ, 1 XZY, 2 YZX, 3 YXZ, 4 ZXY, 5 ZYX (first letter applied first)
    seq = {0: (X, Y, Z), 1: (X, Z, Y), 2: (Y, Z, X), 3: (Y, X, Z), 4: (Z, X, Y), 5: (Z, Y, X)}[int(order)]
    M = np.eye(3)
    for m in seq:
        M = m @ M
    return M


def T4(t=(0, 0, 0), R=None, s=None):
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    if s is not None:
        M[:3, :3] = M[:3, :3] @ np.diag(s)
    M[:3, 3] = t
    return M


def local_tf(p):
    g = lambda k, d: np.asarray(p.get(k, d), float)
    order = int(p.get("RotationOrder", 0) or 0)
    T = T4(g("Lcl Translation", (0, 0, 0)))
    Roff = T4(g("RotationOffset", (0, 0, 0)))
    Rp = T4(g("RotationPivot", (0, 0, 0)))
    Rpre = T4(R=euler(g("PreRotation", (0, 0, 0)), 0))
    Rr = T4(R=euler(g("Lcl Rotation", (0, 0, 0)), order))
    Rpost = T4(R=euler(g("PostRotation", (0, 0, 0)), 0))
    Soff = T4(g("ScalingOffset", (0, 0, 0)))
    Sp = T4(g("ScalingPivot", (0, 0, 0)))
    S = T4(s=g("Lcl Scaling", (1, 1, 1)))
    L = T @ Roff @ Rp @ Rpre @ Rr @ np.linalg.inv(Rpost) @ np.linalg.inv(Rp) @ Soff @ Sp @ S @ np.linalg.inv(Sp)
    G = T4(g("GeometricTranslation", (0, 0, 0)), euler(g("GeometricRotation", (0, 0, 0)), 0), g("GeometricScaling", (1, 1, 1)))
    return L, G


def layer_map(geo, layer, key, idx_key, poly_start, poly_len, n_cv):
    """Per polygon-vertex (or per polygon) values of a LayerElement (normal / UV / material)."""
    le = geo.find(layer)
    if le is None:
        return None, None
    mapping = le.find("MappingInformationType").props[0] if le.find("MappingInformationType") else ""
    ref = le.find("ReferenceInformationType").props[0] if le.find("ReferenceInformationType") else "Direct"
    vals = le.find(key).props[0] if le.find(key) else None
    idx = le.find(idx_key).props[0] if (idx_key and le.find(idx_key)) else None
    return (mapping, ref, vals, idx), le


class Scene:
    def __init__(self, path):
        self.path = path
        self.version, self.root = read_tree(path)
        gs = props70(self.root.find("GlobalSettings"))
        self.global_settings = gs
        self.unit_cm = float(gs.get("UnitScaleFactor", 1.0))
        objs = self.root.find("Objects")
        self.objects = {}
        for o in objs.children:
            oid = o.props[0]
            self.objects[oid] = o
        self.parent = {}
        self.children = {}
        self.op = []
        for c in self.root.find("Connections").findall("C"):
            kind, child, par = c.props[0], c.props[1], c.props[2]
            if kind == "OO":
                self.children.setdefault(par, []).append(child)
                self.parent.setdefault(child, []).append(par)
            else:
                self.op.append((child, par, c.props[3] if len(c.props) > 3 else None))
                self.children.setdefault(par, []).append(child)

    def axes(self):
        g = self.global_settings
        return dict(up=(int(g.get("UpAxis", 1)), int(g.get("UpAxisSign", 1))),
                    front=(int(g.get("FrontAxis", 2)), int(g.get("FrontAxisSign", 1))),
                    coord=(int(g.get("CoordAxis", 0)), int(g.get("CoordAxisSign", 1))),
                    unit_cm=self.unit_cm,
                    orig_up=(g.get("OriginalUpAxis"), g.get("OriginalUpAxisSign")),
                    orig_unit=g.get("OriginalUnitScaleFactor"))

    def models(self):
        return {i: o for i, o in self.objects.items() if o.name == "Model"}

    def world(self, mid, cache=None):
        cache = {} if cache is None else cache
        if mid in cache:
            return cache[mid]
        o = self.objects[mid]
        L, _ = local_tf(props70(o))
        par = [p for p in self.parent.get(mid, []) if p in self.objects and self.objects[p].name == "Model"]
        W = (self.world(par[0], cache) @ L) if par else L
        cache[mid] = W
        return W

    def mesh_of(self, mid):
        for c in self.children.get(mid, []):
            o = self.objects.get(c)
            if o is not None and o.name == "Geometry" and len(o.props) > 2 and o.props[2] == "Mesh":
                return o
        return None

    def materials_of(self, mid):
        return [clean(self.objects[c].props[1]) for c in self.children.get(mid, []) if c in self.objects and self.objects[c].name == "Material"]

    def textures_of_material(self, mat_id):
        out = []
        for c in self.children.get(mat_id, []):
            o = self.objects.get(c)
            if o is not None and o.name == "Texture":
                fn = o.find("RelativeFilename") or o.find("FileName")
                out.append(fn.props[0] if fn else "")
        return out

    def geometry_arrays(self, geo):
        V = geo.find("Vertices").props[0].reshape(-1, 3).astype(np.float64)
        pvi = geo.find("PolygonVertexIndex").props[0].astype(np.int64)
        ends = pvi < 0
        idx = np.where(ends, ~pvi, pvi)
        end_pos = np.nonzero(ends)[0]
        starts = np.concatenate([[0], end_pos[:-1] + 1])
        lens = end_pos - starts + 1
        # material per polygon
        mat = np.zeros(len(starts), np.int64)
        le = geo.find("LayerElementMaterial")
        if le is not None and le.find("Materials") is not None:
            m = le.find("Materials").props[0].astype(np.int64)
            mapping = le.find("MappingInformationType").props[0]
            if mapping == "ByPolygon" and len(m) == len(starts):
                mat = m
            elif len(m):
                mat[:] = m[0]
        # UVs per polygon-vertex
        uv = None
        le = geo.find("LayerElementUV")
        if le is not None and le.find("UV") is not None:
            vals = le.find("UV").props[0].reshape(-1, 2)
            mapping = le.find("MappingInformationType").props[0]
            ref = le.find("ReferenceInformationType").props[0]
            ii = le.find("UVIndex").props[0].astype(np.int64) if le.find("UVIndex") is not None else None
            if mapping == "ByPolygonVertex":
                uv = vals[ii] if (ref == "IndexToDirect" and ii is not None) else vals[: len(idx)]
            elif mapping in ("ByVertice", "ByVertex"):
                uv = (vals[ii] if (ref == "IndexToDirect" and ii is not None) else vals)[idx]
        return V, idx, starts, lens, mat, uv

    def triangulate(self, idx, starts, lens):
        tris, poly_of = [], []
        for p, (s, n) in enumerate(zip(starts, lens)):
            for k in range(1, n - 1):
                tris.append((s, s + k, s + k + 1))
                poly_of.append(p)
        tris = np.array(tris, np.int64).reshape(-1, 3)
        return tris, np.array(poly_of, np.int64)


def summary(path, obj_dir=None, to_m=True):
    sc = Scene(path)
    ax = sc.axes()
    s = 0.01 * sc.unit_cm if to_m else 1.0
    print(f"FBX {sc.version}  unit = {sc.unit_cm} cm  axes {ax}")
    gs = sc.global_settings
    for k in ("OriginalUnitScaleFactor", "UnitScaleFactor", "UpAxis", "FrontAxis", "CoordAxis"):
        print(f"  {k} = {gs.get(k)}")
    fi = sc.root.find("FBXHeaderExtension")
    if fi is not None:
        si = fi.find("SceneInfo")
        if si is not None:
            p = props70(si)
            for k in ("Original|ApplicationVendor", "Original|ApplicationName", "Original|ApplicationVersion"):
                print(f"  {k} = {p.get(k)}")
    kinds = {}
    for o in sc.objects.values():
        kinds[o.name] = kinds.get(o.name, 0) + 1
    print("  objects:", kinds)
    cache = {}
    allV = []
    rows = []
    for mid, m in sc.models().items():
        name = clean(m.props[1]); typ = m.props[2] if len(m.props) > 2 else ""
        W = sc.world(mid, cache)
        _, G = local_tf(props70(m))
        geo = sc.mesh_of(mid)
        par = [p for p in sc.parent.get(mid, []) if p in sc.objects]
        pname = clean(sc.objects[par[0]].props[1]) if par else "<root>"
        if geo is None:
            rows.append((name, typ, pname, 0, 0, None, None))
            continue
        V, idx, starts, lens, mat, uv = sc.geometry_arrays(geo)
        M = W @ G
        Vw = (V @ M[:3, :3].T + M[:3, 3]) * s
        allV.append(Vw)
        lo, hi = Vw.min(0), Vw.max(0)
        mats = sc.materials_of(mid)
        rows.append((name, typ, pname, len(V), len(starts), lo, hi, mats, np.bincount(lens).nonzero()[0].tolist(), uv is not None))
        if obj_dir:
            os.makedirs(obj_dir, exist_ok=True)
            tris, poly_of = sc.triangulate(idx, starts, lens)
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name) or f"m{mid}"
            with open(os.path.join(obj_dir, f"{safe}.obj"), "w") as f:
                for v in Vw:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                if uv is not None:
                    for t in uv:
                        f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
                for tri, p in zip(tris, poly_of):
                    a, b, c = idx[tri] + 1
                    if uv is not None:
                        ta, tb, tc = tri + 1
                        f.write(f"f {a}/{ta} {b}/{tb} {c}/{tc}\n")
                    else:
                        f.write(f"f {a} {b} {c}\n")
    for r in rows:
        if r[3] == 0:
            print(f"  [{r[1]:>6s}] {r[0]!r:40s} parent {r[2]!r}")
        else:
            name, typ, pname, nv, npoly, lo, hi, mats, plens, has_uv = r
            print(f"  [{typ:>6s}] {name!r:40s} parent {pname!r:20s} v {nv:7d} poly {npoly:7d} (sides {plens}) uv {has_uv}  "
                  f"bbox lo {np.round(lo, 4).tolist()} hi {np.round(hi, 4).tolist()} size {np.round(hi - lo, 4).tolist()}  mats {mats}")
    if allV:
        A = np.concatenate(allV)
        print(f"  SCENE bbox lo {np.round(A.min(0), 4).tolist()} hi {np.round(A.max(0), 4).tolist()} size {np.round(A.max(0) - A.min(0), 4).tolist()} (m if unit applied)")
    mats = {i: o for i, o in sc.objects.items() if o.name == "Material"}
    for i, o in mats.items():
        p = props70(o)
        print(f"  material {clean(o.props[1])!r}: diffuse {p.get('DiffuseColor')} opacity {p.get('Opacity')} textures {sc.textures_of_material(i)}")
    vids = [o for o in sc.objects.values() if o.name == "Video"]
    for v in vids:
        cont = v.find("Content")
        n = len(cont.props[0]) if (cont is not None and cont.props and isinstance(cont.props[0], (bytes, bytearray))) else 0
        fn = v.find("RelativeFilename") or v.find("Filename")
        print(f"  video {clean(v.props[1])!r}: file {fn.props[0] if fn else ''!r} embedded {n} bytes")
    return sc


if __name__ == "__main__":
    p = sys.argv[1]
    od = sys.argv[sys.argv.index("--obj") + 1] if "--obj" in sys.argv else None
    summary(p, od)
