#!/usr/bin/env python
"""Path-trace a cucumber_cut recording with Mitsuba 3 (CUDA).

The MPM cucumber is reconstructed from its particles every frame (pysplashsurf) and coloured by where each surface point
sat in the uncut cucumber (the 2 mm skin dark green, the flesh pale green, the seed core paler), so a cut face shows flesh.
With --pieces (prep_pieces.py) it is drawn instead from its own surface mesh, cut where the simulation cut it and moved
with the particles (the "deform pieces" way of rendering MPM fracture): flat cut faces with a cross-section texture.
The knife (steel blade, black handle) and every FR3 link are drawn at their SIMULATED poses: the recorded rigid motion of
the link bodies carries the URDF visual meshes from their load pose, so the hand stays on the knife whatever the arm's
tracking error. The board is wood. Frames -> PNG -> mp4 (+ an optional close-up inset of the cut).

    <python with mitsuba, pysplashsurf, trimesh(+pycollada) or a visuals cache> examples/cutting/render_cucumber.py \\
        out/cutting/<run> --out out/cutting/<run>/render [--start 0 --end 0 --stride 1 --spp 64]

--visuals-cache: the FR3 visual meshes (link frame) as an npz, written on first use; a machine without pycollada can
render from a cache written elsewhere.
"""
import argparse
import json
import math
import os
import os.path as osp
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", ".."))
sys.path.insert(0, osp.join(REPO, "examples", "lab_tactile"))
from lab_rec import load_recording  # noqa: E402
from fr3_kin import Fr3Arm, xyz_rpy_T  # noqa: E402

URDF = osp.join(REPO, "Assets", "fr3", "fr3_franka_hand.urdf")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
SKIN = (0.10, 0.24, 0.07)       # sRGB: dark cucumber green
FLESH = (0.80, 0.86, 0.60)      # pale green flesh
CORE = (0.90, 0.91, 0.74)       # seed core


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def load_visuals(cache):
    """URDF link -> [(V in the link frame, F, rgb)], from the visual DAE meshes (trimesh + pycollada) or the cache"""
    if cache and osp.exists(cache):
        z = np.load(cache, allow_pickle=False)
        idx = json.loads(str(z["index"]))
        return {link: [(z[f"{k}_V"], z[f"{k}_F"], z[f"{k}_c"]) for k in keys] for link, keys in idx.items()}
    import trimesh
    root = ET.parse(URDF).getroot()
    out = {}
    for link in root.findall("link"):
        vis = link.find("visual")
        mesh = vis.find("geometry/mesh") if vis is not None else None
        if mesh is None:
            continue
        path = mesh.get("filename")
        if not osp.isabs(path):
            path = osp.join(osp.dirname(URDF), path)
        if not osp.exists(path):
            continue
        o = vis.find("origin")
        T0 = np.eye(4)
        if o is not None:
            T0 = xyz_rpy_T([float(x) for x in (o.get("xyz") or "0 0 0").split()], [float(x) for x in (o.get("rpy") or "0 0 0").split()])
        s = trimesh.load(path, force="scene")
        parts = []
        for node in s.graph.nodes_geometry:
            T, gname = s.graph[node]
            g = s.geometry[gname].copy().apply_transform(T0 @ T)
            m = getattr(g.visual, "material", None)
            col = None
            for attr in ("baseColorFactor", "diffuse", "main_color"):
                c = getattr(m, attr, None) if m is not None else None
                if c is not None:
                    col = np.asarray(c, float)[:3]
                    break
            col = np.array([0.9, 0.9, 0.9]) if col is None else (col / 255.0 if col.max() > 1.0 else col)
            parts.append((np.asarray(g.vertices, np.float32), np.asarray(g.faces, np.int32), np.asarray(col, np.float32)))
        out[link.get("name")] = parts
    if cache:
        flat, idx = {}, {}
        for link, parts in out.items():
            idx[link] = []
            for j, (V, F, c) in enumerate(parts):
                k = f"{link}__{j}"
                idx[link].append(k)
                flat[f"{k}_V"], flat[f"{k}_F"], flat[f"{k}_c"] = V, F, c
        np.savez_compressed(cache, index=np.array(json.dumps(idx)), **flat)
    return out


def vertex_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], n)
    return N / np.maximum(np.linalg.norm(N, axis=1), 1e-30)[:, None]


def rest_depth(P, V, F, h):
    """distance from each rest particle to the cucumber surface (nearest of dense surface samples)"""
    from scipy.spatial import cKDTree
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    n = np.maximum(1, np.ceil(area / (h / 3.0) ** 2).astype(int))
    rng = np.random.default_rng(1)
    idx = np.repeat(np.arange(len(F)), n)
    r1, r2 = rng.random(len(idx)), rng.random(len(idx))
    sq = np.sqrt(r1)
    S = (1 - sq)[:, None] * A[idx] + (sq * (1 - r2))[:, None] * B[idx] + (sq * r2)[:, None] * C[idx]
    return cKDTree(S).query(P)[0]


def particle_colours(x0, skin, Vc, Fc, spacing):
    """linear RGB per particle: skin / flesh / seed core (deeper than ~60 % of the local radius)"""
    d = rest_depth(x0, Vc, Fc, spacing)
    zb = np.floor((x0[:, 2] - x0[:, 2].min()) / 0.004).astype(int)
    rmax = np.zeros(zb.max() + 1)
    np.maximum.at(rmax, zb, d)
    u = np.clip((d / np.maximum(rmax[zb], 1e-6) - 0.45) / 0.25, 0.0, 1.0)
    u = u * u * (3 - 2 * u)
    col = (1 - u)[:, None] * srgb_to_linear(FLESH) + u[:, None] * srgb_to_linear(CORE)
    col[skin] = srgb_to_linear(SKIN)
    return col.astype(np.float32)


def surface(P, spacing, sl, cube, iso, smooth):
    import pysplashsurf as ps
    P = np.ascontiguousarray(P, np.float32)
    out = ps.reconstruction_pipeline(P, particle_radius=0.5 * spacing, rest_density=1000.0, smoothing_length=sl, cube_size=cube,
                                     iso_surface_threshold=iso, multi_threading=True, mesh_cleanup=True, decimate_barnacles=True,
                                     mesh_smoothing_iters=smooth, mesh_smoothing_weights=True, compute_normals=False)
    m = out[0] if isinstance(out, tuple) else out
    while hasattr(m, "mesh"):
        m = m.mesh
    return np.asarray(m.vertices, float), np.asarray(m.triangles, np.int64)


def knife_handle_faces(V, F, thick=0.006):
    """faces of the handle: where the knife (principal axes of its rest mesh) is thicker than the 4 mm blade"""
    c = V.mean(0)
    _, _, W = np.linalg.svd(V - c, full_matrices=False)
    a, t = (V - c) @ W[0], (V - c) @ W[2]
    b = np.floor((a - a.min()) / 0.002).astype(int)
    lo = np.full(b.max() + 1, np.inf); hi = np.full(b.max() + 1, -np.inf)
    np.minimum.at(lo, b, t); np.maximum.at(hi, b, t)
    th = (hi - lo)[b]
    return th[F].mean(1) > thick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="cucumber_cut output dir (recording.npz, particles.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=0); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--spp", type=int, default=64); ap.add_argument("--bounces", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--eye", type=float, nargs=3, default=(0.86, 0.30, -0.44)); ap.add_argument("--target", type=float, nargs=3, default=(0.45, 0.08, -0.07))
    ap.add_argument("--fov", type=float, default=0.62); ap.add_argument("--exposure", type=float, default=1.0)
    ap.add_argument("--inset-eye", type=float, nargs=3, default=None, help="fixed close-up camera drawn as an inset at the bottom right")
    ap.add_argument("--inset-target", type=float, nargs=3, default=(0.50, 0.04, -0.09)); ap.add_argument("--inset-fov", type=float, default=0.40)
    ap.add_argument("--inset-size", type=int, nargs=2, default=(480, 300)); ap.add_argument("--inset-label", default="")
    ap.add_argument("--inset-pos", default="bl", choices=["bl", "br", "tl", "tr"], help="inset corner")
    ap.add_argument("--follow", action="store_true", help="the main camera rides with the knife: --eye/--target z are relative to "
                    "the blade's mid-plane (clamped to the planes of the cuts), so a camera in that plane sees the blade edge-on")
    ap.add_argument("--inset-follow", action="store_true", help="the same for the inset camera (--inset-eye/--inset-target)")
    ap.add_argument("--label", default="", help="caption drawn at the top left")
    ap.add_argument("--sl", type=float, default=2.2, help="splashsurf smoothing length (x particle radius)")
    ap.add_argument("--cube", type=float, default=0.5, help="splashsurf marching-cubes cell (x particle radius)")
    ap.add_argument("--iso", type=float, default=0.6); ap.add_argument("--smooth", type=int, default=10)
    ap.add_argument("--visuals-cache", default=osp.join(REPO, "out", "cutting", "fr3_visuals_cache.npz"))
    ap.add_argument("--no-denoise", action="store_true"); ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--variant", default="cuda_ad_rgb", help="mitsuba variant: llvm_ad_rgb renders on the CPU (set DRJIT_LIBLLVM_PATH) where the driver is too old for mitsuba's OptiX programs (the pod's 535)")
    ap.add_argument("--fps", type=float, default=0.0, help="video fps (default: the recording rate / stride)")
    ap.add_argument("--bump", type=float, default=0.0003, help="--pieces: wart bump height on the skin (m)")
    ap.add_argument("--cap-look", type=float, nargs=4, default=(0.45, 0.35, 0.1, 0.65), metavar=("ROUGH", "SPEC", "COAT", "GAIN"),
                    help="--pieces: cut-face roughness, specular, clearcoat and albedo gain (a face turned to the lamp can saturate)")
    ap.add_argument("--pieces", default="", help="deform-pieces data from prep_pieces.py: draw the cucumber's own mesh, cut along the "
                    "simulated cuts and moved with the particles, instead of a surface reconstructed from the particles")
    args = ap.parse_args()
    fdir = osp.join(args.out, "frames"); os.makedirs(fdir, exist_ok=True)
    import mitsuba as mi
    mi.set_variant(args.variant)
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial import cKDTree

    z = load_recording(osp.join(args.run, "recording.npz"))
    meta = z["meta"]
    pz = np.load(osp.join(args.run, "particles.npz"))
    spacing = float(meta.get("particle_spacing", 0.0015))
    pskin = pz["skin"].astype(bool)
    pcol = particle_colours(pz["x0"].astype(np.float64), pskin, pz["cucumber_V"].astype(np.float64), pz["cucumber_F"], spacing)
    pcol_skin = srgb_to_linear(SKIN).astype(np.float32)
    names, kinds = [str(n) for n in z["body_names"]], [str(k) for k in z["body_kind"]]
    rest, faces, motion = z["body_rest"], z["body_faces"], np.asarray(z["motion"], float)
    t_rec = np.asarray(z["t"], float)
    L = z["liquid_pts"]
    n_rec = len(t_rec)
    frames = list(range(args.start, args.end or n_rec, args.stride))
    rec_fps = 1.0 / max(float(np.median(np.diff(t_rec))), 1e-9)
    fps = args.fps or rec_fps / args.stride

    # arm: URDF visuals carried by the recorded motion of their link bodies from the load pose (FK of the first record)
    visuals = load_visuals(args.visuals_cache)
    arms = meta.get("arms") or []
    q_arms = np.asarray(z["q_arms"], float) if "q_arms" in z else None
    gap_arms = np.asarray(z["gap_arms"], float) if "gap_arms" in z else None
    link_draw = []                                  # (body index, link name, T_link at record 0 in the load frame)
    if arms and q_arms is not None and q_arms.shape[1] > 0:
        kin = Fr3Arm(URDF, np.asarray(arms[0]["base_T"], float))
        poses0 = kin.link_poses(q_arms[0, 0], float(np.clip(gap_arms[0, 0] / 2000.0, 0.0, 0.04)) if gap_arms is not None and gap_arms.shape[1] else 0.0)
        for i, n in enumerate(names):
            for link in visuals:
                if (n == link or n.endswith("_" + link) or n.endswith("/" + link)) and link in poses0:
                    M0 = np.eye(4); M0[:3, :3] = motion[0, i][:3, :3]; M0[:3, 3] = motion[0, i][:3, 3]
                    link_draw.append((i, link, np.linalg.inv(M0) @ poses0[link]))
                    break
    drawn_arm = {i for i, _, _ in link_draw}
    print(f"[render] {len(frames)} frames of {n_rec} ({rec_fps:.1f} fps rec -> {fps:.1f} fps video); arm links drawn from motion: "
          f"{[l for _, l, _ in link_draw]}; bodies {list(zip(names, kinds))}", flush=True)
    ik = names.index("knife") if "knife" in names else -1
    handle = knife_handle_faces(np.asarray(rest[ik], float), np.asarray(faces[ik])) if ik >= 0 else None

    def M4(i, fr):
        M = np.eye(4); M[:3, :3] = motion[fr, i][:3, :3]; M[:3, 3] = motion[fr, i][:3, 3]; return M

    def world(i, fr):
        M = motion[fr, i]; return np.asarray(rest[i], float) @ M[:3, :3].T + M[:3, 3]

    def mesh(name, V, F, bsdf, N=None, colors=None, uv=None):
        V = np.ascontiguousarray(V, dtype=np.float32); F = np.ascontiguousarray(F, dtype=np.uint32)
        props = mi.Properties(); props["bsdf"] = mi.load_dict(bsdf) if isinstance(bsdf, dict) else bsdf
        m = mi.Mesh(name, vertex_count=len(V), face_count=len(F), has_vertex_normals=True, has_vertex_texcoords=uv is not None, props=props)
        if colors is not None:
            m.add_attribute("vertex_color", 3, np.ascontiguousarray(colors, np.float32).ravel())
        p = mi.traverse(m); p["vertex_positions"] = V.ravel(); p["faces"] = F.ravel()
        if uv is not None:
            p["vertex_texcoords"] = np.ascontiguousarray(uv, np.float32).ravel()
        if N is None:
            N = vertex_normals(V.astype(np.float64), F.astype(np.int64))
        p["vertex_normals"] = np.ascontiguousarray(N, dtype=np.float32).ravel(); p.update()
        return m

    def principled(rgb, rough=0.5, spec=0.5, metal=0.0, two=True):
        b = {"type": "principled", "base_color": {"type": "rgb", "value": [float(x) for x in rgb]}, "roughness": rough,
             "specular": spec, "metallic": metal}
        return {"type": "twosided", "material": b} if two else b
    cucumber_bsdf = {"type": "twosided", "material": {"type": "principled", "base_color": {"type": "mesh_attribute", "name": "vertex_color"},
                                                      "roughness": 0.38, "specular": 0.55}}
    steel = {"type": "twosided", "material": {"type": "roughconductor", "material": "Cr", "distribution": "ggx", "alpha": 0.10}}
    pieces = None
    if args.pieces:
        from scipy import sparse
        pc = np.load(args.pieces)
        nV, NP = len(pc["V"]), int(pc["N"])
        if NP != len(pskin):
            raise SystemExit(f"--pieces was built for {NP} particles, this run has {len(pskin)}")
        seg_p = [str(x) for x in pc["segment"]]
        spans = []                                   # records of "cut j" per cut plane: the crack front runs down then
        for j in range(int(pc["n_plane"])):
            idx = [i for i, sname in enumerate(seg_p) if sname == f"cut {j + 1}"]
            spans.append((min(idx), max(idx)) if idx else (10 ** 9, 10 ** 9))
        skin_parts, cap_parts = [], []
        for i in range(int(pc["n_skin"])):
            F_ = pc[f"skin{i}_F"].astype(np.int64); u = np.unique(F_); rm = np.full(nV, -1, np.int64); rm[u] = np.arange(len(u))
            skin_parts.append((u, rm[F_]))
        for i in range(int(pc["n_cap"])):
            UV = pc[f"cap{i}_UV"]; v0 = int(pc[f"cap{i}_v0"])
            cap_parts.append((np.arange(v0, v0 + len(UV)), pc[f"cap{i}_F"].astype(np.int64) - v0, UV, int(pc[f"cap{i}_plane"])))
        cr, cs, cc, cg = args.cap_look
        cap_bsdfs = [mi.load_dict({"type": "twosided", "material": {
            "type": "principled", "roughness": cr, "specular": cs, "clearcoat": cc, "clearcoat_gloss": 0.85,
            "base_color": {"type": "bitmap", "bitmap": mi.Bitmap(cg * pc[f"tex{j}"].astype(np.float32) / 65535.0),
                           "filter_type": "bilinear", "wrap_mode": "clamp"}}}) for j in range(int(pc["n_plane"]))]
        skin_bsdf = mi.load_dict({"type": "twosided", "material": {   # procedural skin colour + raised warts (bump, 0.3 mm)
            "type": "bumpmap", "scale": args.bump,
            "arbitrary": {"type": "bitmap", "bitmap": mi.Bitmap((pc["skin_bump"].astype(np.float32) / 65535.0)[..., None]), "raw": True,
                          "wrap_mode": "repeat", "filter_type": "bilinear"},
            "bsdf": {"type": "principled", "roughness": 0.30, "specular": 0.5, "clearcoat": 0.45, "clearcoat_gloss": 0.75,
                     "base_color": {"type": "bitmap", "bitmap": mi.Bitmap(pc["skin_tex"].astype(np.float32) / 65535.0),
                                    "wrap_mode": "repeat", "filter_type": "bilinear"}}}})
        pieces = dict(So=sparse.csr_matrix((pc["So_data"], pc["So_indices"], pc["So_indptr"]), shape=(nV, NP)),
                      Su=sparse.csr_matrix((pc["Su_data"], pc["Su_indices"], pc["Su_indptr"]), shape=(nV, NP)),
                      origin=np.asarray(pc["origin"], float), vplane=pc["vplane"].astype(int), edge=np.asarray(pc["edge"], float),
                      lead=float(pc["lead"]), spans=spans, skin=skin_parts, caps=cap_parts, rgb=pc["skin_rgb"], uv=pc["skin_uv"])
        print(f"[render] deform pieces: {nV} vertices, {len(skin_parts)} pieces, {len(cap_parts)} cut faces, cuts at records {spans}", flush=True)
    eye, tgt = np.array(args.eye), np.array(args.target)

    def sensor(eye_, tgt_, fov, w, h, spp):
        return mi.load_dict({"type": "perspective", "fov": math.degrees(fov), "fov_axis": "y", "near_clip": 0.005, "far_clip": 50.0,
                             "to_world": mi.ScalarTransform4f().look_at(origin=list(map(float, eye_)), target=list(map(float, tgt_)), up=[0, 1, 0]),
                             "film": {"type": "hdrfilm", "width": w, "height": h, "pixel_format": "rgb",
                                      "rfilter": {"type": "box" if not args.no_denoise else "gaussian"}},
                             "sampler": {"type": "independent", "sample_count": spp}})
    cam = sensor(eye, tgt, args.fov, args.width, args.height, args.spp)
    cam_in = sensor(args.inset_eye, args.inset_target, args.inset_fov, *args.inset_size, args.spp) if args.inset_eye is not None else None
    zfol = None
    if (args.follow or args.inset_follow) and ik >= 0:
        # the blade's mid-plane z next to the cucumber, clamped to the planes of the cuts: between cuts the camera moves on with
        # the knife, before the first cut and after the last it stays at the first / last plane
        xc = float(np.median(np.asarray(L[0], np.float64)[:, 0]))
        bv = np.unique(np.asarray(faces[ik])[~handle])

        def blade_z(fr):
            V = world(ik, fr)[bv]; m = np.abs(V[:, 0] - xc) < 0.04
            return float(V[m, 2].mean() if m.any() else V[:, 2].mean())
        seg_r = [str(s) for s in z["segment"]]
        zc = []
        for c in range(1, 100):
            idx = [i for i, s in enumerate(seg_r) if s == f"cut {c}"]
            if not idx:
                break
            zc.append(blade_z(idx[len(idx) // 2]))
        if zc:
            def zfol(fr):
                return float(np.clip(blade_z(fr), min(zc), max(zc)))
            print(f"[render] cameras follow the blade plane, cuts at z {[round(v * 1e3, 1) for v in zc]} mm", flush=True)
    static = {"type": "scene", "integrator": {"type": "path", "max_depth": args.bounces},
              "sky": {"type": "constant", "radiance": {"type": "rgb", "value": [0.52, 0.54, 0.58]}},
              "lamp": {"type": "rectangle", "to_world": mi.ScalarTransform4f().look_at(origin=[0.75, 1.6, 0.65], target=[0.45, 0.05, -0.08], up=[0, 0, 1]).scale([0.45, 0.45, 1.0]),
                       "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [16.0, 15.5, 14.6]}}},
              "floor": mesh("floor", np.array([[-0.8, 0, -1.2], [1.6, 0, -1.2], [1.6, 0, 1.2], [-0.8, 0, 1.2]]), np.array([[0, 2, 1], [0, 3, 2]]),
                            principled(srgb_to_linear((0.80, 0.79, 0.76)), 0.6, 0.3, two=False))}
    denoiser = None
    if not args.no_denoise:   # Intel Open Image Denoise (CPU)
        try:
            import oidn
            dev = oidn.NewDevice(); oidn.CommitDevice(dev)

            def denoiser(img):
                a = np.ascontiguousarray(np.asarray(img), dtype=np.float32); o = np.zeros_like(a)
                f = oidn.NewFilter(dev, "RT")
                oidn.SetSharedFilterImage(f, "color", a, oidn.FORMAT_FLOAT3, a.shape[1], a.shape[0])
                oidn.SetSharedFilterImage(f, "output", o, oidn.FORMAT_FLOAT3, a.shape[1], a.shape[0])
                try:
                    oidn.RawFunctions.oidnSetFilterBool(f, b"hdr", True)
                except Exception:
                    pass
                oidn.CommitFilter(f); oidn.ExecuteFilter(f); oidn.ReleaseFilter(f)
                return o
        except Exception as e:
            print("denoiser unavailable:", e, flush=True)

    def to_img(arr):
        arr = np.asarray(arr)
        if denoiser is not None:
            arr = denoiser(arr)
        arr = np.clip(arr * args.exposure, 0, None)
        arr = np.where(arr <= 0.0031308, 12.92 * arr, 1.055 * np.power(np.clip(arr, 0, 1), 1 / 2.4) - 0.055)
        return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    try:
        font = ImageFont.truetype(FONT, 22); font_s = ImageFont.truetype(FONT, 16)
    except Exception:
        font = font_s = ImageFont.load_default()
    t0 = time.time()
    for k, fr in enumerate(frames):
        objs = dict(static)
        for i, n in enumerate(names):
            if i in drawn_arm:
                continue
            V = world(i, fr)
            if i == ik:
                objs["knife_blade"] = mesh("knife_blade", V, np.asarray(faces[i])[~handle], steel)
                objs["knife_handle"] = mesh("knife_handle", V, np.asarray(faces[i])[handle], principled((0.02, 0.02, 0.022), 0.35, 0.5))
            elif n == "board":
                objs["board"] = mesh("board", V, faces[i], principled(srgb_to_linear((0.76, 0.62, 0.46)), 0.6, 0.3))
            elif kinds[i].startswith("arm"):
                objs[f"b_{n}"] = mesh(n, V, faces[i], principled(srgb_to_linear(z["body_color"][i][:3]), 0.45))
            else:
                objs[f"b_{n}"] = mesh(n, V, faces[i], principled(srgb_to_linear(z["body_color"][i][:3]), 0.5))
        for i, link, Tl0 in link_draw:
            T = M4(i, fr) @ Tl0
            for j, (Vl, Fl, col) in enumerate(visuals[link]):
                objs[f"arm_{link}_{j}"] = mesh(f"{link}_{j}", np.asarray(Vl, float) @ T[:3, :3].T + T[:3, 3], Fl,
                                               principled(srgb_to_linear(col), 0.42))
        P = np.asarray(L[fr], np.float64)
        if pieces is not None:
            # every render vertex is a fixed affine-MLS combination of nearby particles of its piece; next to a cut, of both
            # pieces until the crack front (edge - lead) has run past it, so a face opens only where the knife has cut
            Pl = P - pieces["origin"]
            Xo = pieces["So"] @ Pl + pieces["origin"]; Xu = pieces["Su"] @ Pl + pieces["origin"]
            uni = np.zeros(len(Xo), bool)
            for j, (a0, a1) in enumerate(pieces["spans"]):
                near = pieces["vplane"] == j
                if fr < a0:
                    uni |= near
                elif fr <= a1:
                    uni |= near & (Xu[:, 1] < pieces["edge"][fr] - pieces["lead"])
            X = np.where(uni[:, None], Xu, Xo)
            for i, (u, F_) in enumerate(pieces["skin"]):
                objs[f"cuc_skin{i}"] = mesh(f"cuc_skin{i}", X[u], F_, skin_bsdf, uv=pieces["uv"][u])
            for i, (u, F_, UV, j) in enumerate(pieces["caps"]):
                objs[f"cuc_cap{i}"] = mesh(f"cuc_cap{i}", X[u], F_, cap_bsdfs[j], uv=UV)
            Vs, Fs = X, []
        else:
            Vs, Fs = surface(P, spacing, args.sl, args.cube, args.iso, args.smooth)
        if len(Fs):   # mean colour of the 4 nearest particles; pure skin where at least 2 of them are skin (an even skin, while a
            # lone skin particle the knife dragged onto a cut face blends in instead of printing a dark fleck)
            _, nn = cKDTree(P).query(Vs, k=4)
            vc = pcol[nn].mean(1); vc[pskin[nn].sum(1) >= 2] = pcol_skin
            objs["cucumber"] = mesh("cucumber", Vs, Fs, cucumber_bsdf, colors=vc)
        scene = mi.load_dict(objs)
        if zfol is not None:
            sh = np.array([0.0, 0.0, zfol(fr)])
            if args.follow:
                cam = sensor(eye + sh, tgt + sh, args.fov, args.width, args.height, args.spp)
            if args.inset_follow and args.inset_eye is not None:
                cam_in = sensor(np.array(args.inset_eye) + sh, np.array(args.inset_target) + sh, args.inset_fov, *args.inset_size, args.spp)
        img = to_img(mi.render(scene, sensor=cam, spp=args.spp, seed=args.seed))
        d = ImageDraw.Draw(img)
        if cam_in is not None:
            ins = to_img(mi.render(scene, sensor=cam_in, spp=args.spp, seed=args.seed))
            IW, IH = args.inset_size
            ix = 16 if args.inset_pos[1] == "l" else args.width - IW - 16
            iy = 16 + (52 if args.label and args.inset_pos[1] == "l" else 0) if args.inset_pos[0] == "t" else args.height - IH - 16
            img.paste(ins, (ix, iy))
            d.rectangle([ix - 3, iy - 3, ix + IW + 2, iy + IH + 2], outline=(250, 250, 250), width=3)
            if args.inset_label:
                tw = d.textlength(args.inset_label, font=font_s)
                d.rectangle([ix, iy, ix + tw + 14, iy + 26], fill=(28, 32, 40)); d.text((ix + 7, iy + 3), args.inset_label, fill=(240, 240, 240), font=font_s)
        if args.label:
            tw = d.textlength(args.label, font=font)
            d.rectangle([16, 16, 16 + tw + 20, 16 + 36], fill=(28, 32, 40)); d.text((26, 20), args.label, fill=(240, 240, 240), font=font)
        img.save(osp.join(fdir, f"f{fr:05d}.png"))   # named by record, so several processes can split one run (--start/--end)
        if k % 20 == 0 or len(frames) < 5:
            print(f"  frame {k}/{len(frames)} (record {fr}, t {t_rec[fr]:.2f} s) cucumber {len(Vs)} v, {(time.time()-t0)/(k+1):.2f} s/frame", flush=True)
    if not args.no_video and len(frames) > 1:
        mp4 = osp.join(args.out, "cucumber_cut.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-pattern_type", "glob", "-i", osp.join(fdir, "f*.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", mp4], check=True)
        print("->", mp4, flush=True)


if __name__ == "__main__":
    main()
