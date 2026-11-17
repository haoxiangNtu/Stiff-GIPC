#!/usr/bin/env python3
"""Export a cucumber_cut.py run to a single .usdc, the scene render_cucumber.py --pieces draws.

- FR3: the URDF visual meshes, static in their link frames, carried by the recorded link motion (time-sampled transforms).
- Knife (blade + handle), board and any other rigid body: static meshes under time-sampled transforms.
- Cucumber: the deform-pieces meshes of prep_pieces.py (the cucumber's own mesh cut along the simulated crack planes, every
  vertex a fixed affine-MLS combination of nearby particles), time-sampled points; per-vertex displayColor baked from the
  skin / cut-face textures, st texcoords kept. The cut-face pattern (rind, flesh, seeds) is a texture: the simulated flesh is
  one homogeneous material.
- Optionally the MPM particles themselves as a Points prim, every --particle-stride records.
Stage: Y up, metres, timeCodesPerSecond = recording fps / --stride.

    <pxr-python> export_cucumber_usd.py <run dir> --pieces render_pieces.npz --out cucumber_cut.usdc [--particle-stride 5]
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
from render_cucumber import REPO, URDF, knife_handle_faces, load_visuals, srgb_to_linear  # noqa: E402
from lab_rec import load_recording  # noqa: E402
from fr3_kin import Fr3Arm  # noqa: E402


def _mat(T):
    from pxr import Gf
    return Gf.Matrix4d(*np.asarray(T, np.float64).T.flatten().tolist())


def bilinear(img, uv, wrap):
    """sample an (H, W, C) image at texcoords uv (Mitsuba convention: row 0 = v 0), wrap = repeat or clamp"""
    H, W = img.shape[:2]
    u, v = uv[:, 0], uv[:, 1]
    if wrap:
        u, v = u % 1.0, v % 1.0
    x = np.clip(u * W - 0.5, 0, W - 1); y = np.clip(v * H - 0.5, 0, H - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x1] + (1 - fx) * fy * img[y1, x0] + fx * fy * img[y1, x1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="cucumber_cut output dir (recording.npz, particles.npz)")
    ap.add_argument("--pieces", required=True, help="render npz of prep_pieces.py")
    ap.add_argument("--out", required=True, help=".usdc path")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth record for the meshes")
    ap.add_argument("--particle-stride", type=int, default=0, help="also write the MPM particles every Nth record (0 = no)")
    ap.add_argument("--visuals-cache", default=osp.join(REPO, "out", "cutting", "fr3_visuals_cache.npz"))
    ap.add_argument("--doc", default="", help="extra sentence for the /World documentation")
    args = ap.parse_args()
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    from scipy import sparse

    z = load_recording(osp.join(args.run, "recording.npz"))
    meta = z["meta"]
    pz = np.load(osp.join(args.run, "particles.npz"))
    names, kinds = [str(n) for n in z["body_names"]], [str(k) for k in z["body_kind"]]
    rest, faces, motion = z["body_rest"], z["body_faces"], np.asarray(z["motion"], float)
    t_rec = np.asarray(z["t"], float)
    L = z["liquid_pts"]
    seg = [str(s) for s in z["segment"]]
    n_rec = len(t_rec)
    frames = list(range(0, n_rec, args.stride))
    rec_fps = 1.0 / max(float(np.median(np.diff(t_rec))), 1e-9)
    fps = rec_fps / args.stride

    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(1); stage.SetEndTimeCode(len(frames))
    stage.SetTimeCodesPerSecond(fps); stage.SetFramesPerSecond(fps)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    a = meta.get("args", {})
    mat = meta.get("material", {})
    # material zones (2026-11-12: per-particle Young's modulus / damage factors); absent or all 1 = the homogeneous cucumber
    zone_keys = ("skin", "skin_young_x", "skin_damage_x", "core_young_x", "core_damage_x", "core_frac", "core_blend")
    zones = {k: a.get(k) for k in zone_keys if a.get(k) is not None}
    zoned = any(abs(float(a.get(k, 1.0) or 1.0) - 1.0) > 1e-9 for k in ("skin_young_x", "skin_damage_x", "core_young_x", "core_damage_x"))
    if zoned:
        flesh_doc = ("the simulated cucumber has three material zones: skin (within %g mm of the surface, Young's modulus x %g, break "
                     "stretch threshold x %g), seed core (inside %g of the local radius, x %g, x %g) and the flesh between them (x 1)"
                     % (float(a.get("skin", 0.0)) * 1e3, float(a.get("skin_young_x", 1.0)), float(a.get("skin_damage_x", 1.0)),
                        float(a.get("core_frac", 0.0)), float(a.get("core_young_x", 1.0)), float(a.get("core_damage_x", 1.0))))
    else:
        flesh_doc = "the simulated flesh is homogeneous"
    world.GetPrim().SetDocumentation(
        "StiffGIPC knife cuts a cucumber: an FR3 holds a kitchen knife and presses it straight down through an MPM cucumber "
        "on a cutting board, re-aiming before each cut at where the cucumber has been pushed to. The cucumber is the engine's "
        "MPM elastic solid inside the IPC Newton solve; the knife's mid-plane splits the MPM grid (two-field cutter CPIC) so "
        "the halves separate along it. Rigid bodies animate by transform over static meshes. The cucumber mesh is the "
        "original model cut along the simulated crack planes and moved with the particles (affine MLS); its cut-face pattern "
        "is a texture, " + flesh_doc + ". " + args.doc)
    cd = world.GetPrim().GetCustomData() or {}
    cd["simulation"] = json.dumps({k: meta.get(k) for k in ("dt", "particle_spacing", "board_top") if k in meta} |
                                  {"young": mat.get("young"), "poisson": mat.get("poisson"), "rho": a.get("rho"),
                                   "cuts": a.get("cuts"), "speed": a.get("speed")} | ({"zones": zones} if zoned else {}), default=float)
    world.GetPrim().SetCustomData(cd)

    def mesh(path, V, F, color=None, colors=None, st=None):
        m = UsdGeom.Mesh.Define(stage, path)
        F = np.asarray(F)
        m.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
        m.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(F.ravel().astype(np.int32)))
        m.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(V, np.float32)))
        pv = UsdGeom.PrimvarsAPI(m)
        if colors is not None:
            p = pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
            p.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(colors, np.float32)))
        elif color is not None:
            m.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(c) for c in color[:3]])]))
        if st is not None:
            p = pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            p.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(st, np.float32)))
        return m

    def xform(path):
        x = UsdGeom.Xform.Define(stage, path)
        return x.AddTransformOp()

    # FR3 visuals: static in the link frame, carried by the recorded link-body motion from the load pose (as the renderer)
    visuals = load_visuals(args.visuals_cache)
    arms = meta.get("arms") or []
    q_arms = np.asarray(z["q_arms"], float) if "q_arms" in z else None
    gap_arms = np.asarray(z["gap_arms"], float) if "gap_arms" in z else None
    anim = []                                       # (transform op, body index, T_link at record 0 in the body frame)
    drawn = set()
    UsdGeom.Xform.Define(stage, "/World/Robot")
    if arms and q_arms is not None and q_arms.shape[1] > 0:
        kin = Fr3Arm(URDF, np.asarray(arms[0]["base_T"], float))
        g0 = float(np.clip(gap_arms[0, 0] / 2000.0, 0.0, 0.04)) if gap_arms is not None and gap_arms.shape[1] else 0.0
        poses0 = kin.link_poses(q_arms[0, 0], g0)
        for i, n in enumerate(names):
            for link in visuals:
                if (n == link or n.endswith("_" + link) or n.endswith("/" + link)) and link in poses0:
                    M0 = np.eye(4); M0[:3, :3] = motion[0, i][:3, :3]; M0[:3, 3] = motion[0, i][:3, 3]
                    op = xform(f"/World/Robot/{link}")
                    for j, (Vl, Fl, col) in enumerate(visuals[link]):
                        mesh(f"/World/Robot/{link}/visual_{j}", Vl, Fl, color=srgb_to_linear(col))
                    anim.append((op, i, np.linalg.inv(M0) @ poses0[link]))
                    drawn.add(i)
                    break
    UsdGeom.Xform.Define(stage, "/World/Props")
    for i, n in enumerate(names):
        if i in drawn:
            continue
        pn = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in n)
        op = xform(f"/World/Props/{pn}")
        V, F = np.asarray(rest[i], float), np.asarray(faces[i])
        if n == "knife":
            h = knife_handle_faces(V, F)
            mesh(f"/World/Props/{pn}/blade", V, F[~h], color=(0.55, 0.56, 0.58))
            mesh(f"/World/Props/{pn}/handle", V, F[h], color=(0.02, 0.02, 0.022))
        elif n == "board":
            mesh(f"/World/Props/{pn}/mesh", V, F, color=srgb_to_linear((0.76, 0.62, 0.46)))
        else:
            mesh(f"/World/Props/{pn}/mesh", V, F, color=srgb_to_linear(z["body_color"][i][:3]))
        anim.append((op, i, np.eye(4)))
    mesh("/World/Table", np.array([[-0.8, 0, -1.2], [1.6, 0, -1.2], [1.6, 0, 1.2], [-0.8, 0, 1.2]], np.float32),
         np.array([[0, 2, 1], [0, 3, 2]]), color=srgb_to_linear((0.80, 0.79, 0.76)))

    # cucumber: deform pieces
    pc = np.load(args.pieces)
    nV, NP = len(pc["V"]), int(pc["N"])
    if NP != len(pz["skin"]):
        raise SystemExit(f"--pieces was built for {NP} particles, this run has {len(pz['skin'])}")
    seg_p = [str(x) for x in pc["segment"]]
    spans = []
    for j in range(int(pc["n_plane"])):
        idx = [i for i, s in enumerate(seg_p) if s == f"cut {j + 1}"]
        spans.append((min(idx), max(idx)) if idx else (10 ** 9, 10 ** 9))
    So = sparse.csr_matrix((pc["So_data"], pc["So_indices"], pc["So_indptr"]), shape=(nV, NP))
    Su = sparse.csr_matrix((pc["Su_data"], pc["Su_indices"], pc["Su_indptr"]), shape=(nV, NP))
    origin, vplane = np.asarray(pc["origin"], float), pc["vplane"].astype(int)
    edge, lead = np.asarray(pc["edge"], float), float(pc["lead"])
    skin_tex = pc["skin_tex"].astype(np.float32) / 65535.0
    UsdGeom.Xform.Define(stage, "/World/Cucumber")
    parts = []                                      # (mesh, vertex ids)
    for i in range(int(pc["n_skin"])):
        F_ = pc[f"skin{i}_F"].astype(np.int64); u = np.unique(F_); rm = np.full(nV, -1, np.int64); rm[u] = np.arange(len(u))
        uv = pc["skin_uv"][u]
        m = mesh(f"/World/Cucumber/skin_{i}", pc["V"][u], rm[F_], colors=bilinear(skin_tex, uv, True), st=uv)
        parts.append((m, u))
    for i in range(int(pc["n_cap"])):
        UV = pc[f"cap{i}_UV"]; v0 = int(pc[f"cap{i}_v0"]); j = int(pc[f"cap{i}_plane"])
        u = np.arange(v0, v0 + len(UV))
        tex = pc[f"tex{j}"].astype(np.float32) / 65535.0
        m = mesh(f"/World/Cucumber/cut_face_{i}", pc["V"][u], pc[f"cap{i}_F"].astype(np.int64) - v0, colors=bilinear(tex, UV, False), st=UV)
        parts.append((m, u))
    pts = None
    if args.particle_stride > 0:
        pts = UsdGeom.Points.Define(stage, "/World/Cucumber/particles")
        sp = float(meta.get("particle_spacing", 0.001))
        pts.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(np.full(NP, sp, np.float32)))
        col = np.where(pz["skin"].astype(bool)[:, None], srgb_to_linear([[0.10, 0.24, 0.07]]),
                       srgb_to_linear([[0.80, 0.86, 0.60]])).astype(np.float32)
        p = UsdGeom.PrimvarsAPI(pts).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        p.Set(Vt.Vec3fArray.FromNumpy(col))
        pts.GetPrim().SetDocumentation(f"MPM particles (spacing {sp * 1e3:g} mm), every {args.particle_stride} records of the "
                                       f"meshes' time line; hidden by default")
        UsdGeom.Imageable(pts).MakeInvisible()

    tel = stage.DefinePrim("/World/Telemetry")
    a_seg = tel.CreateAttribute("segment", Sdf.ValueTypeNames.String)
    a_f = tel.CreateAttribute("knife_force_N", Sdf.ValueTypeNames.Float)
    Fk = None
    if osp.exists(osp.join(args.run, "knife_force.npz")):
        Fk = np.linalg.norm(np.load(osp.join(args.run, "knife_force.npz"))["F"], axis=1)

    for k, fr in enumerate(frames):
        t = Usd.TimeCode(k + 1)
        for op, i, Tl in anim:
            M = np.eye(4); M[:3, :3] = motion[fr, i][:3, :3]; M[:3, 3] = motion[fr, i][:3, 3]
            op.Set(_mat(M @ Tl), t)
        P = np.asarray(L[fr], np.float64)
        Pl = P - origin
        Xo = So @ Pl + origin; Xu = Su @ Pl + origin
        uni = np.zeros(nV, bool)
        for j, (a0, a1) in enumerate(spans):
            near = vplane == j
            if fr < a0:
                uni |= near
            elif fr <= a1:
                uni |= near & (Xu[:, 1] < edge[fr] - lead)
        X = np.where(uni[:, None], Xu, Xo).astype(np.float32)
        for m, u in parts:
            m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(X[u])), t)
        if pts is not None and fr % args.particle_stride == 0:
            pts.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(P, np.float32)), t)
        a_seg.Set(seg[fr], t)
        if Fk is not None and fr < len(Fk):
            a_f.Set(float(Fk[fr]), t)
        if k % 50 == 0:
            print(f"  frame {k}/{len(frames)}", flush=True)
    stage.GetRootLayer().Save()
    print(f"-> {args.out} ({os.path.getsize(args.out) / 2 ** 20:.1f} MiB, {len(frames)} frames at {fps:g} fps)", flush=True)


if __name__ == "__main__":
    main()
