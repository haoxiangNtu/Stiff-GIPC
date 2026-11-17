#!/usr/bin/env python3
"""Export a Recorder .npz (ABD bodies + FEM cloth + liquid particles) to a time-sampled USD stage.

    PYTHONPATH=. python examples/unified/export_usd.py out/unified/case13/recording.npz --out case13.usdc

Layout: /World/Rigid/<body> (UsdGeom.Mesh with animated xformOp:transform), /World/Deformables/<cloth> (Mesh with
time-sampled points), /World/Liquid/particles (UsdGeom.Points with time-sampled points and constant widths).
"""
from __future__ import annotations
import argparse, os.path as osp, sys
import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, "..", "lab_tactile"))
from lab_rec import load_recording  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--liquid-width", type=float, default=0.0, help="point width (m); 0 = particle spacing")
    args = ap.parse_args()
    out = args.out or args.npz.replace(".npz", ".usdc")
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    z = load_recording(args.npz)
    meta = z["meta"]
    fps = float(meta.get("fps", 50.0)) / args.stride
    frames = list(range(0, len(z["t"]), args.stride))
    if args.max:
        frames = frames[: args.max]
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(0); stage.SetEndTimeCode(len(frames) - 1); stage.SetTimeCodesPerSecond(fps); stage.SetFramesPerSecond(fps)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    def mesh(path, verts, faces, color):
        m = UsdGeom.Mesh.Define(stage, path)
        m.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(verts, np.float32)))
        m.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        m.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.asarray(faces, np.int32).reshape(-1)))
        m.GetDoubleSidedAttr().Set(True)
        m.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(c) for c in color[:3]])]))
        return m

    names = list(z["body_names"]); colors = z["body_color"]
    UsdGeom.Xform.Define(stage, "/World/Rigid")
    xforms = []
    for i, n in enumerate(names):
        xf = UsdGeom.Xform.Define(stage, f"/World/Rigid/{n}")
        op = xf.AddTransformOp()
        mesh(f"/World/Rigid/{n}/mesh", z["body_rest"][i], z["body_faces"][i], colors[i])
        xforms.append(op)
    fem_names = list(z["fem_names"]) if "fem_names" in z else []
    fem_colors = meta.get("fem_colors", [[0.15, 0.45, 0.85]] * len(fem_names))
    fem_meshes = []
    if fem_names:
        UsdGeom.Xform.Define(stage, "/World/Deformables")
        for i, n in enumerate(fem_names):
            fem_meshes.append(mesh(f"/World/Deformables/{n}", z["fem_verts"][i][0], z["fem_faces"][i], fem_colors[i]))
    has_liquid = "liquid_pts" in z and len(z["liquid_pts"]) == len(z["t"])
    if has_liquid:
        UsdGeom.Xform.Define(stage, "/World/Liquid")
        liq = UsdGeom.Points.Define(stage, "/World/Liquid/particles")
        w = args.liquid_width or float(meta.get("particle_spacing", 0.01))
        n0 = len(z["liquid_pts"][0])
        liq.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(np.full(n0, w, np.float32)))
        liq.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.10, 0.42, 0.95)]))
    for t, fr in enumerate(frames):
        for i, op in enumerate(xforms):
            T = np.asarray(z["motion"][fr, i], np.float64)
            op.Set(Gf.Matrix4d(*T.T.reshape(-1).tolist()), t)      # USD matrices are row-vector (transpose)
        for i, m in enumerate(fem_meshes):
            m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(z["fem_verts"][i][fr], np.float32)), t)
        if has_liquid:
            liq.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(z["liquid_pts"][fr], np.float32)), t)
    stage.GetRootLayer().Save()
    print(f"[done] {len(frames)} frames, {len(names)} rigid, {len(fem_names)} deformable, liquid={has_liquid} -> {out}")


if __name__ == "__main__":
    main()
