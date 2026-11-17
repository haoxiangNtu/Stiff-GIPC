#!/usr/bin/env python3
"""Export the recorded lab demo (lab_demo.npz) to USD.

Rigid bodies (arm links, glassware) animate by time-sampled transforms over
static meshes; FEM bodies (gel pads, dropper) carry time-sampled points.
Stage: Y-up, metres, timeCodesPerSecond = recording fps (25).

    <pxr-python> examples/lab_tactile/export_lab_usd.py out/lab_tactile/lab_demo.npz
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from lab_rec import load_recording  # noqa: E402


def _mat(T):
    from pxr import Gf
    return Gf.Matrix4d(*np.asarray(T, np.float64).T.flatten().tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    out = args.out or args.npz.replace(".npz", ".usd")

    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    z = load_recording(args.npz)
    meta = z["meta"]
    names = list(z["body_names"])
    kinds = list(z["body_kind"])
    colors = z["body_color"]
    rest, faces = z["body_rest"], z["body_faces"]
    motion = z["motion"]
    frames = list(range(0, len(z["t"]), args.stride))
    has_fem = "fem_names" in z

    stem, ext = osp.splitext(out)          # ".usdc" must not become ".usdac" (replacing ".usd" first did that)
    usda = out if ext == ".usda" else stem + ".usda"
    stage = Usd.Stage.CreateNew(usda)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(1)
    stage.SetEndTimeCode(len(frames))
    stage.SetTimeCodesPerSecond(args.fps / args.stride)
    stage.SetFramesPerSecond(args.fps / args.stride)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    world.GetPrim().SetDocumentation(
        "StiffGIPC tactile lab-bench demo: FR3 arm with VBTS finger pads performs "
        "dropper -> test-tube pour -> stir -> shake -> pour-to-waste. Rigid bodies animate by "
        "transform over static meshes; gel pads and the rubber-bulb dropper are FEM points "
        "animation. Physics dt=%g s, recorded every %d steps." % (meta.get("dt", 0), meta.get("rec_every", 1)))
    cd = world.GetPrim().GetCustomData() or {}
    cd["youngs_modulus_estimates"] = json.dumps(meta.get("estimates", []), default=float)
    world.GetPrim().SetCustomData(cd)

    def mesh(path, verts, tris, color):
        m = UsdGeom.Mesh.Define(stage, path)
        m.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(tris)))
        m.GetFaceVertexIndicesAttr().Set(Vt.IntArray(np.asarray(tris).ravel().astype(int).tolist()))
        m.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(verts, np.float32)))
        m.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(c) for c in color])]))
        return m

    groups = {"arm": "/World/Robot", "prop": "/World/Props", "fixed": "/World/Fixtures"}
    for g in groups.values():
        UsdGeom.Xform.Define(stage, g)
    ops = []
    for i, n in enumerate(names):
        grp = groups.get(str(kinds[i]), "/World/Props")
        prim_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in n)
        xf = UsdGeom.Xform.Define(stage, f"{grp}/{prim_name}")
        op = xf.AddTransformOp()
        mesh(f"{grp}/{prim_name}/mesh", rest[i], faces[i], colors[i])
        ops.append(op)
    # ground / table
    mesh("/World/Table", np.array([[x, 0.0, zz] for x in (-0.35, 1.0) for zz in (-0.6, 0.6)], np.float32),
         np.array([[0, 1, 3], [0, 3, 2]]), (0.78, 0.76, 0.72))

    fem_meshes = []
    if has_fem:
        UsdGeom.Xform.Define(stage, "/World/Deformables")
        fem_color = {"gel_left": (0.16, 0.25, 0.56), "gel_right": (0.16, 0.25, 0.56),
                     "dropper": (0.25, 0.45, 0.85)}
        meta_cols = meta.get("fem_colors") or []
        for i, n in enumerate(z["fem_names"]):
            default = tuple(meta_cols[i]) if i < len(meta_cols) else (0.5, 0.5, 0.5)
            m = mesh(f"/World/Deformables/{n}", z["fem_verts"][i][0], z["fem_faces"][i],
                     fem_color.get(str(n), default))
            fem_meshes.append((i, m))

    liq = None
    if "liquid_pts" in z and len(z["liquid_pts"]) == len(z["t"]):
        UsdGeom.Xform.Define(stage, "/World/Liquid")
        liq = UsdGeom.Points.Define(stage, "/World/Liquid/water")
        liq.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.12, 0.45, 0.95)]))
        w = float(meta.get("liquid_dx", 0.005)) * 0.6
        n0 = len(z["liquid_pts"][0])
        liq.GetWidthsAttr().Set(Vt.FloatArray([w] * n0))
        liq.GetPrim().SetDocumentation("MPM Lite water particles (one-way coupled to the rigid/soft surfaces); "
                                       "point = particle position, width = 0.6 * grid dx")
    metap = stage.DefinePrim("/World/Telemetry")
    a_gap = metap.CreateAttribute("gripper_gap_mm", Sdf.ValueTypeNames.Float)
    a_fl = metap.CreateAttribute("pad_force_left_N", Sdf.ValueTypeNames.Float)
    a_fr = metap.CreateAttribute("pad_force_right_N", Sdf.ValueTypeNames.Float)
    a_stage = metap.CreateAttribute("stage", Sdf.ValueTypeNames.String)
    a_seg = metap.CreateAttribute("segment", Sdf.ValueTypeNames.String)
    a_est = metap.CreateAttribute("young_estimate", Sdf.ValueTypeNames.String)

    for k, fr in enumerate(frames):
        t = Usd.TimeCode(k + 1)
        for i, op in enumerate(ops):
            op.Set(_mat(motion[fr, i]), t)
        for i, m in fem_meshes:
            m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(z["fem_verts"][i][fr], np.float32)), t)
        if liq is not None:
            liq.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(z["liquid_pts"][fr], np.float32)), t)
        a_gap.Set(float(z["gap_mm"][fr]), t)
        a_fl.Set(float(z["fn_left"][fr]), t)
        a_fr.Set(float(z["fn_right"][fr]), t)
        a_stage.Set(str(z["stage"][fr]), t)
        a_seg.Set(str(z["segment"][fr]), t)
        a_est.Set(str(z["etext"][fr]), t)
    stage.GetRootLayer().Save()
    usdc = usda.replace(".usda", ".usdc")
    stage.GetRootLayer().Export(usdc)
    print(f"-> {usdc} ({os.path.getsize(usdc)/2**20:.1f} MiB, {len(frames)} frames)  (+ {usda})")


if __name__ == "__main__":
    main()
