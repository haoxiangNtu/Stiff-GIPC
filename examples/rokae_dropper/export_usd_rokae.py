"""Export a rokae_dropper recording to USD (Y-up, metres, 25 time codes per second).

    <python with pxr> export_usd_rokae.py out/rokae_dropper/<run> [--rec rec.npz] [--out file.usdc] [--stride 1]

/World/Robot/<link>   AR5 + Wuji visual meshes, time-sampled transforms
/World/Props/flask, beaker, pipette_stem   reference glass meshes, time-sampled transforms
/World/Props/pipette_bulb                  silicone bulb, time-sampled points (simulated FEM surface; --bulb ref = reference mesh)
/World/Water/blue, yellow                  MPM particles (UsdGeomPoints, width 3 mm), time-sampled positions
"""
import argparse
import json
import os.path as osp
import sys

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)


def abd_world_tf(C):
    T = np.array(C, np.float64)
    T[:3, :3] = C[:3, :3].T
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--rec", default="rec.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--bulb", choices=("fem", "ref"), default="fem",
                    help="fem: the simulated bulb surface (small file); ref: the 57k-vertex reference bulb embedded in it")
    a = ap.parse_args()
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    import trimesh
    from robot_assets import URDFKin
    import rokae_dropper as RD
    from render_mitsuba_rokae import tet_embedding

    sc = np.load(osp.join(a.run, "scene.npz"), allow_pickle=True)
    rec = np.load(osp.join(a.run, a.rec), allow_pickle=True)
    names = [str(n) for n in sc["body_names"]]
    kinds = [str(k) for k in sc["body_kind"]]
    C0inv, LT0 = sc["C0inv"], sc["link_T0"]
    prop_T = {k: np.array(v) for k, v in json.loads(str(sc["prop_T"])).items()}
    kin = URDFKin(str(sc["urdf_vis"]))
    t = rec["t"]
    frames = list(range(0, len(t), a.stride))
    fps = 1.0 / (np.median(np.diff(t)) * a.stride)
    out = a.out or osp.join(a.run, "rokae_dropper.usdc")
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(frames) - 1)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    def mat(T):
        return Gf.Matrix4d(*np.asarray(T, np.float64).T.flatten().tolist())

    def mesh(path, V, F, rgb):
        m = UsdGeom.Mesh.Define(stage, path)
        m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(V, np.float32)))
        m.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(F)))
        m.GetFaceVertexIndicesAttr().Set(Vt.IntArray(np.asarray(F, np.int32).ravel().tolist()))
        m.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*rgb)]))
        m.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        return m

    def xform(path):
        x = UsdGeom.Xform.Define(stage, path)
        return x.AddTransformOp()

    ops = []
    for i, n in enumerate(names):
        if kinds[i] == "robot":
            if n not in kin.meshes:
                continue
            path, Tv = kin.meshes[n]
            m = trimesh.load(path, force="mesh")
            p = f"/World/Robot/{n.replace('-', '_')}"
            op = xform(p)
            black = ("finger" in n) or ("palm" in n)
            mesh(p + "/mesh", np.asarray(m.vertices) @ Tv[:3, :3].T + Tv[:3, 3], m.faces,
                 (0.05, 0.05, 0.055) if black else (0.92, 0.92, 0.93))
            ops.append((i, op, LT0[i]))
        elif n in ("flask", "beaker", "stem"):
            V, F = RD.load_ref_mesh("pipette_stem" if n == "stem" else n)
            p = f"/World/Props/{'pipette_stem' if n == 'stem' else n}"
            op = xform(p)
            mesh(p + "/mesh", V, F, (0.85, 0.9, 0.95))
            ops.append((i, op, prop_T[n]))
    st, bu = RD.build_pipette()
    T_d = prop_T["dropper"]
    Vb_ref, Fb_ref = RD.load_ref_mesh("pipette_bulb")
    if a.bulb == "ref":
        emb_t, emb_w = tet_embedding(Vb_ref + T_d[:3, 3], bu["verts"] + T_d[:3, 3], bu["tets"])
        bulb = mesh("/World/Props/pipette_bulb", Vb_ref + T_d[:3, 3], Fb_ref, (0.10, 0.30, 0.85))
    else:
        Fd = np.asarray(sc["dropper_faces"])
        used = np.unique(Fd)
        remap = np.full(int(Fd.max()) + 1, -1)
        remap[used] = np.arange(len(used))
        bulb = mesh("/World/Props/pipette_bulb", rec["drop"][0][used], remap[Fd], (0.10, 0.30, 0.85))
    water = {}
    if "liq_i16" in rec.files:
        lab = sc["liq_label"]
        for k, (nm, rgb) in enumerate((("blue", (0.25, 0.5, 1.0)), ("yellow", (1.0, 0.85, 0.25)))):
            pts = UsdGeom.Points.Define(stage, f"/World/Water/{nm}")
            sel = lab == k
            pts.GetWidthsAttr().Set(Vt.FloatArray([0.003] * int(sel.sum())))
            pts.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*rgb)]))
            pts.GetPrim().SetDocumentation("StiffGIPC MPM liquid particles (2.5 mm spacing), two-way coupled in the IPC solve")
            water[k] = (pts, sel)
    origin = sc["liq_origin"]
    for fi, j in enumerate(frames):
        for i, op, T0 in ops:
            op.Set(mat(abd_world_tf(rec["abd"][j][i]) @ C0inv[i] @ T0), fi)
        Vd = rec["drop"][j].astype(np.float64)
        Vb = np.einsum("nk,nkd->nd", emb_w, Vd[bu["tets"][emb_t]]) if a.bulb == "ref" else Vd[used]
        bulb.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(Vb.astype(np.float32)), fi)
        if water:
            P = (rec["liq_i16"][j].astype(np.float64) * 1e-4 + origin).astype(np.float32)
            for k, (pts, sel) in water.items():
                pts.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P[sel]), fi)
        if fi % 100 == 0:
            print(f"frame {fi}/{len(frames)}", flush=True)
    stage.GetRootLayer().Save()
    print("wrote", out, f"{len(frames)} frames @ {fps:.0f} fps")


if __name__ == "__main__":
    main()
