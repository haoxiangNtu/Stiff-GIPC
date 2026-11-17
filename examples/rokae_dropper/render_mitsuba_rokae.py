"""Path-traced video of a rokae_dropper recording (Mitsuba 3, CUDA).

    LD_PRELOAD=<env>/lib/libstdc++.so.6 python render_mitsuba_rokae.py out/rokae_dropper/<run> --out <dir> [--spp 128]

* robot: full-resolution AR5 / Wuji meshes (ar5_wuji_vis.urdf) moved by the recorded ABD body motions;
* flask, beaker: the reference glass meshes as a dielectric (IOR 1.5);
* pipette: the reference glass stem moved rigidly with the simulated stem (Kabsch fit of the FEM stem vertices) and
  the reference silicone bulb deformed with the simulated FEM bulb (each visual vertex follows its rest-pose
  nearest FEM tetrahedron, barycentric);
* water: one surface per colour (blue = from the flask, yellow = from the beaker), reconstructed from the recorded
  particles with the level-set method of the lab demo (aniso_surface.sdf_mesh, CAUSTICA env var points at it),
  rendered as tinted water (IOR 1.333).
"""
import argparse
import json
import math
import os
import os.path as osp
import subprocess
import sys
import time

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("CAUSTICA", osp.join(HERE, "caustica")))
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def abd_world_tf(C):
    T = np.array(C, np.float64)
    T[:3, :3] = C[:3, :3].T
    return T


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def vertex_normals(V, F):
    n = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for j in range(3):
        np.add.at(n, F[:, j], fn)
    l = np.linalg.norm(n, axis=1)
    l[l == 0] = 1
    return n / l[:, None]


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def tet_embedding(Xq, verts, tets):
    """For query points Xq: index of the containing (or nearest-centroid) rest tet and barycentric weights."""
    from scipy.spatial import cKDTree
    cen = verts[tets].mean(1)
    tree = cKDTree(cen)
    _, cand = tree.query(Xq, k=12)
    best = np.zeros(len(Xq), int)
    W = np.zeros((len(Xq), 4))
    score = np.full(len(Xq), -np.inf)
    for c in range(cand.shape[1]):
        t = tets[cand[:, c]]
        P = verts[t]
        T = np.stack([P[:, 0] - P[:, 3], P[:, 1] - P[:, 3], P[:, 2] - P[:, 3]], -1)
        lam = np.linalg.solve(T, (Xq - P[:, 3])[..., None])[..., 0]
        w = np.c_[lam, 1 - lam.sum(1)]
        s = w.min(1)
        better = s > score
        score[better], best[better], W[better] = s[better], cand[better, c], w[better]
    return best, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--rec", default="rec.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--eye", type=float, nargs=3, default=(1.18, 0.30, -0.36))
    ap.add_argument("--target", type=float, nargs=3, default=(0.66, 0.10, 0.09))
    ap.add_argument("--fov", type=float, default=0.62)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--no-water", action="store_true")
    ap.add_argument("--aniso", default="vox=0.0008,close=3,sigma=3.0,lam=0.3,r_p=0.6,r_sparse=1.2,iso=0.0")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb")
    import trimesh
    from PIL import Image, ImageDraw, ImageFont
    from robot_assets import URDFKin
    import rokae_dropper as RD

    os.makedirs(osp.join(args.out, "frames"), exist_ok=True)
    sc = np.load(osp.join(args.run, "scene.npz"), allow_pickle=True)
    rec = np.load(osp.join(args.run, args.rec), allow_pickle=True)
    rest = np.load(osp.join(args.run, "rest.npz"))
    names = [str(n) for n in sc["body_names"]]
    kinds = [str(k) for k in sc["body_kind"]]
    C0inv, LT0 = sc["C0inv"], sc["link_T0"]
    prop_T = {k: np.array(v) for k, v in json.loads(str(sc["prop_T"])).items()}
    kin = URDFKin(str(sc["urdf_vis"]))
    t = rec["t"]
    abd, drop = rec["abd"], rec["drop"]
    liq = rec["liq_i16"] if ("liq_i16" in rec.files and not args.no_water) else None
    label = sc["liq_label"]
    origin = sc["liq_origin"]
    pump = rec["pump"]

    # ---- static meshes
    vis = {}
    for i, n in enumerate(names):
        if kinds[i] == "robot" and n in kin.meshes:
            path, Tv = kin.meshes[n]
            m = trimesh.load(path, force="mesh")
            vis[i] = (np.asarray(m.vertices) @ Tv[:3, :3].T + Tv[:3, 3], np.asarray(m.faces), LT0[i])
    ref = {k: RD.load_ref_mesh(k) for k in ("flask", "beaker", "pipette_stem", "pipette_bulb")}
    st, bu = RD.build_pipette()
    T_d = prop_T["dropper"]
    bulb_rest_fem = bu["verts"] + T_d[:3, 3]
    iS = names.index("stem")
    # the reference stem mesh in the stem body's rest pose; the reference bulb embedded in the FEM bulb's rest tets
    stem_vis_rest = ref["pipette_stem"][0] + prop_T["stem"][:3, 3]
    bulb_vis_rest = ref["pipette_bulb"][0] + T_d[:3, 3]
    t0 = time.time()
    emb_t, emb_w = tet_embedding(bulb_vis_rest, bulb_rest_fem, bu["tets"])
    print(f"bulb embedding {len(bulb_vis_rest)} vertices in {time.time()-t0:.1f}s", flush=True)
    bulb_N_faces = ref["pipette_bulb"][1]

    def bsdf_glass(ior=1.5, tint=(1, 1, 1)):
        return {"type": "dielectric", "int_ior": ior, "ext_ior": 1.000277,
                "specular_transmittance": {"type": "rgb", "value": [float(x) for x in tint]}}

    def bsdf_plastic(rgb, rough=0.4, spec=0.4, metal=0.0):
        return {"type": "twosided", "material": {"type": "principled", "base_color": {"type": "rgb", "value": [float(x) for x in rgb]},
                                                 "roughness": rough, "specular": spec, "metallic": metal}}

    def mesh(name, V, F, bsdf):
        V = np.ascontiguousarray(V, np.float32)
        F = np.ascontiguousarray(F, np.uint32)
        props = mi.Properties()
        props["bsdf"] = mi.load_dict(bsdf)
        m = mi.Mesh(name, vertex_count=len(V), face_count=len(F), has_vertex_normals=True, has_vertex_texcoords=False, props=props)
        p = mi.traverse(m)
        p["vertex_positions"] = V.ravel()
        p["faces"] = F.ravel()
        p["vertex_normals"] = np.ascontiguousarray(vertex_normals(V.astype(np.float64), F.astype(np.int64)), np.float32).ravel()
        p.update()
        return m

    eye, tgt = np.array(args.eye), np.array(args.target)
    sensor = {"type": "perspective", "fov": math.degrees(args.fov), "fov_axis": "y", "near_clip": 0.01, "far_clip": 50.0,
              "to_world": mi.ScalarTransform4f().look_at(origin=eye.tolist(), target=tgt.tolist(), up=[0, 1, 0]),
              "film": {"type": "hdrfilm", "width": args.width, "height": args.height, "pixel_format": "rgb", "rfilter": {"type": "box"}},
              "sampler": {"type": "independent", "sample_count": args.spp}}
    table_rgb = srgb_to_linear((0.16, 0.40, 0.34))
    static = {"type": "scene", "integrator": {"type": "path", "max_depth": 16}, "sensor": sensor,
              "sky": {"type": "constant", "radiance": {"type": "rgb", "value": [0.60, 0.62, 0.66]}},
              "lamp": {"type": "rectangle", "to_world": mi.ScalarTransform4f().look_at(origin=[1.0, 2.0, -0.4], target=[0.6, 0.1, 0.05], up=[0, 0, 1]).scale([0.8, 0.8, 1.0]),
                       "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [8.5, 8.3, 8.0]}}},
              "table": mesh("table", np.array([[0.2, 0, -0.6], [1.4, 0, -0.6], [1.4, 0, 0.7], [0.2, 0, 0.7]]), np.array([[0, 2, 1], [0, 3, 2]]),
                            {"type": "principled", "base_color": {"type": "rgb", "value": table_rgb.tolist()}, "roughness": 0.45, "specular": 0.3})}
    # denoiser (OIDN on the CPU)
    try:
        import oidn
        dev = oidn.NewDevice()
        oidn.CommitDevice(dev)

        def denoise(img):
            a = np.ascontiguousarray(np.asarray(img), dtype=np.float32)
            o = np.zeros_like(a)
            f = oidn.NewFilter(dev, "RT")
            oidn.SetSharedFilterImage(f, "color", a, oidn.FORMAT_FLOAT3, a.shape[1], a.shape[0])
            oidn.SetSharedFilterImage(f, "output", o, oidn.FORMAT_FLOAT3, a.shape[1], a.shape[0])
            try:
                oidn.RawFunctions.oidnSetFilterBool(f, b"hdr", True)
            except Exception:
                pass
            oidn.CommitFilter(f)
            oidn.ExecuteFilter(f)
            oidn.ReleaseFilter(f)
            return o
    except Exception as e:
        print("no denoiser:", e)
        denoise = None
    from aniso_surface import sdf_mesh
    akw = {k: float(v) for k, v in (kv.split("=") for kv in args.aniso.split(",") if kv)}
    font = ImageFont.truetype(FONT, 22)
    font_s = ImageFont.truetype(FONT, 17)
    water_tint = {0: (0.30, 0.55, 1.00), 1: (1.00, 0.88, 0.30)}
    phases = [(0, 12.0, "伸向滴管"), (12.0, 16.5, "捏住胶头"), (16.5, 21.0, "松开吸液 · 提起"), (21.0, 23.5, "移到烧杯上方"),
              (23.5, 27.5, "挤压滴液"), (27.5, 32.0, "放回烧瓶"), (32.0, 35.5, "松手"), (35.5, 40.0, "左手抓烧杯"),
              (40.0, 53.0, "提起摇晃"), (53.0, 58.0, "放回桌面"), (58.0, 99.0, "复位")]
    frames = [j for j in range(0, len(t), args.stride) if args.start <= t[j] <= args.end]
    wall = time.time()
    for k, j in enumerate(frames):
        objs = dict(static)
        for i, n in enumerate(names):
            M = abd_world_tf(abd[j][i]) @ C0inv[i]
            if kinds[i] == "robot":
                if i not in vis:
                    continue
                V, F, T0 = vis[i]
                T = M @ T0
                black = ("finger" in n) or ("palm" in n)
                b = bsdf_plastic(srgb_to_linear((0.05, 0.05, 0.055)), 0.55) if black else bsdf_plastic(srgb_to_linear((0.92, 0.92, 0.93)), 0.35)
                objs[f"r_{i}"] = mesh(n, V @ T[:3, :3].T + T[:3, 3], F, b)
            elif n == "stem":
                continue
            else:
                V, F = ref[n]
                T = M @ prop_T[n]
                objs[f"p_{n}"] = mesh(n, V @ T[:3, :3].T + T[:3, 3], F, bsdf_glass(1.5))
        Ms = abd_world_tf(abd[j][iS]) @ C0inv[iS]
        objs["stem"] = mesh("stem", stem_vis_rest @ Ms[:3, :3].T + Ms[:3, 3], ref["pipette_stem"][1], bsdf_glass(1.47, (0.95, 0.97, 1.0)))
        Vd = drop[j].astype(np.float64)
        Vb = np.einsum("nk,nkd->nd", emb_w, Vd[bu["tets"][emb_t]])
        objs["bulb"] = mesh("bulb", Vb, bulb_N_faces, bsdf_plastic(srgb_to_linear((0.10, 0.30, 0.85)), 0.45, 0.35))
        n_in_beaker = None
        if liq is not None:
            P = liq[j].astype(np.float64) * 1e-4 + origin
            iB = names.index("beaker")
            TB = abd_world_tf(abd[j][iB]) @ C0inv[iB] @ prop_T["beaker"]
            q = (P - TB[:3, 3]) @ TB[:3, :3]
            inb = (np.hypot(q[:, 0], q[:, 2]) < 0.032) & (q[:, 1] > -0.002) & (q[:, 1] < 0.075)
            n_in_beaker = int((inb & (label[:len(P)] == 0)).sum())
            for lab in (0, 1):
                sel = label[:len(P)] == lab
                if sel.sum() < 4:
                    continue
                Vw, Fw = sdf_mesh(P[sel], **akw)
                if len(Fw):
                    objs[f"water{lab}"] = mesh(f"water{lab}", Vw, Fw, bsdf_glass(1.333, water_tint[lab]))
        scene = mi.load_dict(objs)
        img = np.asarray(mi.render(scene, spp=args.spp, seed=k))
        if denoise is not None:
            img = denoise(img)
        img = np.clip(img, 0, None)
        img = np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(img, 1 / 2.4) - 0.055)
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        if not args.no_hud:
            dr = ImageDraw.Draw(im)
            ph = next((p for a, b, p in phases if a <= t[j] < b), "")
            dr.rectangle((0, 0, args.width, 44), fill=(20, 22, 26))
            dr.text((16, 9), f"t = {t[j]:5.2f} s   {ph}", fill=(240, 240, 240), font=font)
            pv = pump[j]
            if np.isfinite(pv[0]):
                txt = f"胶头腔体 {pv[0]*1e6:5.2f} ml   管内水 {pv[1]*1e6:4.2f} ml"
                if n_in_beaker is not None:
                    txt += f"   烧杯中蓝水 {n_in_beaker} 粒子 ({n_in_beaker * 0.0156:.2f} ml)"
                dr.text((args.width - 760, 12), txt, fill=(200, 210, 230), font=font_s)
        im.save(osp.join(args.out, "frames", f"{k:05d}.png"))
        if k % 10 == 0:
            el = time.time() - wall
            print(f"frame {k+1}/{len(frames)} t {t[j]:.2f}  {el/(k+1):.1f} s/frame", flush=True)
    if not args.no_video:
        out = osp.join(args.out, "rokae_dropper_pathtraced.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{args.fps / args.stride:.3f}", "-i",
                        osp.join(args.out, "frames", "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                        "-movflags", "+faststart", out], check=True)
        print("wrote", out)


if __name__ == "__main__":
    main()
