"""Preview video of a rokae_dropper recording (pyrender, rasterised; water as particles).

    PYOPENGL_PLATFORM=egl python render_preview.py out/rokae_dropper/<run> [--cam main|close|top] [--every 1]

Robot links use the full-resolution meshes (ar5_wuji_vis.urdf) moved by the recorded ABD body motions;
props use their simulated surfaces; the dropper uses its FEM surface.
"""
import argparse
import json
import os
import os.path as osp
import subprocess
import sys

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)


def abd_world_tf(C):
    T = np.array(C, np.float64)
    T[:3, :3] = C[:3, :3].T
    return T


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, u, -f, eye
    return T


CAMS = {
    # world: X forward from the robot, Y up (table top = 0), Z = robot right
    "main": dict(eye=(1.30, 0.45, -0.55), target=(0.62, 0.08, 0.08), yfov=0.62),
    "close": dict(eye=(1.02, 0.22, -0.30), target=(0.70, 0.10, 0.08), yfov=0.50),
    "front": dict(eye=(1.45, 0.30, 0.10), target=(0.62, 0.10, 0.08), yfov=0.60),
    "wide": dict(eye=(1.50, 0.62, -0.42), target=(0.62, 0.18, 0.06), yfov=0.62),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--cam", default="main")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    ap.add_argument("--out", default=None)
    ap.add_argument("--rec", default="rec.npz")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh
    from PIL import Image, ImageDraw, ImageFont
    from robot_assets import URDFKin

    sc = np.load(osp.join(args.run, "scene.npz"), allow_pickle=True)
    rec = np.load(osp.join(args.run, args.rec), allow_pickle=True)
    rest = np.load(osp.join(args.run, "rest.npz"))
    names = [str(n) for n in sc["body_names"]]
    kind = [str(k) for k in sc["body_kind"]]
    C0inv = sc["C0inv"]
    link_T0 = sc["link_T0"]
    # the URDF files are shared and rewritten by every run with ITS left-arm calibration shift: rebuild them for this run
    # before loading, otherwise the left hand is rendered with another run's mount offset (scattered finger meshes)
    import json
    import robot_assets as RA
    cal = json.loads(str(sc["cal"])) if "cal" in sc.files and sc["cal"].shape == () else {}
    RA.build(left_shift=tuple(float(v) for v in cal.get("left_shift", (0.0, 0.0, 0.0))))
    kin = URDFKin(str(sc["urdf_vis"]))

    scene = pyrender.Scene(bg_color=(0.93, 0.94, 0.96, 1.0), ambient_light=(0.35, 0.35, 0.37))
    # table
    tab = trimesh.creation.box((1.2, 0.02, 1.2))
    tab.apply_translation((0.75, -0.0105, 0.0))
    scene.add(pyrender.Mesh.from_trimesh(tab, material=pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.16, 0.42, 0.36, 1.0), roughnessFactor=0.8)))
    nodes = []
    mat_white = pyrender.MetallicRoughnessMaterial(baseColorFactor=(0.93, 0.93, 0.94, 1.0), roughnessFactor=0.5)
    mat_black = pyrender.MetallicRoughnessMaterial(baseColorFactor=(0.10, 0.10, 0.11, 1.0), roughnessFactor=0.6)
    mat_glass = pyrender.MetallicRoughnessMaterial(baseColorFactor=(0.75, 0.85, 0.95, 0.35), roughnessFactor=0.1,
                                                   alphaMode="BLEND")
    for i, (n, k) in enumerate(zip(names, kind)):
        if k == "robot":
            if n not in kin.meshes:
                continue
            path, Tv = kin.meshes[n]
            m = trimesh.load(path, force="mesh")
            local = Tv                      # mesh -> link frame
            base = link_T0[i] @ local       # load pose of the mesh in the world
            mat = mat_black if ("finger" in n or "palm" in n) else mat_white
            node = scene.add(pyrender.Mesh.from_trimesh(m, material=mat, smooth=False))
            nodes.append((i, node, base))
        else:
            m = trimesh.Trimesh(rest[f"v_{i}"], rest[f"f_{i}"], process=False)
            node = scene.add(pyrender.Mesh.from_trimesh(m, material=mat_glass, smooth=False))
            nodes.append((i, node, np.eye(4)))
    dfaces = sc["dropper_faces"]
    dyoung = None
    label = sc["liq_label"]
    origin = sc["liq_origin"]
    col =np.array([[0.10, 0.30, 0.95, 1.0], [0.95, 0.80, 0.10, 1.0]])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=look_at((0.9, 1.5, -0.6), (0.6, 0, 0)))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.2), pose=look_at((0.3, 1.0, 0.9), (0.6, 0, 0)))
    c = CAMS[args.cam]
    cam = pyrender.PerspectiveCamera(yfov=c["yfov"], aspectRatio=args.size[0] / args.size[1])
    scene.add(cam, pose=look_at(c["eye"], c["target"]))
    r = pyrender.OffscreenRenderer(*args.size)
    t = rec["t"]
    abd = rec["abd"]
    drop = rec["drop"]
    liq = rec["liq_i16"] if "liq_i16" in rec.files else None
    pump = rec["pump"]
    frames_dir = osp.join(args.run, f"frames_{args.cam}")
    os.makedirs(frames_dir, exist_ok=True)
    dnode = lnode = None
    font = ImageFont.load_default()
    idx = list(range(0, len(t), args.every))
    for fi, j in enumerate(idx):
        for i, node, base in nodes:
            M = abd_world_tf(abd[j][i]) @ C0inv[i]
            scene.set_pose(node, M @ base)
        if dnode is not None:
            scene.remove_node(dnode)
        dm = trimesh.Trimesh(drop[j], dfaces, process=False)
        dnode = scene.add(pyrender.Mesh.from_trimesh(dm, material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.20, 0.45, 0.90, 0.85), roughnessFactor=0.4, alphaMode="BLEND"), smooth=False))
        if liq is not None:
            if lnode is not None:
                scene.remove_node(lnode)
            p = liq[j].astype(np.float64) * 1e-4 + origin
            tf = np.tile(np.eye(4), (len(p), 1, 1))
            tf[:, :3, 3] = p
            meshes = []
            for lab in (0, 1):
                sel = label[:len(p)] == lab
                if not sel.any():
                    continue
                s_ = trimesh.creation.icosphere(subdivisions=1, radius=1.3e-3)
                s_.visual.vertex_colors = (col[lab] * 255).astype(np.uint8)
                tfl = tf[sel]
                meshes.append(pyrender.Mesh.from_trimesh(s_, poses=tfl, smooth=False))
            lnode = []
            for mm in meshes:
                lnode.append(scene.add(mm))
        img, _ = r.render(scene)
        if isinstance(lnode, list):
            for nn in lnode:
                scene.remove_node(nn)
            lnode = None
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        pv = pump[j]
        d.text((12, 10), f"t = {t[j]:5.2f} s (recording time)", fill=(20, 20, 20), font=font)
        if np.isfinite(pv[0]):
            d.text((12, 26), f"bulb cavity {pv[0]*1e6:5.2f} ml   water in bore {pv[1]*1e6:5.2f} ml   tip {'in water' if pv[4] else 'in air'}",
                   fill=(20, 20, 20), font=font)
        im.save(osp.join(frames_dir, f"{fi:05d}.png"))
    r.delete()
    fps = 1.0 / max(np.median(np.diff(t)) * args.every, 1e-6) * args.speed
    out = args.out or osp.join(args.run, f"preview_{args.cam}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:.3f}", "-i", osp.join(frames_dir, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", out], check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
