"""Preview video of a rokae_dropper recording (pyrender, rasterised; water as particles) - render_preview.py with a time
window, a free or prop-relative camera, a readable time stamp and its own output folder, so several views of one run can be
rendered side by side (2026-11-12, for the "before / after the tracking fix" comparison).

    PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=<n> python render_preview2.py out/rokae_dropper/<run> --tag hover \
        --cam beaker --t0 21 --t1 33 [--every 1] [--out file.mp4]

--cam main|close|front|wide (fixed, as render_preview.py) | beaker | neck (the viewer's views: relative to the prop's position)
--eye/--target/--yfov override the camera. Frames go to <run>/frames2_<tag>/, the video to <run>/preview2_<tag>.mp4.
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


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    T = np.eye(4)
    T[:3, 0] = s; T[:3, 1] = u; T[:3, 2] = -f; T[:3, 3] = eye
    return T


CAMS = {
    # world: X forward from the robot, Y up (table top = 0), Z = robot right
    "main": dict(eye=(1.30, 0.45, -0.55), target=(0.62, 0.08, 0.08), yfov=0.62),
    "close": dict(eye=(1.02, 0.22, -0.30), target=(0.70, 0.10, 0.08), yfov=0.50),
    "front": dict(eye=(1.45, 0.30, 0.10), target=(0.62, 0.10, 0.08), yfov=0.60),
    "wide": dict(eye=(1.50, 0.62, -0.42), target=(0.62, 0.18, 0.06), yfov=0.62),
}
REL = {   # the interactive viewer's two close views, relative to a prop: (prop, eye offset (x, absolute y, z), target y)
    "beaker": ("beaker", (0.30, 0.34, -0.24), 0.10),
    "neck": ("flask", (0.33, 0.36, 0.28), 0.16),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--cam", default="main")
    ap.add_argument("--eye", type=float, nargs=3, default=None)
    ap.add_argument("--target", type=float, nargs=3, default=None)
    ap.add_argument("--yfov", type=float, default=None)
    ap.add_argument("--zoom", type=float, default=1.0, help="move the eye towards the target by this factor (> 1 = closer)")
    ap.add_argument("--t0", type=float, default=-1e9)
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rec", default="rec.npz")
    ap.add_argument("--stem-rgba", type=float, nargs=4, default=None,
                    help="colour of the dropper's glass stem (default: the faint glass of the other props); e.g. 0.25 0.30 0.40 0.8")
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
    import robot_assets as RA
    cal = json.loads(str(sc["cal"])) if "cal" in sc.files and sc["cal"].shape == () else {}
    RA.build(left_shift=tuple(float(v) for v in cal.get("left_shift", (0.0, 0.0, 0.0))))
    kin = URDFKin(str(sc["urdf_vis"]))
    t = rec["t"]
    abd = rec["abd"]
    drop = rec["drop"]
    liq = rec["liq_i16"] if "liq_i16" in rec.files else None
    idx = [j for j in range(0, len(t), args.every) if args.t0 - 1e-9 <= t[j] <= args.t1 + 1e-9]

    scene = pyrender.Scene(bg_color=(0.93, 0.94, 0.96, 1.0), ambient_light=(0.35, 0.35, 0.37))
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
            base = link_T0[i] @ Tv
            mat = mat_black if ("finger" in n or "palm" in n) else mat_white
            node = scene.add(pyrender.Mesh.from_trimesh(m, material=mat, smooth=False))
            nodes.append((i, node, base))
        else:
            m = trimesh.Trimesh(rest[f"v_{i}"], rest[f"f_{i}"], process=False)
            mat = mat_glass
            if n == "stem" and args.stem_rgba is not None:
                mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=tuple(args.stem_rgba), roughnessFactor=0.3, alphaMode="BLEND")
            node = scene.add(pyrender.Mesh.from_trimesh(m, material=mat, smooth=False))
            nodes.append((i, node, np.eye(4)))
    dfaces = sc["dropper_faces"]
    label = sc["liq_label"]
    origin = sc["liq_origin"]
    col = np.array([[0.10, 0.30, 0.95, 1.0], [0.95, 0.80, 0.10, 1.0]])
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=look_at((0.9, 1.5, -0.6), (0.6, 0, 0)))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.2), pose=look_at((0.3, 1.0, 0.9), (0.6, 0, 0)))

    if args.cam in REL:
        prop, off, ty = REL[args.cam]
        ip = names.index(prop)
        j0 = idx[0]
        c = (abd_world_tf(abd[j0][ip]) @ C0inv[ip] @ np.r_[rest[f"v_{ip}"].mean(0), 1.0])[:3]
        eye = np.array([c[0] + off[0], off[1], c[2] + off[2]]); target = np.array([c[0], ty, c[2]]); yfov = 0.55
    else:
        c = CAMS[args.cam]; eye = np.array(c["eye"], float); target = np.array(c["target"], float); yfov = c["yfov"]
    if args.eye is not None:
        eye = np.array(args.eye, float)
    if args.target is not None:
        target = np.array(args.target, float)
    if args.yfov is not None:
        yfov = args.yfov
    eye = target + (eye - target) / args.zoom
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=args.size[0] / args.size[1], znear=0.02, zfar=20.0)
    scene.add(cam, pose=look_at(eye, target))
    print(f"[render] {len(idx)} frames, t {t[idx[0]]:.2f}..{t[idx[-1]]:.2f} s, eye {np.round(eye, 3)}, target {np.round(target, 3)}, yfov {yfov}", flush=True)
    r = pyrender.OffscreenRenderer(*args.size)
    tag = args.tag or args.cam
    frames_dir = osp.join(args.run, f"frames2_{tag}")
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        os.remove(osp.join(frames_dir, f))
    dnode = None
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    for fi, j in enumerate(idx):
        for i, node, base in nodes:
            scene.set_pose(node, abd_world_tf(abd[j][i]) @ C0inv[i] @ base)
        if dnode is not None:
            scene.remove_node(dnode)
        dm = trimesh.Trimesh(drop[j], dfaces, process=False)
        dnode = scene.add(pyrender.Mesh.from_trimesh(dm, material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.20, 0.45, 0.90, 0.85), roughnessFactor=0.4, alphaMode="BLEND"), smooth=False))
        lnode = []
        if liq is not None:
            p = liq[j].astype(np.float64) * 1e-4 + origin
            tf = np.tile(np.eye(4), (len(p), 1, 1))
            tf[:, :3, 3] = p
            for lab in (0, 1):
                sel = label[:len(p)] == lab
                if not sel.any():
                    continue
                s_ = trimesh.creation.icosphere(subdivisions=1, radius=1.3e-3)
                s_.visual.vertex_colors = (col[lab] * 255).astype(np.uint8)
                lnode.append(scene.add(pyrender.Mesh.from_trimesh(s_, poses=tf[sel], smooth=False)))
        img, _ = r.render(scene)
        for nn in lnode:
            scene.remove_node(nn)
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        d.rectangle([0, args.size[1] - 44, 190, args.size[1]], fill=(255, 255, 255))
        d.text((12, args.size[1] - 38), f"t = {t[j]:5.2f} s", fill=(20, 20, 20), font=font)
        im.save(osp.join(frames_dir, f"{fi:05d}.png"))
        if fi % 100 == 0:
            print(f"  frame {fi}/{len(idx)}", flush=True)
    r.delete()
    fps = 1.0 / max(np.median(np.diff(t)) * args.every, 1e-6)
    out = args.out or osp.join(args.run, f"preview2_{tag}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:.3f}", "-i", osp.join(frames_dir, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", out], check=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
