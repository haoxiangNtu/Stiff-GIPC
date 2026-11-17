#!/usr/bin/env python3
"""Render the recorded lab demo (lab_demo.npz) to an mp4: a shaded 3D view of
the bench plus a side panel with the two finger-pad tactile maps, grip
state and the Young's-modulus estimates.

    PYTHONPATH=. <solver-python> examples/lab_tactile/render_lab_video.py out/lab_tactile/lab_demo.npz
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import subprocess
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from lab_rec import load_recording  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if not osp.exists(FONT):
    FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"


def viridis(u):
    vir = np.array([[0.267, 0.005, 0.329], [0.254, 0.265, 0.530], [0.164, 0.471, 0.558],
                    [0.135, 0.659, 0.518], [0.478, 0.821, 0.318], [0.993, 0.906, 0.144]])
    x = np.clip(u, 0, 1) * (len(vir) - 1)
    i = np.clip(x.astype(int), 0, len(vir) - 2)
    f = (x - i)[..., None]
    return vir[i] * (1 - f) + vir[i + 1] * f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth record")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--eye", type=float, nargs=3, default=(1.12, 0.78, 0.98))
    ap.add_argument("--target", type=float, nargs=3, default=(0.40, 0.21, 0.02))
    args = ap.parse_args()
    out_mp4 = args.out or args.npz.replace(".npz", ".mp4")

    from PIL import Image, ImageDraw, ImageFont
    from stiff_physics.sensors.offscreen import SceneRenderer

    z = load_recording(args.npz)
    meta = z["meta"]
    names = list(z["body_names"])
    colors = z["body_color"]
    rest, faces = z["body_rest"], z["body_faces"]
    motion = z["motion"]
    n_fr = len(z["t"])
    has_fem = "fem_names" in z
    fem_names = list(z["fem_names"]) if has_fem else []
    fem_faces = z["fem_faces"] if has_fem else []
    fem_verts = z["fem_verts"] if has_fem else []
    has_maps = has_fem and len(z["pad_map_idx"]) > 0
    map_idx = z["pad_map_idx"] if has_maps else np.zeros(0, int)
    fem_color = {"gel_left": (0.16, 0.25, 0.56), "gel_right": (0.16, 0.25, 0.56), "dropper": (0.25, 0.45, 0.85)}

    W, H = args.width, args.height
    PW = 340
    R = SceneRenderer(W, H, bg=(0.93, 0.94, 0.96))
    glass = {"flask", "beaker_mix", "beaker_waste", "test_tube", "test_tube2"}
    for i, n in enumerate(names):
        node = R.add_mesh(n, rest[i], faces[i], color=tuple(colors[i]), merge=("arm" in str(z["body_kind"][i])))
        if n in glass and "liquid_pts" in z:            # see the liquid through the glassware
            for prim in node.mesh.primitives:
                prim.material.baseColorFactor = [*tuple(colors[i])[:3], 0.42]
                prim.material.alphaMode = "BLEND"
    gx = np.array([[x, -0.0005, zz] for x in (-0.35, 1.0) for zz in (-0.6, 0.6)])
    gf = np.array([[0, 1, 3], [0, 3, 2], [0, 3, 1], [0, 2, 3]], np.int32)
    R.add_mesh("table", gx, gf, color=(0.78, 0.76, 0.72), merge=False)
    R.look_at(args.eye, args.target, up=(0, 1, 0), yfov=0.70)

    font = ImageFont.truetype(FONT, 18)
    font_s = ImageFont.truetype(FONT, 14)
    font_b = ImageFont.truetype(FONT, 22)
    tdir = osp.join(osp.dirname(out_mp4), "_frames")
    os.makedirs(tdir, exist_ok=True)
    for f in os.listdir(tdir):
        os.remove(osp.join(tdir, f))

    fem_nodes = []
    has_liquid = "liquid_pts" in z and len(z["liquid_pts"]) == n_fr
    if has_liquid:
        import pyrender
        R.r.point_size = max(2.0, 0.0045 * W / 960 * 1000 * float(meta.get("liquid_dx", 0.005)))
    liq_node = None
    last_map = {"left": None, "right": None}
    map_pos = {int(k): j for j, k in enumerate(map_idx)}
    frames = list(range(0, n_fr, args.stride))
    if args.max:
        frames = frames[:args.max]
    est_lines = []
    # close-up inset (picture in picture) that follows the dropper tip while it draws / drips
    INSET_SEGS = ("aspirate", "lower over beaker", "squeeze bulb", "release squeeze")
    IW, IH = 360, 270
    inset_fov, main_fov = 0.34, 0.70
    for k, fr in enumerate(frames):
        for i, n in enumerate(names):
            R.set_pose(n, motion[fr, i].astype(np.float64))
        for nd in fem_nodes:
            R.scene.remove_node(nd)
        fem_nodes = []
        for i, n in enumerate(fem_names):
            fem_nodes.append(R.add_mesh(f"fem{i}", fem_verts[i][fr], fem_faces[i], color=fem_color.get(n, (0.5, 0.5, 0.5)),
                                        merge=False))
            if n == "dropper" and has_liquid:            # see the liquid column inside the hollow tube
                for prim in fem_nodes[-1].mesh.primitives:
                    prim.material.baseColorFactor = [*fem_color["dropper"], 0.55]
                    prim.material.alphaMode = "BLEND"
        if has_liquid:
            if liq_node is not None:
                R.scene.remove_node(liq_node)
            pts = np.asarray(z["liquid_pts"][fr], np.float32)
            if len(pts):
                col = np.tile(np.array([[0.12, 0.45, 0.95, 1.0]], np.float32), (len(pts), 1))   # opaque: drawn before the blended glass
                liq_node = R.scene.add(pyrender.Mesh.from_points(pts, colors=col))
            else:
                liq_node = None
        inset = None
        seg_now = str(z["segment"][fr])
        if has_fem and "dropper" in fem_names and any(s_ in seg_now for s_ in INSET_SEGS):
            dv = fem_verts[fem_names.index("dropper")][fr]
            tip = dv[np.argmin(dv[:, 1])].astype(np.float64)
            if "aspirate" in seg_now:                    # tube above the flask water: watch the column rise
                focus = tip + np.array([0.0, 0.055, 0.0]); off = np.array([0.17, 0.11, 0.19])
            else:                                        # drops fall below the tip into the beaker
                focus = tip + np.array([0.0, -0.035, 0.0]); off = np.array([0.15, 0.06, 0.17])
            cam = R._nodes["cam"].camera
            cam.yfov = inset_fov
            R.look_at(focus + off, focus, up=(0, 1, 0), yfov=inset_fov)
            ps = R.r.point_size
            R.r.point_size = max(ps * 3.0, 6.0)
            inset = Image.fromarray(R.render()).resize((IW, IH), Image.LANCZOS)
            R.r.point_size = ps
            cam.yfov = main_fov
            R.look_at(args.eye, args.target, up=(0, 1, 0), yfov=main_fov)
        img3d = Image.fromarray(R.render())
        canvas = Image.new("RGB", (W + PW, H), (255, 255, 255))
        canvas.paste(img3d, (0, 0))
        d = ImageDraw.Draw(canvas)
        if inset is not None:
            ix, iy = W - IW - 12, 76
            canvas.paste(inset, (ix, iy))
            d.rectangle([ix - 2, iy - 2, ix + IW + 1, iy + IH + 1], outline=(28, 32, 40), width=2)
            d.rectangle([ix - 2, iy - 24, ix + IW + 1, iy - 2], fill=(28, 32, 40))
            d.text((ix + 6, iy - 22), "滴管特写 close-up", fill=(240, 240, 240), font=font_s)
        # ---- header on the 3D view
        stage, seg = str(z["stage"][fr]), str(z["segment"][fr])
        d.rectangle([0, 0, W, 64], fill=(28, 32, 40))
        d.text((14, 8), f"StiffGIPC 触觉实验台演示  |  {stage}", fill=(240, 240, 240), font=font_b)
        d.text((14, 38), f"{seg}    t = {float(z['t'][fr]):5.1f} s", fill=(200, 210, 230), font=font)
        # ---- side panel
        x0 = W + 12
        d.text((x0, 10), "指尖触觉 (VBTS) 压痕图", fill=(30, 30, 30), font=font)
        if fr in map_pos:
            j = map_pos[fr]
            last_map["left"], last_map["right"] = z["pad_map_left"][j], z["pad_map_right"][j]
        y = 40
        for side in ("left", "right"):
            m = last_map[side]
            if m is None:
                m = np.zeros((200, 176), np.uint8)
            rgb = (viridis(m.astype(np.float32) / 255.0) * 255).astype(np.uint8)
            im = Image.fromarray(np.ascontiguousarray(np.rot90(rgb)))      # pad x (finger length) horizontal
            im = im.resize((300, int(300 * im.height / im.width)), Image.BILINEAR)
            canvas.paste(im, (x0, y))
            d.text((x0 + 4, y + 2), f"PAD-{'L' if side == 'left' else 'R'}  max {m.max()/255*2.0:.2f} mm",
                   fill=(255, 255, 255), font=font_s)
            y += im.height + 8
        d.text((x0, y + 4), "夹爪状态", fill=(30, 30, 30), font=font)
        y += 32
        gap = float(z["gap_mm"][fr])
        fl, fr_ = float(z["fn_left"][fr]), float(z["fn_right"][fr])
        d.text((x0, y), f"pad 间距  {gap:6.1f} mm", fill=(50, 50, 50), font=font_s); y += 20
        d.text((x0, y), f"法向力  L {fl:5.2f} N   R {fr_:5.2f} N", fill=(50, 50, 50), font=font_s); y += 20
        # force bars
        for i, (lab, v) in enumerate((("L", fl), ("R", fr_))):
            bx = x0 + 30 + i * 150
            d.rectangle([bx, y, bx + 100, y + 10], outline=(120, 120, 120))
            d.rectangle([bx, y, bx + int(min(v, 10.0) / 10.0 * 100), y + 10], fill=(43, 108, 176))
            d.text((bx - 24, y - 3), lab, fill=(50, 50, 50), font=font_s)
        y += 30
        d.text((x0, y), "杨氏模量估计 (Hertz, 触觉)", fill=(30, 30, 30), font=font); y += 28
        et = str(z["etext"][fr])
        if et and (not est_lines or est_lines[-1].split(":")[0] != et.split(":")[0]):
            est_lines.append(et)
        elif et and est_lines and est_lines[-1].split(":")[0] == et.split(":")[0]:
            est_lines[-1] = et
        for line in est_lines[-5:]:
            # wrap at ~44 chars
            parts = [line[i:i + 44] for i in range(0, len(line), 44)]
            for pp in parts:
                d.text((x0, y), pp, fill=(40, 40, 40), font=font_s); y += 18
            y += 4
        canvas.save(osp.join(tdir, f"f{k:05d}.png"))
        if k % 100 == 0:
            print(f"  frame {k}/{len(frames)}")
    R.close()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps / args.stride),
                    "-i", osp.join(tdir, "f%05d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-pix_fmt", "yuv420p", "-crf", "20", out_mp4], check=True)
    print(f"-> {out_mp4}  ({len(frames)} frames @ {args.fps/args.stride:.1f} fps)")


if __name__ == "__main__":
    main()
