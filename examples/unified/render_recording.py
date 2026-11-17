#!/usr/bin/env python3
"""Render a Recorder .npz (ABD bodies + FEM cloth + liquid particles) to an MP4 with the pyrender/EGL offscreen renderer.

    PYTHONPATH=. python examples/unified/render_recording.py out/unified/case13/recording.npz --out case13.mp4
"""
from __future__ import annotations
import argparse, os, os.path as osp, subprocess, sys
import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, "..", "lab_tactile"))
from lab_rec import load_recording  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=0.0, help="0 = recording fps from meta")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--eye", type=float, nargs=3, default=(3.6, 2.6, 4.4))
    ap.add_argument("--target", type=float, nargs=3, default=(0.0, 0.9, 0.0))
    ap.add_argument("--yfov", type=float, default=0.78)
    ap.add_argument("--liquid-color", type=float, nargs=3, default=(0.10, 0.42, 0.95))
    ap.add_argument("--point-px", type=float, default=0.0, help="liquid point size in pixels (0 = from spacing)")
    ap.add_argument("--ground", type=float, default=0.0, help="y of the ground plane; nan = none")
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()
    out_mp4 = args.out or args.npz.replace(".npz", ".mp4")
    from PIL import Image
    import pyrender
    from stiff_physics.sensors.offscreen import SceneRenderer

    z = load_recording(args.npz)
    meta = z["meta"]
    fps = args.fps or float(meta.get("fps", 50.0))
    names = list(z["body_names"]); colors = z["body_color"]
    rest, faces, motion = z["body_rest"], z["body_faces"], z["motion"]
    n_fr = len(z["t"])
    fem_names = list(z["fem_names"]) if "fem_names" in z else []
    fem_faces = z["fem_faces"] if fem_names else []
    fem_verts = z["fem_verts"] if fem_names else []
    fem_colors = meta.get("fem_colors", [[0.15, 0.45, 0.85]] * len(fem_names))
    has_liquid = "liquid_pts" in z and len(z["liquid_pts"]) == n_fr

    R = SceneRenderer(args.width, args.height, bg=(0.94, 0.95, 0.97))
    for i, n in enumerate(names):
        R.add_mesh(n, rest[i], faces[i], color=tuple(float(c) for c in colors[i]), merge=False)
    if not np.isnan(args.ground):
        g = 2.6
        gx = np.array([[-g, args.ground - 0.002, -g], [g, args.ground - 0.002, -g], [g, args.ground - 0.002, g], [-g, args.ground - 0.002, g]])
        R.add_mesh("ground", gx, np.array([[0, 2, 1], [0, 3, 2]], np.int32), color=(0.80, 0.80, 0.78), merge=False)
    R.look_at(args.eye, args.target, up=(0, 1, 0), yfov=args.yfov)
    if has_liquid:
        spacing = float(meta.get("particle_spacing", 0.01))
        R.r.point_size = args.point_px if args.point_px > 0 else max(2.0, spacing * args.width * 0.55)

    tdir = osp.join(osp.dirname(osp.abspath(out_mp4)), "_frames_" + osp.splitext(osp.basename(out_mp4))[0])
    os.makedirs(tdir, exist_ok=True)
    for f in os.listdir(tdir):
        os.remove(osp.join(tdir, f))
    frames = list(range(0, n_fr, args.stride))
    if args.max:
        frames = frames[: args.max]
    fem_nodes, liq_node = [], None
    liq_col = np.array([[*args.liquid_color, 1.0]], np.float32)
    for k, fr in enumerate(frames):
        for i, n in enumerate(names):
            R.set_pose(n, motion[fr, i].astype(np.float64))
        for nd in fem_nodes:
            R.scene.remove_node(nd)
        fem_nodes = [R.add_mesh(f"fem{i}", fem_verts[i][fr], fem_faces[i], color=tuple(fem_colors[i]), merge=False)
                     for i in range(len(fem_names))]
        if has_liquid:
            if liq_node is not None:
                R.scene.remove_node(liq_node)
            pts = np.asarray(z["liquid_pts"][fr], np.float32)
            liq_node = R.scene.add(pyrender.Mesh.from_points(pts, colors=np.repeat(liq_col, len(pts), 0))) if len(pts) else None
        img = R.render()
        Image.fromarray(img).save(osp.join(tdir, f"f{k:05d}.png"))
        if k % 25 == 0:
            print(f"rendered {k+1}/{len(frames)}", flush=True)
    R.close()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps / args.stride), "-i", osp.join(tdir, "f%05d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_mp4], check=True)
    if not args.keep_frames:
        for f in os.listdir(tdir):
            os.remove(osp.join(tdir, f))
        os.rmdir(tdir)
    print(f"[done] {len(frames)} frames -> {out_mp4}")


if __name__ == "__main__":
    main()
