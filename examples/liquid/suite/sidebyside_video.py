"""Side-by-side particle videos of two (or more) runs that dumped frames/f_<step>.npz (x, v[, block]) — our engine and
the recovered Unified_GIPC engine on the same scene — rendered with matplotlib (no GPU), faststart mp4.

    python sidebyside_video.py --scene m2 --out m2_tri.mp4 --panel "ours (trilinear)" out/ugrun/v_m2_tri \
                                                           --panel "recovered (trilinear)" ug_ref/out/v_m2_tri

Each panel is a side view (x horizontal, y vertical) of the particles coloured by speed, with the container outline;
the frame time is taken from the dump. Frames are matched by step (dumps must use the same --dump-every and dt).
"""
import argparse, glob, os, subprocess, sys, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCENES = {   # (xlim, ylim, container polylines (x, y) in metres, speed colour scale m/s, projection)
    "m2":    dict(xlim=(0.13, 0.27), ylim=(0.0, 0.20), vmax=0.3,
                  lines=[[(0.135, 0.01), (0.135, 0.18), (0.14, 0.18), (0.14, 0.02), (0.26, 0.02), (0.26, 0.18), (0.265, 0.18), (0.265, 0.01), (0.135, 0.01)]]),
    "m6":    dict(xlim=(0.16, 0.24), ylim=(-0.005, 0.08), vmax=0.15, proj="xz->z",   # beaker: axis at z = 0.20; view along x so the tilt (about x) shows
                  lines=[[(0.1763, 0.0005), (0.1763, 0.0696), (0.2237, 0.0696), (0.2237, 0.0005), (0.1763, 0.0005)]]),
    "float": dict(xlim=(0.13, 0.27), ylim=(0.0, 0.14), vmax=0.3,
                  lines=[[(0.135, 0.01), (0.135, 0.18), (0.14, 0.18), (0.14, 0.02), (0.26, 0.02), (0.26, 0.18), (0.265, 0.18), (0.265, 0.01), (0.135, 0.01)]]),
    "m12":   dict(xlim=(-0.02, 0.45), ylim=(-0.005, 0.15), vmax=1.5,
                  lines=[[(-0.01, 0.0), (-0.01, 0.144), (0.0, 0.144), (0.0, 0.0)], [(0.0, 0.0), (0.45, 0.0)]]),
}


def load_frames(run):
    fs = sorted(glob.glob(os.path.join(run, "frames", "f_*.npz")))
    return {int(os.path.basename(f)[2:7]): f for f in fs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=sorted(SCENES))
    ap.add_argument("--panel", nargs=2, action="append", metavar=("TITLE", "RUN"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--pt", type=float, default=3.0, help="marker size")
    ap.add_argument("--title", default="")
    ap.add_argument("--vmax", type=float, default=0.0, help="speed colour scale (m/s); 0 = the scene default")
    a = ap.parse_args()
    sc = dict(SCENES[a.scene])
    if a.vmax > 0:
        sc["vmax"] = a.vmax
    panels = [(t, load_frames(r)) for t, r in a.panel]
    steps = sorted(set.intersection(*[set(p[1]) for p in panels]))
    if a.max_frames:
        steps = steps[: a.max_frames]
    if not steps:
        sys.exit("no common frames")
    tmp = tempfile.mkdtemp(prefix="sbs_")
    n = len(panels)
    fh = 3.0 if a.scene == "m12" else 5.2
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, fh), dpi=a.dpi)
    axes = np.atleast_1d(axes)
    for k, st in enumerate(steps):
        for ax, (title, frames) in zip(axes, panels):
            ax.cla()
            d = np.load(frames[st])
            x = d["x"]; v = d["v"]; sp = np.linalg.norm(v, axis=1)
            if sc.get("proj") == "xz->z":
                h, w = x[:, 2], x[:, 1]
            else:
                h, w = x[:, 0], x[:, 1]
            order = np.argsort(h)             # painter's order is irrelevant for a side view; sort by depth for stable look
            ax.scatter(h, w, c=sp, s=a.pt, cmap="viridis", vmin=0, vmax=sc["vmax"], linewidths=0, rasterized=True)
            for pl in sc["lines"]:
                px, py = zip(*pl); ax.plot(px, py, "-", color="0.35", lw=1.2)
            b = None
            if "block" in d.files:
                b = d["block"]
            elif a.scene == "float" and "mesh" in d.files and d["mesh"].shape[0] >= 24:
                b = d["mesh"][16:24]            # recovered runner: cup (16 vertices) then the block (8)
            if b is not None:
                bx = b[:, 2] if sc.get("proj") == "xz->z" else b[:, 0]
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(np.c_[bx, b[:, 1]]); vs = list(hull.vertices) + [hull.vertices[0]]
                    ax.plot(bx[vs], b[vs, 1], "-", color="crimson", lw=1.6)
                except Exception:
                    pass
            ax.set_xlim(*sc["xlim"]); ax.set_ylim(*sc["ylim"]); ax.set_aspect("equal")
            ax.set_title(f"{title}   t = {float(d['t']):.2f} s", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        if a.title:
            fig.suptitle(a.title, fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(tmp, f"{k:05d}.png"))
        if k % 50 == 0:
            print(f"[sbs] frame {k}/{len(steps)}", flush=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(a.fps), "-i", os.path.join(tmp, "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-movflags", "+faststart", a.out], check=True)
    print(f"[sbs] wrote {a.out} ({len(steps)} frames)")


if __name__ == "__main__":
    main()
