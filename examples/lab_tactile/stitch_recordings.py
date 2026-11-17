#!/usr/bin/env python3
"""Concatenate several lab-demo recordings (same scene) into one, dropping
the pre-settle stage from all but the first, so a continuous USD / video can
be produced from independently validated stage runs.

    python stitch_recordings.py out.npz a.npz b.npz c.npz ...
"""
import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from lab_rec import load_recording, save_recording  # noqa: E402


def main():
    out_path, paths = sys.argv[1], sys.argv[2:]
    recs = [load_recording(p) for p in paths]
    base = recs[0]
    keep = []
    for i, z in enumerate(recs):
        stage = np.array([str(s) for s in z["stage"]])
        m = np.ones(len(stage), bool) if i == 0 else ~np.char.startswith(stage, "0")
        keep.append(m)
    out = dict(body_names=base["body_names"], body_kind=base["body_kind"], body_color=base["body_color"],
               body_rest=base["body_rest"], body_faces=base["body_faces"],
               fem_names=base["fem_names"], fem_faces=base["fem_faces"])
    t_off, cat = 0.0, {k: [] for k in ("motion", "t", "stage", "segment", "etext", "gap_mm", "fn_left",
                                       "fn_right", "tcp", "q", "pad_map_left", "pad_map_right", "pad_map_idx")}
    fem = [[] for _ in base["fem_names"]]
    liq = []
    n_rec, ests = 0, []
    for z, m in zip(recs, keep):
        idx = np.nonzero(m)[0]
        t = z["t"][idx]
        cat["t"].append(t - t[0] + t_off)
        t_off = cat["t"][-1][-1] + float(np.median(np.diff(t))) if len(t) > 1 else t_off
        for k in ("motion", "stage", "segment", "etext", "gap_mm", "fn_left", "fn_right", "tcp", "q"):
            cat[k].append(z[k][idx])
        for j in range(len(fem)):
            fem[j].append(z["fem_verts"][j][idx])
        if "liquid_pts" in z:                                   # engine-liquid recordings: per-record particle arrays
            liq.extend([np.asarray(z["liquid_pts"][k], np.float32) for k in idx])   # idx is already an index array
        # pad maps: keep those whose record index is kept, re-index
        pm_idx = np.asarray(z["pad_map_idx"])
        sel = [k for k, r in enumerate(pm_idx) if m[r]]
        remap = {r: n for n, r in enumerate(idx)}
        cat["pad_map_left"].append(np.asarray(z["pad_map_left"])[sel])
        cat["pad_map_right"].append(np.asarray(z["pad_map_right"])[sel])
        cat["pad_map_idx"].append(np.array([remap[pm_idx[k]] + n_rec for k in sel], np.int32))
        n_rec += len(idx)
        ests += z["meta"].get("estimates", [])
    for k, v in cat.items():
        out[k] = np.concatenate(v) if len(v[0].shape) else np.array(v)
    out["fem_verts"] = [np.concatenate(f) for f in fem]
    if liq:
        out["liquid_pts"] = liq
    out["meta"] = dict(base["meta"], estimates=ests, stitched_from=paths, frames=int(n_rec))
    save_recording(out_path, out)
    print(f"-> {out_path}: {n_rec} records, {out['t'][-1]:.1f} s, {len(ests)} estimates")


if __name__ == "__main__":
    main()
