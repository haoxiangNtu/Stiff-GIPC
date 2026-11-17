"""Recording file format for the lab demo (plain npz, no pickled objects, so
both the solver Python (numpy 2) and the USD Python (numpy 1.x) can read it).

Per-body arrays are stored as numbered keys (body_rest_<i>, body_faces_<i>,
fem_verts_<i>, fem_faces_<i>); ``load_recording`` rebuilds the lists."""

from __future__ import annotations

import json

import numpy as np


def save_recording(path, out: dict):
    flat = {}
    for k, v in out.items():
        if isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], np.ndarray):
            flat[f"{k}__n"] = np.array(len(v))
            for i, a in enumerate(v):
                flat[f"{k}_{i}"] = np.asarray(a)
        elif isinstance(v, (dict,)):
            flat[k] = np.array(json.dumps(v, default=float))
        else:
            flat[k] = np.asarray(v)
    np.savez_compressed(path, **flat)
    return path


def load_recording(path) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {}
    lists = {}
    for k in z.files:
        if k.endswith("__n"):
            lists[k[:-3]] = int(z[k])
    for k in z.files:
        if k.endswith("__n"):
            continue
        base = k.rsplit("_", 1)[0]
        if base in lists and k[len(base) + 1:].isdigit():
            continue
        out[k] = z[k]
    for base, n in lists.items():
        out[base] = [z[f"{base}_{i}"] for i in range(n)]
    if "meta" in out:
        out["meta"] = json.loads(str(out["meta"]))
    return out
