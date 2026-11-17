"""Replace the convex-hull collision geometry of URDF robot links in a lab recording by the links' VISUAL meshes, for rendering.

The recording stores per-body rest vertices in WORLD coordinates at load time plus a per-frame rigid motion; the hull and the
visual mesh of a link share the link frame, so the visual mesh placed with the same load pose moves with the recorded motion.
The load pose of each link is rebuilt with the URDF FK (tools/urdf_kin.py) from the robot's base transform and initial joint
angles (stored in meta["robot"] by the policy scripts, or given on the command line), and checked against the recorded hull.

    python3 examples/nuka/tools/swap_visual_meshes.py recording.npz --urdf Assets/nuka/go2/go2_hull.urdf --prefix go2/ \
        --base-y 0.376 --init go2_default --out recording_visual.npz
"""
import argparse, json, math, os.path as osp, sys
import numpy as np
import trimesh

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "..", "lab_tactile"))
from urdf_kin import UrdfKin, rpy_to_R  # noqa: E402
from lab_rec import load_recording, save_recording  # noqa: E402

R_CONV = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
GO2_DEFAULT = {f"{l}_{j}_joint": v for l in ("FL", "FR", "RL", "RR") for j, v in zip(("hip", "thigh", "calf"), (0.0, 0.8, -1.5))}
for l, h in (("FL", 0.1), ("FR", -0.1), ("RL", 0.1), ("RR", -0.1)):
    GO2_DEFAULT[f"{l}_hip_joint"] = h
GO2_DEFAULT["RL_thigh_joint"] = 1.0; GO2_DEFAULT["RR_thigh_joint"] = 1.0


def visual_parts(urdf_path, link_el):
    """(vertices, faces) of a link's visual meshes in the LINK frame (binary STL copies next to the URDF)."""
    import xml.etree.ElementTree as ET  # noqa: F401
    base = osp.dirname(urdf_path)
    V, F = [], []
    for vis in link_el.findall("visual"):
        geo = vis.find("geometry"); mesh = geo.find("mesh") if geo is not None else None
        if mesh is None:
            continue
        path = osp.join(base, mesh.get("filename"))
        if not osp.isfile(path):
            continue
        m = trimesh.load(path, force="mesh")
        sc = mesh.get("scale")
        if sc:
            m.apply_scale([float(v) for v in sc.split()])
        o = vis.find("origin")
        T = np.eye(4)
        if o is not None:
            T[:3, :3] = rpy_to_R([float(v) for v in o.get("rpy", "0 0 0").split()]); T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
        m.apply_transform(T)
        F.append(np.asarray(m.faces, np.int32) + sum(len(v) for v in V)); V.append(np.asarray(m.vertices, float))
    if not V:
        return None, None
    return np.vstack(V), np.vstack(F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--prefix", default="go2/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-y", type=float, default=None)
    ap.add_argument("--base-xyz", type=str, default=None, help="engine (y-up) base position x,y,z; overrides --base-y")
    ap.add_argument("--yaw0", type=float, default=0.0, help="z-up yaw (rad) applied before the z-up -> y-up rotation")
    ap.add_argument("--init", default=None, help="'go2_default', a JSON file of joint angles, or 'meta' (from meta['robot'])")
    ap.add_argument("--color", type=str, default=None, help="r,g,b for the robot bodies")
    args = ap.parse_args()
    import xml.etree.ElementTree as ET
    rec = load_recording(args.recording)
    meta = rec.get("meta", {})
    robot_meta = meta.get("robot") if isinstance(meta, dict) else None
    if robot_meta and args.init in (None, "meta"):
        T = np.array(robot_meta["T"], float); init = {k: float(v) for k, v in robot_meta["init"].items()}
    else:
        T = np.eye(4)
        Rz = np.array([[math.cos(args.yaw0), -math.sin(args.yaw0), 0], [math.sin(args.yaw0), math.cos(args.yaw0), 0], [0, 0, 1]])
        T[:3, :3] = R_CONV @ Rz
        if args.base_xyz:
            T[:3, 3] = [float(v) for v in args.base_xyz.split(",")]
        elif args.base_y is not None:
            T[1, 3] = args.base_y
        init = GO2_DEFAULT if args.init == "go2_default" else (json.load(open(args.init)) if args.init else {})
    kin = UrdfKin(args.urdf)
    fk = kin.fk(init, base_T=T)
    links = {l.get("name"): l for l in ET.parse(args.urdf).getroot().findall("link")}
    names = [str(n) for n in rec["body_names"]]
    rest, faces = list(rec["body_rest"]), list(rec["body_faces"])
    n_swapped = 0; worst = 0.0
    for i, n in enumerate(names):
        if not n.startswith(args.prefix):
            continue
        link = n[len(args.prefix):]
        if link not in links or link not in fk:
            print(f"[skip] {n}: no URDF link / FK"); continue
        V, F = visual_parts(args.urdf, links[link])
        if V is None:
            print(f"[skip] {n}: no visual mesh"); continue
        Tl = fk[link]
        Vw = (Tl @ np.c_[V, np.ones(len(V))].T).T[:, :3]
        # self-check: the recorded hull (world, load pose) must sit inside / around the visual mesh bbox
        h = np.asarray(rest[i], float)
        d = np.linalg.norm(h.mean(0) - Vw.mean(0)); worst = max(worst, d)
        rest[i] = Vw.astype(np.float32); faces[i] = F.astype(np.int32); n_swapped += 1
    rec["body_rest"], rec["body_faces"] = rest, faces
    if args.color:
        col = np.array([float(v) for v in args.color.split(",")], np.float32)
        bc = np.array(rec["body_color"], np.float32)
        for i, n in enumerate(names):
            if n.startswith(args.prefix):
                bc[i] = col
        rec["body_color"] = bc
    save_recording(args.out, rec)
    print(f"[swap] {n_swapped} bodies replaced by visual meshes; worst hull/visual centroid distance {worst*1e3:.1f} mm -> {args.out}")


if __name__ == "__main__":
    main()
