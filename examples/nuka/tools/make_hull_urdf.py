"""Build a StiffGIPC-loadable URDF from a vendor URDF: per-link convex-hull collision meshes (binary STL), visual meshes copied as
binary STL, floating root joints removed (the engine's importer supports revolute / prismatic / fixed / continuous only).

    python3 examples/nuka/tools/make_hull_urdf.py --src <unitree_ros>/robots/g1_description --urdf g1_29dof_with_hand_rev_1_0.urdf \
        --out Assets/nuka/g1 --name g1_hull.urdf
"""
import argparse, os, os.path as osp, shutil, math
import xml.etree.ElementTree as ET
import numpy as np
import trimesh


def rpy_to_R(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def origin_T(el):
    T = np.eye(4)
    if el is None:
        return T
    xyz = np.array([float(v) for v in el.get("xyz", "0 0 0").split()])
    rpy = np.array([float(v) for v in el.get("rpy", "0 0 0").split()])
    T[:3, :3] = rpy_to_R(rpy); T[:3, 3] = xyz
    return T


def resolve(src, fn):
    fn = fn.replace("package://", "")
    for cand in (osp.join(src, fn), osp.join(src, osp.basename(fn)), osp.join(src, "meshes", osp.basename(fn)),
                 osp.join(src, "/".join(fn.split("/")[1:]))):
        if osp.isfile(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="robot_hull.urdf")
    ap.add_argument("--max-hull-faces", type=int, default=400)
    args = ap.parse_args()
    os.makedirs(osp.join(args.out, "meshes"), exist_ok=True)
    tree = ET.parse(osp.join(args.src, args.urdf)); root = tree.getroot()
    # drop floating joints and their (virtual) parent links
    drop_links = set()
    for j in list(root.findall("joint")):
        if j.get("type") == "floating":
            drop_links.add(j.find("parent").get("link")); root.remove(j)
            print(f"[urdf] removed floating joint {j.get('name')} (parent link {j.find('parent').get('link')})")
    for l in list(root.findall("link")):
        if l.get("name") in drop_links and l.find("visual") is None and l.find("inertial") is None:
            root.remove(l); print(f"[urdf] removed virtual link {l.get('name')}")
    n_hull = 0; n_nocol = 0; stats = []
    for link in root.findall("link"):
        name = link.get("name")
        parts = []
        for vis in link.findall("visual"):
            geo = vis.find("geometry"); mesh = geo.find("mesh") if geo is not None else None
            if mesh is None:
                continue
            path = resolve(args.src, mesh.get("filename"))
            if path is None:
                print(f"[warn] {name}: visual mesh {mesh.get('filename')} not found"); continue
            m = trimesh.load(path, force="mesh")
            sc = mesh.get("scale")
            if sc:
                m.apply_scale([float(v) for v in sc.split()])
            m.apply_transform(origin_T(vis.find("origin")))
            parts.append(m)
            dst = osp.join(args.out, "meshes", osp.splitext(osp.basename(path))[0] + ".stl")
            if not osp.isfile(dst):
                trimesh.load(path, force="mesh").export(dst)            # binary STL copy for the renderer
            mesh.set("filename", "meshes/" + osp.basename(dst))
            if sc is None:
                pass
        for col in list(link.findall("collision")):
            link.remove(col)
        if not parts:
            n_nocol += 1
            continue
        allm = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
        hull = allm.convex_hull
        if len(hull.faces) > args.max_hull_faces:
            try:
                hull = hull.simplify_quadric_decimation(face_count=args.max_hull_faces).convex_hull
            except Exception as e:   # noqa: BLE001
                print(f"[warn] {name}: decimation failed ({e}); keeping the full hull ({len(hull.faces)} faces)")
        hull_path = osp.join(args.out, "meshes", f"{name}_hull.stl")
        hull.export(hull_path)
        col = ET.SubElement(link, "collision")
        ET.SubElement(col, "origin", xyz="0 0 0", rpy="0 0 0")
        geo = ET.SubElement(col, "geometry"); ET.SubElement(geo, "mesh", filename=f"meshes/{name}_hull.stl")
        n_hull += 1
        ext = hull.extents
        stats.append((name, len(hull.vertices), len(hull.faces), float(hull.volume), ext))
    out_path = osp.join(args.out, args.name)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"[urdf] {out_path}: {n_hull} links with hull collisions, {n_nocol} links without geometry; "
          f"{len(root.findall('joint'))} joints ({sum(1 for j in root.findall('joint') if j.get('type') == 'revolute')} revolute)")
    for name, nv, nf, vol, ext in stats:
        print(f"  {name:32s} hull {nv:4d} v {nf:4d} f  vol {vol*1e6:8.1f} cm3  extent {np.round(ext*1e3, 0).tolist()} mm")


if __name__ == "__main__":
    main()
