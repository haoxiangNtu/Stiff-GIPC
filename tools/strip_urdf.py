#!/usr/bin/env python3
"""strip_urdf.py — remove specified link nodes (+ joints referencing them)
from a URDF, preserving the rest of the kinematic tree.

Example (remove all 8 finger + soft_material links from ridgeback_dual_panda2):
    python tools/strip_urdf.py \
        Assets/sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2.urdf \
        Assets/sim_data/urdf/ridgeback_dual_panda_soft/ridgeback_dual_panda2_no_finger.urdf \
        left_arm_leftfinger left_arm_rightfinger \
        left_arm_leftfinger_soft_material left_arm_rightfinger_soft_material \
        right_arm_leftfinger right_arm_rightfinger \
        right_arm_leftfinger_soft_material right_arm_rightfinger_soft_material
"""
import sys
import xml.etree.ElementTree as ET


def strip_urdf(urdf_in: str, urdf_out: str, links_to_remove: list) -> dict:
    tree = ET.parse(urdf_in)
    root = tree.getroot()
    remove_set = set(links_to_remove)

    # Find joints that reference any removed link (parent or child)
    joints_to_remove = []
    for j in root.findall('joint'):
        p_el = j.find('parent')
        c_el = j.find('child')
        if p_el is None or c_el is None:
            continue
        p, c = p_el.get('link'), c_el.get('link')
        if p in remove_set or c in remove_set:
            joints_to_remove.append(j)

    # Remove
    removed_links = []
    for link in list(root.findall('link')):
        if link.get('name') in remove_set:
            root.remove(link)
            removed_links.append(link.get('name'))

    removed_joints = []
    for j in joints_to_remove:
        root.remove(j)
        removed_joints.append(j.get('name'))

    # Preserve XML formatting (pretty print)
    try:
        ET.indent(tree, space='  ')
    except AttributeError:
        pass  # Python < 3.9
    tree.write(urdf_out, xml_declaration=True, encoding='utf-8')

    return dict(
        removed_links=removed_links,
        removed_joints=removed_joints,
        n_links_remaining=len(root.findall('link')),
        n_joints_remaining=len(root.findall('joint')),
    )


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <in.urdf> <out.urdf> <link1> [link2 ...]")
        sys.exit(1)
    info = strip_urdf(sys.argv[1], sys.argv[2], sys.argv[3:])
    print(f"[strip_urdf] removed {len(info['removed_links'])} links, "
          f"{len(info['removed_joints'])} joints", flush=True)
    print(f"  removed links: {info['removed_links']}", flush=True)
    print(f"  removed joints: {info['removed_joints']}", flush=True)
    print(f"  remaining: {info['n_links_remaining']} links, "
          f"{info['n_joints_remaining']} joints", flush=True)


if __name__ == "__main__":
    main()
