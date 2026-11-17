import warp as wp
from .types import *
from .sp_grid import (
    block_coords_from_node, local_coords_in_block, local2global,
)
from .kernel.constant import _0

@wp.func
def proj_boundary_vel(
    v: vec3,
    btype: int,
    normal: vec3,
    velocity: vec3,
) -> vec3:
    if btype == 0: return v
    if btype == 1: return velocity  # sticky
    if btype == 3: # separable (StiffGIPC patch): only block the approaching normal component
        vn = wp.dot(v - velocity, normal)
        if vn < real(0.0):
            v = v - vn * normal
        return v
    else: # slippy
        v = v - wp.dot(v - velocity, normal) * normal
        return v


@wp.func
def proj_boundary_vel_with_friction(
    v: vec3,
    btype: int,
    normal: vec3,
    velocity: vec3,
    friction: real = 1.0,
) -> vec3:
    if btype == 0: return v
    if btype == 1: return velocity  # sticky
    else: # slippy
        vn_obj = wp.dot(velocity, normal) * normal
        vt_component = v - wp.dot(v, normal) * normal
        v = vn_obj + friction * vt_component
        return v


@wp.func
def proj_boundary_delta_vel(
    dv: vec3,
    btype: int,
    normal: vec3,
) -> vec3:
    if btype == 0: return dv
    if btype == 1: return vec3(real(0)) # delta component should be zero for sticky
    else: # slippy (type 3 separable is treated as slip inside the implicit iterations)
        return dv - wp.dot(dv, normal) * normal # there should be no normal component in delta v


@wp.kernel
def paint_bc_kernel(
    boundary_ijk: wp.array(dtype=wp.vec3i, ndim=1),
    boundary_type: wp.array(dtype=int, ndim=1),
    boundary_norm: wp.array(dtype=vec3, ndim=1),
    boundary_velo: wp.array(dtype=vec3, ndim=1),
    bc_block2bid: wp.array(dtype=int, ndim=3),
    bc_type: wp.array(dtype=int, ndim=4),
    bc_norm: wp.array(dtype=vec3, ndim=4),
    bc_velo: wp.array(dtype=vec3, ndim=4),
):
    dof = wp.tid()
    bc_node = boundary_ijk[dof]
    bc_bcoords = block_coords_from_node(bc_node.x, bc_node.y, bc_node.z)
    bc_bid = bc_block2bid[bc_bcoords.x, bc_bcoords.y, bc_bcoords.z]
    if bc_bid < 0: return
    bc_lijk = local_coords_in_block(bc_node.x, bc_node.y, bc_node.z)
    bc_type[bc_bid, bc_lijk.x, bc_lijk.y, bc_lijk.z] = boundary_type[dof]
    bc_norm[bc_bid, bc_lijk.x, bc_lijk.y, bc_lijk.z] = boundary_norm[dof]
    bc_velo[bc_bid, bc_lijk.x, bc_lijk.y, bc_lijk.z] = boundary_velo[dof]

@wp.func
def query_bc_sp(
    node: wp.vec3i,
    bc_block2bid: wp.array(dtype=int, ndim=3),
    bc_type: wp.array(dtype=int, ndim=4),
    bc_norm: wp.array(dtype=vec3, ndim=4),
    bc_velo: wp.array(dtype=vec3, ndim=4),
    hf_bc_p: wp.array(dtype=vec3),
    hf_bc_n: wp.array(dtype=vec3),
    hf_bc_v: wp.array(dtype=vec3),
    hf_bc_type: wp.array(dtype=int),
    num_hf: wp.int32,
    dx: real,
) -> Tuple[wp.int32, vec3, vec3]:
    # x = vec3((real(node.x) + real(0.5)) * dx,
    #          (real(node.y) + real(0.5)) * dx,
    #          (real(node.z) + real(0.5)) * dx)
    x = dx * vec3(real(node.x), real(node.y), real(node.z))

    for i in range(num_hf):
        p = hf_bc_p[i]
        n = hf_bc_n[i]
        v = hf_bc_v[i]
        if wp.dot(x - p, n) <= real(0):
            t = wp.int32(hf_bc_type[i])
            return t, n, v

    bc_bcoords = block_coords_from_node(node.x, node.y, node.z)
    bc_bid = bc_block2bid[bc_bcoords.x, bc_bcoords.y, bc_bcoords.z]
    if bc_bid >= 0:
        bc_lijk = local_coords_in_block(node.x, node.y, node.z)
        bc_li, bc_lj, bc_lk = bc_lijk.x, bc_lijk.y, bc_lijk.z
        t = bc_type[bc_bid, bc_li, bc_lj, bc_lk]
        if t != 0:
            return t, bc_norm[bc_bid, bc_li, bc_lj, bc_lk], bc_velo[bc_bid, bc_li, bc_lj, bc_lk]
    
    return wp.int32(0), vec3(real(0)), vec3(real(0)) # not a boundary


@wp.kernel # project
def boundary_projection_kernel(
    n_active_nodes: wp.array(dtype=int, ndim=1),
    ndof2bijk: wp.array(dtype=wp.vec2i, ndim=1),
    block_xyz_by_id: wp.array(dtype=wp.vec3i, ndim=1),
    node_vector: wp.array(dtype=vec3, ndim=1),

    bc_block2bid: wp.array(dtype=int, ndim=3),
    bc_type: wp.array(dtype=int, ndim=4),
    bc_norm: wp.array(dtype=vec3, ndim=4),
    bc_velo: wp.array(dtype=vec3, ndim=4),

    hf_bc_p: wp.array(dtype=vec3, ndim=1),
    hf_bc_n: wp.array(dtype=vec3, ndim=1),
    hf_bc_v: wp.array(dtype=vec3, ndim=1),
    hf_bc_type: wp.array(dtype=int, ndim=1),
    num_hf: wp.int32,

    grid_size    : wp.vec3i,
    dx           : real,
):
    dof = wp.tid()
    if dof >= n_active_nodes[0]:
        node_vector[dof] = vec3(_0)
        return
    bid, local_n1d = ndof2bijk[dof].x, ndof2bijk[dof].y
    node = local2global(block_xyz_by_id[bid], local_n1d)
    if not (node.x >= 0 and node.x < grid_size.x and
            node.y >= 0 and node.y < grid_size.y and
            node.z >= 0 and node.z < grid_size.z):
        node_vector[dof] = vec3(_0)
        return

    bct, bcn, bcv = query_bc_sp(node, bc_block2bid, bc_type, bc_norm, bc_velo, hf_bc_p, hf_bc_n, hf_bc_v, hf_bc_type, num_hf, dx)
    if bct > 0:
        node_vector[dof] = proj_boundary_delta_vel(node_vector[dof], bct, bcn)
