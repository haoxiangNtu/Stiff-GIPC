import warp as wp
from .types import *

# ----------------------------
# block-sparse grid (improved)
# ----------------------------
B = 32 # !!!!!
BLOCK_VOL = B * B * B

@wp.func
def lin_JK(j: int, k: int) -> int:
    return (j << 5) | k

@wp.func
def lin_I_JK(i: int, jk: int) -> int:
    return (i << 10) | jk

@wp.func
def lin_IJK(i: int, j: int, k: int) -> int:
    return (i << 10) | (j << 5) | k

@wp.func
def unlin_JK(jk: int) -> Tuple[int, int]:
    j = (jk >> 5) & 31
    k = jk & 31
    return j, k

@wp.func
def unlin_IJK(ijk: int) -> Tuple[int, int, int]:
    i = (ijk >> 10) & 31
    j = (ijk >> 5) & 31
    k = ijk & 31
    return i, j, k

@wp.func
def unlin_I_JK(ijk: int) -> Tuple[int, int]:
    i = (ijk >> 10) & 31
    jk = ijk & 0x3FF
    return i, jk

@wp.func
def block_coords_from_node(ix: int, iy: int, iz: int) -> wp.vec3i:
    # ix//64 == ix>>6
    # return wp.vec3i(ix // B, iy // B, iz // B)
    return wp.vec3i(ix >> 5, iy >> 5, iz >> 5)
    # return wp.vec3i(ix >> 3, iy >> 3, iz >> 3)


@wp.func
def local_coords_in_block(ix: int, iy: int, iz: int) -> wp.vec3i:
    # # Use subtract with multiply to avoid slow modulo; same as before
    # bx = ix // B; by = iy // B; bz = iz // B
    # return wp.vec3i(ix - bx * B, iy - by * B, iz - bz * B)
    # ix % 32 == ix & 31
    return wp.vec3i(ix & 31, iy & 31, iz & 31)
    # return wp.vec3i(ix & 7, iy & 7, iz & 7)

@wp.func
def local_coords_in_block3(ix: int, iy: int, iz: int) -> Tuple[int, int, int]:
    # # Use subtract with multiply to avoid slow modulo; same as before
    # bx = ix // B; by = iy // B; bz = iz // B
    # return wp.vec3i(ix - bx * B, iy - by * B, iz - bz * B)
    # ix % 32 == ix & 31
    return ix & 31, iy & 31, iz & 31
    # return wp.vec3i(ix & 7, iy & 7, iz & 7)

@wp.func
def local_linear(local_coords: wp.vec3i) -> int:
    lx = local_coords.x; ly = local_coords.y; lz = local_coords.z
    # For B=32, this could be bit shifts, but keep generic:
    # return lx + B * (ly + B * lz)
    # lx + 32*(ly + 32*lz) == lx + (ly<<5) + (lz<<10)
    return (lx << 10) | (ly << 5) | lz
    # return (lx << 6) + (ly << 3) + lz


@wp.func
def local2global(block_xyz: wp.vec3i, local1d: int) -> wp.vec3i:
    lx = (local1d >> 10) & 31
    ly = (local1d >> 5) & 31
    lz = local1d & 31
    return wp.vec3i(
        (block_xyz.x << 5) + lx,
        (block_xyz.y << 5) + ly,
        (block_xyz.z << 5) + lz
    )
    # lx = local1d >> 6 
    # ly = (local1d >> 3) & 7
    # lz = local1d & 7
    # return wp.vec3i(
    #     (block_xyz.x << 3) + lx,
    #     (block_xyz.y << 3) + ly,
    #     (block_xyz.z << 3) + lz
    # )


@wp.kernel
def activate_blocks(
    x: wp.array(dtype=vec3),
    dx: real,
    grid_size: wp.vec3i,
    block_activated: wp.array(dtype=int, ndim=3),
):
    p = wp.tid()

    pos_c = x[p] / dx - vec3(real(0.5))
    base = wp.vec3i(int(wp.floor(pos_c[0])), int(wp.floor(pos_c[1])), int(wp.floor(pos_c[2])))

    for gx in range(3):
        ix = base[0] + gx
        if ix < 0 or ix >= grid_size[0]: continue
        for gy in range(3):
            iy = base[1] + gy
            if iy < 0 or iy >= grid_size[1]: continue
            for gz in range(3):
                iz = base[2] + gz
                if iz < 0 or iz >= grid_size[2]: continue

                bc = block_coords_from_node(ix, iy, iz)
                block_activated[bc.x, bc.y, bc.z] = 1

@wp.kernel
def activate_blocks_by_ijk(
    ijk: wp.array(dtype=wp.vec3i),
    n_ijk: int,
    block_activated: wp.array(dtype=int, ndim=3),
):
    thread_idx = wp.tid()
    if thread_idx >= n_ijk: return
    node_ijk = ijk[thread_idx]

    bc = block_coords_from_node(node_ijk[0], node_ijk[1], node_ijk[2])
    block_activated[bc.x, bc.y, bc.z] = 1

@wp.kernel
def get_active_block_num(
    block_activated: wp.array(dtype=int, ndim=3),
    block_count: wp.array(dtype=int, ndim=1),
):
    i, j, k = wp.tid()
    if block_activated[i, j, k] > 0:
        wp.atomic_add(block_count, 0, 1)