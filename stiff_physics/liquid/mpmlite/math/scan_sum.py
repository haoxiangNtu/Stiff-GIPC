"""
3D block prefix-sum (scan) used to hand out dense DOF indices to the
"active" cells of a sparse 3D grid.

Pipeline:
  * Flatten the [nx, ny, nz] active mask to 1D in C (row-major) order.
  * Run an inclusive prefix sum over it (hierarchical, tile based).
  * build_reverse_map_kernel turns the inclusive scan into a 0-based DOF id
    for every active cell and fills both the forward (cell -> dof) and the
    reverse (dof -> cell) maps.
"""

import warp as wp
import numpy as np

# Number of elements each tile (one thread block) processes.
TILE = wp.constant(256)


@wp.kernel
def scan_tiles_kernel(
    x: wp.array(dtype=wp.int32),
    y: wp.array(dtype=wp.int32),
    block_sums: wp.array(dtype=wp.int32),
    inclusive: int,
):
    """Scan one tile locally and emit that tile's total.

    Each thread block owns TILE consecutive elements. It writes the local
    (within-tile) scan into `y` and the tile's grand total into
    `block_sums[bid]`. The total comes from tile_sum, so it is correct
    regardless of whether the local scan is inclusive or exclusive.
    """
    bid = wp.tid()
    base = bid * TILE

    # Out-of-range lanes in the final partial tile are zero-padded, which
    # leaves both the scan of the valid lanes and the tile sum correct.
    t = wp.tile_load(x, shape=(TILE,), offset=(base,), storage="register", bounds_check=True)

    if inclusive != 0:
        s = wp.tile_scan_inclusive(t)
    else:
        s = wp.tile_scan_exclusive(t)

    wp.tile_store(y, s, offset=(base,), bounds_check=True)

    tsum = wp.tile_sum(t)
    wp.tile_store(block_sums, tsum, offset=(bid,), bounds_check=False)


@wp.kernel
def add_offsets_kernel(
    y: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
):
    """Add each tile's running offset to every element of that tile.

    `offsets[bid]` is the sum of all elements in tiles 0..bid-1, so adding it
    to locally-scanned tile `bid` promotes the local scan to a global one.
    Reading and writing `y` in place is safe: every block touches only its
    own disjoint slice.
    """
    bid = wp.tid()
    base = bid * TILE

    t = wp.tile_load(y, shape=(TILE,), offset=(base,), storage="register", bounds_check=True)

    off = offsets[bid]
    off_tile = wp.tile_full(shape=(TILE,), value=off, dtype=wp.int32, storage="register")

    out = wp.tile_map(wp.add, t, off_tile)
    wp.tile_store(y, out, offset=(base,), bounds_check=True)


@wp.kernel
def build_reverse_map_kernel(
    active: wp.array(dtype=wp.int32),
    scan: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    block2bid: wp.array3d(dtype=wp.int32),
    bdof2block: wp.array(dtype=wp.vec3i),
):
    """Turn the inclusive scan into DOF ids and fill the forward/reverse maps.

    Flattening is C-order (row-major): i = (x*ny + y)*nz + z, i.e. z is the
    fastest-varying axis. For an active cell the 0-based DOF id is the
    *exclusive* prefix, recovered from the inclusive scan as
    scan[i] - active[i].
    """
    ix, iy, iz = wp.tid()
    i = (ix * ny + iy) * nz + iz

    if active[i] > 0:
        dof = scan[i] - active[i]                 # inclusive -> exclusive
        block2bid[ix, iy, iz] = dof               # forward map: cell -> dof
        bdof2block[dof] = wp.vec3i(ix, iy, iz)     # reverse map: dof -> cell
    else:
        block2bid[ix, iy, iz] = -1                 # inactive cells get -1


def prefix_scan_sum(x: wp.array, inclusive: bool = True) -> wp.array:
    """Hierarchical prefix sum over a 1D int32 array.

    Classic "scan, then propagate", expressed as a single recursion:

      1. Split into tiles of TILE; scan_tiles_kernel produces the local scan
         `y` plus each tile's total in `block_sums`.
      2. One tile   -> `y` is already the answer.
      3. Many tiles -> recursively take the EXCLUSIVE scan of `block_sums`
         (an offset is "the sum of everything in earlier tiles"), then
         add_offsets_kernel folds those offsets back into `y`.

    Only this top-level call honours `inclusive`; every recursive call asks
    for an exclusive scan, because that is what tile offsets require.

    NOTE on the rewrite: the previous version mixed an explicit level-building
    `while` loop with a per-level recursive call. Both computed the *same*
    hierarchy, so the loop's work for every level above the bottom was thrown
    away and the upper levels were rebuilt ~O(levels^2) times. Collapsing it
    all into this one recursion gives identical results with linear work and
    far fewer kernel launches.
    """
    if x.ndim != 1:
        raise ValueError("only 1D arrays are supported")

    n = x.shape[0]
    if n == 0:
        return wp.empty(shape=(0,), dtype=wp.int32, device=x.device)

    num_tiles = (n + int(TILE) - 1) // int(TILE)

    y = wp.empty(shape=(n,), dtype=wp.int32, device=x.device)
    block_sums = wp.empty(shape=(num_tiles,), dtype=wp.int32, device=x.device)

    # Step 1: local scan inside each tile + per-tile totals.
    wp.launch_tiled(
        scan_tiles_kernel,
        dim=[num_tiles],
        inputs=[x, y, block_sums, 1 if inclusive else 0],
        block_dim=int(TILE),
        device=x.device,
    )

    # Step 2: a single tile is already fully scanned.
    if num_tiles == 1:
        return y

    # Step 3: offsets = exclusive scan of the tile totals, then fold them in.
    offsets = prefix_scan_sum(block_sums, inclusive=False)
    wp.launch_tiled(
        add_offsets_kernel,
        dim=[num_tiles],
        inputs=[y, offsets],
        block_dim=int(TILE),
        device=x.device,
    )
    return y


@wp.kernel
def get_total_kernel(
    active: wp.array(dtype=wp.int32),
    scan: wp.array(dtype=wp.int32),
    N: int,
    out: wp.array(dtype=wp.int32),
):
    """Read back the total active count from the inclusive scan.

    Single-threaded helper: walk backwards to the last active entry; its
    inclusive-scan value is the total number of active cells. (Utility; not
    exercised by the test below.)
    """
    out[0] = 0
    for i in range(N - 1, -1, -1):
        if active[i] > 0:
            out[0] = scan[i]
            break


class PrefixSum3D:
    """Prefix sum over a 3D block grid, used to allocate DOF indices."""

    def __init__(self, device=None):
        self.device = device

    def compute(
        self,
        active_mask: wp.array(dtype=int, ndim=3),  # [nx, ny, nz]
        bdof2block: wp.array(dtype=wp.vec3i, ndim=1),
        block2bid: wp.array(dtype=int, ndim=3),    # [nx, ny, nz]
    ):
        """Run the scan and build the maps.

        Fills:
            block2bid : [nx, ny, nz], DOF id (>= 0) for active cells, else -1
            bdof2block: [num_active] vec3i, DOF id -> (x, y, z)
        """
        if active_mask.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {active_mask.shape}")

        nx, ny, nz = active_mask.shape
        N = nx * ny * nz

        # Flatten to 1D in C-order (z fastest). reshape returns a view, so no
        # data is copied here.
        active_view = active_mask.reshape(N)

        # Inclusive prefix sum over the flattened mask.
        scan_result = prefix_scan_sum(active_view, inclusive=True)

        # Turn the scan into DOF ids and fill both maps.
        wp.launch(
            kernel=build_reverse_map_kernel,
            dim=(nx, ny, nz),
            device=self.device,
            inputs=(
                active_view,
                scan_result,
                nx, ny, nz,
                block2bid,
                bdof2block,
            ),
        )


# -------------------------
# Tests
# -------------------------

def _test_prefix_sum_3d():
    """Sanity-check the 3D prefix sum on a random grid."""
    wp.init()
    device = wp.get_preferred_device()

    nx, ny, nz = 61, 127, 57

    # Random activation pattern.
    active_np = np.random.randint(0, 2, size=(nx, ny, nz), dtype=np.int32)
    active_mask = wp.array(active_np, dtype=wp.int32, device=device)

    print(f"Grid: {nx}x{ny}x{nz}")
    print(f"Active blocks: {active_np.sum()} / {nx * ny * nz}")

    ps3d = PrefixSum3D(device=device)

    # Upper bound on DOFs is "every cell active".
    bdof2block = wp.empty(nx * ny * nz, dtype=wp.vec3i, device=device)
    block2bid = wp.empty((nx, ny, nz), dtype=wp.int32, device=device)
    block2bid.fill_(-1)

    with wp.ScopedTimer("compute", cuda_filter=wp.TIMING_ALL):
        ps3d.compute(active_mask, bdof2block, block2bid)

    b2d_np = block2bid.numpy()
    bd2b_np = bdof2block.numpy()

    print(b2d_np.shape)
    print(bd2b_np.shape)

    print("\nValidation:")
    print(f"Allocated DOFs: {(b2d_np >= 0).sum()}")
    print(f"Expected DOFs : {active_np.sum()}")

    # DOF ids of the active cells should be exactly 0..k-1.
    active_blocks = np.argwhere(active_np > 0)
    dof_list = []
    for x, y, z in active_blocks:
        dof = b2d_np[x, y, z]
        if dof < 0:
            print(f"✗ active block ({x},{y},{z}) has DOF = {dof}")
        else:
            dof_list.append(dof)

    if sorted(dof_list) == list(range(len(dof_list))):
        print(f"✓ DOFs are contiguous 0..{len(dof_list) - 1}")
    else:
        print("✗ DOF allocation is wrong")

    n_dofs = active_np.sum()

    # The reverse map must invert the forward map.
    errors = 0
    for dof in range(n_dofs):
        bx, by, bz = bd2b_np[dof]
        if b2d_np[bx, by, bz] != dof:
            errors += 1

    if errors == 0:
        print("✓ reverse map is correct")
    else:
        print(f"✗ reverse map has {errors} errors")

    # Inactive cells must stay -1.
    inactive_blocks = np.argwhere(active_np == 0)
    for x, y, z in inactive_blocks[:5]:
        dof = b2d_np[x, y, z]
        if dof >= 0:
            print(f"✗ inactive block ({x},{y},{z}) has DOF = {dof}")


def _benchmark():
    """Throughput sweep over a few grid sizes."""
    wp.init()
    device = wp.get_preferred_device()

    sizes = [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256), (291, 313, 271)]

    print("Benchmark:")
    print("-" * 60)

    for nx, ny, nz in sizes:
        N = nx * ny * nz

        active_np = np.random.randint(0, 2, size=(nx, ny, nz), dtype=np.int32)
        active_mask = wp.array(active_np, dtype=wp.int32, device=device)

        ps3d = PrefixSum3D(device=device)
        bdof2block = wp.empty(N, dtype=wp.vec3i, device=device)
        block2bid = wp.empty((nx, ny, nz), dtype=wp.int32, device=device)

        # Warm-up.
        ps3d.compute(active_mask, bdof2block, block2bid)

        import time
        wp.synchronize()
        t0 = time.perf_counter()

        num_runs = 10
        for _ in range(num_runs):
            ps3d.compute(active_mask, bdof2block, block2bid)

        wp.synchronize()
        t1 = time.perf_counter()

        avg_ms = (t1 - t0) / num_runs * 1000.0
        throughput = N / (avg_ms / 1000.0) / 1e6  # M elements/s

        print(f"{nx}x{ny}x{nz} ({N:>9} elements): "
              f"{avg_ms:6.2f} ms  ({throughput:6.1f} M elem/s)")


if __name__ == "__main__":
    _test_prefix_sum_3d()
    print("\n" + "=" * 60 + "\n")
    _benchmark()