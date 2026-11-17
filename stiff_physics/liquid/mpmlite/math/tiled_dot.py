import functools
import warp as wp
from ..types import *
from typing import Any, Type

def _as_2d_vec(x: wp.array) -> wp.array:
    # allow 1D (N,) -> (1, N)
    if x.ndim == 1:
        return x.reshape((1, -1))
    return x

def view2d(arr: wp.array, shape0: int, shape1: int) -> wp.array:
    # arr must be 2D contiguous in last dim (typical for wp.array2d)
    view = wp.array(
        ptr=arr.ptr,
        shape=(int(shape0), int(shape1)),
        dtype=arr.dtype,
        strides=arr.strides,
        device=arr.device,
    )
    view._ref = arr
    return view

@functools.lru_cache(maxsize=None)
def _create_vec_tiled_dot_kernels(tile_size: int, vec_dtype: Type, scalar_dtype: Type):
    """
    Build kernels specialized for vec_dtype (vec2 or vec3).
    Cached by (tile_size, vec_dtype).
    """
    @wp.kernel
    def block_dot_kernel(
        a: wp.array2d(dtype=vec_dtype),        # [C, N] but dtype is vec_dtype at runtime
        b: wp.array2d(dtype=vec_dtype),        # [C, N]
        partial: wp.array2d(dtype=scalar_dtype) # [C, num_blocks]
    ):
        col, block_id, _tid = wp.tid()
        start = block_id * tile_size

        a_tile = wp.tile_load(a[col], shape=tile_size, offset=start, bounds_check=True)
        b_tile = wp.tile_load(b[col], shape=tile_size, offset=start, bounds_check=True)

        # ✅ Built-in dot: works for both vec2 and vec3.
        prod = wp.tile_map(wp.dot, a_tile, b_tile)
        s = wp.tile_sum(prod)

        wp.tile_store(partial[col], s, offset=block_id)

    @wp.kernel
    def block_sum_kernel(
        data: wp.array2d(dtype=scalar_dtype),    # [C, L]
        partial: wp.array2d(dtype=scalar_dtype), # [C, ceil(L/tile)]
    ):
        col, block_id, _tid = wp.tid()
        start = block_id * tile_size

        t = wp.tile_load(data[col], shape=tile_size, offset=start, bounds_check=True)
        s = wp.tile_sum(t)
        wp.tile_store(partial[col], s, offset=block_id)

    return block_dot_kernel, block_sum_kernel


class VecTiledDot:
    """
    Tile-based dot product for vector arrays (vec2 or vec3):
        out[c] = sum_i dot(a[c,i], b[c,i])

    Notes:
    - vec_dtype must be vec2 or vec3 (or any Warp vector type that wp.dot supports).
    - scalar_type should match your solver scalar (wp.float32 or wp.float64).
    - Graph-friendly: uses recorded wp.Launch commands.
    """

    def __init__(
        self,
        max_length: int,
        vec_dtype: Type,
        scalar_type: Type,
        device=None,
        tile_size: int = 512,
        max_column_count: int = 1,
    ):
        self.max_length = int(max_length)
        self.device = device
        self.tile_size = int(tile_size)
        self.max_column_count = int(max_column_count)
        self.vec_dtype = vec_dtype
        self.scalar_type = scalar_type

        # Worst-case first stage blocks
        num_blocks0 = (max_length + self.tile_size - 1) // self.tile_size

        # ping-pong buffers: [2, C, num_blocks0]
        scratch = wp.empty((2, self.max_column_count, num_blocks0), dtype=self.scalar_type, device=self.device)
        self.partial_a = scratch[0]
        self.partial_b = scratch[1]

        # kernels specialized per (tile_size, vec_dtype)
        self.dot_kernel, self.sum_kernel = _create_vec_tiled_dot_kernels(self.tile_size, self.vec_dtype, self.scalar_type)

        # how many reduction rounds until length==1 (worst-case)
        rounds = 0
        length = num_blocks0
        while length > 1:
            length = (length + self.tile_size - 1) // self.tile_size
            rounds += 1
        self.rounds = rounds

        self._output = self.partial_a if (rounds % 2 == 0) else self.partial_b

        # Placeholders for recording (must have correct dtype/ndim)
        dummy_vec = wp.empty((self.max_column_count, 1), dtype=self.vec_dtype, device=self.device)

        self.dot_launch: wp.Launch = wp.launch(
            kernel=self.dot_kernel,
            dim=(self.max_column_count, num_blocks0, self.tile_size),
            device=self.device,
            inputs=(dummy_vec, dummy_vec, self.partial_a),
            block_dim=self.tile_size,
            record_cmd=True,
        )

        self.sum_launch: wp.Launch = wp.launch(
            kernel=self.sum_kernel,
            dim=(self.max_column_count, num_blocks0, self.tile_size),
            device=self.device,
            inputs=(self.partial_a, self.partial_b),
            block_dim=self.tile_size,
            record_cmd=True,
        )

    def compute(self, a: wp.array, b: wp.array, col_offset: int = 0) -> wp.array:
        """
        a, b:
          - 1D vector array: shape (N,) -> treated as (1, N)
          - 2D vector array: shape (C, N)

        Returns:
          device view of shape (C, 1) holding results.
        """
        a2 = _as_2d_vec(a)
        b2 = _as_2d_vec(b)

        if a2.shape != b2.shape:
            raise ValueError(f"shape mismatch: {a2.shape} vs {b2.shape}")

        # dtype check (helps catch mistakes early)
        if a2.dtype != self.vec_dtype or b2.dtype != self.vec_dtype:
            raise TypeError(f"expected dtype {self.vec_dtype}, got {a2.dtype} / {b2.dtype}")

        C, N = a2.shape
        if col_offset + C > self.max_column_count:
            raise ValueError("col_offset + C exceeds max_column_count")

        num_blocks = (N + self.tile_size - 1) // self.tile_size

        out0 = self.partial_a[col_offset : col_offset + C]
        out1 = self.partial_b[col_offset : col_offset + C]

        out0.zero_()
        out1.zero_()

        # Stage 1: block dot -> out0
        self.dot_launch.set_param_at_index(0, a2)
        self.dot_launch.set_param_at_index(1, b2)
        self.dot_launch.set_param_at_index(2, out0)
        self.dot_launch.set_dim((C, num_blocks, self.tile_size))
        self.dot_launch.launch()

        # Reduce: out0 -> out1 -> out0 -> ...
        data_in, data_out = out0, out1
        cur_blocks = num_blocks

        for _ in range(self.rounds):
            # data_out.zero_()
            L = cur_blocks
            cur_blocks = (L + self.tile_size - 1) // self.tile_size

            xin  = view2d(data_in,  C, L)          # ✅ actual length
            xout = view2d(data_out, C, cur_blocks) # ✅ output length

            self.sum_launch.set_param_at_index(0, xin)
            self.sum_launch.set_param_at_index(1, xout)
            self.sum_launch.set_dim((C, cur_blocks, self.tile_size))
            self.sum_launch.launch()

            data_in, data_out = data_out, data_in
            if cur_blocks <= 1:
                break

        return data_in[:, :1]

    def col(self, c: int = 0) -> wp.array:
        """Device scalar array shape (1,) for column c."""
        return self._output[c][:1]


# -------------------------
# Random self-checking tests
# -------------------------


def _np_dtype_for(scalar_type):
    return np.float64 if scalar_type == wp.float64 else np.float32


def _tol_for(scalar_type):
    # (rtol, machine factor for atol). float32 accumulates, so stay lenient;
    # atol is scaled by sum(|a*b|) at the call site.
    return (1e-9, 1e-12) if scalar_type == wp.float64 else (2e-3, 1e-5)


def _reference_dot(a_np, b_np):
    # a_np, b_np: (C, N, dim) -> returns (C,) with out[c] = sum_i dot(a[c,i], b[c,i])
    return np.einsum("cnd,cnd->c", a_np.astype(np.float64), b_np.astype(np.float64))


def _random_test(vec_dtype, scalar_type, dim, N, C, tile_size, seed=0):
    """Run one random (C, N) case through compute() and compare to numpy."""
    device = wp.get_preferred_device()
    rng = np.random.default_rng(seed)
    np_dtype = _np_dtype_for(scalar_type)

    # numpy (C, N, dim) -> warp array of shape (C, N) whose element dtype is the vector
    a_np = rng.standard_normal((C, N, dim)).astype(np_dtype)
    b_np = rng.standard_normal((C, N, dim)).astype(np_dtype)
    a = wp.array(a_np, dtype=vec_dtype, device=device)
    b = wp.array(b_np, dtype=vec_dtype, device=device)

    td = VecTiledDot(
        max_length=N,
        vec_dtype=vec_dtype,
        scalar_type=scalar_type,
        device=device,
        tile_size=tile_size,
        max_column_count=C,
    )

    got = td.compute(a, b).numpy().reshape(C).astype(np.float64)  # (C,)
    exp = _reference_dot(a_np, b_np)

    rtol, mfac = _tol_for(scalar_type)
    scale = float(np.sum(np.abs(a_np).astype(np.float64) * np.abs(b_np).astype(np.float64)))
    atol = mfac * scale + 1e-30
    ok = np.allclose(got, exp, rtol=rtol, atol=atol)

    name = getattr(vec_dtype, "__name__", str(vec_dtype))
    print(f"{'✓' if ok else '✗'} {name:8s} N={N:<7d} C={C} tile={tile_size:<4d} "
          f"max|err|={np.max(np.abs(got - exp)):.2e}")
    assert ok, f"mismatch: got={got} exp={exp}"


def _demo_pattern_test(vec_dtype, scalar_type, dim, N=5000, tile_size=512, seed=1):
    """Mirror the original demo: a·b -> col 0, a·a -> col 1, read back via .col()."""
    device = wp.get_preferred_device()
    rng = np.random.default_rng(seed)
    np_dtype = _np_dtype_for(scalar_type)

    a_np = rng.standard_normal((N, dim)).astype(np_dtype)
    b_np = rng.standard_normal((N, dim)).astype(np_dtype)
    a = wp.array(a_np, dtype=vec_dtype, device=device)  # (N,) -> treated as (1, N)
    b = wp.array(b_np, dtype=vec_dtype, device=device)

    td = VecTiledDot(
        max_length=N, vec_dtype=vec_dtype, scalar_type=scalar_type,
        device=device, tile_size=tile_size, max_column_count=2,
    )
    td.compute(a, b, col_offset=0)  # column 0 = a·b
    td.compute(a, a, col_offset=1)  # column 1 = a·a

    got_ab = float(td.col(0).numpy()[0])
    got_aa = float(td.col(1).numpy()[0])
    exp_ab = float(np.einsum("nd,nd->", a_np.astype(np.float64), b_np.astype(np.float64)))
    exp_aa = float(np.einsum("nd,nd->", a_np.astype(np.float64), a_np.astype(np.float64)))

    rtol, _ = _tol_for(scalar_type)
    ok = (abs(got_ab - exp_ab) <= rtol * abs(exp_ab) + 1e-3 and
          abs(got_aa - exp_aa) <= rtol * abs(exp_aa) + 1e-3)
    name = getattr(vec_dtype, "__name__", str(vec_dtype))
    print(f"{'✓' if ok else '✗'} {name:8s} demo-pattern N={N}: "
          f"a·b={got_ab:.6g} (exp {exp_ab:.6g}) | a·a={got_aa:.6g} (exp {exp_aa:.6g})")
    assert ok


def _run_all(vec_dtype, dim, scalar_type=real):
    # (N, C, tile_size): tiny / sub-tile / exact tile / just-over / multi-column /
    # small tile to force several reduction rounds / large N (multi-round @ tile=512)
    configs = [
        (1,      1, 512),
        (511,    1, 512),
        (512,    1, 512),
        (513,    1, 512),
        (5000,   3, 512),
        (5000,   3, 32),
        (300000, 2, 512),
    ]
    for (N, C, tile_size) in configs:
        _random_test(vec_dtype, scalar_type, dim, N, C, tile_size)
    _demo_pattern_test(vec_dtype, scalar_type, dim)


if __name__ == "__main__":
    wp.init()
    _run_all(vec2, dim=2)
    _run_all(vec3, dim=3)
    print("all random tests passed")
