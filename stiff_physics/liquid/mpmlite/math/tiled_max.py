import functools
from typing import Type

import warp as wp

from ..types import *


@functools.lru_cache(maxsize=None)
def _create_vec_absmax_kernels(tile_size: int, vec_dtype: Type, scalar_dtype: Type):
    """Build the stage-0 (per-vector abs-max) and reduction kernels for a given
    vector type, scalar type and tile size.

    Cached so each unique (tile_size, vec_dtype, scalar_dtype) combination is
    only compiled once.
    """

    @wp.kernel
    def block_vec2_absmax_kernel(
        x: wp.array(dtype=vec_dtype),           # [N]
        partial: wp.array(dtype=scalar_dtype),  # [num_blocks]
        N: int,
        neg_inf: scalar_dtype,
    ):
        block_id, tid = wp.tid()
        idx = block_id * tile_size + tid

        v = neg_inf
        if idx < N:
            a = x[idx]
            v = wp.max(wp.abs(a[0]), wp.abs(a[1]))

        t = wp.tile(v)
        m = wp.tile_max(t)

        if tid == 0:
            wp.tile_store(partial, m, offset=block_id)

    @wp.kernel
    def block_vec3_absmax_kernel(
        x: wp.array(dtype=vec_dtype),           # [N]
        partial: wp.array(dtype=scalar_dtype),  # [num_blocks]
        N: int,
        neg_inf: scalar_dtype,
    ):
        block_id, tid = wp.tid()
        idx = block_id * tile_size + tid

        v = neg_inf
        if idx < N:
            a = x[idx]
            v = wp.max(wp.max(wp.abs(a[0]), wp.abs(a[1])), wp.abs(a[2]))

        t = wp.tile(v)
        m = wp.tile_max(t)

        if tid == 0:
            wp.tile_store(partial, m, offset=block_id)

    @wp.kernel
    def block_max_reduce_kernel(
        data: wp.array(dtype=scalar_dtype),     # [L]
        partial: wp.array(dtype=scalar_dtype),  # [ceil(L / tile_size)]
        L: int,
        neg_inf: scalar_dtype,
    ):
        block_id, tid = wp.tid()
        idx = block_id * tile_size + tid

        v = neg_inf
        if idx < L:
            v = data[idx]

        t = wp.tile(v)
        m = wp.tile_max(t)

        if tid == 0:
            wp.tile_store(partial, m, offset=block_id)

    stage0_kernel = block_vec2_absmax_kernel if vec_dtype == vec2 else block_vec3_absmax_kernel
    return stage0_kernel, block_max_reduce_kernel


class VecTiledAbsMax:
    """Compute the maximum absolute component over an array of vec2/vec3.

    For ``x`` of shape ``(N,)`` this returns::

        max_i max(|x[i][0]|, |x[i][1]|, |x[i][2]|)

    (the third component is only used for vec3). The reduction runs entirely on
    the device using a tiled, multi-pass max reduction with two ping-pong
    scratch buffers.
    """

    def __init__(self, max_length: int, vec_dtype: Type, scalar_type: Type, device=None, tile_size: int = 512):
        self.device = device
        self.tile_size = int(tile_size)
        self.max_length = int(max_length)
        self.vec_dtype = vec_dtype
        self.scalar_type = scalar_type

        num_blocks0 = (self.max_length + self.tile_size - 1) // self.tile_size

        # Two scratch buffers used as ping-pong storage during the reduction.
        scratch = wp.empty((2, num_blocks0), dtype=self.scalar_type, device=self.device)
        self.partial_a = scratch[0]
        self.partial_b = scratch[1]

        self.stage0_kernel, self.reduce_kernel = _create_vec_absmax_kernels(
            self.tile_size, self.vec_dtype, self.scalar_type
        )

        # Sentinel value used for out-of-range lanes.
        if self.scalar_type in (wp.float16, wp.float32, wp.float64):
            self.neg_inf = self.scalar_type(-wp.inf)
        else:
            self.neg_inf = self.scalar_type(-(2**30))

        # Worst-case number of reduction passes needed to collapse the stage-0
        # output (num_blocks0 values) down to a single value.
        rounds = 0
        L = num_blocks0
        while L > 1:
            L = (L + self.tile_size - 1) // self.tile_size
            rounds += 1
        self.rounds = rounds

        # Buffer holding the final scalar after compute(); updated on each call.
        self._output = self.partial_a

        # Pre-record both launches so repeated compute() calls only patch
        # parameters instead of re-recording the command.
        dummy = wp.empty((1,), dtype=self.vec_dtype, device=self.device)

        self.stage0_launch: wp.Launch = wp.launch(
            kernel=self.stage0_kernel,
            dim=(num_blocks0, self.tile_size),
            device=self.device,
            inputs=(dummy, self.partial_a, 1, self.neg_inf),
            block_dim=self.tile_size,
            record_cmd=True,
        )

        self.reduce_launch: wp.Launch = wp.launch(
            kernel=self.reduce_kernel,
            dim=(num_blocks0, self.tile_size),
            device=self.device,
            inputs=(self.partial_a, self.partial_b, 1, self.neg_inf),
            block_dim=self.tile_size,
            record_cmd=True,
        )

    def compute(self, x: wp.array) -> wp.array:
        if x.dtype != self.vec_dtype:
            raise TypeError(f"expected dtype {self.vec_dtype}, got {x.dtype}")
        if x.ndim != 1:
            raise ValueError("expected a 1D array of vec2/vec3 (shape (N,))")

        N = x.shape[0]
        if N > self.max_length:
            raise ValueError(f"input length {N} exceeds max_length {self.max_length}")

        num_blocks = (N + self.tile_size - 1) // self.tile_size

        out0, out1 = self.partial_a, self.partial_b
        out0.zero_()
        out1.zero_()

        # Stage 0: per-vector abs-max, producing one value per block.
        self.stage0_launch.set_param_at_index(0, x)
        self.stage0_launch.set_param_at_index(1, out0)
        self.stage0_launch.set_param_at_index(2, int(N))
        self.stage0_launch.set_param_at_index(3, self.neg_inf)
        self.stage0_launch.set_dim((num_blocks, self.tile_size))
        self.stage0_launch.launch()

        # Reduction: repeatedly collapse the block values down to a single one.
        data_in, data_out = out0, out1
        cur = num_blocks
        for _ in range(self.rounds):
            L = cur
            cur = (L + self.tile_size - 1) // self.tile_size

            self.reduce_launch.set_param_at_index(0, data_in)
            self.reduce_launch.set_param_at_index(1, data_out)
            self.reduce_launch.set_param_at_index(2, int(L))
            self.reduce_launch.set_param_at_index(3, self.neg_inf)
            self.reduce_launch.set_dim((cur, self.tile_size))
            self.reduce_launch.launch()

            data_in, data_out = data_out, data_in
            if cur <= 1:
                break

        self._output = data_in
        return data_in[:1]  # shape (1,)

    def value(self) -> float:
        """Return the scalar result of the most recent compute() as a float."""
        return float(self._output.numpy()[0])


if __name__ == "__main__":
    wp.init()
    device = wp.get_preferred_device()

    N = 262144 * 8

    # Random vec2 or vec3 input (2 -> vec2, 3 -> vec3).
    components = np.random.randint(2, 4)
    x_np = (np.random.randn(N, components) * 10.0).astype(scalar)
    vectype = vec2 if components == 2 else vec3
    x = wp.array(x_np, dtype=vectype, device=device)

    op = VecTiledAbsMax(max_length=N, vec_dtype=vectype, scalar_type=real, device=device, tile_size=512)
    out = op.compute(x)

    got = out.numpy()[0]
    expected = np.max(np.abs(x_np))
    print("got:", got, "expected:", expected)