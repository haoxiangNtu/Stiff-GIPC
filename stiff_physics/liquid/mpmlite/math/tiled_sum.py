import functools
from typing import Type

import numpy as np
import warp as wp

from ..types import *


def _as_2d_vec(x: wp.array) -> wp.array:
    """Promote a 1D array of shape (N,) to a 2D array of shape (1, N)."""
    if x.ndim == 1:
        return x.reshape((1, -1))
    return x


def view2d(arr: wp.array, shape0: int, shape1: int) -> wp.array:
    """Return a 2D view of `arr` with the given shape, sharing its storage."""
    view = wp.array(
        ptr=arr.ptr,
        shape=(int(shape0), int(shape1)),
        dtype=arr.dtype,
        strides=arr.strides,
        device=arr.device,
    )
    view._ref = arr  # keep the backing array alive
    return view


@functools.lru_cache(maxsize=None)
def _create_vec_tiled_sum_kernel(tile_size: int, dtype: Type):
    """
    Build a block-sum kernel specialized for a given tile size and dtype.
    Cached by (tile_size, dtype).
    """

    @wp.kernel
    def block_sum_kernel(
        data: wp.array2d(dtype=dtype),     # [C, L]
        partial: wp.array2d(dtype=dtype),  # [C, ceil(L / tile_size)]
    ):
        col, block_id, _tid = wp.tid()
        start = block_id * tile_size

        t = wp.tile_load(data[col], shape=tile_size, offset=start, bounds_check=True)
        s = wp.tile_sum(t)
        wp.tile_store(partial[col], s, offset=block_id)

    return block_sum_kernel


class VecTiledSum:
    """
    Tile-based reduction (sum) for scalar or vector arrays:

        out[c] = sum_i a[c, i]

    For vector dtypes (e.g. vec2, vec3) the sum is computed component-wise.

    Notes:
    - `dtype` may be a scalar type (wp.float32 / wp.float64) or a Warp vector
      type that wp.tile_sum supports.
    - Graph-friendly: reuses a single recorded wp.Launch command.
    """

    def __init__(
        self,
        max_length: int,
        dtype: Type,
        device=None,
        tile_size: int = 512,
        max_column_count: int = 1,
    ):
        self.max_length = int(max_length)
        self.device = device
        self.tile_size = int(tile_size)
        self.max_column_count = int(max_column_count)
        self.dtype = dtype

        # Worst-case number of blocks in the first stage.
        num_blocks0 = (self.max_length + self.tile_size - 1) // self.tile_size

        # Ping-pong buffers: [2, C, num_blocks0].
        scratch = wp.empty(
            (2, self.max_column_count, num_blocks0),
            dtype=self.dtype,
            device=self.device,
        )
        self.partial_a = scratch[0]
        self.partial_b = scratch[1]

        # Kernel specialized per (tile_size, dtype).
        self.sum_kernel = _create_vec_tiled_sum_kernel(self.tile_size, self.dtype)

        # Worst-case number of reduction rounds until length == 1.
        rounds = 0
        length = num_blocks0
        while length > 1:
            length = (length + self.tile_size - 1) // self.tile_size
            rounds += 1
        self.rounds = rounds

        self._output = self.partial_a if (rounds % 2 == 0) else self.partial_b

        self.sum_launch: wp.Launch = wp.launch(
            kernel=self.sum_kernel,
            dim=(self.max_column_count, num_blocks0, self.tile_size),
            device=self.device,
            inputs=(self.partial_a, self.partial_b),
            block_dim=self.tile_size,
            record_cmd=True,
        )

    def compute(self, a: wp.array, col_offset: int = 0) -> wp.array:
        """
        Sum `a` along its last axis.

        Args:
            a: a 1D array of shape (N,) (treated as (1, N)) or a 2D array of
               shape (C, N).
            col_offset: column offset into the internal scratch buffers.

        Returns:
            A device view of shape (C, 1) holding the per-row sums.
        """
        a2 = _as_2d_vec(a)

        # dtype check (helps catch mistakes early).
        if a2.dtype != self.dtype:
            raise TypeError(f"expected dtype {self.dtype}, got {a2.dtype}")

        C, N = a2.shape
        if col_offset + C > self.max_column_count:
            raise ValueError("col_offset + C exceeds max_column_count")

        num_blocks = (N + self.tile_size - 1) // self.tile_size

        out0 = self.partial_a[col_offset : col_offset + C]
        out1 = self.partial_b[col_offset : col_offset + C]
        out0.zero_()
        out1.zero_()

        # Stage 1: block sum of the input -> out0.
        self.sum_launch.set_param_at_index(0, a2)
        self.sum_launch.set_param_at_index(1, out0)
        self.sum_launch.set_dim((C, num_blocks, self.tile_size))
        self.sum_launch.launch()

        # Iteratively reduce: out0 -> out1 -> out0 -> ...
        data_in, data_out = out0, out1
        cur_blocks = num_blocks

        for _ in range(self.rounds):
            L = cur_blocks
            cur_blocks = (L + self.tile_size - 1) // self.tile_size

            xin = view2d(data_in, C, L)             # current length
            xout = view2d(data_out, C, cur_blocks)  # reduced length

            self.sum_launch.set_param_at_index(0, xin)
            self.sum_launch.set_param_at_index(1, xout)
            self.sum_launch.set_dim((C, cur_blocks, self.tile_size))
            self.sum_launch.launch()

            data_in, data_out = data_out, data_in
            if cur_blocks <= 1:
                break

        return data_in[:, :1]

    def col(self, c: int = 0) -> wp.array:
        """Return the device scalar array of shape (1,) for column `c`."""
        return self._output[c][:1]


if __name__ == "__main__":
    wp.init()
    device = "cuda:0"

    # Vector (vec3d) reduction.
    ts = VecTiledSum(
        max_length=277773,
        dtype=wp.vec3d,
        device=device,
        tile_size=512,
        max_column_count=2,
    )

    for _ in range(3):
        a = 10 * np.random.rand(77773, 3).astype(np.float64)
        expected_sum = a.sum(0)
        a = wp.array(a, dtype=wp.vec3d, device=device)
        calculated_sum = ts.compute(a).numpy()[0]
        print("err:", np.linalg.norm(expected_sum - calculated_sum) / np.linalg.norm(expected_sum))

    # Scalar reduction.
    ts = VecTiledSum(
        max_length=277773,
        dtype=real,
        device=device,
        tile_size=512,
        max_column_count=2,
    )

    for _ in range(3):
        a = 10 * np.random.rand(77773).astype(np.float64)
        expected_sum = a.sum(0)
        a = wp.array(a, dtype=real, device=device)
        calculated_sum = ts.compute(a).numpy()[0][0]
        print("err:", np.abs(expected_sum - calculated_sum) / np.abs(expected_sum))