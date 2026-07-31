#!/usr/bin/env python3
"""G17e: per-env masked GPU-RL reset.

Two independent FEM cloths fall under gravity in one merged world. After a few
GPU-native frames, a device-side int32 mask resets ONLY env 0 to the
prepare-time snapshot. The gate asserts bitwise: env 0 returns exactly to the
snapshot, env 1 keeps exactly its evolved state. The mask lives in device
memory, so an RL loop can flip it from a done-flag kernel with zero host
transfers.
"""
import ctypes
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class Cuda:
    def __init__(self) -> None:
        for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise RuntimeError("libcudart not found")
        self.D2H, self.H2D = 2, 1

    def check(self, status: int, what: str) -> None:
        assert status == 0, f"{what} -> cudaError {status}"

    def malloc(self, size: int) -> int:
        pointer = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(pointer), size), "malloc")
        return pointer.value

    def memcpy(self, dst: int, src: int, size: int, kind: int) -> None:
        self.check(
            self.lib.cudaMemcpy(
                ctypes.c_void_p(dst), ctypes.c_void_p(src), size, kind
            ),
            "memcpy",
        )

    def read_ints(self, pointer: int, count: int) -> np.ndarray:
        out = np.empty(count, dtype=np.int32)
        self.memcpy(out.ctypes.data, pointer, count * 4, self.D2H)
        return out

    def write_ints(self, pointer: int, values: np.ndarray) -> None:
        v = np.ascontiguousarray(values, dtype=np.int32)
        self.memcpy(pointer, v.ctypes.data, v.nbytes, self.H2D)

    def sync(self) -> None:
        self.check(self.lib.cudaDeviceSynchronize(), "sync")


def main() -> None:
    from stiff_physics.engine import Config, Engine

    cfg = Config(
        dt=0.01,
        gravity=(0.0, -9.8, 0.0),
        ground_offset=0.0,
        cloth_thickness=1e-3,
        cloth_young_modulus=1e4,
        bend_young_modulus=1e3,
        cloth_density=200,
        strain_rate=100,
        poisson_rate=0.49,
        friction_rate=0.4,
        relative_dhat=1e-3,
        assets_dir=os.path.join(ROOT, "Assets") + "/",
        multienv_mode="merged",
        skip_all_collision=False,
    )
    engine = Engine(cfg)
    for x in (-0.6, 0.6):
        tf = np.eye(4)
        tf[:3, :3] *= 0.4
        tf[0, 3] = x
        tf[1, 3] = 0.05
        engine.load_mesh(
            "triMesh/cloth_30x30.obj",
            dimensions=2,
            body_type="FEM",
            transform=tf,
        )
    n_total = 961 * 2
    engine.finalize()
    engine.set_vertex_env_ids([0] * 961 + [1] * 961)
    engine.native.set_log_level(0)
    engine.step()  # warm every lazy workspace before the episode graph

    cuda = Cuda()
    engine.native.prepare_gpu_rl()
    abi = engine.native.get_gpu_rl_device_abi()
    d_pos = int(abi["positions"])

    def positions() -> np.ndarray:
        out = np.empty(n_total * 3, dtype=np.float64)
        cuda.memcpy(out.ctypes.data, d_pos, out.nbytes, cuda.D2H)
        return out.reshape(n_total, 3)
    # get_vertices returns load order (the multienv examples slice it by
    # contiguous per-mesh ranges), so env 0 is the first cloth's 961 rows.
    env0 = np.zeros(n_total, dtype=bool)
    env0[:961] = True
    env1 = ~env0
    p_start = np.asarray(engine.get_vertices(), dtype=np.float64).copy()

    for _ in range(5):
        engine.native.launch_gpu_rl_async(0)
    cuda.sync()
    # get_vertices is episode-stale by design on the D path; the ABI slot is
    # the device truth (a D2D publish of mesh positions each launch).
    p_moved = positions()
    st = np.empty(int(abi["status_bytes"]), dtype=np.uint8)
    cuda.memcpy(st.ctypes.data, int(abi["statuses"]), st.nbytes, cuda.D2H)
    import struct as _st
    print(f"[dbg] last frame result={_st.unpack_from('<i', st, 0)[0]} "
          f"invalid=0x{_st.unpack_from('<I', st, 8)[0]:x}")
    fell = float(np.abs(p_moved - p_start).max())
    assert fell > 1e-5, f"cloths did not move ({fell:.3e})"

    d_mask = cuda.malloc(8)
    cuda.write_ints(d_mask, np.array([int(os.environ.get("MASK0","1")), int(os.environ.get("MASK1","0"))], dtype=np.int32))
    engine.launch_gpu_rl_reset_masked_async(d_mask, 0)
    cuda.sync()
    # End the episode so the getters re-attach to the live mesh arrays; the
    # engine keeps the current (post-reset) state for subsequent step()s.
    engine.native.end_gpu_rl()
    p_after = np.asarray(engine.get_vertices(), dtype=np.float64).copy()

    err0 = float(np.abs(p_after[env0] - p_start[env0]).max())
    keep1 = float(np.abs(p_after[env1] - p_moved[env1]).max())
    print(
        f"GPU-RL-MASK-RESET: env0 |after-snapshot|={err0:.3e} "
        f"env1 |after-evolved|={keep1:.3e} moved={fell:.3e}"
    )
    assert err0 == 0.0, "reset env did not return bitwise to the snapshot"
    assert keep1 == 0.0, "unmasked env was perturbed by the masked reset"

    engine.step()  # the mixed post-reset state must be steppable
    print("GPU-RL-MASK-RESET-GATE: PASS")


if __name__ == "__main__":
    main()
