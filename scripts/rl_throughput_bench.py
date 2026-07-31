#!/usr/bin/env python3
"""Phase-D throughput: episode launches vs host step() on the D4 contact scene."""
import importlib.util, os, sys, time
import numpy as np
ROOT = "/home/ps/Downloads/Stiff-GIPC-c1-ls-graph"
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location(
    "d4", os.path.join(ROOT, "scripts", "gpu_rl_contact_steady.py"))
d4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d4)

N = int(os.environ.get("BENCH_STEPS", "150"))
rev, pri = d4.actions(N + 20)

# --- host step() ---
eng = d4.make_engine()
for i in range(5):
    eng.set_revolute_target(0, float(rev[i, 0]))
    eng.set_prismatic_target(0, float(pri[i, 0]))
    eng.step()
t0 = time.perf_counter()
for i in range(5, 5 + N):
    eng.set_revolute_target(0, float(rev[i, 0]))
    eng.set_prismatic_target(0, float(pri[i, 0]))
    eng.step()
host_ms = (time.perf_counter() - t0) * 1000.0 / N
eng.reset()

# --- episode launches ---
eng = d4.make_engine()
cuda = d4.Cudart()
eng.prepare_gpu_rl()
abi = eng.get_gpu_rl_device_abi()
# per-step bytes: joints x 3 float64 components (D4 has 1 rev + 1 pri joint,
# actions() returns (N,3) per family)
rb = h_rev_shape = 3 * 8
pb = 3 * 8
h_rev = np.ascontiguousarray(rev); h_pri = np.ascontiguousarray(pri)
def publish(i):
    cuda.memcpy(int(abi["revolute_actions"]), h_rev[i].ctypes.data, rb, 1)
    cuda.memcpy(int(abi["prismatic_actions"]), h_pri[i].ctypes.data, pb, 1)
for i in range(5):
    publish(i); eng.launch_gpu_rl_async(0)
eng.synchronize_gpu_rl()
t0 = time.perf_counter()
for i in range(5, 5 + N):
    publish(i); eng.launch_gpu_rl_async(0)
eng.synchronize_gpu_rl()
ep_ms = (time.perf_counter() - t0) * 1000.0 / N
print(f"RL-THROUGHPUT: host_step={host_ms:.2f} ms/frame  "
      f"episode={ep_ms:.2f} ms/frame  speedup={host_ms/ep_ms:.2f}x  (N={N})")
