#!/usr/bin/env python3
"""M5: programmatic joint2 drive test for case_27 softgripper hard-pin.

Monkey-patches polyscope so importing case_27 doesn't open a window, then
runs a custom step loop that drives left_arm_joint2 angle each frame.
Logs per-step timing + total elapsed.

Usage:
    cd /home/ps/Downloads/Stiff-GIPC-dailyv2
    PYTHONPATH=. /home/ps/Downloads/Stiff-GIPC/.venv/bin/python3 \
        examples/m5_drive_joint2_test.py [USE_HARD_PIN=1] [STEPS=30] [JOINT2_DELTA_DEG=1.0]

Env (also picked up directly from os.environ):
    USE_HARD_PIN=0/1
    FEM_BLOBAL=0/1
    STEPS=N
    JOINT2_DELTA_DEG=D
    JOINT2_NAME=left_arm_joint2 (substring match, case-insensitive)
"""
import sys, os, math, time, importlib

# Parse args of form KEY=VAL into env
for arg in sys.argv[1:]:
    if '=' in arg:
        k, v = arg.split('=', 1)
        os.environ[k] = v

# Ensure dailyv2 engine resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------ Monkey-patch polyscope so case_27 doesn't open a window --------
class _FakePsShow(Exception):
    """Raised inside case_27 main() when ps.show() is called, to bail out
    after engine setup and let us run our custom loop."""
    pass

import polyscope as ps
import polyscope.imgui as psim

# Bypass init/show but keep register_* working (they're no-ops in headless)
_orig_init = ps.init
def _no_init(*a, **kw):
    pass
ps.init = _no_init
ps.set_up_dir = lambda *a, **kw: None
ps.set_ground_plane_mode = lambda *a, **kw: None
ps.register_surface_mesh = lambda *a, **kw: type('m', (), {
    'set_color': lambda *a, **kw: None,
    'set_radius': lambda *a, **kw: None,
    'update_vertex_positions': lambda *a, **kw: None,
})()
ps.register_curve_network = lambda *a, **kw: type('c', (), {
    'set_color': lambda *a, **kw: None,
    'set_radius': lambda *a, **kw: None,
    'update_node_positions': lambda *a, **kw: None,
})()
ps.set_user_callback = lambda *a, **kw: None

# Captured engine + robot from case_27.main scope via globals trick:
captured = {'eng': None, 'robot': None}

# We hijack ps.show to grab `eng` and `robot` from caller's frame, then
# raise to bail.
def _show_capture():
    import inspect
    frame = inspect.currentframe().f_back
    # case_27 main() defines `eng` and `robot` locals
    captured['eng']   = frame.f_locals.get('eng')
    captured['robot'] = frame.f_locals.get('robot')
    raise _FakePsShow()
ps.show = _show_capture

# --- Run case_27 main ---
case27 = importlib.import_module("examples.case_27_mobile_s1_softgripper_cup")
print("[m5] running case_27 main() up to ps.show ...", flush=True)
try:
    case27.main()
except _FakePsShow:
    pass

eng   = captured['eng']
robot = captured['robot']
if eng is None or robot is None:
    print("[m5] FAILED to capture eng/robot from case_27.main", flush=True)
    sys.exit(1)

# --- Run programmatic drive loop ---
N = int(os.environ.get("STEPS", "30"))
DELTA_DEG = float(os.environ.get("JOINT2_DELTA_DEG", "1.0"))
JNAME = os.environ.get("JOINT2_NAME", "left_arm_joint2").lower()

# Find the joint by name substring
joint_idx = None
joint_obj = None
for i, ji in enumerate(robot.revolute_joints):
    if JNAME in ji.name.lower():
        joint_idx = i; joint_obj = ji
        break

if joint_idx is None:
    print(f"[m5] WARN no revolute joint name contains '{JNAME}', falling back to index 0", flush=True)
    joint_idx = 0; joint_obj = robot.revolute_joints[0]

cur_deg = robot.get_revolute_target_deg(joint_idx)
hi_deg = math.degrees(joint_obj.upper_limit)
lo_deg = math.degrees(joint_obj.lower_limit)
print(f"[m5] driving joint[{joint_idx}] {joint_obj.name} : "
      f"cur={cur_deg:.1f}°  range=[{lo_deg:.1f}, {hi_deg:.1f}]  delta_per_step={DELTA_DEG:.2f}°",
      flush=True)
print(f"[m5] USE_HARD_PIN={os.environ.get('USE_HARD_PIN','0')}  "
      f"FEM_BLOBAL={os.environ.get('FEM_BLOBAL','0')}  STEPS={N}",
      flush=True)

# Step loop
direction = 1
times_ms = []
for k in range(N):
    target = cur_deg + DELTA_DEG * direction
    if target >= hi_deg or target <= lo_deg:
        direction *= -1
        target = cur_deg + DELTA_DEG * direction
    cur_deg = target
    robot.set_revolute_position(joint_idx, cur_deg, degree=True)
    t0 = time.perf_counter()
    eng.step()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    times_ms.append(dt_ms)
    print(f"[m5] step {k:3d}  joint2={cur_deg:+7.2f}°  step_time={dt_ms:7.1f} ms",
          flush=True)

print(f"[m5] DONE  total={sum(times_ms)/1000:.1f}s  avg={sum(times_ms)/len(times_ms):.1f}ms  "
      f"max={max(times_ms):.1f}ms  min={min(times_ms):.1f}ms",
      flush=True)
