import os, numpy as np
os.environ.setdefault("CASE39ME_HEADLESS", "1")
import sys; sys.path.insert(0, os.getcwd())
from stiff_physics.engine import Engine, Config
import stiff_physics.engine as em
cfg = Config(assets_dir="Assets/")
e = Engine(cfg)
e.load_mesh("triMesh/cloth_30x30.obj", dimensions=2, body_type="FEM", young_modulus=1e4)
e.finalize()
acts = np.random.rand(10, 4)
ptr = e.upload_episode_actions(acts)
assert ptr != 0
seen = []
def on_obs(f, obs):
    assert obs.shape[1] == 3 and np.isfinite(obs).all()
    seen.append((f, float(obs[:, 1].mean())))
e.episode_loop(10, on_obs=on_obs)
assert len(seen) == 10 and seen[0][0] == 0 and seen[-1][0] == 9
ys = [y for _, y in seen]
assert ys[-1] < ys[0], f"cloth should fall: {ys[0]} -> {ys[-1]}"
print("OBS-SMOKE PASS frames=%d y0=%.4f y9=%.4f" % (len(seen), ys[0], ys[-1]))
