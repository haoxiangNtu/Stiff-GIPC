"""LeRobotDataset export: multimodal sim episodes -> the standard
robot-learning dataset, so simulated tactile data drops straight into LeRobot
training workflows (the same output the commercial handheld data-capture rigs
produce).

Thin wrapper over the OFFICIAL ``LeRobotDataset.create / add_frame /
save_episode`` API (lerobot >= 0.4, dataset format v3.0): the library writes
its own parquet/video layout, so the format can never drift from upstream.
H.264 is used for the videos (this machine's ffmpeg has no SVT-AV1).

Sim frames are synchronous by construction — every modality is sampled at the
same engine step — which is the property the hardware rigs advertise their
synchronised timelines for.

    w = LeRobotWriter(root, fps=200, robot_type="sim_parallel_gripper",
                      state_names=[...], action_names=[...],
                      video_keys=["observation.images.tactile_left"],
                      image_shape=(504, 252))
    w.begin_episode("grasp and lift the M20 bolt")
    w.add_frame(state_vec, action_vec, {"observation.images.tactile_left": img})
    w.end_episode()
    w.finish()          # -> root, loadable by LeRobotDataset(root=...)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


class LeRobotWriter:
    def __init__(self, root: str, fps: float, robot_type: str,
                 state_names: list[str], action_names: list[str],
                 video_keys: list[str], image_shape: tuple[int, int]):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = Path(root)
        if root.exists():
            shutil.rmtree(root)
        h, w = image_shape
        feats = {
            "observation.state": dict(dtype="float32",
                                      shape=(len(state_names),),
                                      names=state_names),
            "action": dict(dtype="float32", shape=(len(action_names),),
                           names=action_names),
        }
        for k in video_keys:
            feats[k] = dict(dtype="video", shape=(h, w, 3),
                            names=["height", "width", "channels"])
        self.video_keys = list(video_keys)
        self.ds = LeRobotDataset.create(
            repo_id=f"sim/{Path(root).name}", fps=int(round(fps)),
            features=feats, root=root, robot_type=robot_type,
            use_videos=True, vcodec="h264")
        self._task = None

    def begin_episode(self, task: str):
        self._task = task

    def add_frame(self, state, action, images: dict):
        frame = {
            "observation.state": np.asarray(state, np.float32),
            "action": np.asarray(action, np.float32),
            "task": self._task,
        }
        for k in self.video_keys:
            im = np.asarray(images[k])
            assert im.dtype == np.uint8 and im.ndim == 3 and im.shape[2] == 3
            frame[k] = im
        self.ds.add_frame(frame)

    def end_episode(self):
        self.ds.save_episode()

    def finish(self) -> str:
        self.ds.finalize()
        return str(self.ds.root)
