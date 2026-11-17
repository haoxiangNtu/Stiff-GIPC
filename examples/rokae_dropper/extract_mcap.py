"""Convert the teleoperation recording (ROS 2 mcap) into a small npz the simulation replays.

    pip install mcap mcap-ros2-support
    python extract_mcap.py Rokae_AR5_260615_145413_0.mcap --out Assets/rokae_wuji/teleop_dropper.npz

Topics used:
  io_teleop/joint_states              measured arm joints, 14 = left 1..7 + right 1..7 (100 Hz)
  /io_teleop/joint_cmd_finger_left    commanded Wuji finger joints, 20 (about 70 Hz; the hands publish no state)
  /io_teleop/joint_cmd_finger_right
Time is relative to the first arm state message.
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mcap")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    topics = {"io_teleop/joint_states": "arm", "/io_teleop/joint_cmd_finger_left": "left",
              "/io_teleop/joint_cmd_finger_right": "right"}
    data = {k: [] for k in topics.values()}
    names = {}
    with open(args.mcap, "rb") as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, ch, msg, dec in r.iter_decoded_messages(topics=list(topics)):
            k = topics[ch.topic]
            names.setdefault(k, list(dec.name))
            data[k].append([msg.log_time * 1e-9] + list(dec.position))
    t0 = data["arm"][0][0]
    out = {}
    for k, rows in data.items():
        a = np.asarray(rows, np.float64)
        out[f"{k}_t"] = a[:, 0] - t0
        out[f"{k}_q"] = a[:, 1:]
        out[f"{k}_names"] = np.array(names[k])
        print(f"{k}: {len(a)} samples, {a[-1, 0] - a[0, 0]:.1f} s, {a.shape[1] - 1} joints")
    np.savez_compressed(args.out, **out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
