"""Regenerate nj4_poses.json: random joints inside the NJ4 limits and their FK poses."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nj4 import LIMITS, H, P, make_robot

SEED = 20261013
N = 1000


def main() -> None:
    rng = np.random.default_rng(SEED)
    joints = rng.uniform(LIMITS[:, 0], LIMITS[:, 1], size=(N, 6))
    robot = make_robot(degrees=True)
    poses = robot.batch_forward(joints).as_matrix().reshape(N, 16)
    out = {
        "description": "Comau NJ4-220-3.0 poses from random joints inside the limits, "
        "flattened row-major 4x4 in mm, joints in degrees (model zero pose).",
        "seed": SEED,
        "model": {"h": H.tolist(), "p": P.tolist(), "offsets": [0.0] * 6, "flip_axes": [False] * 6},
        "limits_deg": LIMITS.tolist(),
        "joints_deg": joints.tolist(),
        "poses": poses.tolist(),
    }
    target = Path(__file__).with_name("nj4_poses.json")
    target.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {N} poses to {target}")


if __name__ == "__main__":
    main()
