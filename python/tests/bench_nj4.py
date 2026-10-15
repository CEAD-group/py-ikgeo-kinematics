"""
Benchmark of the public API on the Comau NJ4-220: batch_forward, cold inverse, tracked
batch_inverse on a printing path, and reach single and multi threaded.

Not collected by pytest; run with ``uv run python python/tests/bench_nj4.py``.
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import RigidTransform, Rotation

sys.path.insert(0, str(Path(__file__).parent))
from nj4 import LIMITS, make_robot

HOME = (0.0, 0.0, 0.0, 0.0, 90.0, 0.0)
RETRACTS = 5


def timed(fn, repeats: int = 3):
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def printing_path() -> tuple[RigidTransform, list[int]]:
    """50 layers of two 1 m lines at 1 mm pitch, tool down, five 300 mm retracts."""
    down = Rotation.from_euler("y", 90, degrees=True)
    points = []
    retracts = []
    for layer in range(50):
        z = 600.0 + 2.0 * layer
        y0 = -200.0 + 8.0 * layer
        xs = 1300.0 + np.arange(1000)
        points.append(np.stack([xs, np.full(1000, y0), np.full(1000, z)], axis=1))
        points.append(np.stack([xs[::-1], np.full(1000, y0 + 1.0), np.full(1000, z)], axis=1))
        if layer % 10 == 5:
            last = points[-1][-1]
            n = sum(len(p) for p in points)
            retracts += [n, n + 1]
            points.append(np.array([[last[0], last[1], last[2] + 300.0], last]))
    translations = np.concatenate(points)
    rotation = Rotation.concatenate([down] * len(translations))
    return RigidTransform.from_components(rotation=rotation, translation=translations), retracts


def random_poses(robot, n: int, seed: int) -> RigidTransform:
    rng = np.random.default_rng(seed)
    joints = rng.uniform(LIMITS[:, 0], LIMITS[:, 1], size=(n, 6))
    return robot.batch_forward(joints)


def main() -> None:
    robot = make_robot(degrees=True)
    rng = np.random.default_rng(1)
    print(f"family: {robot.family}")

    joints = rng.uniform(LIMITS[:, 0], LIMITS[:, 1], size=(100_000, 6))
    _, t_fwd = timed(lambda: robot.batch_forward(joints))
    print(f"batch_forward:            {t_fwd / len(joints) * 1e6:8.3f} us/pose  (100k joints)")

    cold = random_poses(robot, 1000, 2)
    mats = cold.as_matrix()
    single = [RigidTransform.from_matrix(m) for m in mats]

    def run_cold():
        return sum(len(robot.inverse(p)) for p in single)

    roots, t_cold = timed(run_cold)
    print(
        f"inverse (cold):           {t_cold / len(single) * 1e6:8.1f} us/pose  "
        f"(1000 random poses, {roots / len(single):.2f} roots per pose, Python call overhead included)"
    )

    path, retracts = printing_path()
    tracked, t_track = timed(lambda: robot.batch_inverse(path, current_joints=HOME))
    nan_rows = int(np.isnan(tracked[:, 0]).sum())
    steps = np.abs(np.diff(tracked, axis=0)).max(axis=1)
    mask = np.ones(len(steps), dtype=bool)
    for r in retracts:
        mask[r - 1 : r + 1] = False
    print(
        f"batch_inverse (tracked):  {t_track / len(path) * 1e6:8.2f} us/pose  "
        f"({len(path)} poses, NaN rows {nan_rows}, largest printing step {steps[mask].max():.4f} deg, "
        f"largest retract step {steps[~mask].max():.2f} deg)"
    )

    reach_poses = random_poses(robot, 20_000, 3)
    res1, t_r1 = timed(lambda: robot.reach(reach_poses, LIMITS, threads=1), repeats=2)
    res0, t_r0 = timed(lambda: robot.reach(reach_poses, LIMITS, threads=0), repeats=2)
    same = np.array_equal(res1.joints, res0.joints, equal_nan=True)
    print(
        f"reach threads=1:          {t_r1 / len(reach_poses) * 1e6:8.1f} us/pose  "
        f"(20k poses, mean {res1.n_solutions.mean():.2f} roots, {np.mean(res1.n_solutions > 0):.3f} reachable)"
    )
    print(
        f"reach threads=0:          {t_r0 / len(reach_poses) * 1e6:8.1f} us/pose  "
        f"(speedup {t_r1 / t_r0:.1f}x, identical to threads=1: {same})"
    )


if __name__ == "__main__":
    main()
