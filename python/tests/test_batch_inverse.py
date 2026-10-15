"""Tracked batch_inverse: seam consistency, equality with seeded inverse, NaN rows, py-opw parity."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import py_ikgeo_kinematics as ikg
import py_opw_kinematics as opw
import pytest
from nj4 import make_robot
from scipy.spatial.transform import RigidTransform, Rotation

FIXTURE = Path(__file__).parent / "fixtures" / "nj4_poses.json"
NJ165: dict[str, Any] = {
    "a1": 400.333,
    "a2": -251.449,
    "b": 0,
    "c1": 830,
    "c2": 1177.556,
    "c3": 1443.593,
    "c4": 230,
    "offsets": (0, 0, 0, 0, 0, 0),
    "flip_axes": (True, False, True, True, False, True),
}
TOL_DEG = 1e-9
PARITY_TOL_DEG = 1e-6


@pytest.fixture(scope="module")
def nj4_poses() -> RigidTransform:
    data = json.loads(FIXTURE.read_text())
    return RigidTransform.from_matrix(np.array(data["poses"]).reshape(-1, 4, 4))


@pytest.fixture(params=[None, "ee"])
def ee_transform(request: pytest.FixtureRequest) -> RigidTransform | None:
    if request.param is None:
        return None
    rot = Rotation.from_euler("xyz", [10, -90, 25], degrees=True)
    return RigidTransform.from_components(rotation=rot, translation=[12.0, -7.5, 150.0])


def smooth_path(robot: Any, ee_transform: RigidTransform | None, n: int = 2000) -> RigidTransform:
    """Joint-space sweep through a comfortable NJ165 region, so every pose has a near root."""
    s = np.linspace(0.0, 1.0, n)
    joints = np.column_stack([20 * s, 10 + 25 * s, -80 + 30 * s, 15 * s - 5, 40 - 20 * s, 30 * s])
    return robot.batch_forward(joints, ee_transform=ee_transform)


def test_seam_on_random_poses(nj4_poses: RigidTransform) -> None:
    robot = make_robot(degrees=True)
    seed = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    whole = robot.batch_inverse(nj4_poses, current_joints=seed)
    assert whole.shape == (len(nj4_poses), 6)
    assert not np.isnan(whole).any()
    parts = []
    prev: Any = seed
    for start in range(0, len(nj4_poses), 100):
        part = robot.batch_inverse(nj4_poses[start : start + 100], current_joints=prev)
        parts.append(part)
        prev = tuple(part[-1])
    assert np.allclose(np.vstack(parts), whole, atol=TOL_DEG)


def test_equals_seeded_inverse_loop(nj4_poses: RigidTransform) -> None:
    robot = make_robot(degrees=True)
    seed = (10.0, -20.0, 30.0, 40.0, -50.0, 60.0)
    batch = robot.batch_inverse(nj4_poses[:200], current_joints=seed)
    prev: Any = seed
    for i in range(200):
        expect = robot.inverse(nj4_poses[i], current_joints=prev)[0]
        assert np.allclose(batch[i], expect, atol=TOL_DEG), i
        prev = tuple(batch[i])


def test_nan_poses_give_nan_rows_and_keep_the_chain(nj4_poses: RigidTransform) -> None:
    robot = make_robot(degrees=True)
    matrices = nj4_poses[:50].as_matrix().copy()
    matrices[[3, 17], 0, 3] = np.nan
    broken = robot.batch_inverse(RigidTransform.from_matrix(matrices), current_joints=(0,) * 6)
    assert np.isnan(broken[[3, 17]]).all()
    keep = [i for i in range(50) if i not in (3, 17)]
    # A NaN pose leaves the chain seed untouched, so skipping those rows gives the same path.
    skipped = robot.batch_inverse(nj4_poses[keep], current_joints=(0,) * 6)
    assert np.allclose(broken[keep], skipped, atol=TOL_DEG)


def test_nan_seed_is_treated_as_no_seed(nj4_poses: RigidTransform) -> None:
    robot = make_robot(degrees=True)
    plain = robot.batch_inverse(nj4_poses[:20])
    nan_seeded = robot.batch_inverse(nj4_poses[:20], current_joints=(float("nan"),) * 6)
    assert np.allclose(plain, nan_seeded, atol=TOL_DEG)
    assert robot.inverse(nj4_poses[0], current_joints=(float("nan"),) * 6) == robot.inverse(
        nj4_poses[0]
    )


def test_unreachable_pose_gives_nan_row() -> None:
    robot = make_robot(degrees=True)
    far = RigidTransform.from_components(
        rotation=Rotation.identity(), translation=[10_000.0, 0.0, 0.0]
    )
    poses = RigidTransform.concatenate([robot.forward((0,) * 6), far, robot.forward((0,) * 6)])
    joints = robot.batch_inverse(poses, current_joints=(0,) * 6)
    assert np.isnan(joints[1]).all()
    assert np.allclose(joints[0], 0.0, atol=TOL_DEG)
    assert np.allclose(joints[2], 0.0, atol=TOL_DEG)


def test_parity_with_py_opw_seeded(ee_transform: RigidTransform | None) -> None:
    theirs = opw.Robot(opw.KinematicModel(**NJ165), degrees=True)
    ours = ikg.Robot(ikg.KinematicModel.from_opw(**NJ165), degrees=True)
    poses = smooth_path(theirs, ee_transform)
    seed = (1.0, 12.0, -78.0, -4.0, 38.0, 1.0)
    a = theirs.batch_inverse(poses, current_joints=seed, ee_transform=ee_transform)
    b = ours.batch_inverse(poses, current_joints=seed, ee_transform=ee_transform)
    assert np.allclose(a, b, atol=PARITY_TOL_DEG)


def test_parity_with_py_opw_unseeded(ee_transform: RigidTransform | None) -> None:
    # Without a seed the two libraries pick a different first root (branch order versus
    # nearest to home), so parity is: our first row is one of py-opw's roots, and the rest
    # of the chain equals py-opw's chain seeded with that row.
    theirs = opw.Robot(opw.KinematicModel(**NJ165), degrees=True)
    ours = ikg.Robot(ikg.KinematicModel.from_opw(**NJ165), degrees=True)
    poses = smooth_path(theirs, ee_transform)
    b = ours.batch_inverse(poses, ee_transform=ee_transform)
    roots = theirs.inverse(poses[0], ee_transform=ee_transform)
    assert any(np.allclose(b[0], r, atol=PARITY_TOL_DEG) for r in roots)
    a = theirs.batch_inverse(poses, current_joints=tuple(b[0]), ee_transform=ee_transform)
    assert np.allclose(a, b, atol=PARITY_TOL_DEG)


def test_dataframe_seed_round_trips_type() -> None:
    pl = pytest.importorskip("polars")
    robot = make_robot(degrees=True)
    poses = robot.batch_forward(np.zeros((5, 6)))
    seed = pl.DataFrame({c: [0.0] for c in ["J1", "J2", "J3", "J4", "J5", "J6"]})
    out = robot.batch_inverse(poses, current_joints=seed)
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ["J1", "J2", "J3", "J4", "J5", "J6"]
    assert np.allclose(out.to_numpy(), 0.0, atol=TOL_DEG)


def test_degrees_flag(nj4_poses: RigidTransform) -> None:
    deg = make_robot(degrees=True)
    rad = make_robot(degrees=False)
    seed_deg = (10.0, -20.0, 30.0, 40.0, -50.0, 60.0)
    a = deg.batch_inverse(nj4_poses[:100], current_joints=seed_deg)
    b = rad.batch_inverse(nj4_poses[:100], current_joints=tuple(np.deg2rad(seed_deg)))
    assert np.allclose(a, np.rad2deg(b), atol=1e-9)


def test_tracking_keeps_continuation_near_the_fold() -> None:
    """Near J5 = 180 two branches run close; tracking stays on the seed's branch, the cold
    first pick is the nearest by summed distance. Both are valid roots (Codex review)."""
    robot = make_robot(True)
    q = (156.384, -26.590, -76.231, -8.423, 177.843, 162.321)
    seed = (148.219, -29.289, -50.288, -11.880, 180.024, 173.753)
    pose = robot.forward(q)
    tracked = robot.batch_inverse(pose, current_joints=seed)[0]
    nearest = np.asarray(robot.inverse(pose, current_joints=seed)[0])
    assert np.allclose(tracked, q, atol=1e-6)

    def dist(j: Any) -> float:
        return float(np.abs(np.asarray(j) - np.asarray(seed)).sum())

    assert dist(nearest) <= dist(tracked)
    for j in (tracked, nearest):
        err = np.linalg.norm(robot.forward(tuple(j)).translation - pose.translation)
        assert err < 1e-9
