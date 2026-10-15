"""Cold inverse on the Comau NJ4-220 fixture: coverage and FK round trip of every root."""

import json
from pathlib import Path

import numpy as np
import pytest
from nj4 import make_robot
from py_ikgeo_kinematics import KinematicModel, Robot
from scipy.spatial.transform import RigidTransform

FIXTURE = Path(__file__).parent / "fixtures" / "nj4_poses.json"
POSITION_TOL_MM = 1e-9
ANGLE_TOL_DEG = 1e-9
MIN_HIT_RATE = 0.99


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def robot() -> Robot:
    return make_robot(degrees=True)


def test_family(robot: Robot) -> None:
    assert robot.family == "two_intersecting_two_parallel"


def test_fixture_matches_model(fixture: dict, robot: Robot) -> None:
    model = robot.kinematic_model
    assert np.allclose(model.h, fixture["model"]["h"])
    assert np.allclose(model.p, fixture["model"]["p"])
    poses = robot.batch_forward(np.array(fixture["joints_deg"])).as_matrix().reshape(-1, 16)
    assert np.allclose(poses, fixture["poses"], atol=1e-9)


def test_every_pose_has_roots_that_round_trip(fixture: dict, robot: Robot) -> None:
    poses = np.array(fixture["poses"]).reshape(-1, 4, 4)
    joints = np.array(fixture["joints_deg"])
    hits = 0
    for pose, truth in zip(poses, joints):
        target = RigidTransform.from_matrix(pose)
        sols = np.array(robot.inverse(target))
        assert len(sols) >= 1, f"no root for joints {truth}"
        assert len(sols) <= 16
        diff = (sols - truth + 180.0) % 360.0 - 180.0
        if np.any(np.all(np.abs(diff) < 1e-6, axis=1)):
            hits += 1
        for sol in sols:
            got = robot.forward(tuple(sol))
            assert np.linalg.norm(got.translation - target.translation) < POSITION_TOL_MM
            angle = np.rad2deg((got.rotation.inv() * target.rotation).magnitude())
            assert angle < ANGLE_TOL_DEG
    assert hits >= MIN_HIT_RATE * len(poses), f"generating joints found for {hits} of {len(poses)}"


def test_seed_orders_solutions(fixture: dict, robot: Robot) -> None:
    poses = np.array(fixture["poses"]).reshape(-1, 4, 4)
    joints = np.array(fixture["joints_deg"])
    # A cuspidal arm can have another root a few degrees away, so the seed stays close.
    for pose, truth in zip(poses[:100], joints[:100]):
        # Near a fold two roots merge, so the root nearest the seed need not be the truth.
        if np.linalg.svd(robot.jacobian(tuple(truth)), compute_uv=False)[-1] < 0.01:
            continue
        seed = tuple(truth + 0.1)
        sols = robot.inverse(RigidTransform.from_matrix(pose), current_joints=seed)
        dist = [np.abs(np.array(s) - seed).sum() for s in sols]
        assert dist == sorted(dist)
        assert np.allclose(sols[0], truth, atol=1e-6)


def test_unsolvable_chain_is_rejected() -> None:
    # Axes 2/3 and 4/5 both parallel: no vendored solver, so construction must fail loudly.
    h = np.array([[0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 1]], dtype=float)
    p = np.array(
        [
            [0, 0.3, 0.1, 0.8, 0.2, 0.1, 0.1],
            [0, 0.2, 0.0, 0.1, 0.3, 0.15, 0.1],
            [0.5, 0.4, 0.9, 0.3, 0.1, 0.2, 0.0],
        ],
        dtype=float,
    )
    with pytest.raises(ValueError, match="both parallel"):
        Robot(KinematicModel(h, p))
