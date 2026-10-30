"""reach on the Comau NJ4-220: wrist folds, limits, threads and the fixture poses."""

import json
from pathlib import Path

import numpy as np
import pytest
from nj4 import LIMITS, make_robot
from py_ikgeo_kinematics import Robot
from scipy.spatial.transform import RigidTransform

FIXTURE = Path(__file__).parent / "fixtures" / "nj4_poses.json"


@pytest.fixture(scope="module")
def robot() -> Robot:
    return make_robot(degrees=True)


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def column_of(result_joints: np.ndarray, truth: np.ndarray) -> int:
    diff = (result_joints - truth + 180.0) % 360.0 - 180.0
    hits = np.where(np.all(np.abs(diff) < 1e-6, axis=1))[0]
    assert len(hits) == 1, f"{truth} found in columns {hits}"
    return int(hits[0])


def test_wrist_margin_zero_at_both_folds(robot: Robot) -> None:
    rng = np.random.default_rng(3)
    base = rng.uniform(LIMITS[:, 0], LIMITS[:, 1], size=(40, 6))
    for j5 in (0.0, 180.0, -180.0):
        joints = base.copy()
        joints[:, 4] = j5
        result = robot.reach(robot.batch_forward(joints))
        for i, truth in enumerate(joints):
            c = column_of(result.joints[i], truth)
            assert abs(result.wrist[i, c]) < 1e-6, (j5, result.wrist[i, c])
    # Half way between the folds the margin is the 60 degree bend from either fold.
    joints = base.copy()
    joints[:, 4] = 90.0
    result = robot.reach(robot.batch_forward(joints))
    for i, truth in enumerate(joints):
        c = column_of(result.joints[i], truth)
        assert result.wrist[i, c] > 20.0


def sigma_at(robot: Robot, joints: np.ndarray) -> np.ndarray:
    result = robot.reach(robot.batch_forward(joints))
    return np.array(
        [result.sigma_min[i, column_of(result.joints[i], q)] for i, q in enumerate(joints)]
    )


def test_sigma_min_drops_at_both_folds(robot: Robot) -> None:
    # At J5 = 0 and J5 = 180 the three wrist axes are coplanar, so the wrist alone cannot
    # rotate about the normal of that plane. The arm axes still complete the rank, so the
    # 6x6 Jacobian is badly conditioned there but not exactly singular: sigma_min falls by
    # more than an order of magnitude against the half-bent wrist, and reach must agree
    # with a plain SVD of the Jacobian.
    rng = np.random.default_rng(4)
    base = rng.uniform(LIMITS[:, 0], LIMITS[:, 1], size=(60, 6))
    # Keep the 4/5 point off axis 1, where the arm itself is singular whatever J5 is.
    wrist = np.array([robot.joint_poses(tuple(q))[4].translation for q in base])
    base = base[np.hypot(wrist[:, 0], wrist[:, 1]) > 200.0]
    sigmas = {}
    for j5 in (0.0, 90.0, 180.0):
        joints = base.copy()
        joints[:, 4] = j5
        sigmas[j5] = sigma_at(robot, joints)
        svd = np.array(
            [np.linalg.svd(robot.jacobian(tuple(q)), compute_uv=False)[-1] for q in joints]
        )
        # sigma_min comes from the eigenvalues of J^T J (as in py-opw), whose absolute
        # floor is about sqrt(eps) * sigma_max, 1e-4 for a 3 m arm in mm.
        assert np.allclose(sigmas[j5], svd, rtol=1e-6, atol=1e-3)
    assert np.median(sigmas[180.0]) < 0.1 * np.median(sigmas[90.0])
    assert np.median(sigmas[0.0]) < 0.1 * np.median(sigmas[90.0])
    assert np.all(sigmas[90.0] > 0.01)
    assert np.all(sigmas[180.0] > 0.0) and np.all(sigmas[0.0] > 0.0)


def test_fixture_poses_contain_their_joints_with_margins(fixture: dict, robot: Robot) -> None:
    poses = RigidTransform.from_matrix(np.array(fixture["poses"]).reshape(-1, 4, 4))
    joints = np.array(fixture["joints_deg"])
    result = robot.reach(poses, LIMITS)
    assert result.joints.shape == (len(joints), 16, 6)
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]
    # py-opw semantics: joints are wrapped to (-180, 180] before the limit comparison, so
    # a fixture joint at 195 is measured as -165 against the +-200 window.
    wrapped = (joints + 180.0) % 360.0 - 180.0
    wrapped[wrapped == -180.0] = 180.0
    expected_margin = np.minimum(wrapped - lo, hi - wrapped).min(axis=1)
    for i, truth in enumerate(joints):
        c = column_of(result.joints[i], truth)
        assert result.limit_margin[i, c] == pytest.approx(expected_margin[i], abs=1e-6)
        assert result.sigma_min[i, c] >= 0.0
        assert result.extension[i] > 0.0
        n = result.n_solutions[i]
        assert 1 <= n <= 16
        assert np.all(np.isnan(result.joints[i, n:, 0]))
        assert np.all(~np.isnan(result.joints[i, :n, 0]))
        # Out-of-limit columns have no conditioning number.
        out = result.limit_margin[i, :n] < 0
        assert np.all(np.isnan(result.sigma_min[i, :n][out]))
        assert np.all(~np.isnan(result.sigma_min[i, :n][~out]))
    # No limits: every margin is +inf.
    free = robot.reach(poses[:50])
    exists = ~np.isnan(free.joints[:, :, 0])
    assert np.all(np.isinf(free.limit_margin[exists]))


def branch_key(robot: Robot, q: np.ndarray) -> tuple[int, int, int, float]:
    """(shoulder, elbow, wrist_flip, J4) from the joint frames, as ReachResult documents."""
    frames = robot.joint_poses(tuple(q))
    origins = frames.translation
    r1 = frames.rotation[0].as_matrix()
    w = origins[3]  # the NJ4 model's joint-4 point is the 4/5 intersection
    shoulder = 0 if (r1.T @ (w - origins[0]))[0] >= 0 else 1
    h2 = r1 @ np.array([0.0, 1.0, 0.0])
    bend = h2 @ np.cross(origins[2] - origins[1], w - origins[2])
    elbow = 0 if bend >= 0 else 1
    return (shoulder, elbow, 0 if q[4] >= 0 else 1, float(q[3]))


def test_columns_are_sorted_by_key(fixture: dict, robot: Robot) -> None:
    poses = RigidTransform.from_matrix(np.array(fixture["poses"][:200]).reshape(-1, 4, 4))
    result = robot.reach(poses)
    seen = {0: 0, 1: 0}
    for i in range(len(poses)):
        n = result.n_solutions[i]
        keys = [branch_key(robot, q) for q in result.joints[i, :n]]
        assert keys == sorted(keys), keys
        for k in keys:
            seen[k[0]] += 1
        # Determinism: the same pose solved again gives identical columns.
        again = robot.reach(poses[i])
        assert np.array_equal(again.joints[0], result.joints[i], equal_nan=True)
    assert seen[0] > 0 and seen[1] > 0


def test_threads_match_and_extension_is_per_pose(fixture: dict, robot: Robot) -> None:
    poses = RigidTransform.from_matrix(np.array(fixture["poses"][:200]).reshape(-1, 4, 4))
    one = robot.reach(poses, LIMITS, threads=1)
    many = robot.reach(poses, LIMITS)
    two = robot.reach(poses, LIMITS, threads=2)
    for a, b in ((one, many), (one, two)):
        assert np.array_equal(a.joints, b.joints, equal_nan=True)
        assert np.array_equal(a.sigma_min, b.sigma_min, equal_nan=True)
        assert np.array_equal(a.extension, b.extension, equal_nan=True)
        assert np.array_equal(a.n_solutions, b.n_solutions)
    # Far out of reach: no roots, negative extension of about the overshoot.
    far = RigidTransform.from_translation([[6000.0, 0.0, 2000.0]])
    out = robot.reach(far)
    assert out.n_solutions[0] == 0
    assert out.extension[0] < -1000.0
    assert np.all(np.isnan(out.joints[0]))


def test_radians_flag(fixture: dict) -> None:
    deg = make_robot(degrees=True)
    rad = make_robot(degrees=False)
    poses = RigidTransform.from_matrix(np.array(fixture["poses"][:30]).reshape(-1, 4, 4))
    a = deg.reach(poses, LIMITS)
    b = rad.reach(poses, np.deg2rad(LIMITS))
    assert np.allclose(a.joints, np.rad2deg(b.joints), equal_nan=True, atol=1e-9)
    assert np.allclose(a.limit_margin, np.rad2deg(b.limit_margin), equal_nan=True, atol=1e-9)
    assert np.allclose(a.wrist, np.rad2deg(b.wrist), equal_nan=True, atol=1e-9)
    assert np.allclose(a.sigma_min, b.sigma_min, equal_nan=True)
