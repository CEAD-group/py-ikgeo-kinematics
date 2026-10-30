"""The per-call threads argument of reach: None, 1, n and the deprecated 0."""

import warnings

import numpy as np
import pytest
from nj4 import LIMITS, make_robot
from py_ikgeo_kinematics import ReachResult, Robot


@pytest.fixture(scope="module")
def robot() -> Robot:
    return make_robot(degrees=True)


@pytest.fixture(scope="module")
def joints() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.uniform(LIMITS[:, 0], LIMITS[:, 1], size=(64, 6))


def assert_same(a: ReachResult, b: ReachResult) -> None:
    for name in ("joints", "limit_margin", "extension", "sigma_min", "wrist", "n_solutions"):
        assert np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True), name


@pytest.mark.parametrize("threads", [None, 4])
def test_results_do_not_depend_on_threads(robot: Robot, joints: np.ndarray, threads: int) -> None:
    poses = robot.batch_forward(joints)
    assert_same(robot.reach(poses, LIMITS, threads=1), robot.reach(poses, LIMITS, threads=threads))


def test_zero_is_a_deprecated_alias_for_none(robot: Robot, joints: np.ndarray) -> None:
    poses = robot.batch_forward(joints)
    with pytest.warns(DeprecationWarning, match="threads=None"):
        zero = robot.reach(poses, LIMITS, threads=0)
    assert_same(zero, robot.reach(poses, LIMITS, threads=1))


def test_none_does_not_warn(robot: Robot, joints: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        robot.reach(robot.batch_forward(joints[:2]), LIMITS)


@pytest.mark.parametrize("threads", [-1, -8])
def test_below_one_raises(robot: Robot, joints: np.ndarray, threads: int) -> None:
    with pytest.raises(ValueError, match="threads"):
        robot.reach(robot.batch_forward(joints[:2]), threads=threads)


def test_threads_is_keyword_only(robot: Robot, joints: np.ndarray) -> None:
    with pytest.raises(TypeError):
        robot.reach(robot.batch_forward(joints[:2]), LIMITS, None, 1)  # type: ignore[misc]


def test_internal_rejects_zero(robot: Robot, joints: np.ndarray) -> None:
    # Only the public wrapper keeps the deprecated alias.
    rows = np.ascontiguousarray(robot.batch_forward(joints[:2]).as_matrix().reshape(-1, 16))
    with pytest.raises(ValueError, match="threads"):
        robot._robot.reach(rows, None, None, threads=0)


@pytest.mark.parametrize("n", [0, 1, 2, 3])
@pytest.mark.parametrize("threads", [None, 1, 2, 4, 16])
def test_small_inputs(robot: Robot, joints: np.ndarray, n: int, threads: int | None) -> None:
    rows = np.ascontiguousarray(robot.batch_forward(joints[:4]).as_matrix().reshape(-1, 16))[:n]
    out = robot._robot.reach(rows, None, None, threads=threads)
    ref = robot._robot.reach(rows, None, None, threads=1)
    assert out[0].shape == (n, 16, 6)
    for x, y in zip(out, ref):
        assert np.array_equal(x, y, equal_nan=True)


def test_repeated_calls_reuse_the_pool(robot: Robot, joints: np.ndarray) -> None:
    poses = robot.batch_forward(joints)
    first = robot.reach(poses, LIMITS, threads=3)
    for _ in range(5):
        assert_same(first, robot.reach(poses, LIMITS, threads=3))
