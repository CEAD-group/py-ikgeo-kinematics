"""Inverse kinematics of an OPW robot given as h, p must return py-opw-kinematics' solution set."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import py_ikgeo_kinematics as ikg
import py_opw_kinematics as opw
import pytest
from scipy.spatial.transform import RigidTransform, Rotation

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
N = 500
TOL_DEG = 1e-6


@pytest.fixture(scope="module")
def robots() -> tuple[opw.Robot, ikg.Robot]:
    return (
        opw.Robot(opw.KinematicModel(**NJ165), degrees=True),
        ikg.Robot(ikg.KinematicModel.from_opw(**NJ165), degrees=True),
    )


@pytest.fixture(params=[None, "ee"])
def ee_transform(request: pytest.FixtureRequest) -> RigidTransform | None:
    if request.param is None:
        return None
    rot = Rotation.from_euler("xyz", [10, -90, 25], degrees=True)
    return RigidTransform.from_components(rotation=rot, translation=[12.0, -7.5, 150.0])


@pytest.fixture(scope="module")
def joints() -> np.ndarray:
    rng = np.random.default_rng(4321)
    return rng.uniform(-180, 180, size=(N, 6))


def same_mod_360(a: Sequence[float], b: Sequence[float]) -> bool:
    diff = (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0
    return bool(np.all(np.abs(diff) < TOL_DEG))


def assert_same_set(ours: Sequence[Sequence[float]], theirs: Sequence[Sequence[float]]) -> None:
    assert len(ours) == len(theirs)
    for t in theirs:
        assert any(same_mod_360(o, t) for o in ours), f"{t} missing from {ours}"


def test_solution_sets_match(
    robots: tuple[opw.Robot, ikg.Robot], joints: np.ndarray, ee_transform: RigidTransform | None
) -> None:
    a, b = robots
    for j in joints:
        pose = a.forward(tuple(j), ee_transform=ee_transform)
        assert_same_set(
            b.inverse(pose, ee_transform=ee_transform), a.inverse(pose, ee_transform=ee_transform)
        )


def test_seeded_order_and_unwrapping_match(
    robots: tuple[opw.Robot, ikg.Robot], joints: np.ndarray, ee_transform: RigidTransform | None
) -> None:
    a, b = robots
    rng = np.random.default_rng(99)
    for j in joints[:200]:
        pose = a.forward(tuple(j), ee_transform=ee_transform)
        seed = tuple(j + rng.uniform(-20, 20, size=6) + 360.0 * rng.integers(-1, 2, size=6))
        theirs = a.inverse(pose, current_joints=seed, ee_transform=ee_transform)
        ours = b.inverse(pose, current_joints=seed, ee_transform=ee_transform)
        assert_same_set(ours, theirs)
        # The seed-nearest solution is first in both, in the same unwrapped turn.
        assert np.allclose(ours[0], theirs[0], atol=TOL_DEG)


def test_degrees_flag(joints: np.ndarray) -> None:
    deg = ikg.Robot(ikg.KinematicModel.from_opw(**NJ165), degrees=True)
    rad = ikg.Robot(ikg.KinematicModel.from_opw(**NJ165), degrees=False)
    for j in joints[:50]:
        pose = deg.forward(tuple(j))
        a = np.array(deg.inverse(pose))
        b = np.rad2deg(np.array(rad.inverse(pose)))
        assert a.shape == b.shape
        assert np.allclose(np.sort(a, axis=0), np.sort(b, axis=0), atol=1e-9)


def test_unseeded_order_is_distance_to_home(robots: tuple[opw.Robot, ikg.Robot]) -> None:
    _, ours = robots
    rng = np.random.default_rng(3)
    for _ in range(50):
        joints = rng.uniform(-170, 170, 6)
        pose = ours.forward(tuple(joints))
        sols = np.array(ours.inverse(pose))
        home = np.array(ours.inverse(pose, current_joints=(0.0,) * 6))
        assert np.allclose(sols, home)
        dist = np.abs(sols).sum(axis=1)
        assert np.all(np.diff(dist) >= -1e-9)
        assert np.all(np.abs(sols) <= 180 + 1e-9)


def _perturbed_nj165() -> tuple[np.ndarray, np.ndarray, float, float]:
    """NJ165 with 1e-7 rad skew between axes 2 and 3 and 1e-5 mm misses in the wrist."""
    model = ikg.KinematicModel.from_opw(**NJ165)
    h = np.array(model.h)
    p = np.array(model.p)
    skew = 1e-7
    miss_mm = 1e-5
    # Tilting axis 3 about z (the offset direction) makes the pair skew, not coplanar.
    h[:, 2] = Rotation.from_rotvec([0.0, 0.0, skew]).apply(h[:, 2])
    # Axes 4 (z), 5 (y) and 6 (z) have common normal x: x offsets between them are misses.
    p[0, 4] += miss_mm
    p[0, 5] += miss_mm
    return h, p, skew, miss_mm


def test_axis_tolerance_rescues_a_noisy_model() -> None:
    h, p, skew, miss_mm = _perturbed_nj165()
    scale = float(np.linalg.norm(p, axis=0).max())
    with pytest.warns(UserWarning, match="axis_tolerance"):
        strict = ikg.Robot(
            ikg.KinematicModel(h, p, NJ165["offsets"], NJ165["flip_axes"]), degrees=True
        )
    assert strict.family == "general"
    report = strict.axis_report()
    assert [r.pair for r in report] == ["1-2", "2-3", "3-4", "4-5", "5-6"]
    assert report[1].parallel_deviation == pytest.approx(skew, rel=1e-3)
    assert not report[1].parallel
    for pair in (3, 4):
        assert report[pair].intersection_deviation == pytest.approx(miss_mm / scale, rel=1e-3)
        assert not report[pair].intersecting
    assert report[0].intersection_deviation > 0.1

    loose = ikg.Robot(
        ikg.KinematicModel(h, p, NJ165["offsets"], NJ165["flip_axes"]),
        degrees=True,
        axis_tolerance=1e-6,
    )
    assert loose.family == "spherical_two_parallel"
    assert loose.axis_report()[1].parallel
    assert loose.axis_report()[3].intersecting
    assert loose.axis_report()[4].intersecting
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(200):
        joints = tuple(rng.uniform(-170, 170, 6))
        pose = loose.forward(joints)
        sols = loose.inverse(pose)
        assert sols, "no root on the snapped model"
        assert any(np.allclose(s, joints, atol=1e-6) for s in sols)
        for s in sols:
            err = np.linalg.norm(loose.forward(s).translation - pose.translation)
            worst = max(worst, float(err))
    # Roots are polished on the true (noisy) model, so they round-trip far below the noise.
    assert worst < skew * scale


def test_axis_tolerance_validation() -> None:
    model = ikg.KinematicModel.from_opw(**NJ165)
    with pytest.raises(ValueError):
        ikg.Robot(model, axis_tolerance=-1.0)
    with pytest.raises(ValueError):
        ikg.Robot(model, axis_tolerance=float("nan"))


def test_exact_grid_pose_is_not_misconverted(robots: tuple[opw.Robot, ikg.Robot]) -> None:
    # nalgebra's iterative quaternion-from-matrix once converged to the wrong quaternion here.
    theirs, ours = robots
    joints = (0.0, -25.0, -95.0, -120.0, 110.0, 0.0)
    pose = ours.forward(joints)
    sols = ours.inverse(pose)
    assert len(sols) == 8
    assert len(theirs.inverse(pose)) == 8
    for s in sols:
        assert np.linalg.norm(ours.forward(s).translation - pose.translation) < 1e-9
    result = ours.reach(pose)
    assert int(result.n_solutions[0]) == 8
    assert np.isfinite(result.sigma_min[0, :8]).all()


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_half_turn_rotation_matrices_round_trip(
    robots: tuple[opw.Robot, ikg.Robot], axis: str
) -> None:
    _, ours = robots
    rot = Rotation.from_euler(axis, 180, degrees=True)
    pose = RigidTransform.from_components(rotation=rot, translation=[1500.0, 200.0, 1200.0])
    assert np.allclose(np.abs(pose.as_matrix()[:3, :3]), np.eye(3))
    sols = ours.inverse(pose)
    assert sols
    for s in sols:
        back = ours.forward(s)
        assert np.linalg.norm(back.translation - pose.translation) < 1e-9
        assert (back.rotation.inv() * rot).magnitude() < 1e-9
    result = ours.reach(pose)
    assert int(result.n_solutions[0]) == len(sols)
