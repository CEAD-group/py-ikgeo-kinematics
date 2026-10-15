"""Forward kinematics of an OPW robot given as h, p must match py-opw-kinematics."""

from typing import Any

import numpy as np
import py_ikgeo_kinematics as ikg
import py_opw_kinematics as opw
import pytest
from scipy.spatial.transform import RigidTransform, Rotation

OPW_PARAMS: list[dict[str, Any]] = [
    {
        "a1": 400,
        "a2": -250,
        "b": 0,
        "c1": 830,
        "c2": 1175,
        "c3": 1444,
        "c4": 230,
        "offsets": (0, 0, 0, 0, 0, 0),
        "flip_axes": (True, False, True, True, False, True),
    },
    {
        "a1": 350,
        "a2": -41,
        "b": 120,
        "c1": 675,
        "c2": 1150,
        "c3": 1200,
        "c4": 215,
        "offsets": (10, -20, 30, 5, -5, 15),
        "flip_axes": (False, True, False, True, True, False),
    },
]
N = 10_000


def pair(params: dict[str, Any], degrees: bool) -> tuple[opw.Robot, ikg.Robot]:
    a = opw.Robot(opw.KinematicModel(**params), degrees=degrees)
    b = ikg.Robot(ikg.KinematicModel.from_opw(**params), degrees=degrees)
    return a, b


@pytest.fixture(params=[None, "ee"])
def ee_transform(request: pytest.FixtureRequest) -> RigidTransform | None:
    if request.param is None:
        return None
    rot = Rotation.from_euler("xyz", [10, -90, 25], degrees=True)
    return RigidTransform.from_components(rotation=rot, translation=[12.0, -7.5, 150.0])


@pytest.fixture
def joints() -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.uniform(-180, 180, size=(N, 6))


def assert_poses_equal(a: RigidTransform, b: RigidTransform) -> None:
    ma, mb = a.as_matrix(), b.as_matrix()
    assert ma.shape == mb.shape
    np.testing.assert_allclose(ma[..., :3, :3], mb[..., :3, :3], atol=1e-9, rtol=0)
    np.testing.assert_allclose(ma[..., :3, 3], mb[..., :3, 3], atol=1e-9, rtol=0)


@pytest.mark.parametrize("params", OPW_PARAMS)
def test_family_of_opw_model(params: dict) -> None:
    _, robot = pair(params, degrees=True)
    assert robot.family == "spherical_two_parallel"


@pytest.mark.parametrize("params", OPW_PARAMS)
def test_batch_forward_matches(
    params: dict[str, Any], joints: np.ndarray, ee_transform: RigidTransform | None
) -> None:
    a, b = pair(params, degrees=True)
    assert_poses_equal(a.batch_forward(joints, ee_transform), b.batch_forward(joints, ee_transform))


@pytest.mark.parametrize("params", OPW_PARAMS)
def test_forward_matches(
    params: dict[str, Any], joints: np.ndarray, ee_transform: RigidTransform | None
) -> None:
    a, b = pair(params, degrees=True)
    for row in joints[:200]:
        j = tuple(row)
        assert_poses_equal(a.forward(j, ee_transform), b.forward(j, ee_transform))


@pytest.mark.parametrize("params", OPW_PARAMS)
def test_joint_poses_match(
    params: dict[str, Any], joints: np.ndarray, ee_transform: RigidTransform | None
) -> None:
    a, b = pair(params, degrees=True)
    assert_poses_equal(
        a.batch_joint_poses(joints, ee_transform), b.batch_joint_poses(joints, ee_transform)
    )
    for row in joints[:50]:
        j = tuple(row)
        assert_poses_equal(a.joint_poses(j, ee_transform), b.joint_poses(j, ee_transform))


@pytest.mark.parametrize("params", OPW_PARAMS)
def test_radians_flag(params: dict[str, Any], joints: np.ndarray) -> None:
    a, b = pair(params, degrees=False)
    rad = np.deg2rad(joints[:500])
    assert_poses_equal(a.batch_forward(rad), b.batch_forward(rad))


def test_forward_frames_translations(joints: np.ndarray) -> None:
    params = OPW_PARAMS[0]
    a, b = pair(params, degrees=True)
    for row in joints[:50]:
        j = tuple(row)
        with pytest.warns(DeprecationWarning):
            fa = a.forward_frames(j).translation
        fb = b.forward_frames(j)
        assert len(fb) == 8
        # py-opw's frame for J2 ignores b and J1 sits at the origin; compare the rest.
        np.testing.assert_allclose(
            fa[[0, 3, 4, 5, 6, 7]], fb.translation[[0, 3, 4, 5, 6, 7]], atol=1e-6
        )
        np.testing.assert_allclose(fb.translation[7], b.forward(j).translation, atol=1e-9)


def test_jacobian_against_finite_differences() -> None:
    _, robot = pair(OPW_PARAMS[1], degrees=True)
    ee = RigidTransform.from_translation([0.0, 0.0, 100.0])
    j0 = np.array([12.0, -30.0, 40.0, 25.0, 50.0, -70.0])
    jac = robot.jacobian(tuple(j0), ee)
    eps = 1e-6
    for k in range(6):
        jp, jm = j0.copy(), j0.copy()
        jp[k] += eps
        jm[k] -= eps
        tp, tm = robot.forward(tuple(jp), ee), robot.forward(tuple(jm), ee)
        dv = (tp.translation - tm.translation) / np.deg2rad(2 * eps)
        dr = (tp.rotation.as_matrix() - tm.rotation.as_matrix()) / np.deg2rad(2 * eps)
        w_hat = dr @ robot.forward(tuple(j0), ee).rotation.as_matrix().T
        w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])
        np.testing.assert_allclose(jac[:3, k], dv, atol=1e-3)
        np.testing.assert_allclose(jac[3:, k], w, atol=1e-6)


def test_model_validation() -> None:
    with pytest.raises(ValueError):
        ikg.KinematicModel(np.zeros((3, 5)), np.zeros((3, 7)))
    with pytest.raises(ValueError):
        ikg.KinematicModel(np.eye(3)[:, [0, 1, 1, 2, 1, 2]], np.zeros((3, 6)))
    h = [[0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1]]
    p = [[0.0, 1, 0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0, 0, 0], [1.0, 0, 1, 1, 0, 0, 0.1]]
    model = ikg.KinematicModel(h, p)
    assert model.h.shape == (3, 6)
    assert model.p.shape == (3, 7)
    assert ikg.Robot(model).family == "spherical_two_parallel"
