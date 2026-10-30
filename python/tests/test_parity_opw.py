"""
API-contract tests run unchanged against py-opw-kinematics and py-ikgeo-kinematics.

Ported from py-opw's test_robot.py and test_reach.py. Every test is parametrised
over both backends and makes the same assertions, so a consumer can switch
packages without branching. Where py-opw's own tests rely on its eight stable
branch slots or its ``wrist = |sin J5|`` definition, the assertion is written in
backend-neutral terms and the difference is noted in a comment.
"""

import itertools
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import py_ikgeo_kinematics
import py_opw_kinematics
import pytest
from scipy.spatial.transform import RigidTransform, Rotation

Joints = tuple[float, float, float, float, float, float]

NJ165_MM: dict[str, Any] = {
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
# Comau NJ165 in metres, kinematic-frame J3, as py-opw's reach tests parameterise it.
NJ165_M: dict[str, Any] = {
    "a1": 0.400,
    "a2": -0.250,
    "b": 0.0,
    "c1": 0.830,
    "c2": 1.175,
    "c3": 1.444,
    "c4": 0.230,
    "offsets": (0, 0, 0, 0, 0, 0),
    "flip_axes": (True, False, True, True, False, True),
}
NJ165_LIMITS = np.array(
    [(-180, 180), (-75, 95), (-160, -30), (-2700, 2700), (-125, 125), (-2700, 2700)],
    dtype=float,
)
HOME = np.array([0.0, 0.0, -90.0, 0.0, 0.0, 0.0])
EE_ROTATION = Rotation.from_euler("xyz", [0, -90, 0], degrees=True)


@dataclass(frozen=True)
class Backend:
    name: str
    module: Any

    def model(self, **opw: Any) -> Any:
        if self.name == "opw":
            return self.module.KinematicModel(**opw)
        return self.module.KinematicModel.from_opw(**opw)

    def robot(self, degrees: bool = True, **opw: Any) -> Any:
        return self.module.Robot(self.model(**opw), degrees=degrees)


BACKENDS = [Backend("opw", py_opw_kinematics), Backend("ikgeo", py_ikgeo_kinematics)]


@pytest.fixture(params=BACKENDS, ids=[b.name for b in BACKENDS], scope="module")
def backend(request: pytest.FixtureRequest) -> Backend:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def robot(backend: Backend) -> Any:
    return backend.robot(**NJ165_MM)


@pytest.fixture(scope="module")
def robot_m(backend: Backend) -> Any:
    return backend.robot(**NJ165_M)


def _j(q: Any) -> Joints:
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]), float(q[5]))


def _margin(joints: np.ndarray, limits: np.ndarray) -> np.ndarray:
    lo, hi = limits[:, 0], limits[:, 1]
    return np.minimum(joints - lo, hi - joints).min(axis=-1)


def _grid_joints() -> np.ndarray:
    # Inside the limits, away from wrap-around on J1, J4, J6 so IK normalisation
    # cannot move a sample to an equivalent angle. J5 = 0 is excluded: at the
    # wrist singularity the J4/J6 split is not unique.
    j1 = np.linspace(-150, 150, 5)
    j2 = np.linspace(-60, 80, 5)
    j3 = np.linspace(-150, -40, 5)
    j4 = np.array([-120.0, 30.0, 150.0])
    j5 = np.linspace(-110, 110, 6)
    j6 = np.array([-90.0, 0.0, 60.0])
    return np.array(list(itertools.product(j1, j2, j3, j4, j5, j6)), dtype=float)


def _slot_of(joints_sample: np.ndarray, slots: np.ndarray, tol: float = 1e-6) -> int:
    # Compare modulo a full turn: batch_inverse keeps angles near the previous row.
    diff = (slots - joints_sample + 180.0) % 360.0 - 180.0
    matches = np.where(np.all(np.abs(diff) < tol, axis=1))[0]
    assert len(matches) >= 1, f"{joints_sample} not found in {slots}"
    return int(matches[0])


def _existing(joints: np.ndarray) -> np.ndarray:
    return ~np.isnan(joints[..., 0])


def _reach_fields(result: Any) -> list[str]:
    # n_solutions is py-ikgeo only; compare whatever the backend provides.
    return [
        f
        for f in ("joints", "limit_margin", "extension", "sigma_min", "wrist", "n_solutions")
        if hasattr(result, f)
    ]


# ---------------------------------------------------------------------------
# Module surface


def test_public_names(backend: Backend) -> None:
    for name in ("KinematicModel", "ReachResult", "RigidTransform", "Robot", "interpolate_poses"):
        assert name in backend.module.__all__
        assert hasattr(backend.module, name)
    assert backend.module.RigidTransform is RigidTransform


def test_robot_properties_and_repr(backend: Backend) -> None:
    robot = backend.robot(**NJ165_MM)
    assert robot.degrees is True
    assert backend.robot(degrees=False, **NJ165_MM).degrees is False
    assert "Robot" in repr(robot)
    assert "KinematicModel" in repr(backend.model(**NJ165_MM))


# ---------------------------------------------------------------------------
# Ported from py-opw test_robot.py


@pytest.mark.parametrize(
    "joints, expected_position, expected_orientation",
    [
        ([0, 0, -90, 0, 0, 0], [2073.926, 0.0, 2259.005], None),
        ([10, 0, -90, 0, 0, 0], [2042.418, -360.133, 2259.005], None),
        ([10, 10, -80, 0, 0, 0], [2243.792, -395.641, 2241.115], None),
        ([0, 0, -90, 0, 10, 0], [2070.432, 0.0, 2219.066], None),
        ([0, 0, -90, 10, 10, 0], [2070.432, -6.935, 2219.673], None),
        ([10, 20, -70, 30, 20, 10], [2418.558, -466.396, 2119.864], [-37.346, 25.987, -4.814]),
    ],
)
def test_forward_kinematics(
    robot: Any, joints: list[float], expected_position: list[float], expected_orientation: Any
) -> None:
    ee = RigidTransform.from_components(rotation=EE_ROTATION, translation=[0, 0, 0])
    result = robot.forward(joints=_j(joints), ee_transform=ee)
    assert np.allclose(result.translation, expected_position, atol=1e-3)
    if expected_orientation:
        euler = result.rotation.as_euler("XYZ", degrees=True)
        assert np.allclose(euler, expected_orientation, atol=1e-3)


@pytest.mark.parametrize(
    "joints, expected_position, expected_orientation",
    [
        ([0, 0, -90, 0, 0, 0], [2308.022, 132.200, 1707.280], [0, 0, 0]),
        ([10, 20, -70, 30, 20, 10], [2396.467, -743.091, 1572.479], [-37.346, 25.987, -4.814]),
    ],
)
def test_forward_kinematics_with_ee_translation(
    robot: Any, joints: list[float], expected_position: list[float], expected_orientation: Any
) -> None:
    ee = RigidTransform.from_components(
        rotation=EE_ROTATION, translation=[234.096, 132.2, -551.725]
    )
    result = robot.forward(joints=_j(joints), ee_transform=ee)
    assert np.allclose(result.translation, expected_position, atol=1e-3)
    euler = result.rotation.as_euler("XYZ", degrees=True)
    assert np.allclose(euler, expected_orientation, atol=1e-3)


def test_inverse_with_ee_translation(robot: Any) -> None:
    ee = RigidTransform.from_components(
        rotation=EE_ROTATION, translation=[234.096, 132.2, -551.725]
    )
    expected_joints = [10, 20, -70, 30, 20, 10]
    target = RigidTransform.from_components(
        rotation=Rotation.from_euler("XYZ", (-37.346, 25.987, -4.814), degrees=True),
        translation=[2396.467, -743.091, 1572.479],
    )
    solutions = robot.inverse(target, ee_transform=ee)
    assert any(np.allclose(s, expected_joints, atol=1e-2) for s in solutions)


@pytest.mark.parametrize("joints", [[-10, 0, -30, 10, 10, -10], [10, 20, -90, 30, 20, 10]])
@pytest.mark.parametrize("ee_translation", [[0, 0, 0], [100, 200, 300]])
@pytest.mark.parametrize("ee_rotation", [[0, 0, 0], [0, -90, 0], [30, 40, 60]])
@pytest.mark.parametrize("degrees", [True, False])
@pytest.mark.parametrize(
    "flip_axes",
    [(False, False, False, False, False, False), (False, True, False, False, True, False)],
)
@pytest.mark.parametrize("offsets", [(0, 0, 0, 0, 0, 0), (1, 2, 3, 4, 5, 6)])
def test_kinematics_roundtrip(
    backend: Backend,
    joints: list[float],
    ee_translation: list[float],
    ee_rotation: list[float],
    degrees: bool,
    flip_axes: tuple[bool, ...],
    offsets: tuple[float, ...],
) -> None:
    robot = backend.robot(
        degrees=degrees,
        a1=1,
        a2=2,
        b=3,
        c1=4,
        c2=5,
        c3=6,
        c4=7,
        offsets=offsets,
        flip_axes=flip_axes,
    )
    q = np.asarray(joints, dtype=float) if degrees else np.deg2rad(joints)
    ee = RigidTransform.from_components(
        rotation=Rotation.from_euler("xyz", ee_rotation, degrees=True), translation=ee_translation
    )
    forward_result = robot.forward(joints=_j(q), ee_transform=ee)
    solutions = robot.inverse(forward_result, ee_transform=ee)

    assert any(np.allclose(s, q, atol=1e-3) for s in solutions), (q, solutions)
    for s in solutions:
        back = robot.forward(joints=_j(s), ee_transform=ee)
        assert np.allclose(forward_result.translation, back.translation, atol=1e-3)
        assert np.allclose(
            forward_result.rotation.as_matrix(), back.rotation.as_matrix(), atol=1e-3
        )


@pytest.mark.parametrize(
    "initial_translation, joint_angles, expected_diff",
    [
        ([0, 0, 0], [0, 0, -90, 0, 0, 0], [0, 0, 0]),
        ([10, 20, 30], [0, 0, -90, 0, 0, 0], [10, 20, 30]),
        ([10, 20, 30], [90, 0, -90, 0, 0, 0], [20, -10, 30]),
    ],
)
def test_ee_translation(
    robot: Any,
    initial_translation: list[float],
    joint_angles: list[float],
    expected_diff: list[float],
) -> None:
    ee0 = RigidTransform.from_components(rotation=EE_ROTATION, translation=[0, 0, 0])
    ee1 = RigidTransform.from_components(rotation=EE_ROTATION, translation=initial_translation)
    initial = robot.forward(joints=_j(joint_angles), ee_transform=ee0)
    updated = robot.forward(joints=_j(joint_angles), ee_transform=ee1)
    assert np.allclose(updated.translation - initial.translation, expected_diff)


def test_batch_forward_random(robot: Any) -> None:
    ee = RigidTransform.from_rotation(EE_ROTATION)
    rng = np.random.default_rng(42)
    lo = [-180, -90, -180, -180, -90, -180]
    hi = [180, 90, 180, 180, 90, 180]
    joints_array = rng.uniform(lo, hi, size=(50, 6))
    poses = robot.batch_forward(joints_array, ee_transform=ee)
    assert len(poses) == 50
    for i in (0, 17, 49):
        single = robot.forward(_j(joints_array[i]), ee_transform=ee)
        assert np.allclose(poses[i].as_matrix(), single.as_matrix(), atol=1e-9)


def test_batch_inverse_random(robot: Any) -> None:
    ee = RigidTransform.from_rotation(EE_ROTATION)
    rng = np.random.default_rng(42)
    positions = rng.uniform([1500, -1000, 1000], [2500, 1000, 2500], size=(50, 3))
    euler = rng.uniform([-180, -90, -180], [180, 90, 180], size=(50, 3))
    poses = RigidTransform.from_components(
        rotation=Rotation.from_euler("xyz", euler, degrees=True), translation=positions
    )
    result = robot.batch_inverse(poses, ee_transform=ee)
    assert isinstance(result, np.ndarray)
    assert result.shape == (50, 6)
    # Every solved row reproduces its pose; unreachable poses are NaN rows, never dropped.
    solved = ~np.isnan(result[:, 0])
    back = robot.batch_forward(result[solved], ee_transform=ee).as_matrix()
    assert np.allclose(back, poses.as_matrix()[solved], atol=1e-6)


def test_batch_roundtrip(robot: Any) -> None:
    ee = RigidTransform.from_components(rotation=EE_ROTATION, translation=[100, 0, -500])
    n = 21
    joints_array = np.column_stack(
        [
            np.linspace(-180, 100, n),
            np.linspace(-80, 90, n),
            np.linspace(-170, 180, n),
            np.linspace(-600, 0, n),
            np.linspace(-80, 90, n),
            np.linspace(600, 0, n),
        ]
    )
    poses = robot.batch_forward(joints_array, ee_transform=ee)
    result = robot.batch_inverse(poses, current_joints=joints_array[0], ee_transform=ee)
    assert np.isclose(joints_array, result, atol=1e-3).all()


def test_batch_inverse_current_joints_signatures(backend: Backend, robot: Any) -> None:
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars")
    ee = RigidTransform.from_components(rotation=EE_ROTATION, translation=[0, 0, 0])
    poses = robot.batch_forward(
        np.array([[0, 0, -90, 0, 0, 0], [10, 0, -90, 0, 0, 0]]), ee_transform=ee
    )
    start = (0, 0, -90, 0, 0, 0)
    cols = ["J1", "J2", "J3", "J4", "J5", "J6"]

    result = robot.batch_inverse(poses, ee_transform=ee)
    assert isinstance(result, np.ndarray) and result.shape == (2, 6)
    result = robot.batch_inverse(poses, current_joints=start, ee_transform=ee)
    assert isinstance(result, np.ndarray) and result.shape == (2, 6)
    result = robot.batch_inverse(poses, current_joints=np.array(start), ee_transform=ee)
    assert isinstance(result, np.ndarray) and result.shape == (2, 6)

    pl_joints = pl.DataFrame({c: [float(v)] for c, v in zip(cols, start, strict=True)})
    result = robot.batch_inverse(poses, current_joints=pl_joints, ee_transform=ee)
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (2, 6) and list(result.columns) == cols

    if backend.name == "opw" and hasattr(pd.DataFrame, "select"):
        pytest.xfail("released py-opw-kinematics reads a pandas >= 3.1 frame as polars")
    pd_joints = pd.DataFrame({c: [float(v)] for c, v in zip(cols, start, strict=True)})
    result = robot.batch_inverse(poses, current_joints=pd_joints, ee_transform=ee)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 6) and list(result.columns) == cols


def test_batch_forward_accepts_dataframes(robot: Any) -> None:
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars")
    cols = ["J1", "J2", "J3", "J4", "J5", "J6"]
    joints = np.array([[0, 0, -90, 0, 0, 0], [10, 20, -70, 30, 20, 10]], dtype=float)
    expected = robot.batch_forward(joints).as_matrix()
    for frame in (pd.DataFrame(joints, columns=cols), pl.DataFrame(joints, schema=cols)):
        assert np.allclose(robot.batch_forward(frame).as_matrix(), expected)
        assert np.allclose(
            robot.batch_joint_poses(frame).as_matrix(),
            robot.batch_joint_poses(joints).as_matrix(),
        )


SIX_POSTURES = [
    (0, 0, -90, 0, 0, 0),
    (10, 0, -90, 0, 0, 0),
    (10, 10, -80, 0, 0, 0),
    (0, 0, -90, 0, 10, 0),
    (0, 0, -90, 10, 10, 0),
    (10, 20, -70, 30, 20, 10),
]


@pytest.mark.parametrize("joints", SIX_POSTURES)
def test_forward_frames_tcp(backend: Backend, robot: Any, joints: Joints) -> None:
    tcp = robot.forward(joints)
    with warnings.catch_warnings():
        # py-opw's forward_frames is deprecated; py-ikgeo's is not.
        warnings.simplefilter("ignore", DeprecationWarning)
        frames = robot.forward_frames(joints)
    assert len(frames) == 8
    assert np.allclose(tcp.translation, frames[-1].translation, atol=1e-9)
    if backend.name == "ikgeo":
        # py-opw's deprecated forward_frames has wrong rotations (its own tests xfail this).
        assert np.allclose(tcp.as_matrix(), frames[-1].as_matrix(), atol=1e-10)


@pytest.mark.parametrize("joints", SIX_POSTURES)
def test_joint_poses_tcp_matches_forward(robot: Any, joints: Joints) -> None:
    tcp = robot.forward(joints)
    poses = robot.joint_poses(joints)
    assert len(poses) == 6
    assert np.allclose(tcp.as_matrix(), poses[-1].as_matrix(), atol=1e-10)


def test_joint_poses_with_ee_transform(robot: Any) -> None:
    joints = (10, 20, -70, 30, 20, 10)
    ee = RigidTransform.from_components(rotation=EE_ROTATION, translation=[100, 200, 300])
    tcp = robot.forward(joints, ee_transform=ee)
    poses = robot.joint_poses(joints, ee_transform=ee)
    assert len(poses) == 7
    assert np.allclose(tcp.as_matrix(), poses[-1].as_matrix(), atol=1e-10)
    poses_no_ee = robot.joint_poses(joints)
    for i in range(6):
        assert np.allclose(poses[i].as_matrix(), poses_no_ee[i].as_matrix(), atol=1e-10)


@pytest.mark.parametrize("with_ee", [False, True])
def test_batch_joint_poses(robot: Any, with_ee: bool) -> None:
    ee = (
        RigidTransform.from_components(rotation=EE_ROTATION, translation=[100, 200, 300])
        if with_ee
        else None
    )
    joints_list = [(0, 0, -90, 0, 0, 0), (10, 10, -80, 0, 0, 0), (10, 20, -70, 30, 20, 10)]
    per_config = 7 if with_ee else 6
    batch = robot.batch_joint_poses(np.array(joints_list, dtype=float), ee_transform=ee)
    batch_matrices = batch.as_matrix().reshape(len(joints_list), per_config, 4, 4)
    for i, joints in enumerate(joints_list):
        single = robot.joint_poses(joints, ee_transform=ee).as_matrix()
        assert np.allclose(single, batch_matrices[i])


def test_interpolate_poses(backend: Backend) -> None:
    interpolate_poses = backend.module.interpolate_poses
    pose_start = RigidTransform.from_translation([0, 0, 0])
    pose_end = RigidTransform.from_components(
        rotation=Rotation.from_euler("X", 90, degrees=True), translation=[100, 200, 300]
    )
    keyframes = RigidTransform.concatenate([pose_start, pose_end])
    trajectory = interpolate_poses([0, 1], keyframes, np.linspace(0, 1, 11))
    assert len(trajectory) == 11
    assert np.allclose(trajectory[0].translation, pose_start.translation)
    assert np.allclose(trajectory[-1].translation, pose_end.translation)
    assert np.allclose(trajectory[5].translation, [50, 100, 150], atol=1e-6)
    assert np.allclose(trajectory[5].rotation.as_euler("xyz", degrees=True), [45, 0, 0])


def test_interpolate_poses_multi_keyframe(backend: Backend) -> None:
    interpolate_poses = backend.module.interpolate_poses
    poses = RigidTransform.from_components(
        rotation=Rotation.from_euler("ZYX", [0, 90, 180], degrees=True),
        translation=[[0, 0, 0], [100, 0, 0], [100, 100, 0]],
    )
    trajectory = interpolate_poses([0, 1, 2], poses, [0, 0.5, 1, 1.5, 2])
    assert len(trajectory) == 5
    assert np.allclose(trajectory[0].translation, [0, 0, 0])
    assert np.allclose(trajectory[2].translation, [100, 0, 0])
    assert np.allclose(trajectory[4].translation, [100, 100, 0])


# ---------------------------------------------------------------------------
# Ported from py-opw test_reach.py (NJ165 in metres)


def test_home_pose(backend: Backend, robot_m: Any) -> None:
    pose = robot_m.forward(_j(HOME))
    assert np.allclose(pose.translation, [0.400 + 1.444 + 0.230, 0.0, 0.830 + 1.175 + 0.250])

    result = robot_m.reach(pose)
    assert isinstance(result, backend.module.ReachResult)
    # py-opw has 8 stable slots, py-ikgeo up to 16 sorted columns.
    assert result.joints.shape[0] == 1 and result.joints.shape[2] == 6
    assert result.joints.shape[1] in (8, 16)
    assert result.limit_margin.shape == result.joints.shape[:2]
    assert result.sigma_min.shape == result.joints.shape[:2]
    assert result.wrist.shape == result.joints.shape[:2]
    assert result.extension.shape == (1,)
    _slot_of(HOME, result.joints[0])

    exists = _existing(result.joints[0])
    for q in result.joints[0][exists]:
        back = robot_m.forward(_j(q))
        assert np.allclose(back.translation, pose.translation, atol=1e-9)
        assert np.allclose(back.rotation.as_matrix(), pose.rotation.as_matrix(), atol=1e-9)

    # J5 == 0 on the home branch is the wrist singularity. py-opw fills both wrist-flip
    # slots with the same joints; py-ikgeo deduplicates them into one column.
    at_singularity = exists & (np.abs(result.joints[0, :, 4]) < 1e-9)
    assert at_singularity.sum() >= 1
    assert np.all(result.wrist[0, at_singularity] < 1e-12)
    assert np.all(result.sigma_min[0, at_singularity] < 1e-6)
    assert np.all(np.isinf(result.limit_margin[0, exists]))


def test_round_trip_through_fk(backend: Backend, robot_m: Any) -> None:
    samples = _grid_joints()
    poses = robot_m.batch_forward(samples)
    result = robot_m.reach(poses, NJ165_LIMITS)

    assert result.joints.shape[0] == len(samples)
    expected_margin = _margin(samples, NJ165_LIMITS)
    slots = np.array([_slot_of(q, result.joints[i]) for i, q in enumerate(samples)])
    found_margin = result.limit_margin[np.arange(len(samples)), slots]
    assert np.allclose(found_margin, expected_margin, atol=1e-6)

    if backend.name == "opw":
        # py-opw's slot index is a function of the sign choices only; py-ikgeo sorts
        # columns per pose instead (see its ReachResult docstring), so only py-opw is
        # checked against the slot formula.
        matrices = poses.as_matrix()
        wrist_centre = matrices[:, :3, 3] - 0.230 * matrices[:, :3, 2]
        theta1_front = np.degrees(np.arctan2(wrist_centre[:, 1], wrist_centre[:, 0]))
        d1 = (-samples[:, 0] - theta1_front + 180.0) % 360.0 - 180.0
        shoulder = (np.abs(d1) > 90.0).astype(int)
        raw_theta3 = -samples[:, 2] + np.degrees(np.arctan2(-0.250, 1.444))
        elbow = (raw_theta3 < 0).astype(int)
        wrist_flip = (samples[:, 4] < 0).astype(int)
        assert np.array_equal(slots, 4 * wrist_flip + 2 * shoulder + elbow)

    # No sample was near a boundary, so every existing column reproduces the pose.
    exists = _existing(result.joints)
    back = robot_m.batch_forward(result.joints[exists]).as_matrix()
    target = np.repeat(poses.as_matrix()[:, None], result.joints.shape[1], axis=1)[exists]
    assert np.allclose(back, target, atol=1e-9)


def test_superset_of_batch_inverse(robot_m: Any) -> None:
    samples = _grid_joints()[::7]
    poses = robot_m.batch_forward(samples)
    single = np.asarray(robot_m.batch_inverse(poses, current_joints=_j(HOME)))
    result = robot_m.reach(poses)
    for i in range(len(samples)):
        if np.isnan(single[i, 0]):
            continue
        _slot_of(single[i], result.joints[i])
        assert np.any(_existing(result.joints[i]))


def test_branch_existence_is_order_independent(robot_m: Any) -> None:
    rng = np.random.default_rng(0)
    samples = _grid_joints()[::5]
    poses = robot_m.batch_forward(samples)
    perm = rng.permutation(len(samples))
    shuffled = poses[perm]
    a = robot_m.reach(poses, NJ165_LIMITS)
    b = robot_m.reach(shuffled, NJ165_LIMITS)
    inv = np.argsort(perm)
    for name in _reach_fields(a):
        assert np.array_equal(getattr(a, name), getattr(b, name)[inv], equal_nan=True)


def test_extension_margin(robot_m: Any) -> None:
    # Walk the TCP radially outward from the shoulder at the shoulder height.
    rotation = robot_m.forward(_j(HOME)).rotation
    radius = np.linspace(2.2, 3.5, 1301)
    translations = np.stack([radius, np.zeros_like(radius), np.full_like(radius, 0.830)], axis=1)
    poses = RigidTransform.from_components(
        rotation=Rotation.concatenate([rotation] * len(radius)), translation=translations
    )
    result = robot_m.reach(poses)

    ext = result.extension
    assert np.all(np.diff(ext) < 0)
    d = radius - 0.230 - 0.400
    length = np.hypot(-0.250, 1.444)
    expected = np.minimum(1.175 + length - d, d - abs(1.175 - length))
    assert np.allclose(ext, expected, atol=1e-12)

    exists = np.any(_existing(result.joints), axis=1)
    last = np.max(np.where(exists)[0])
    assert ext[last] >= 0
    assert ext[last + 1] < 0
    assert not np.any(exists[last + 1 :])
    step = np.diff(ext)
    assert np.allclose(step, step[0], atol=1e-12)


def test_extension_back_shoulder(robot_m: Any) -> None:
    # Reachable only through the back-shoulder branches: extension must not be negative.
    pose = RigidTransform.from_components(
        rotation=robot_m.forward(_j(HOME)).rotation, translation=[-0.45, 0.0, 0.80]
    )
    result = robot_m.reach(pose)
    exists = _existing(result.joints[0])
    assert exists.any()
    # Backend-neutral form of py-opw's slot check: every existing branch reaches back,
    # that is J1 points away from the wrist centre by more than 90 degrees.
    wrist_centre = pose.translation - 0.230 * pose.rotation.as_matrix()[:, 2]
    theta1_front = np.degrees(np.arctan2(wrist_centre[1], wrist_centre[0]))
    d1 = (-result.joints[0, exists, 0] - theta1_front + 180.0) % 360.0 - 180.0
    assert np.all(np.abs(d1) > 90.0)
    assert result.extension[0] >= 0


def test_wrist_centre_inside_b_cylinder(backend: Backend) -> None:
    robot = backend.robot(**{**NJ165_M, "b": 0.15})
    pose = RigidTransform.from_components(
        rotation=Rotation.identity(), translation=[0.0, 0.0, 2.0 + 0.230]
    )
    result = robot.reach(pose)
    assert np.isfinite(result.extension[0])
    assert result.extension[0] < 0
    assert np.all(np.isnan(result.joints[0]))


def test_reach_result_equality_does_not_raise(robot_m: Any) -> None:
    a = robot_m.reach(robot_m.forward(_j(HOME)))
    b = robot_m.reach(robot_m.forward(_j(HOME)))
    assert bool(a == b) is False
    assert a != b
    hash(a)


def _numeric_jacobian(robot: Any, q: np.ndarray, ee: RigidTransform | None = None) -> np.ndarray:
    jac = np.zeros((6, 6))
    h = 1e-4
    for j in range(6):
        plus, minus = q.copy(), q.copy()
        plus[j] += h
        minus[j] -= h
        fp = robot.forward(_j(plus), ee_transform=ee)
        fm = robot.forward(_j(minus), ee_transform=ee)
        jac[:3, j] = (fp.translation - fm.translation) / np.radians(2 * h)
        jac[3:, j] = (fp.rotation * fm.rotation.inv()).as_rotvec() / np.radians(2 * h)
    return jac


def test_sigma_min_matches_finite_difference(robot_m: Any) -> None:
    rng = np.random.default_rng(1)
    lo = np.maximum(NJ165_LIMITS[:, 0], -170)
    hi = np.minimum(NJ165_LIMITS[:, 1], 170)
    samples = rng.uniform(lo, hi, size=(200, 6))
    # Keep the wrist away from its singularity so sigma_min is not dominated by it.
    samples[:, 4] = np.where(np.abs(samples[:, 4]) < 10, 20.0, samples[:, 4])
    result = robot_m.reach(robot_m.batch_forward(samples))
    for i, q in enumerate(samples):
        s = _slot_of(q, result.joints[i])
        numeric = np.linalg.svd(_numeric_jacobian(robot_m, q), compute_uv=False).min()
        assert abs(result.sigma_min[i, s] - numeric) / numeric < 1e-3


def test_sigma_min_with_ee_transform(robot_m: Any) -> None:
    ee = RigidTransform.from_components(rotation=EE_ROTATION, translation=[0.3, 0.1, 0.05])
    q = np.array([20.0, 15.0, -70.0, 40.0, 50.0, -30.0])
    result = robot_m.reach(robot_m.forward(_j(q), ee_transform=ee), ee_transform=ee)
    s = _slot_of(q, result.joints[0])
    numeric = np.linalg.svd(_numeric_jacobian(robot_m, q, ee), compute_uv=False).min()
    assert abs(result.sigma_min[0, s] - numeric) / numeric < 1e-3


def test_singular_configurations(robot_m: Any) -> None:
    # Wrist singularity: J5 == 0. py-opw fills both wrist-flip slots with the same
    # joints; py-ikgeo deduplicates them into one column.
    q = np.array([30.0, 20.0, -100.0, 45.0, 0.0, 10.0])
    pose = robot_m.forward(_j(q))
    result = robot_m.reach(pose)
    exists = _existing(result.joints[0])
    at_singularity = exists & (np.abs(result.joints[0, :, 4]) < 1e-9)
    assert at_singularity.sum() >= 1
    for s in np.where(at_singularity)[0]:
        assert result.wrist[0, s] < 1e-12
        assert result.sigma_min[0, s] < 1e-6
        # The J4/J6 split is arbitrary at the singularity; the pose must still hold.
        back = robot_m.forward(_j(result.joints[0, s]))
        assert np.allclose(back.as_matrix(), pose.as_matrix(), atol=1e-9)

    # Straight arm: place the wrist centre exactly on the outer reach circle. The
    # summation order is py-opw's own; the boundary case is sensitive to the last ulp.
    length = np.hypot(-0.250, 1.444)
    d = 1.175 + length
    wrist_x = 0.400 + d
    tcp_x = wrist_x + 0.230
    pose = RigidTransform.from_components(
        rotation=robot_m.forward(_j(HOME)).rotation, translation=[tcp_x, 0.0, 0.830]
    )
    result = robot_m.reach(pose)
    assert abs(result.extension[0]) < 1e-12
    exists = _existing(result.joints[0])
    assert np.any(exists)
    assert np.all(result.sigma_min[0, exists] < 1e-6)


def test_limits_are_reported_not_enforced(robot_m: Any) -> None:
    q = np.array([10.0, 80.0, -100.0, 20.0, 40.0, 30.0])
    tight = NJ165_LIMITS.copy()
    tight[1] = (-75, 75)
    result = robot_m.reach(robot_m.forward(_j(q)), tight)
    s = _slot_of(q, result.joints[0])
    assert np.isclose(result.limit_margin[0, s], -5.0, atol=1e-9)


def test_nan_pose_row(robot_m: Any) -> None:
    matrices = robot_m.batch_forward(np.array([HOME, HOME])).as_matrix()
    matrices[1, 0, 3] = np.nan
    # Bypass RigidTransform validation by going through the internal call. py-ikgeo
    # returns one extra array (n_solutions).
    joints, margin, ext, *_rest = robot_m._robot.reach(
        np.ascontiguousarray(matrices.reshape(-1, 16)), None, None
    )
    assert np.all(np.isnan(joints[1]))
    assert np.isnan(ext[1])
    assert np.all(np.isnan(margin[1]))
    assert not np.all(np.isnan(joints[0]))


def test_batch_inverse_nan_row(robot_m: Any) -> None:
    # Away from J5 = 0, where the J4/J6 split is not unique.
    q = np.array([10.0, 20.0, -70.0, 30.0, 20.0, 10.0])
    matrices = robot_m.batch_forward(np.array([q, q, q])).as_matrix()
    matrices[1, 0, 3] = np.nan
    joints = robot_m._robot.batch_inverse(
        np.ascontiguousarray(matrices.reshape(-1, 16)), _j(q), None
    )
    assert joints.shape == (3, 6)
    assert np.all(np.isnan(joints[1]))
    assert np.allclose(joints[0], q, atol=1e-6)
    assert np.allclose(joints[2], q, atol=1e-6)


def test_radians_robot(backend: Backend) -> None:
    robot_rad = backend.robot(degrees=False, **NJ165_M)
    q = np.radians(np.array([10.0, 20.0, -70.0, 30.0, 20.0, 10.0]))
    limits = np.radians(NJ165_LIMITS)
    result = robot_rad.reach(robot_rad.forward(_j(q)), limits)
    diff = (result.joints[0] - q + np.pi) % (2 * np.pi) - np.pi
    s = int(np.where(np.all(np.abs(diff) < 1e-6, axis=1))[0][0])
    assert np.isclose(result.limit_margin[0, s], _margin(q, limits))
    # The same pose in degrees gives the same branch set.
    robot_deg = backend.robot(**NJ165_M)
    result_deg = robot_deg.reach(robot_deg.forward(_j(np.degrees(q))))
    assert np.array_equal(_existing(result.joints), _existing(result_deg.joints))


def test_joint_limits_shape_validation(robot_m: Any) -> None:
    with pytest.raises(ValueError):
        robot_m.reach(robot_m.forward(_j(HOME)), np.zeros((5, 2)))


def test_sigma_min_skipped_outside_limits(robot_m: Any) -> None:
    q = np.array([10.0, 20.0, -70.0, 30.0, 40.0, 10.0])
    tight = NJ165_LIMITS.copy()
    # Keeps the front-shoulder branches only; the back-shoulder ones need J1 near 180.
    tight[0] = (-30, 30)
    pose = robot_m.forward(_j(q))
    limited = robot_m.reach(pose, tight)
    unlimited = robot_m.reach(pose)

    outside = limited.limit_margin[0] < 0
    assert np.any(outside)
    assert np.all(np.isnan(limited.sigma_min[0, outside]))
    inside = limited.limit_margin[0] >= 0
    assert np.any(inside)
    assert np.array_equal(limited.sigma_min[0, inside], unlimited.sigma_min[0, inside])


# None is omitted rather than passed: released py-opw-kinematics still types threads as int.
@pytest.mark.parametrize("threads", [None, 2, 4])
def test_threads_give_identical_results(robot_m: Any, threads: int | None) -> None:
    poses = robot_m.batch_forward(_grid_joints()[::3])
    single = robot_m.reach(poses, NJ165_LIMITS, threads=1)
    kwargs = {} if threads is None else {"threads": threads}
    threaded = robot_m.reach(poses, NJ165_LIMITS, **kwargs)
    for name in _reach_fields(single):
        assert np.array_equal(getattr(single, name), getattr(threaded, name), equal_nan=True)


def test_threads_release_the_gil(robot_m: Any) -> None:
    # A second Python thread must keep running while reach is in the kernel.
    poses = robot_m.batch_forward(_grid_joints())
    # Repeat the grid so the call outlasts coarse sleep timers on macOS CI runners.
    poses = poses[np.tile(np.arange(len(poses)), 8)]
    ticks = 0
    done = threading.Event()

    def spin() -> None:
        nonlocal ticks
        while not done.is_set():
            ticks += 1
            time.sleep(0.001)

    spinner = threading.Thread(target=spin)
    spinner.start()
    try:
        while ticks == 0:
            time.sleep(0.001)
        before = ticks
        t0 = time.perf_counter()
        robot_m.reach(poses, NJ165_LIMITS)
        elapsed = time.perf_counter() - t0
    finally:
        done.set()
        spinner.join()
    # The spinner sleeps 1 ms per tick, so a GIL-holding kernel would stall it.
    assert elapsed > 0.1, "kernel too fast to tell; use a larger grid"
    assert ticks - before > 5


def test_negative_threads_rejected(robot_m: Any) -> None:
    with pytest.raises(ValueError):
        robot_m.reach(robot_m.forward(_j(HOME)), threads=-1)
