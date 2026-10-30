"""reach on an OPW robot given as h, p must contain py-opw-kinematics' branches with equal margins."""

from typing import Any

import numpy as np
import py_ikgeo_kinematics as ikg
import py_opw_kinematics as opw
import pytest
from scipy.spatial.transform import RigidTransform, Rotation

# Comau NJ165 in metres as py-opw's own reach tests parameterise it.
NJ165: dict[str, Any] = {
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
# A lateral offset b exercises the cylinder clamp of extension.
OFFSET_B: dict[str, Any] = {**NJ165, "b": 0.3, "offsets": (5.0, -10.0, 15.0, 0.0, 20.0, -30.0)}
N = 500
TOL_DEG = 1e-6


def make_pair(params: dict[str, Any]) -> tuple[opw.Robot, ikg.Robot]:
    return (
        opw.Robot(opw.KinematicModel(**params), degrees=True),
        ikg.Robot(ikg.KinematicModel.from_opw(**params), degrees=True),
    )


@pytest.fixture(scope="module", params=["nj165", "offset_b"])
def robots(request: pytest.FixtureRequest) -> tuple[opw.Robot, ikg.Robot]:
    return make_pair(NJ165 if request.param == "nj165" else OFFSET_B)


@pytest.fixture(params=[None, "ee"])
def ee_transform(request: pytest.FixtureRequest) -> RigidTransform | None:
    if request.param is None:
        return None
    rot = Rotation.from_euler("xyz", [10, -90, 25], degrees=True)
    return RigidTransform.from_components(rotation=rot, translation=[0.012, -0.0075, 0.150])


@pytest.fixture(params=[None, "limits"])
def limits(request: pytest.FixtureRequest) -> np.ndarray | None:
    return None if request.param is None else NJ165_LIMITS


@pytest.fixture(scope="module")
def joints() -> np.ndarray:
    rng = np.random.default_rng(777)
    return rng.uniform(-180, 180, size=(N, 6))


def match_columns(theirs: np.ndarray, ours: np.ndarray) -> list[tuple[int, int]]:
    """Pairs (py-opw slot, our column) for every non-NaN py-opw branch."""
    pairs = []
    for s in range(theirs.shape[0]):
        if np.isnan(theirs[s, 0]):
            continue
        diff = (ours - theirs[s] + 180.0) % 360.0 - 180.0
        hits = np.where(np.all(np.abs(diff) < TOL_DEG, axis=1))[0]
        assert len(hits) == 1, f"slot {s} {theirs[s]} matched {hits} in {ours}"
        pairs.append((s, int(hits[0])))
    return pairs


def test_branches_and_margins_match(
    robots: tuple[opw.Robot, ikg.Robot],
    joints: np.ndarray,
    ee_transform: RigidTransform | None,
    limits: np.ndarray | None,
) -> None:
    a, b = robots
    poses = a.batch_forward(joints, ee_transform=ee_transform)
    theirs = a.reach(poses, limits, ee_transform=ee_transform)
    ours = b.reach(poses, limits, ee_transform=ee_transform)
    assert ours.joints.shape == (N, 16, 6)
    assert ours.limit_margin.shape == (N, 16)
    assert ours.n_solutions.shape == (N,)
    assert np.all(np.isnan(ours.extension) == np.isnan(theirs.extension))
    assert np.allclose(ours.extension, theirs.extension, atol=1e-9)
    for i in range(N):
        pairs = match_columns(theirs.joints[i], ours.joints[i])
        assert ours.n_solutions[i] == len(pairs)
        assert np.all(np.isnan(ours.joints[i, ours.n_solutions[i] :, 0]))
        for s, c in pairs:
            assert ours.joints[i, c, 0] == pytest.approx(ours.joints[i, c, 0])
            assert np.all(np.abs(ours.joints[i, c]) <= 180.0 + 1e-12)
            assert ours.limit_margin[i, c] == pytest.approx(theirs.limit_margin[i, s], abs=1e-6)
            if np.isnan(theirs.sigma_min[i, s]):
                assert np.isnan(ours.sigma_min[i, c])
                assert theirs.limit_margin[i, s] < 0
            else:
                assert ours.sigma_min[i, c] == pytest.approx(theirs.sigma_min[i, s], rel=1e-6)
            assert np.sin(np.deg2rad(ours.wrist[i, c])) == pytest.approx(
                theirs.wrist[i, s], abs=1e-9
            )
            assert ours.wrist[i, c] >= -1e-12


def test_extension_inside_offset_cylinder() -> None:
    a, b = make_pair(OFFSET_B)
    # Wrist centres closer to axis 1 than b: both report the negative distance to the cylinder.
    rng = np.random.default_rng(5)
    centres = np.column_stack(
        [rng.uniform(-0.2, 0.2, 50), rng.uniform(-0.2, 0.2, 50), rng.uniform(0.5, 2.5, 50)]
    )
    flange = centres + 0.230 * np.array([0.0, 0.0, 1.0])
    poses = RigidTransform.from_translation(flange)
    theirs = a.reach(poses)
    ours = b.reach(poses)
    assert np.all(ours.extension < 0)
    assert np.allclose(ours.extension, theirs.extension, atol=1e-9)
    assert np.all(ours.n_solutions == 0)


def test_nan_rows_and_threads(robots: tuple[opw.Robot, ikg.Robot], joints: np.ndarray) -> None:
    _, b = robots
    # scipy refuses NaN matrices, so the internal (n, 16) entry point is used directly.
    arr = b.batch_forward(joints[:40]).as_matrix()
    arr[3] = np.nan
    arr[17, 1, 2] = np.nan
    one = b._robot.reach(np.ascontiguousarray(arr.reshape(-1, 16)), None, None, threads=1)
    many = b._robot.reach(np.ascontiguousarray(arr.reshape(-1, 16)), None, None)
    for x, y in zip(one, many):
        assert np.array_equal(x, y, equal_nan=True)
    joints_out, margin, extension, sigma, wrist, count = one
    for i in (3, 17):
        assert np.all(np.isnan(joints_out[i]))
        assert np.all(np.isnan(margin[i])) and np.isnan(extension[i])
        assert np.all(np.isnan(sigma[i])) and np.all(np.isnan(wrist[i]))
        assert count[i] == 0
    assert count[0] > 0


def test_reach_validates_arguments(robots: tuple[opw.Robot, ikg.Robot]) -> None:
    _, b = robots
    pose = b.forward((0.0, 0.0, -90.0, 0.0, 30.0, 0.0))
    with pytest.raises(ValueError):
        b.reach(pose, threads=-1)
    with pytest.raises(ValueError):
        b.reach(pose, joint_limits=np.zeros((5, 2)))
    result = b.reach(pose)
    assert isinstance(result, ikg.ReachResult)
    assert result.joints.shape == (1, 16, 6)
