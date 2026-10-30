"""
py-ikgeo-kinematics: forward and inverse kinematics for general 6R arms.

Same surface as py-opw-kinematics (``Robot``, ``forward``, ``inverse``,
``batch_inverse``, ``reach``), driven by a product-of-exponentials model
instead of OPW lengths, so a wrist whose axes do not meet at one point
(Comau NJ4, Kawasaki KJ, FANUC CRX) is solvable. See PLAN.md.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import RigidTransform

from ._internal import KinematicModel
from ._internal import Robot as _RobotInternal


class _DataFrameLike(Protocol):
    """pandas or polars DataFrame with columns J1..J6."""

    def to_numpy(self) -> np.ndarray: ...


_JOINT_COLS = ["J1", "J2", "J3", "J4", "J5", "J6"]


def _is_polars(frame: object) -> bool:
    # pandas 3.1 added DataFrame.select, so duck typing on it misreads a pandas frame.
    return type(frame).__module__.split(".")[0] == "polars"
Joints: TypeAlias = tuple[float, float, float, float, float, float]
JointsInput: TypeAlias = ArrayLike | _DataFrameLike


def _joints_array(joints: JointsInput) -> np.ndarray:
    frame: Any = joints
    if hasattr(frame, "to_numpy"):
        if _is_polars(frame):
            arr = frame.select(_JOINT_COLS).to_numpy()
        else:
            arr = frame[_JOINT_COLS].to_numpy()
    else:
        arr = np.asarray(joints)
    return np.ascontiguousarray(arr, dtype=np.float64)


def _ee_matrix(ee_transform: RigidTransform | None) -> np.ndarray | None:
    return None if ee_transform is None else ee_transform.as_matrix()


# Pairs this close to parallel or intersecting are named in the ``general`` warning.
_NEAR_MISS = 1e-4


@dataclass(frozen=True)
class AxisPairReport:
    """
    Measured geometry of one consecutive axis pair, before the tolerance snaps it.

    :ivar pair: ``"1-2"`` .. ``"5-6"``.
    :ivar parallel_deviation: angle between the two axis lines in radians (0 for parallel or
        anti-parallel axes).
    :ivar intersection_deviation: shortest distance between the two axis lines divided by the
        longest offset of the model; NaN for a parallel pair.
    :ivar parallel: whether the pair was treated as parallel.
    :ivar intersecting: whether the pair was treated as intersecting.
    """

    pair: str
    parallel_deviation: float
    intersection_deviation: float
    parallel: bool
    intersecting: bool


class Robot:
    """
    Robot kinematics with scipy RigidTransform integration.

    Joint angles are in degrees by default; poses are scipy RigidTransform objects.
    """

    def __init__(
        self,
        kinematic_model: KinematicModel,
        degrees: bool = True,
        *,
        axis_tolerance: float = 1e-9,
    ) -> None:
        """
        :param kinematic_model: Axis directions and offsets of the arm.
        :param degrees: If True, joint angles are in degrees. If False, radians.
        :param axis_tolerance: Consecutive axes within this angle (radians) are treated as
            parallel, and within this fraction of the longest offset as intersecting; the
            model is then snapped to the exact relation for the solver. The default suits
            exact models; a model read from CAD or a calibration may need 1e-6 or so, see
            :meth:`axis_report`.
        """
        self._robot = _RobotInternal(kinematic_model, degrees, axis_tolerance=axis_tolerance)
        self._degrees = degrees
        self._kinematic_model = kinematic_model
        if self._robot.family == "general":
            near = [
                r.pair
                for r in self.axis_report()
                if not (r.parallel or r.intersecting)
                and (r.parallel_deviation < _NEAR_MISS or r.intersection_deviation < _NEAR_MISS)
            ]
            hint = (
                f" Pairs {near} are within {_NEAR_MISS:g} of parallel or intersecting; "
                "raise axis_tolerance or snap the model."
                if near
                else ""
            )
            warnings.warn(
                "no consecutive axes are parallel or intersecting: the general 2D search "
                "solver is slow and misses roots." + hint,
                UserWarning,
                stacklevel=2,
            )

    def axis_report(self) -> list[AxisPairReport]:
        """Measured parallel and intersection deviations of the five consecutive axis pairs."""
        return [AxisPairReport(*row) for row in self._robot.axis_report()]

    @property
    def degrees(self) -> bool:
        """Whether joint angles are in degrees (True) or radians (False)."""
        return self._degrees

    @property
    def kinematic_model(self) -> KinematicModel:
        return self._kinematic_model

    @property
    def family(self) -> str:
        """IK-Geo family detected from the axis geometry, e.g. ``"spherical_two_parallel"``."""
        return self._robot.family

    def __repr__(self) -> str:
        return self._robot.__repr__()

    def forward(self, joints: Joints, ee_transform: RigidTransform | None = None) -> RigidTransform:
        """TCP pose for one joint vector."""
        return RigidTransform.from_matrix(self._robot.forward(joints, _ee_matrix(ee_transform)))

    def inverse(
        self,
        pose: RigidTransform,
        current_joints: Joints | None = None,
        ee_transform: RigidTransform | None = None,
    ) -> list[Joints]:
        """
        Joint solutions found for one TCP pose by the cold solver.

        Search-based families can miss roots. An empty result means no root was found;
        it does not prove the pose is unreachable.

        Each joint is unwrapped to the turn nearest ``current_joints`` and the list is
        ordered by the summed absolute joint distance to it, as py-opw does. Without a seed
        the zero joint vector (home) is the seed, so the first element is the posture
        closest to home and the joints are in [-180, 180), or [-pi, pi) in radians.
        """
        return self._robot.inverse(pose.as_matrix(), current_joints, _ee_matrix(ee_transform))

    def batch_inverse(
        self,
        poses: RigidTransform,
        current_joints: JointsInput | None = None,
        ee_transform: RigidTransform | None = None,
    ) -> Any:
        """
        Joint path for N poses, each row continuing the previous one (tracked solve).

        Every pose is warm-started from the previous row with Gauss-Newton, which follows
        the branch the previous row is on (continuation). When that fails (first pose, a joint
        step above the tracking limit, no convergence) the cold solution nearest the previous
        row by summed joint distance is taken instead, as ``inverse(pose, current_joints=prev)[0]``
        would. Where two branches run close together, for example near the J5 = 180 fold of a
        non-spherical wrist, continuation and nearest can be different valid roots; this method
        keeps continuation, so it is not always equal to a loop over ``inverse``. Unsolvable or
        NaN poses give NaN rows and do not break the chain.

        :param poses: RigidTransform containing N poses.
        :param current_joints: Seed for the first pose: tuple, array, or single-row DataFrame.
        :param ee_transform: End effector transformation (optional).
        :return: (N, 6) joints, as a DataFrame of the same type when the seed was one.
        """
        output_type: Any = None
        output_kwargs: dict[str, Any] = {}
        current_joints_tuple: Joints | None = None
        if current_joints is not None:
            frame: Any = current_joints
            if hasattr(frame, "to_numpy"):
                if _is_polars(frame):
                    arr = frame.select(_JOINT_COLS).to_numpy()
                    output_kwargs = {"schema": _JOINT_COLS}
                else:
                    arr = frame[_JOINT_COLS].to_numpy()
                    output_kwargs = {"columns": _JOINT_COLS}
                output_type = type(current_joints)
            else:
                arr = np.atleast_2d(np.asarray(current_joints))
            row = np.ascontiguousarray(arr, dtype=np.float64)[0]
            current_joints_tuple = (row[0], row[1], row[2], row[3], row[4], row[5])

        matrix_array = np.ascontiguousarray(poses.as_matrix().reshape(-1, 16), dtype=np.float64)
        result = self._robot.batch_inverse(
            matrix_array, current_joints_tuple, _ee_matrix(ee_transform)
        )
        return output_type(result, **output_kwargs) if output_type else result

    def batch_forward(
        self, joints: JointsInput, ee_transform: RigidTransform | None = None
    ) -> RigidTransform:
        """TCP poses for (N, 6) joints, or a DataFrame with columns J1..J6."""
        result = self._robot.batch_forward(_joints_array(joints), _ee_matrix(ee_transform))
        return RigidTransform.from_matrix(result.reshape(-1, 4, 4))

    def joint_poses(
        self, joints: Joints, ee_transform: RigidTransform | None = None
    ) -> RigidTransform:
        """Frames [J1, ..., J6] on the joint axes, plus TCP when ee_transform is given."""
        raw = np.array(self._robot.joint_poses(joints, _ee_matrix(ee_transform)))
        return RigidTransform.from_matrix(raw)

    def batch_joint_poses(
        self, joints: JointsInput, ee_transform: RigidTransform | None = None
    ) -> RigidTransform:
        """Per-joint frames for N joint vectors as N*6 (or N*7) poses."""
        result = self._robot.batch_joint_poses(_joints_array(joints), _ee_matrix(ee_transform))
        return RigidTransform.from_matrix(result.reshape(-1, 4, 4))

    def forward_frames(
        self, joints: Joints, ee_transform: RigidTransform | None = None
    ) -> RigidTransform:
        """Frames [Base, J1, ..., J6, TCP]."""
        raw = np.array(self._robot.forward_frames(joints, _ee_matrix(ee_transform)))
        return RigidTransform.from_matrix(raw)

    def jacobian(self, joints: Joints, ee_transform: RigidTransform | None = None) -> np.ndarray:
        """6x6 geometric Jacobian of the TCP (translation rows first) per radian of each joint."""
        return self._robot.jacobian(joints, _ee_matrix(ee_transform))

    def reach(
        self,
        poses: RigidTransform,
        joint_limits: ArrayLike | None = None,
        ee_transform: RigidTransform | None = None,
        *,
        threads: int | None = None,
    ) -> "ReachResult":
        """
        Inverse-kinematics branches found for each pose, with per-branch margins.

        Unlike :meth:`batch_inverse` there is no continuity selection and no limit
        filtering: found roots are returned, sorted per pose by the key described in
        :class:`ReachResult`, and ``limit_margin`` reports how far inside (or outside)
        the joint limits it is.

        :param poses: RigidTransform containing N poses.
        :param joint_limits: (6, 2) lower/upper joint bounds in the robot's angle unit
            (optional). Without limits every margin is ``+inf``. With limits,
            ``sigma_min`` is left NaN for branches outside them.
        :param ee_transform: End effector transformation (optional).
        :param threads: Keyword-only worker cap for the per-pose loop. None (the default)
            uses the process-wide Rayon pool, so ``RAYON_NUM_THREADS`` applies; 1 runs
            serially in the calling thread; n >= 2 uses a pool of n threads, built once
            per n and kept for the life of the process. Below 1 raises ValueError,
            except 0, a deprecated alias for None. Results do not depend on it, and the
            GIL is released either way.
        :return: ReachResult with per-pose, per-branch arrays.
        """
        if threads == 0:
            warnings.warn(
                "threads=0 is deprecated; use threads=None for the global pool",
                DeprecationWarning,
                stacklevel=2,
            )
            threads = None
        matrix_array = np.ascontiguousarray(poses.as_matrix().reshape(-1, 16), dtype=np.float64)
        limits = None
        if joint_limits is not None:
            limits_array = np.asarray(joint_limits, dtype=np.float64)
            if limits_array.shape != (6, 2):
                raise ValueError("joint_limits must have shape (6, 2)")
            limits = [tuple(row) for row in limits_array]
        joints, limit_margin, extension, sigma_min, wrist, n_solutions = self._robot.reach(
            matrix_array, limits, _ee_matrix(ee_transform), threads=threads
        )
        return ReachResult(
            joints=joints,
            limit_margin=limit_margin,
            extension=extension,
            sigma_min=sigma_min,
            wrist=wrist,
            n_solutions=n_solutions,
        )


# eq=False: the generated __eq__ and __hash__ do not work on ndarray fields.
@dataclass(frozen=True, eq=False)
class ReachResult:
    """
    Found inverse-kinematics branches for each pose, with per-branch margins.

    Up to 16 branches per pose. Within a pose the columns are sorted by
    ``(shoulder, elbow, wrist_flip, J4)``: shoulder 0 has the wrist point, seen in
    the J1 frame, on the side of axis 1 towards which a positive J2 swings the arm
    from the zero pose (py-opw's front shoulder for every OPW model), elbow 0 has a non-negative bend
    between upper arm and forearm about the parallel axes 2/3 (the sign of J3 when
    they are not parallel), wrist_flip 0 has J5 >= 0, and ties are broken by J4
    ascending. The wrist point is the 4/5 axis intersection when axes 4 and 5 meet
    (the wrist centre of a spherical wrist). The order is deterministic per pose,
    but columns do not guarantee the same physical branch at neighbouring poses.
    Callers that need continuity must follow branches themselves.

    :ivar joints: (n, 16, 6) joint angles in the robot's angle unit, each in
        (-180, 180], or (-pi, pi] in radians. A NaN row is an empty column.
        Limits are never applied.
    :ivar limit_margin: (n, 16) ``min(q - lo, hi - q)`` over all joints, in the
        robot's angle unit. Negative means outside the limits. ``+inf`` when no
        limits were given, NaN where the column is empty.
    :ivar extension: (n,) signed distance of the wrist point to the annulus
        reachable by the shoulder, in the model's length unit, taken over both
        shoulder sides and, on a wrist whose point moves with the branch, over
        all branches of the pose (the best one). Positive inside, zero on the reach boundary, negative when
        no branch can exist; for a wrist point closer to axis 1 than the lateral
        arm offset it is the negative distance to that cylinder instead. Only
        defined when axes 2 and 3 are parallel, NaN otherwise. For a pose without
        any root on a non-spherical wrist it is evaluated at the zero-wrist estimate
        of the wrist point.
    :ivar sigma_min: (n, 16) smallest singular value of the 6x6 geometric Jacobian
        of the requested TCP with respect to the joints in radians. Translation rows
        are in the model's length unit per radian, rotation rows unitless. Zero at a
        singularity. NaN where the column is empty and where ``limit_margin`` is
        negative, since the conditioning of an unusable branch is not computed.
    :ivar wrist: (n, 16) wrist bend margin in the robot's angle unit: the distance of
        the angle between axes 4 and 6 from the nearest fold of the wrist,
        ``min(phi - |beta1 - beta2|, min(beta1 + beta2, 360 - beta1 - beta2) - phi)``
        in degrees, with beta the angles between axes 4/5 and 5/6. Zero at a wrist
        fold; the full arm Jacobian can still have rank six. For an orthogonal wrist
        this equals ``asin|sin J5|``. The sine of the margin in radians equals
        py-opw-kinematics' ``wrist``.
    :ivar n_solutions: (n,) number of non-empty columns per pose.
    """

    joints: np.ndarray
    limit_margin: np.ndarray
    extension: np.ndarray
    sigma_min: np.ndarray
    wrist: np.ndarray
    n_solutions: np.ndarray


def interpolate_poses(
    x: ArrayLike,
    poses: RigidTransform,
    xn: ArrayLike,
) -> RigidTransform:
    """
    Interpolate poses at new points using SLERP for rotation and linear for translation.

    API follows scipy.interpolate.interp1d(x, y) pattern, identical to py-opw-kinematics.

    :param x: Array of N values corresponding to each keyframe (e.g., times or distances).
    :param poses: RigidTransform with N keyframe poses.
    :param xn: Array of M values where interpolation is desired.
    :return: RigidTransform containing M interpolated poses.
    """
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Slerp

    x = np.asarray(x)
    xn = np.asarray(xn)
    interp_rotations = Slerp(x, poses.rotation)(xn)
    interp_translations = interp1d(x, poses.translation, axis=0)(xn)
    return RigidTransform.from_components(
        rotation=interp_rotations,
        translation=interp_translations,  # type: ignore[arg-type]
    )


__all__ = [
    "AxisPairReport",
    "KinematicModel",
    "ReachResult",
    "RigidTransform",
    "Robot",
    "interpolate_poses",
]
