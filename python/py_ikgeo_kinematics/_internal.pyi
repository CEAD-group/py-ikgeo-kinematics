from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Joints: TypeAlias = tuple[float, float, float, float, float, float]
Flags: TypeAlias = tuple[bool, bool, bool, bool, bool, bool]

class KinematicModel:
    @property
    def h(self) -> npt.NDArray[np.float64]: ...
    @property
    def p(self) -> npt.NDArray[np.float64]: ...
    @property
    def offsets(self) -> Joints: ...
    @property
    def flip_axes(self) -> Flags: ...

    def __init__(
        self,
        h: npt.ArrayLike,
        p: npt.ArrayLike,
        offsets: Joints = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        flip_axes: Flags = (False, False, False, False, False, False),
    ) -> None:
        """
        Product-of-exponentials model.

        :param h: (3, 6) joint axis directions in the zero pose.
        :param p: (3, 7) offsets base -> J1 -> ... -> J6 -> flange in the zero pose.
        :param offsets: Joint offsets in the robot's angle unit.
        :param flip_axes: Per-joint sign flips.
        """

    @staticmethod
    def from_opw(
        a1: float = 0,
        a2: float = 0,
        b: float = 0,
        c1: float = 0,
        c2: float = 0,
        c3: float = 0,
        c4: float = 0,
        offsets: Joints = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        flip_axes: Flags = (False, False, False, False, False, False),
    ) -> KinematicModel:
        """The OPW parameterisation of py-opw-kinematics as h, p."""

class Robot:
    family: str

    def __init__(
        self, kinematic_model: KinematicModel, degrees: bool = True, *, axis_tolerance: float = 1e-9
    ) -> None: ...
    def axis_report(self) -> list[tuple[str, float, float, bool, bool]]: ...
    def forward(
        self, joints: Joints, ee_transform: npt.NDArray[np.float64] | None = None
    ) -> npt.NDArray[np.float64]: ...
    def inverse(
        self,
        pose: npt.NDArray[np.float64],
        current_joints: Joints | None = None,
        ee_transform: npt.NDArray[np.float64] | None = None,
    ) -> list[Joints]: ...
    def batch_inverse(
        self,
        poses: npt.NDArray[np.float64],
        current_joints: Joints | None = None,
        ee_transform: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def batch_forward(
        self, joints: npt.NDArray[np.float64], ee_transform: npt.NDArray[np.float64] | None = None
    ) -> npt.NDArray[np.float64]: ...
    def joint_poses(
        self, joints: Joints, ee_transform: npt.NDArray[np.float64] | None = None
    ) -> list[npt.NDArray[np.float64]]: ...
    def batch_joint_poses(
        self, joints: npt.NDArray[np.float64], ee_transform: npt.NDArray[np.float64] | None = None
    ) -> npt.NDArray[np.float64]: ...
    def forward_frames(
        self, joints: Joints, ee_transform: npt.NDArray[np.float64] | None = None
    ) -> list[npt.NDArray[np.float64]]: ...
    def jacobian(
        self, joints: Joints, ee_transform: npt.NDArray[np.float64] | None = None
    ) -> npt.NDArray[np.float64]: ...
    def reach(
        self,
        poses: npt.NDArray[np.float64],
        joint_limits: list[tuple[float, float]] | None = None,
        ee_transform: npt.NDArray[np.float64] | None = None,
        *,
        threads: int | None = None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
    ]:
        """
        Every inverse-kinematics branch of (n, 16) poses with margins.

        :param threads: None uses the global pool, 1 runs in the calling thread,
            n >= 2 a cached pool of n threads; below 1 raises ValueError.
        :return: (joints (n, 16, 6), limit_margin (n, 16), extension (n,),
            sigma_min (n, 16), wrist (n, 16), n_solutions (n,)).
        """

__all__: list[str] = ["KinematicModel", "Robot"]
