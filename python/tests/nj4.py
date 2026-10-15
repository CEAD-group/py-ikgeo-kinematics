"""Comau NJ4-220-3.0 test model, the same numbers as ikgeo::robot::fixtures::comau_nj4_220."""

import numpy as np
from py_ikgeo_kinematics import KinematicModel, Robot

BETA = np.deg2rad(60.0)

# Axis directions in the zero pose (Comau calibration pose J2 = 0, J3 = -90), millimetres.
H = np.array(
    [
        [0.0, 0.0, 0.0, 1.0, np.cos(BETA), 1.0],
        [0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0, np.sin(BETA), 0.0],
    ]
)
P = np.array(
    [
        [0.0, 460.0, 0.0, 1515.0, 0.0, 170.0 * np.cos(BETA), 225.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1140.0, 1050.0, 250.0, 0.0, 170.0 * np.sin(BETA), 0.0],
    ]
)

# Model-space limits in degrees: the drawing's axis 3 range -220..75 shifted by the +90
# zero offset of the model; the others as on the drawing.
LIMITS = np.array(
    [
        [-180.0, 180.0],
        [-75.0, 75.0],
        [-130.0, 165.0],
        [-200.0, 200.0],
        [-200.0, 200.0],
        [-200.0, 200.0],
    ]
)


def make_model() -> KinematicModel:
    return KinematicModel(H, P)


def make_robot(degrees: bool = True) -> Robot:
    return Robot(make_model(), degrees=degrees)
