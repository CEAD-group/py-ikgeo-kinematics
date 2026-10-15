# %%
"""
Example: Comau NJ4-220-3.0 built from drawing CR82275505 (working areas).

The NJ4 wrist is not spherical: axes 4, 5 and 6 meet pairwise at 60 degrees
with 170 mm between the two intersection points, so OPW kinematics does not
apply. The model is a product of exponentials: ``h`` holds the six joint axis
directions and ``p`` the seven offsets (base, J1 .. J6, flange) in the zero
pose, in millimetres.
"""

import numpy as np
from py_ikgeo_kinematics import KinematicModel, Robot
from scipy.spatial.transform import RigidTransform, Rotation

np.set_printoptions(precision=3, suppress=True)

# %%
# Zero pose of the model = Comau calibration pose (J2 = 0, J3 = -90): upper arm
# vertical, forearm horizontal along +X, wrist straight. Base frame on the floor
# at the axis 1 centre, X toward the arm, Z up.
BETA = np.deg2rad(60.0)  # angle between axes 4/5 and between axes 5/6 (drawing: 30 deg to vertical)
WRIST = 170.0  # mm along axis 5 between the 4/5 and the 5/6 intersection points
FLANGE = 225.0  # mm from the 5/6 intersection point to the flange face, along axis 6

# fmt: off
H = np.array([
    #  J1     J2     J3     J4     J5             J6
    [  0.0,   0.0,   0.0,   1.0,  np.cos(BETA),   1.0],
    [  0.0,   1.0,  -1.0,   0.0,  0.0,            0.0],
    [ -1.0,   0.0,   0.0,   0.0,  np.sin(BETA),   0.0],
])
# J1 vertical, positive clockwise from above as on the drawing's top view.
# J2 horizontal, positive tips the upper arm forward (+75 arc on the front side).
# J3 parallel to J2, positive tips the forearm backward (+75 limit is backward).
# J4 along the forearm; J5 60 deg from J4, leaning toward the flange and up.
# J6 60 deg from J5, so parallel to J4 again in the zero pose.

P = np.array([
    [   0.0,    0.0,    0.0],  # base -> J1: any point on axis 1, floor level
    [ 460.0,    0.0, 1140.0],  # J1 -> J2: 460 forward of axis 1, 1140 above the base plate
    [   0.0,    0.0, 1050.0],  # J2 -> J3: upper arm, 1050 between axes 2 and 3
    [1515.0,    0.0,  250.0],  # J3 -> 4/5 point: 1515 along the forearm, 250 above axis 3
    [   0.0,    0.0,    0.0],  # J4 -> J5: axes 4 and 5 intersect
    [WRIST * np.cos(BETA), 0.0, WRIST * np.sin(BETA)],  # J5 -> 5/6 point: 85 forward, 147.22 up
    [FLANGE,    0.0,    0.0],  # J6 -> flange face
]).T
# The 400 on the drawing is the forearm's rear overhang behind axis 3, not a link.
# fmt: on

# Comau reports J3 = -90 at the calibration pose where the model's q3 is 0, so
# q = flip * J - offset needs offset -90 on axis 3. The positive sense of J4, J5
# and J6 could not be read from the drawing and must be confirmed on the machine
# through flip_axes; py-opw's NJ165 example (values from a controller) uses
# relative J3 and flips on axes 1, 3, 4 and 6, which is the likely starting
# point here but is unverified for the NJ4. The controller reports J3 absolute
# (NJ290 parallelogram); this model takes the relative J3 = AX.2 + AX.3.
OFFSETS = (0.0, 0.0, -90.0, 0.0, 0.0, 0.0)
FLIP_AXES = (False, False, False, False, False, False)

# Joint limits from the drawing, Comau convention, degrees, for reach() later.
LIMITS = np.array(
    [
        [-180.0, 180.0],
        [-75.0, 75.0],
        [-220.0, 75.0],
        [-200.0, 200.0],
        [-200.0, 200.0],
        [-200.0, 200.0],
    ]
)

kinematic_model = KinematicModel(H, P, offsets=OFFSETS, flip_axes=FLIP_AXES)
robot = Robot(kinematic_model, degrees=True)
print("family:", robot.family)

# %%
# Calibration check against the drawing's working-area table (pos. 7): with
# J2 = 0 and J3 = -90 the 4/5 intersection point, the wrist centre on axis 4,
# is at X = 1975 (460 + 1515) and Z = 2440 (1140 + 1050 + 250). The flange face
# is 85 + 225 further along X and 147.22 higher (calibration view, sheet 1 zone A1).
CALIBRATION = (0.0, 0.0, -90.0, 0.0, 0.0, 0.0)
frames = robot.joint_poses(CALIBRATION)
wrist_centre = frames[4].translation  # frame J5 sits on the 4/5 intersection point
flange = robot.forward(CALIBRATION).translation
expected_wrist = np.array([1975.0, 0.0, 2440.0])
expected_flange = expected_wrist + [WRIST * np.cos(BETA) + FLANGE, 0.0, WRIST * np.sin(BETA)]
wrist_ok = np.allclose(wrist_centre, expected_wrist, atol=1e-6)
flange_ok = np.allclose(flange, expected_flange, atol=1e-6)
print("Calibration pose J =", CALIBRATION)
print(f"  wrist centre (4/5 point): {wrist_centre}  drawing: {expected_wrist}")
print(f"  flange face:              {flange}  expected: {expected_flange}")
print("Calibration check:", "match" if (wrist_ok and flange_ok) else "MISMATCH")

# %%
# Forward then inverse: every root must reproduce the pose, and the seed
# posture must be among the roots.
print("\nForward/inverse round trips:")
for joints in [
    (0.0, 0.0, -90.0, 0.0, 0.0, 0.0),
    (20.0, 30.0, -120.0, 40.0, 60.0, -30.0),
    (-45.0, -20.0, -60.0, 90.0, 120.0, 10.0),
    (120.0, 60.0, -150.0, -150.0, -90.0, 170.0),
]:
    pose = robot.forward(joints)
    solutions = robot.inverse(pose, current_joints=joints)
    back = robot.batch_forward(np.array(solutions))
    worst_pos = np.linalg.norm(back.translation - pose.translation, axis=1).max()
    worst_rot = np.degrees((back.rotation.inv() * pose.rotation).magnitude()).max()
    found = any(np.allclose(sol, joints, atol=1e-6) for sol in solutions)
    print(
        f"  J={joints}: {len(solutions)} roots, worst residual {worst_pos:.1e} mm "
        f"{worst_rot:.1e} deg, seed found: {found}"
    )

# %%
# Tracked solve along a straight line with the tool pointing down. The flange
# normal is +X of the flange frame (axis 6 in the zero pose), so a rotation of
# +90 degrees about Y points it at the floor.
tool_down = Rotation.from_euler("y", 90, degrees=True)
n = 200
line = np.column_stack([np.full(n, 1800.0), np.linspace(-600.0, 600.0, n), np.full(n, 800.0)])
poses = RigidTransform.from_components(
    rotation=Rotation.concatenate([tool_down] * n), translation=line
)
seed = (0.0, 30.0, -120.0, 0.0, 30.0, 0.0)
first = robot.inverse(poses[0], current_joints=seed)[0]
path = robot.batch_inverse(poses, current_joints=first)
steps = np.abs(np.diff(path, axis=0)).max(axis=0)
print("\nStraight line, tool down, 200 poses:")
print("  first joints:", np.array(first))
print("  last joints: ", path[-1])
print("  largest joint step per axis [deg]:", steps)
print("  NaN rows:", int(np.isnan(path).any(axis=1).sum()))
check = robot.batch_forward(path)
print(
    "  worst position error [mm]:",
    f"{np.linalg.norm(check.translation - poses.translation, axis=1).max():.1e}",
)
