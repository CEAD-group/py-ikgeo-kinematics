# py-ikgeo-kinematics: implementation plan

Status (2026-10-15): steps 0 to 7 are implemented. v0.0.2 is the first PyPI release.
The CI configuration builds and tests four platforms on Python 3.11–3.14;
remote results must be checked before release. See README.md for usage and
current solver limitations.

## Working scope

- Work in this repository. Downstream cell planning and controller
  integration belong in their own repositories.
- The reference py-opw-kinematics checkout is at `../py-opw-kinematics`.
  The parity tests use the installed package from the `test` extra.
- Local upstream sources and reference drawings may be present under `tmp/`.
  They are not required to build or run the tests. Do not commit the drawings.
- Keep the vendored BSD-3-Clause license and `ikgeo/NOTICE` with the solver.
- Release tagging and publishing remain the maintainer's responsibility.

## Why

The downstream cell planner's only arm solver is py-opw-kinematics, which
assumes an ortho-parallel base and a spherical wrist. The Comau NJ4-220 (and the
Kawasaki KJ, FANUC CRX, Yaskawa EPX/MPX families) has a wrist whose three axes do
not meet at one point. From Comau drawing CR82275505 (NJ4-220-3.0):

| Quantity | Value |
|---|---|
| Angle axis 4 to axis 5, and axis 5 to axis 6 | 60 deg |
| Distance along axis 5 between the 4/5 and 5/6 intersection points | 170 mm |
| Flange offset above the forearm axis at calibration | 147.22 mm (170 sin 60) |
| 5/6 intersection point to flange face along axis 6 | 225 mm |
| Axis 4 / 5 / 6 stroke | +-200 deg each |

Pieper's closed form needs three consecutive intersecting or parallel axes. This
arm has a parallel pair (2,3) and two intersecting pairs (4,5), (5,6), and no
triple. In IK-Geo's taxonomy that is "two intersecting axes and two parallel
axes": a 1D search over one joint with closed-form subproblems inside. Reported
Rust timing is 115 to 440 us per cold pose, versus about 2 us for OPW.

The Sinumerik side already supports this geometry: ROBX wrist type 3/6 with DH
`a = 0`, `d = 170`, `alpha = +-60`, solved iteratively (`ROBX_ITER_MAXCOUNT`,
default 8). Our solver has to agree with it on the branch; it does not have to
replace it.

## Scope

In scope: a Rust crate that solves forward and inverse kinematics for any 6R arm
given as a product of exponentials, a Python package whose API is the same as
py-opw-kinematics, and the margins the reach analysis needs (limit margin,
sigma_min, wrist distance, extension).

Out of scope here, planned in their own repos afterwards: the downstream
solver class and structural detector, its reach kernel for a
non-slot-continuous branch set, STAT handling for up to 16 roots, the ROBX
machine-data export, and controller validation.

## Design decisions

1. **Vendor, do not depend.** `ikgeo/` is a workspace crate copied from
   rpiRobotics/ik-geo (`rust/`) with Verdant-Evolution's packaging, both BSD-3.
   Reasons: the crates.io `ik-geo` 0.1.2 is unmaintained since 2024, pins
   nalgebra 0.32 against our 0.34, and its 1D/2D search links `nlopt`, a C
   library we do not want in wheels. The paper's own bracketing search replaces
   it. `ikgeo/NOTICE` records provenance; keep the BSD text.
2. **Robot model is a product of exponentials**: `h` (3x6 unit axis directions)
   and `p` (3x7 offsets, base to joint 1 ... joint 6 to flange) in the zero
   pose, in the model's length unit. This is what a cell model built from
   joint origins and axis directions gives: each origin is one column of `p`,
   each axis direction one column of `h`. No DH, no OPW letters. `KinematicModel.from_opw(...)` exists
   only for the parity tests.
3. **Family detection is automatic and reported.** The constructor tests
   consecutive axes for parallel and intersecting (tolerance 1e-9 relative to
   the longest link), picks the decomposition, and exposes it as
   `Robot.family` (`"spherical_two_parallel"`, `"two_intersecting"`, ...).
   It never guesses silently.
4. **Two solve modes, one API.**
   - *Cold*: all solutions for one pose, the IK-Geo search (200 initial samples,
     false-position bracketing per sign change, up to 16 roots), used by
     `inverse` and `reach`.
   - *Tracked*: `batch_inverse` warm-starts each pose from the previous
     solution with Gauss-Newton on the se(3) error (6x6 solve, converge in 2 to
     4 steps, about 5 to 10 us). It falls back to a cold solve, then picks the
     nearest root, when Newton fails to converge, the step exceeds a configured
     joint-space jump, or the pose is the first one. This is the same contract
     as py-opw's `batch_inverse` (`current_joints` seeds the chain), so a
     consumer's batch solve needs no change for the common path.
     Tracking follows the previous row's branch (continuation); only the
     fallback picks the nearest root, so where two branches run close, near
     the J5 = 180 fold, the two can be different valid roots.
5. **Joint limits live in the crate**, applied in `reach` as `limit_margin`
   exactly like py-opw (normalise to (-180, 180], `min(q - lo, hi - q)`), never
   in `inverse` or `batch_inverse`, which return raw solutions. The consumer
   filters, as it does with py-opw.
6. **Branch slots are not stable on this robot class.** py-opw's `reach`
   promises slot `k = 4 wrist + 2 shoulder + elbow` is one physical branch at
   every pose. Non-spherical wrist arms can be cuspidal: branches can connect without
   crossing a singularity. The solver does not provide stable branch labels. `reach` returns `(n, 16, 6)` sorted by a deterministic key
   per pose (shoulder side, elbow side, then J5 sign, then J4 ascending) and
   documents that column continuity is not guaranteed. Branch following
   across neighbouring poses is the consumer's job. Open question, see below.
7. **Singularity margins are numeric.** `sigma_min` from the 6x6 geometric
   Jacobian built from twists (same convention as py-opw `reach.rs`:
   translation rows in model length per radian, rotation rows unitless).
   `wrist` is the bend angle margin `min(phi - phi_min, phi_max - phi)` in the
   robot's angle unit, where phi is the angle between axes 4 and 6
   (`cos phi = cos beta1 cos beta2 + sin beta1 sin beta2 cos J5`),
   `phi_min = |beta1 - beta2|` and `phi_max = min(beta1 + beta2, 2 pi - beta1 -
   beta2)` (axes are lines, so the result does not depend on the sign chosen
   for an axis direction); for an orthogonal
   wrist this reduces to `asin|sin J5|`, so `sin(wrist)` is py-opw's `wrist`.
   On a wrist whose axes are not orthogonal the two margins measure different
   things: at a fold (J5 = 0 or +-180 on the NJ4) the three wrist axes are
   coplanar and the wrist-only orientation Jacobian is singular, so `wrist`
   reads 0, but the arm axes complete the rank of the 6x6 Jacobian and
   `sigma_min` only drops (median 0.011 at the folds against 0.54 at J5 = 90,
   mm units) without reaching zero. `extension` is the
   signed distance of the 4/5 intersection point to the shoulder annulus, which
   is well defined because that point sits on axis 4 at a fixed distance from
   the elbow.
8. **Units and flags as py-opw**: `degrees=True` default, radians internally,
   `ee_transform` optional 4x4, public SciPy `RigidTransform` poses (internally flattened `(n, 16)` matrices),
   NaN rows when no root is found, `threads` on `reach` with the GIL released.
9. **Versioning**: `0.0.0-dev` in Cargo.toml, stamped from the `v*` tag in CI,
   identical to py-opw-kinematics, and published to PyPI from the same
   workflow.

## Python API

```python
class KinematicModel:
    def __init__(self, h: ArrayLike, p: ArrayLike,
                 offsets=(0,)*6, flip_axes=(False,)*6) -> None
    @classmethod
    def from_opw(cls, a1, a2, b, c1, c2, c3, c4, offsets, flip_axes) -> KinematicModel
    h: np.ndarray  # (3, 6)
    p: np.ndarray  # (3, 7)

class Robot:
    def __init__(self, kinematic_model: KinematicModel, degrees: bool = True,
                 *, axis_tolerance: float = 1e-9) -> None   # radians for parallel, fraction of the longest offset for intersecting
    family: str
    def axis_report() -> list[AxisPairReport]   # per pair: parallel_deviation, intersection_deviation, flags; warns when family is general
    def forward(joints, ee_transform=None) -> RigidTransform
    def batch_forward(joints: (n, 6), ee_transform=None) -> RigidTransform
    def joint_poses(joints, ee_transform=None) -> RigidTransform      # 6 or 7 frames
    def batch_joint_poses(joints: (n, 6), ee_transform=None) -> RigidTransform
    def forward_frames(joints, ee_transform=None) -> RigidTransform         # Base, J1..J6, TCP
    def inverse(pose, current_joints=None, ee_transform=None) -> list[tuple6]   # cold, found roots, tuples
    def batch_inverse(poses: RigidTransform, current_joints=None, ee_transform=None) -> (n, 6)  # tracked
    def reach(poses: RigidTransform, joint_limits=None, ee_transform=None, threads=1) -> ReachResult
    def jacobian(joints, ee_transform=None) -> (6, 6)                  # new, exposed for TOPP and viz

@dataclass(frozen=True, eq=False)
class ReachResult:
    joints: (n, 16, 6); limit_margin: (n, 16); extension: (n,)
    sigma_min: (n, 16); wrist: (n, 16); n_solutions: (n,) int64

def interpolate_poses(x, poses, xn) -> RigidTransform   # same as py-opw
```

`KinematicModel.h` and `.p` are read-only numpy getters; `offsets` and
`flip_axes` are tuples, as in py-opw.

Differences from py-opw, all additive: `family`, `jacobian`, `n_solutions`,
16 branch columns instead of 8, and `KinematicModel` taking `h, p`.

## Steps

Each step is one PR against `main`, reviewable alone, with its tests.

0. **Scaffold builds.** Fill the CI workflow from py-opw-kinematics, make
   `maturin develop` produce an importable empty module, `cargo test -p ikgeo`
   runs. Exit: green CI on an empty package.
1. **Vendor ik-geo.** Copy `subproblems/`, `inverse_kinematics/`,
   `solutionset.rs` into `ikgeo/src`, port to nalgebra 0.34, delete the
   hardcoded robot setups except as test fixtures, replace `search_1d` /
   `search_2d` (nlopt) with the paper's bracketing and grid-then-minimise
   search. Exit: upstream correctness tests pass for all eight families;
   timing test records us per family in `tmp/`.
2. **Family detection + FK.** `Kinematics::classify(h, p)`, per-joint frames
   (`joint_poses`), twist Jacobian, `sigma_min`. Exit: FK of an OPW robot given
   as `h, p` matches py-opw `forward_frames` to 1e-9 at 10k random joints.
3. **Cold inverse.** `inverse` over the family's decomposition, roots
   deduplicated, FK-validated, sorted by distance to `current_joints`. Exit:
   on the NJ290 (spherical, two parallel) the solution set equals py-opw's set
   per pose; on the NJ4 model every returned root round-trips through FK to
   1e-9 mm and 1e-9 deg; random-pose coverage finds at least one root wherever
   a seeded FK pose was generated.
4. **Tracked inverse.** Gauss-Newton tracker with fallback; `batch_inverse`.
   Exit: a 100k-point printing path on the NJ4 model tracks with no fallback
   except at retracts; mean under 10 us per point in release; the seam test
   (chunked calls seeded on the previous chunk) produces the same joints as one
   call.
5. **Reach.** `reach` with limits, margins, threads, GIL released; `ReachResult`.
   Exit: on the NJ290 the 8 py-opw branches appear among the 16 columns with
   identical margins; on the NJ4 `wrist` reads 0 at J5 = 0 and at J5 = +-180
   (the 120 deg fold) and `sigma_min` drops by an order of magnitude at both
   folds without reaching zero, because the full Jacobian keeps rank there
   (design decision 7).
6. **Python surface and parity suite.** `__init__.py` wrappers, `.pyi`,
   `py.typed`, mypy clean, `python/tests/test_parity_opw.py` that runs the same
   assertions against both packages. Exit: `pytest` green, wheel builds on the
   four CI targets.
7. **NJ4 example and fixture.** `python/examples/example_comau_nj4_220.py`
   building the model from the drawing. The wrist numbers above are certain;
   the arm lengths (dimensions 460, 1140, 1050, 250, 1515, 400 on the drawing)
   must be assigned from the STEP file and checked against the calibration
   pose in the drawing's table: J2 = 0, J3 = -90 puts the 4/5 intersection
   point (the wrist centre on axis 4) at X = 1975, Z = 2440, which is
   460 + 1515 and 1140 + 1050 + 250; the flange face is 85 + 225 further along
   X and 147.22 higher. Without a STEP file the assignment was checked against
   all seven table rows (0.005 mm) and the restricted-envelope heights on sheet
   2; the 400 is the forearm's rear overhang, not a link. Then a JSON fixture
   of 1k poses with expected joints for regression. Exit: fixture committed,
   used by steps 3 to 5.
8. **Tag v0.1.0, publish.** The API above is the contract for downstream
   integration.

## Performance targets (release build, Apple M-series or i7 class)

| Call | Target | Measured 2026-10-13, M1 Pro |
|---|---|---|
| `batch_forward` per pose | within 2x of py-opw | 0.133 us against py-opw 0.121 us (100k joints, NJ165) |
| `inverse` cold, NJ4 | under 500 us | 95 us mean in Rust, 93 us through Python, 1000 random poses |
| `batch_inverse` tracked, NJ4 | under 10 us per pose | 1.9 us in Rust, 2.1 us through Python, 100k-point printing path, 0 fallbacks |
| `reach` per pose, NJ4, 16 branches | under 1 ms single thread | 107 us in Rust, 99 us through Python; 17.5 us with threads=0 on 10 cores |

Below these, a downstream path simulation stays IK-bound at roughly today's
OPW cost (about 54 us per point end to end). Above them, revisit before step 8.

## Open questions

- **Branch continuity for reach.** Column-stable slots do not exist on a
  cuspidal arm. Options: (a) label branches by continuation from a seed pose
  per voxel neighbourhood inside the consumer's reach kernel, (b) drop
  per-branch maps and report per-pose aggregates (best margin, count), (c)
  label by the ROBX STAT convention plus an A6 seed, which is what the
  controller does. Decide downstream, not here.
- **Which NJ4-220.** Comau lists the 3.0; the 2.4 and 2.7 are Smart5 variants.
  The wrist is the same family but the lengths must come from the matching
  drawing or STEP.
- **Controller.** If the cell runs Comau C5G instead of Sinumerik ROBX, the
  export side (PDL2, frames) is a separate and larger project; this crate is
  unaffected.
- **A4 parallel to A3 variants.** ROBX has `ROBX_A4PAR`; IK-Geo handles it as a
  different family. Not needed for the NJ4, keep the detector general.
- **Axis tolerance for measured models.** The default `axis_tolerance` is
  1e-9 (exact models). A model read from a STEP file or a calibration can
  carry 1e-7 to 1e-6 of axis skew, which at the default classifies as
  `general` (slow 2D search, missed roots); `Robot` then warns and names the
  near-miss pairs in `axis_report()`. Whether a consumer snaps its axes or
  passes a tolerance is its decision, not this crate's.
