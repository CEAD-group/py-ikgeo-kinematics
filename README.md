# py-ikgeo-kinematics

Forward and inverse kinematics for six-axis robot arms, including arms with
non-spherical wrists such as the Comau NJ4. The solver uses IK-Geo subproblem
decomposition in Rust, with a Python API based on NumPy and SciPy
`RigidTransform`.

The main tested models are the Comau NJ4-220-3.0 and OPW-compatible arms. Other
geometries use the detected IK-Geo family; search-based solvers can miss valid
solutions.

## Install

```sh
pip install py-ikgeo-kinematics
```

## Build from source

Install Rust and uv, then run from this checkout:

```sh
uv sync --extra test
uv run --with maturin maturin develop --release
```

Python 3.11–3.14 is covered by the CI configuration. The extension uses PyO3
0.27, which does not support Python 3.15. To choose an interpreter explicitly,
use `uv sync --python 3.14 --extra test`.

## Solve a pose

This example uses the OPW conversion helper. For a non-spherical wrist, pass
axis directions and offsets directly to `KinematicModel(h, p)`, as in the
[Comau NJ4 example](python/examples/example_comau_nj4_220.py).

```python
import numpy as np
from py_ikgeo_kinematics import KinematicModel, Robot

model = KinematicModel.from_opw(a1=0.1, c1=0.2, c2=0.3, c3=0.25, c4=0.05)
robot = Robot(model, degrees=True)
joints = (20.0, 30.0, -40.0, 10.0, 50.0, -20.0)

pose = robot.forward(joints)
solutions = robot.inverse(pose, current_joints=joints)
assert solutions
np.testing.assert_allclose(robot.forward(solutions[0]).as_matrix(), pose.as_matrix(), atol=1e-9)

poses = robot.batch_forward(np.array([joints, joints]))
path = robot.batch_inverse(poses, current_joints=joints)  # Shape (2, 6).
reach = robot.reach(poses)
print(robot.family, reach.n_solutions)
```

## Model and units

`h` has shape `(3, 6)`. Each column is a joint axis direction in the zero pose;
nonzero directions are normalized. `p` has shape `(3, 7)`, with columns for the
base-to-J1 offset, the five offsets between joints, and the J6-to-flange offset.
Both use a common coordinate frame at the zero pose.

All lengths use the model's unit. The example above uses metres; the NJ4
example uses millimetres. Joint angles and model `offsets` use degrees by
default, or radians with `Robot(model, degrees=False)`. Internally,
`q = sign * joint - offset`, where `flip_axes=True` selects a negative sign.

Model values must be finite, and axis directions must be nonzero. The model's
`h` and `p` getters return read-only NumPy copies. Build a new model to change
geometry.

`axis_tolerance` defaults to `1e-9`: radians for parallelism and a fraction of
the longest offset for intersections. `robot.axis_report()` shows the measured
deviations. Raising the tolerance lets the solver treat nearly parallel or
intersecting axes as exact relations; returned roots are checked against the
original model. A `general` family emits a warning because its 2D search is
slow and can miss roots.

## API

Public pose inputs and outputs are SciPy `RigidTransform` objects. Use
`.as_matrix()` for `(4, 4)` or `(n, 4, 4)` NumPy matrices. Flattened `(n, 16)`
arrays belong to the internal extension API.

| Method | Result |
| --- | --- |
| `forward(joints)` | One flange or TCP pose. |
| `batch_forward(joints)` | A stack of poses for `(n, 6)` joints. |
| `inverse(pose, current_joints=None)` | Up to 16 solutions as six-element tuples. An empty list means no root was found. |
| `batch_inverse(poses, current_joints=None)` | A tracked joint path, shape `(n, 6)`. Failed poses produce NaN rows. |
| `joint_poses(joints)` | Six joint frames, plus TCP when a tool is supplied. |
| `batch_joint_poses(joints)` | The joint frames stacked in pose order. |
| `forward_frames(joints)` | Base, six joint frames, and TCP. |
| `jacobian(joints)` | A `(6, 6)` geometric Jacobian, translation rows first, per radian. |
| `reach(poses, joint_limits=None, *, threads=None)` | Solutions and margins described below. |

These methods accept an optional `ee_transform` as a `RigidTransform`. For
compatibility with py-opw-kinematics, a tool with rotation `R` and translation
`t` uses `R @ t` as its flange-frame offset. This differs from ordinary
homogeneous-transform composition when both rotation and translation are set.

`inverse` unwraps joints near `current_joints` and sorts by summed absolute
joint distance. Without a seed it uses zero. `batch_inverse` follows the
previous solution by Newton continuation and falls back to the nearest cold
root when tracking fails. It can therefore choose a different valid branch
from a loop over `inverse`. Failed rows leave the last valid seed intact.
Neither method filters joint limits.

Batch joint inputs also accept pandas or Polars DataFrames with columns
`J1` through `J6`. A DataFrame seed for `batch_inverse` produces the same type
of DataFrame. `interpolate_poses(x, poses, xn)` combines linear translation
interpolation with rotation SLERP.

## Reach results

`reach` returns a frozen `ReachResult`; its NumPy arrays remain mutable.
`threads=None` (the default) uses the process-wide Rayon pool, so
`RAYON_NUM_THREADS` applies. `threads=1` runs serially in the calling thread,
and `threads=n` uses a pool of `n` threads that is cached for the life of the
process. Values below 1 raise `ValueError`; `threads=0` still means `None` but
is deprecated. Results do not depend on `threads`. The GIL is released during
the solve.

| Field | Shape | Meaning |
| --- | --- | --- |
| `joints` | `(n, 16, 6)` | Found solutions, with NaN rows in unused slots. |
| `n_solutions` | `(n,)` | Number of populated slots. |
| `limit_margin` | `(n, 16)` | Minimum distance to a joint bound; negative outside, `+inf` without limits. |
| `sigma_min` | `(n, 16)` | Smallest Jacobian singular value; NaN for empty or out-of-limit branches. |
| `wrist` | `(n, 16)` | Angular distance to the nearest wrist fold, in the robot's angle unit. |
| `extension` | `(n,)` | Signed distance to the shoulder annulus, in model length units; NaN when axes 2 and 3 are not parallel. |

`joint_limits` has shape `(6, 2)` and uses the robot's angle unit. Margins use
joints normalized to `(-180, 180]`, or `(-pi, pi]` in radians. Limits report
usability; they do not remove solutions. Branch columns are deterministic
within a pose but do not promise continuity between poses. Consumers must
track branches themselves.

The `wrist` margin differs from py-opw's dimensionless margin: for an
orthogonal wrist, the sine of this angle in radians equals py-opw's value.
At an NJ4 wrist fold, `wrist` can be zero while the full arm Jacobian still
has rank six. `sigma_min` also depends on the model's length unit.

## Limits of the solver

Returned roots are checked against forward kinematics. This checks their
accuracy, not whether every root was found. NJ4 and OPW models have dedicated
regression and completeness tests. Some other 1D search families miss roots,
and the general 2D search has lower coverage. An empty result alone does not
prove that a pose is unreachable.

The NJ4 example checks geometry against the working-area drawing. Wrist joint
signs and the controller's J3 convention still need confirmation on the
machine. Controller validation and downstream integration are
outside this repository.

## Development

```sh
uv run --with maturin maturin develop --release
cargo test -p ikgeo --release
uv run pytest
uv run mypy python
uvx ruff check .
cargo fmt --check
PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --workspace --all-targets -- -D warnings
```

CI builds and tests wheels for Linux x86_64, Linux ARM64, Windows x64 and
macOS ARM64 on Python 3.11–3.14. These checks run on pull requests, pushes to
`main`, and manual dispatch. Version tags stamp `Cargo.toml`, create a GitHub
release and upload to PyPI after all checks pass; the upload runs while the
`PUBLISH_TO_PYPI` repository variable is `true`. Local PEP 517 builds take a
`git describe` version, while a bare `maturin develop` reports `0.0.0`; in a
consumer checkout, use `uv run --no-sync` to avoid replacing a locally
installed build with the locked wheel.

[PLAN.md](PLAN.md) records design decisions and remaining integration work.
The bindings use the [MIT license](LICENSE); the vendored solver uses
[BSD-3-Clause](ikgeo/LICENSE-upstream), with provenance in
[ikgeo/NOTICE](ikgeo/NOTICE).
