//! Python bindings. The Rust solver lives in the `ikgeo` workspace crate; this
//! crate only converts arrays, applies the degrees flag and releases the GIL.
use ikgeo::nalgebra::{
    Isometry3, Matrix3, Matrix3x6, Matrix4, Rotation3, Translation3, UnitQuaternion, Vector3,
};
use ikgeo::reach::PoseReach;
use ikgeo::robot::{self, Matrix3x7};
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayLike2, PyArrayMethods,
    PyReadonlyArray2,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

type Matrix4x4 = [[f64; 4]; 4];
type ReachArrays = (
    Py<PyArray3<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray1<i64>>,
);
const BRANCHES: usize = ikgeo::inverse::MAX_ROOTS;

fn to_matrix(iso: &Isometry3<f64>) -> Matrix4x4 {
    let m = iso.to_matrix();
    std::array::from_fn(|r| std::array::from_fn(|c| m[(r, c)]))
}

fn from_matrix(m: &Matrix4x4) -> Isometry3<f64> {
    let flat: Vec<f64> = m.iter().flatten().copied().collect();
    let m4 = Matrix4::from_row_slice(&flat);
    let r: Matrix3<f64> = m4.fixed_view::<3, 3>(0, 0).into();
    let t: Vector3<f64> = m4.fixed_view::<3, 1>(0, 3).into();
    Isometry3::from_parts(
        Translation3::from(t),
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r)),
    )
}

/// py-opw reads the ee translation in the rotated tool frame, so the flange-frame offset is R_ee * t_ee.
fn ee_iso(ee_transform: Option<&Matrix4x4>) -> Isometry3<f64> {
    match ee_transform {
        None => Isometry3::identity(),
        Some(m) => {
            let raw = from_matrix(m);
            let offset = raw.rotation * raw.translation.vector;
            Isometry3::from_parts(Translation3::from(offset), raw.rotation)
        }
    }
}

/// The same tool offset as `ee_iso` as (R_ee, R_ee * t_ee), computed once per batch call.
fn ee_rt(ee_transform: Option<&Matrix4x4>) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    ee_transform.map(|m| {
        let r = Matrix3::new(
            m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
        );
        let t = Vector3::new(m[0][3], m[1][3], m[2][3]);
        (r, r * t)
    })
}

fn rt_to_matrix(r: &Matrix3<f64>, t: &Vector3<f64>) -> Matrix4x4 {
    [
        [r[(0, 0)], r[(0, 1)], r[(0, 2)], t[0]],
        [r[(1, 0)], r[(1, 1)], r[(1, 2)], t[1]],
        [r[(2, 0)], r[(2, 1)], r[(2, 2)], t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn flat_rows<'py>(py: Python<'py>, rows: usize, data: Vec<f64>) -> PyResult<Py<PyArray2<f64>>> {
    let arr = Array2::from_shape_vec((rows, 16), data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).into())
}

fn push_matrix(out: &mut Vec<f64>, m: &Matrix4x4) {
    for row in m {
        out.extend_from_slice(row);
    }
}

/// Product-of-exponentials model: `h` (3, 6) unit axis directions and `p` (3, 7)
/// offsets base -> J1 -> ... -> J6 -> flange in the zero pose, in the model's length unit.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct KinematicModel {
    inner: robot::KinematicModel,
    offsets_user: [f64; 6],
}

#[pymethods]
impl KinematicModel {
    #[new]
    #[pyo3(signature = (
        h,
        p,
        offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        flip_axes = (false, false, false, false, false, false),
    ))]
    fn new(
        h: PyArrayLike2<'_, f64, AllowTypeChange>,
        p: PyArrayLike2<'_, f64, AllowTypeChange>,
        offsets: (f64, f64, f64, f64, f64, f64),
        flip_axes: (bool, bool, bool, bool, bool, bool),
    ) -> PyResult<Self> {
        let h = h.as_array();
        let p = p.as_array();
        if h.shape() != [3, 6] {
            return Err(PyValueError::new_err(format!(
                "h must have shape (3, 6), got {:?}",
                h.shape()
            )));
        }
        if p.shape() != [3, 7] {
            return Err(PyValueError::new_err(format!(
                "p must have shape (3, 7), got {:?}",
                p.shape()
            )));
        }
        let hm = Matrix3x6::from_fn(|r, c| h[[r, c]]);
        for c in 0..6 {
            if hm.column(c).norm() < 1e-12 {
                return Err(PyValueError::new_err(format!(
                    "h column {c} is a zero vector"
                )));
            }
        }
        let pm = Matrix3x7::from_fn(|r, c| p[[r, c]]);
        let offsets_user = [
            offsets.0, offsets.1, offsets.2, offsets.3, offsets.4, offsets.5,
        ];
        let flip = [
            flip_axes.0,
            flip_axes.1,
            flip_axes.2,
            flip_axes.3,
            flip_axes.4,
            flip_axes.5,
        ];
        let inner = robot::KinematicModel::new(hm, pm, offsets_user, flip);
        inner
            .validate()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(KinematicModel {
            inner,
            offsets_user,
        })
    }

    /// The OPW parameterisation of py-opw-kinematics as a product of exponentials.
    #[staticmethod]
    #[pyo3(signature = (
        a1 = 0.0,
        a2 = 0.0,
        b = 0.0,
        c1 = 0.0,
        c2 = 0.0,
        c3 = 0.0,
        c4 = 0.0,
        offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        flip_axes = (false, false, false, false, false, false),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_opw(
        a1: f64,
        a2: f64,
        b: f64,
        c1: f64,
        c2: f64,
        c3: f64,
        c4: f64,
        offsets: (f64, f64, f64, f64, f64, f64),
        flip_axes: (bool, bool, bool, bool, bool, bool),
    ) -> PyResult<Self> {
        let offsets_user = [
            offsets.0, offsets.1, offsets.2, offsets.3, offsets.4, offsets.5,
        ];
        let flip = [
            flip_axes.0,
            flip_axes.1,
            flip_axes.2,
            flip_axes.3,
            flip_axes.4,
            flip_axes.5,
        ];
        let inner = robot::KinematicModel::from_opw(a1, a2, b, c1, c2, c3, c4, offsets_user, flip);
        inner
            .validate()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(KinematicModel {
            inner,
            offsets_user,
        })
    }

    #[getter]
    fn h<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        let arr = Array2::from_shape_fn((3, 6), |(r, c)| self.inner.h[(r, c)]);
        let arr = arr.into_pyarray(py);
        arr.readwrite().make_nonwriteable();
        arr.into()
    }

    #[getter]
    fn p<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        let arr = Array2::from_shape_fn((3, 7), |(r, c)| self.inner.p[(r, c)]);
        let arr = arr.into_pyarray(py);
        arr.readwrite().make_nonwriteable();
        arr.into()
    }

    #[getter]
    fn offsets(&self) -> (f64, f64, f64, f64, f64, f64) {
        let o = self.offsets_user;
        (o[0], o[1], o[2], o[3], o[4], o[5])
    }

    #[getter]
    fn flip_axes(&self) -> (bool, bool, bool, bool, bool, bool) {
        let f = self.inner.flip_axes();
        (f[0], f[1], f[2], f[3], f[4], f[5])
    }

    fn __repr__(&self) -> String {
        let cols = |m: &dyn Fn(usize) -> Vector3<f64>, n: usize| -> String {
            (0..n)
                .map(|c| {
                    let v = m(c);
                    format!("({:.6}, {:.6}, {:.6})", v[0], v[1], v[2])
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "KinematicModel(\n    h=[{}],\n    p=[{}],\n    offsets={:?},\n    flip_axes={:?}\n)",
            cols(&|c| self.inner.h.column(c).into(), 6),
            cols(&|c| self.inner.p.column(c).into(), 7),
            self.offsets_user,
            self.inner.flip_axes(),
        )
    }
}

impl KinematicModel {
    /// Offsets are given in the robot's angle unit, converted here like py-opw does.
    fn to_robot(&self, degrees: bool, axis_tolerance: f64) -> PyResult<robot::Robot> {
        let mut model = self.inner.clone();
        if degrees {
            model.offsets = self.offsets_user.map(f64::to_radians);
        }
        robot::Robot::with_tolerance(model, axis_tolerance)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass]
struct Robot {
    robot: robot::Robot,
    degrees: bool,
    kinematic_model: KinematicModel,
}

impl Robot {
    fn joints_rad(&self, joints: [f64; 6]) -> [f64; 6] {
        if self.degrees {
            joints.map(f64::to_radians)
        } else {
            joints
        }
    }

    fn forward_iso(&self, joints: [f64; 6], ee: Option<&Matrix4x4>) -> Isometry3<f64> {
        self.robot.forward(&self.joints_rad(joints)) * ee_iso(ee)
    }

    /// TCP rotation and position with a precomputed tool offset, no quaternion round trip.
    fn forward_rt(
        &self,
        joints: [f64; 6],
        ee: Option<&(Matrix3<f64>, Vector3<f64>)>,
    ) -> (Matrix3<f64>, Vector3<f64>) {
        let (r, t) = self.robot.forward_rt(&self.joints_rad(joints));
        match ee {
            None => (r, t),
            Some((r_ee, offset)) => (r * r_ee, t + r * offset),
        }
    }

    fn joint_isos(&self, joints: [f64; 6], ee: Option<&Matrix4x4>) -> Vec<Isometry3<f64>> {
        let frames = self.robot.joint_frames(&self.joints_rad(joints));
        let mut out: Vec<Isometry3<f64>> = frames.to_vec();
        if let Some(ee) = ee {
            out.push(self.forward_iso(joints, Some(ee)));
        }
        out
    }
}

#[pymethods]
impl Robot {
    #[new]
    #[pyo3(signature = (kinematic_model, degrees=true, *, axis_tolerance=robot::AXIS_TOL))]
    fn new(kinematic_model: KinematicModel, degrees: bool, axis_tolerance: f64) -> PyResult<Self> {
        Ok(Robot {
            robot: kinematic_model.to_robot(degrees, axis_tolerance)?,
            degrees,
            kinematic_model,
        })
    }

    fn __repr__(&self) -> String {
        let km = self
            .kinematic_model
            .__repr__()
            .lines()
            .map(|l| format!("    {l}"))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "Robot(\n    kinematic_model=\n{},\n    degrees={},\n    family={:?}\n)",
            km,
            self.degrees,
            self.robot.family().name()
        )
    }

    /// IK-Geo family detected from the axis geometry, for example "two_intersecting_two_parallel".
    #[getter]
    fn family(&self) -> &'static str {
        self.robot.family().name()
    }

    /// Per consecutive pair (1-2 .. 5-6): angle between the axes in radians, line distance
    /// over the longest offset (NaN when parallel), and the flags the tolerance assigned.
    fn axis_report(&self) -> Vec<(String, f64, f64, bool, bool)> {
        self.robot
            .axis_report()
            .iter()
            .enumerate()
            .map(|(i, r)| {
                (
                    format!("{}-{}", i + 1, i + 2),
                    r.parallel_deviation,
                    r.intersection_deviation,
                    r.parallel,
                    r.intersecting,
                )
            })
            .collect()
    }

    /// Flange (or TCP with ee_transform) pose as a row-major 4x4 matrix.
    #[pyo3(signature = (joints, ee_transform=None))]
    fn forward(&self, joints: [f64; 6], ee_transform: Option<Matrix4x4>) -> Matrix4x4 {
        let (r, t) = self.forward_rt(joints, ee_rt(ee_transform.as_ref()).as_ref());
        rt_to_matrix(&r, &t)
    }

    /// Every inverse-kinematics solution for a 4x4 TCP pose, as py-opw returns them: each
    /// joint is unwrapped near `current_joints` (home when absent) and the list is ordered by closeness.
    #[pyo3(signature = (pose, current_joints=None, ee_transform=None))]
    fn inverse(
        &self,
        pose: Matrix4x4,
        current_joints: Option<[f64; 6]>,
        ee_transform: Option<Matrix4x4>,
    ) -> Vec<(f64, f64, f64, f64, f64, f64)> {
        let flange = from_matrix(&pose) * ee_iso(ee_transform.as_ref()).inverse();
        let seed = current_joints.map(|j| self.joints_rad(j));
        let mut solutions = self.robot.inverse_all(&flange, seed.as_ref());
        if self.degrees {
            for s in solutions.iter_mut() {
                *s = s.map(f64::to_degrees);
            }
        }
        // Tuples, as py-opw's stub declares; a consumer must not need to know the backend.
        solutions
            .into_iter()
            .map(|s| (s[0], s[1], s[2], s[3], s[4], s[5]))
            .collect()
    }

    /// Tracked inverse kinematics for (n, 16) row-major TCP poses, returning (n, 6) joints
    /// with py-opw's contract: each row continues the previous one (Gauss-Newton from it,
    /// cold solve nearest to it as the fallback), a NaN or unsolvable pose gives a NaN row.
    #[pyo3(signature = (poses, current_joints=None, ee_transform=None))]
    fn batch_inverse<'py>(
        &self,
        py: Python<'py>,
        poses: PyReadonlyArray2<'py, f64>,
        current_joints: Option<[f64; 6]>,
        ee_transform: Option<Matrix4x4>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let poses = poses.as_array();
        if poses.ncols() != 16 {
            return Err(PyValueError::new_err("poses must have shape (n, 16)"));
        }
        let ee_inv = ee_iso(ee_transform.as_ref()).inverse();
        let flanges: Vec<Option<Isometry3<f64>>> = poses
            .rows()
            .into_iter()
            .map(|row| {
                if row.iter().any(|v| v.is_nan()) {
                    return None;
                }
                let m: Matrix4x4 = std::array::from_fn(|r| std::array::from_fn(|c| row[4 * r + c]));
                Some(from_matrix(&m) * ee_inv)
            })
            .collect();
        let seed = current_joints.map(|j| self.joints_rad(j));
        let n = flanges.len();
        // The loop is pure Rust on copied data, so the GIL is released for its duration.
        let (joints, _stats) = py.detach(|| self.robot.batch_inverse(&flanges, seed.as_ref()));
        let mut out = Vec::with_capacity(n * 6);
        for row in joints {
            let row = if self.degrees {
                row.map(f64::to_degrees)
            } else {
                row
            };
            out.extend_from_slice(&row);
        }
        let arr = Array2::from_shape_vec((n, 6), out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py).into())
    }

    /// Forward kinematics for (n, 6) joints, returning (n, 16) row-major matrices.
    #[pyo3(signature = (joints, ee_transform=None))]
    fn batch_forward<'py>(
        &self,
        py: Python<'py>,
        joints: PyReadonlyArray2<'py, f64>,
        ee_transform: Option<Matrix4x4>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let joints = joints.as_array();
        if joints.ncols() != 6 {
            return Err(PyValueError::new_err("joints must have shape (n, 6)"));
        }
        let n = joints.nrows();
        let ee = ee_rt(ee_transform.as_ref());
        let mut out = Vec::with_capacity(n * 16);
        for row in joints.rows() {
            if row.iter().any(|v| v.is_nan()) {
                out.extend_from_slice(&[f64::NAN; 16]);
                continue;
            }
            let j: [f64; 6] = std::array::from_fn(|i| row[i]);
            let (r, t) = self.forward_rt(j, ee.as_ref());
            push_matrix(&mut out, &rt_to_matrix(&r, &t));
        }
        flat_rows(py, n, out)
    }

    /// Per-joint frames [J1, ..., J6], plus TCP when ee_transform is given.
    #[pyo3(signature = (joints, ee_transform=None))]
    fn joint_poses(&self, joints: [f64; 6], ee_transform: Option<Matrix4x4>) -> Vec<Matrix4x4> {
        self.joint_isos(joints, ee_transform.as_ref())
            .iter()
            .map(to_matrix)
            .collect()
    }

    /// Per-joint frames for (n, 6) joints as (n*6, 16) or (n*7, 16) with ee_transform.
    #[pyo3(signature = (joints, ee_transform=None))]
    fn batch_joint_poses<'py>(
        &self,
        py: Python<'py>,
        joints: PyReadonlyArray2<'py, f64>,
        ee_transform: Option<Matrix4x4>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let joints = joints.as_array();
        if joints.ncols() != 6 {
            return Err(PyValueError::new_err("joints must have shape (n, 6)"));
        }
        let n = joints.nrows();
        let per = if ee_transform.is_some() { 7 } else { 6 };
        let mut out = Vec::with_capacity(n * per * 16);
        for row in joints.rows() {
            if row.iter().any(|v| v.is_nan()) {
                out.extend(std::iter::repeat_n(f64::NAN, per * 16));
                continue;
            }
            let j: [f64; 6] = std::array::from_fn(|i| row[i]);
            for m in self.joint_poses(j, ee_transform) {
                push_matrix(&mut out, &m);
            }
        }
        flat_rows(py, n * per, out)
    }

    /// Frames [Base, J1, ..., J6, TCP]; unlike py-opw's deprecated version the rotations are correct.
    #[pyo3(signature = (joints, ee_transform=None))]
    fn forward_frames(&self, joints: [f64; 6], ee_transform: Option<Matrix4x4>) -> Vec<Matrix4x4> {
        let mut out = vec![to_matrix(&Isometry3::identity())];
        let frames = self.robot.joint_frames(&self.joints_rad(joints));
        out.extend(frames.iter().map(to_matrix));
        out.push(self.forward(joints, ee_transform));
        out
    }

    /// 6x6 geometric Jacobian of the TCP with respect to the joints in radians (base frame,
    /// translation rows first), regardless of the degrees flag.
    #[pyo3(signature = (joints, ee_transform=None))]
    fn jacobian<'py>(
        &self,
        py: Python<'py>,
        joints: [f64; 6],
        ee_transform: Option<Matrix4x4>,
    ) -> Py<PyArray2<f64>> {
        let tcp = ee_iso(ee_transform.as_ref()).translation.vector;
        let jac = self.robot.jacobian(&self.joints_rad(joints), &tcp);
        Array2::from_shape_fn((6, 6), |(r, c)| jac[(r, c)])
            .into_pyarray(py)
            .into()
    }

    /// Every inverse-kinematics branch of (n, 16) TCP poses with margins, GIL released.
    /// Returns (joints (n, 16, 6), limit_margin (n, 16), extension (n,), sigma_min (n, 16),
    /// wrist (n, 16), n_solutions (n,)). Angles follow the degrees flag; sigma_min is per radian.
    #[pyo3(signature = (poses, joint_limits=None, ee_transform=None, threads=1))]
    fn reach<'py>(
        &self,
        py: Python<'py>,
        poses: PyReadonlyArray2<'py, f64>,
        joint_limits: Option<[[f64; 2]; 6]>,
        ee_transform: Option<Matrix4x4>,
        threads: usize,
    ) -> PyResult<ReachArrays> {
        let poses = poses.as_array();
        if poses.ncols() != 16 {
            return Err(PyValueError::new_err("poses must have shape (n, 16)"));
        }
        let n = poses.nrows();
        let ee = ee_iso(ee_transform.as_ref());
        let ee_inv = ee.inverse();
        let tcp_in_flange = ee.translation.vector;
        let to_rad = if self.degrees {
            std::f64::consts::PI / 180.0
        } else {
            1.0
        };
        let limits = joint_limits.map(|lim| lim.map(|[lo, hi]| (lo * to_rad, hi * to_rad)));
        // Copied out so the solve runs without the GIL and without aliasing Python memory.
        let rows: Vec<f64> = poses.iter().copied().collect();

        let results: Vec<PoseReach> = py.detach(|| {
            // NaN rows are solved as NaN results in place, so the index mapping stays trivial.
            let flanges: Vec<Option<Isometry3<f64>>> = rows
                .chunks(16)
                .map(|row| {
                    if row.iter().any(|v| v.is_nan()) {
                        None
                    } else {
                        let m: Matrix4x4 =
                            std::array::from_fn(|r| std::array::from_fn(|c| row[r * 4 + c]));
                        Some(from_matrix(&m) * ee_inv)
                    }
                })
                .collect();
            let valid: Vec<Isometry3<f64>> = flanges.iter().flatten().copied().collect();
            let solved = self
                .robot
                .reach(&valid, limits.as_ref(), &tcp_in_flange, threads);
            let mut it = solved.into_iter();
            flanges
                .iter()
                .map(|f| match f {
                    Some(_) => it.next().unwrap_or_else(PoseReach::nan),
                    None => PoseReach::nan(),
                })
                .collect()
        });
        if results.len() != n {
            return Err(PyRuntimeError::new_err(
                "reach produced a wrong number of rows",
            ));
        }

        let scale = 1.0 / to_rad;
        let mut joints = Vec::with_capacity(n * BRANCHES * 6);
        let mut margins = Vec::with_capacity(n * BRANCHES);
        let mut extension = Vec::with_capacity(n);
        let mut sigma = Vec::with_capacity(n * BRANCHES);
        let mut wrist = Vec::with_capacity(n * BRANCHES);
        let mut count = Vec::with_capacity(n);
        for res in &results {
            for s in 0..BRANCHES {
                joints.extend(res.joints[s].iter().map(|v| v * scale));
                margins.push(res.limit_margin[s] * scale);
                wrist.push(res.wrist[s] * scale);
            }
            sigma.extend_from_slice(&res.sigma_min);
            extension.push(res.extension);
            count.push(res.n_solutions as i64);
        }
        let err = |e: numpy::ndarray::ShapeError| PyValueError::new_err(e.to_string());
        Ok((
            Array3::from_shape_vec((n, BRANCHES, 6), joints)
                .map_err(err)?
                .into_pyarray(py)
                .into(),
            Array2::from_shape_vec((n, BRANCHES), margins)
                .map_err(err)?
                .into_pyarray(py)
                .into(),
            Array1::from_vec(extension).into_pyarray(py).into(),
            Array2::from_shape_vec((n, BRANCHES), sigma)
                .map_err(err)?
                .into_pyarray(py)
                .into(),
            Array2::from_shape_vec((n, BRANCHES), wrist)
                .map_err(err)?
                .into_pyarray(py)
                .into(),
            Array1::from_vec(count).into_pyarray(py).into(),
        ))
    }
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KinematicModel>()?;
    m.add_class::<Robot>()?;
    Ok(())
}
