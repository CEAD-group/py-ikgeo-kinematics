//! Robot model on top of the vendored solvers: joint conventions, family
//! detection with a solver-ready canonical chain, forward kinematics, per-joint
//! frames and the geometric Jacobian. Angles are radians throughout.

use crate::inverse_kinematics::auxiliary::Kinematics;
pub type Matrix3x7 = crate::inverse_kinematics::auxiliary::Matrix3x7<f64>;
use crate::subproblems::auxiliary::rot;
use nalgebra::{
    Isometry3, Matrix3, Matrix3x6, Matrix6, Rotation3, Translation3, UnitQuaternion, Vector3,
    Vector6,
};

/// Default tolerance for the axis tests: radians for parallelism, fraction of
/// the longest offset for intersection. Pairs inside it are snapped to exact.
pub const AXIS_TOL: f64 = 1e-9;

/// Measured geometry of one consecutive axis pair, before any snapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisPairReport {
    /// Angle between the two axis lines in radians, 0 for parallel or anti-parallel.
    pub parallel_deviation: f64,
    /// Shortest distance between the two axis lines divided by the longest offset; NaN for a parallel pair.
    pub intersection_deviation: f64,
    pub parallel: bool,
    pub intersecting: bool,
}

/// A chain no vendored solver can handle; rejected at construction rather than
/// returning no roots for every pose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelError(pub String);

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ModelError {}

/// Kinematic family as named in the IK-Geo paper (Table I).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Spherical,
    SphericalTwoIntersecting,
    SphericalTwoParallel,
    ThreeParallel,
    ThreeParallelTwoIntersecting,
    ThreeParallelTwoParallel,
    TwoIntersecting,
    TwoIntersectingTwoIntersecting,
    TwoIntersectingTwoParallel,
    TwoParallel,
    TwoParallelTwoParallel,
    General,
}

/// Default tracker step limit: half a radian per joint between consecutive poses.
pub const DEFAULT_MAX_TRACK_STEP: f64 = 0.5;

/// Which vendored solver in `inverse_kinematics` handles a family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Solver {
    Spherical,
    SphericalTwoIntersecting,
    SphericalTwoParallel,
    ThreeParallel,
    ThreeParallelTwoIntersecting,
    TwoIntersecting,
    TwoIntersectingTwoParallel,
    TwoParallel,
    GenSixDof,
}

impl Family {
    pub fn name(self) -> &'static str {
        match self {
            Family::Spherical => "spherical",
            Family::SphericalTwoIntersecting => "spherical_two_intersecting",
            Family::SphericalTwoParallel => "spherical_two_parallel",
            Family::ThreeParallel => "three_parallel",
            Family::ThreeParallelTwoIntersecting => "three_parallel_two_intersecting",
            Family::ThreeParallelTwoParallel => "three_parallel_two_parallel",
            Family::TwoIntersecting => "two_intersecting",
            Family::TwoIntersectingTwoIntersecting => "two_intersecting_two_intersecting",
            Family::TwoIntersectingTwoParallel => "two_intersecting_two_parallel",
            Family::TwoParallel => "two_parallel",
            Family::TwoParallelTwoParallel => "two_parallel_two_parallel",
            Family::General => "general",
        }
    }

    pub fn solver(self) -> Solver {
        match self {
            Family::Spherical => Solver::Spherical,
            Family::SphericalTwoIntersecting => Solver::SphericalTwoIntersecting,
            Family::SphericalTwoParallel => Solver::SphericalTwoParallel,
            // Upstream has no dedicated solver for a second parallel pair next to three parallel axes.
            Family::ThreeParallel | Family::ThreeParallelTwoParallel => Solver::ThreeParallel,
            Family::ThreeParallelTwoIntersecting => Solver::ThreeParallelTwoIntersecting,
            // The 1D search over q4 only uses the 5/6 intersection; a second intersecting pair just shapes it.
            Family::TwoIntersecting | Family::TwoIntersectingTwoIntersecting => {
                Solver::TwoIntersecting
            }
            // Subproblem 5 degenerates for parallel axes 2/3, so this family has its own planar solve.
            Family::TwoIntersectingTwoParallel => Solver::TwoIntersectingTwoParallel,
            Family::TwoParallel | Family::TwoParallelTwoParallel => Solver::TwoParallel,
            Family::General => Solver::GenSixDof,
        }
    }

    /// 0 closed form, 1 one-dimensional search, 2 two-dimensional search.
    pub fn search_dimension(self) -> u8 {
        match self.solver() {
            Solver::Spherical
            | Solver::SphericalTwoIntersecting
            | Solver::SphericalTwoParallel
            | Solver::ThreeParallel
            | Solver::ThreeParallelTwoIntersecting => 0,
            Solver::TwoIntersecting | Solver::TwoIntersectingTwoParallel | Solver::TwoParallel => 1,
            Solver::GenSixDof => 2,
        }
    }
}

/// Consecutive-pair geometry of a six-axis chain; index i is the pair (i+1, i+2) in joint numbering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisPairs {
    pub parallel: [bool; 5],
    pub intersecting: [bool; 5],
}

/// The chain the solver sees. `kin` has intersecting pairs with a zero offset
/// and parallel pairs with identical directions; `axis_sign` and `reversed`
/// map solver angles back to the model's internal angles.
#[derive(Debug, Clone)]
pub struct CanonicalChain {
    pub kin: Kinematics<6, 7>,
    pub axis_sign: [f64; 6],
    pub reversed: bool,
    pub family: Family,
    pub pairs: AxisPairs,
}

impl CanonicalChain {
    /// Solver angles (in the canonical chain order) to internal model angles.
    pub fn to_internal(&self, q_solver: &Vector6<f64>) -> Vector6<f64> {
        let mut q = Vector6::zeros();
        for j in 0..6 {
            let src = if self.reversed { 5 - j } else { j };
            q[j] = self.axis_sign[src] * q_solver[src];
        }
        q
    }

    /// Internal model angles to solver angles.
    pub fn from_internal(&self, q: &Vector6<f64>) -> Vector6<f64> {
        let mut out = Vector6::zeros();
        for j in 0..6 {
            let dst = if self.reversed { 5 - j } else { j };
            out[dst] = self.axis_sign[dst] * q[j];
        }
        out
    }
}

/// Product-of-exponentials model plus the per-joint user conventions.
/// Internal angle q_j = flip_j * user_j - offset_j, as rs-opw-kinematics does.
#[derive(Debug, Clone)]
pub struct KinematicModel {
    pub h: Matrix3x6<f64>,
    pub p: Matrix3x7,
    pub offsets: [f64; 6],
    pub flip: [f64; 6],
}

impl KinematicModel {
    pub fn new(h: Matrix3x6<f64>, p: Matrix3x7, offsets: [f64; 6], flip_axes: [bool; 6]) -> Self {
        let mut h = h;
        for j in 0..6 {
            let n = h.column(j).norm();
            if n > 0.0 {
                let c = h.column(j) / n;
                h.set_column(j, &c);
            }
        }
        KinematicModel {
            h,
            p,
            offsets,
            flip: flip_axes.map(|f| if f { -1.0 } else { 1.0 }),
        }
    }

    /// The OPW layout of rs-opw-kinematics as a product of exponentials, offsets in radians.
    #[allow(clippy::too_many_arguments)]
    pub fn from_opw(
        a1: f64,
        a2: f64,
        b: f64,
        c1: f64,
        c2: f64,
        c3: f64,
        c4: f64,
        offsets: [f64; 6],
        flip_axes: [bool; 6],
    ) -> Self {
        let ey = Vector3::y();
        let ez = Vector3::z();
        let h = Matrix3x6::from_columns(&[ez, ey, ey, ez, ey, ez]);
        let p = Matrix3x7::from_columns(&[
            Vector3::new(0.0, 0.0, c1),
            Vector3::new(a1, b, 0.0),
            Vector3::new(0.0, 0.0, c2),
            Vector3::new(a2, 0.0, 0.0),
            Vector3::new(0.0, 0.0, c3),
            Vector3::new(0.0, 0.0, c4),
            Vector3::zeros(),
        ]);
        KinematicModel::new(h, p, offsets, flip_axes)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        for (name, values) in [
            ("h", self.h.as_slice()),
            ("p", self.p.as_slice()),
            ("offsets", self.offsets.as_slice()),
            ("flip", self.flip.as_slice()),
        ] {
            if values.iter().any(|v| !v.is_finite()) {
                return Err(ModelError(format!(
                    "{name} must contain only finite values"
                )));
            }
        }
        for j in 0..6 {
            let norm = self.h.column(j).norm();
            if !norm.is_finite() || norm < 1e-12 {
                return Err(ModelError(format!(
                    "h column {j} must have a finite, nonzero norm"
                )));
            }
        }
        Ok(())
    }

    pub fn flip_axes(&self) -> [bool; 6] {
        self.flip.map(|f| f < 0.0)
    }

    pub fn kinematics(&self) -> Kinematics<6, 7> {
        Kinematics {
            h: self.h,
            p: self.p,
        }
    }

    pub fn to_internal(&self, user: &[f64; 6]) -> Vector6<f64> {
        Vector6::from_fn(|j, _| self.flip[j] * user[j] - self.offsets[j])
    }

    pub fn to_user(&self, q: &Vector6<f64>) -> [f64; 6] {
        std::array::from_fn(|j| self.flip[j] * (q[j] + self.offsets[j]))
    }
}

fn longest_offset(p: &Matrix3x7) -> f64 {
    let l = (0..7).map(|i| p.column(i).norm()).fold(0.0, f64::max);
    if l > 0.0 {
        l
    } else {
        1.0
    }
}

/// Measured deviations of the five consecutive pairs with the flags `tol` assigns.
pub fn axis_report(kin: &Kinematics<6, 7>, tol: f64) -> [AxisPairReport; 5] {
    let scale = longest_offset(&kin.p);
    std::array::from_fn(|i| {
        let hi: Vector3<f64> = kin.h.column(i).into();
        let hj: Vector3<f64> = kin.h.column(i + 1).into();
        let n = hi.cross(&hj);
        let n_norm = n.norm();
        // Anti-parallel axes count as parallel, so measure the smaller of the two angles.
        let angle = n_norm.atan2(hi.dot(&hj).abs());
        if angle < tol {
            AxisPairReport {
                parallel_deviation: angle,
                intersection_deviation: f64::NAN,
                parallel: true,
                intersecting: false,
            }
        } else {
            // Signed distance between the two lines along their common normal.
            let d = kin.p.column(i + 1).dot(&n) / n_norm;
            AxisPairReport {
                parallel_deviation: angle,
                intersection_deviation: d.abs() / scale,
                parallel: false,
                intersecting: d.abs() < tol * scale,
            }
        }
    })
}

/// Parallel and intersecting flags for the five consecutive pairs.
pub fn axis_pairs(kin: &Kinematics<6, 7>, tol: f64) -> AxisPairs {
    let report = axis_report(kin, tol);
    AxisPairs {
        parallel: report.map(|r| r.parallel),
        intersecting: report.map(|r| r.intersecting),
    }
}

/// Reversed chain: tool end becomes the base, so the forward map is the inverse pose.
pub fn reversed_kinematics(kin: &Kinematics<6, 7>) -> Kinematics<6, 7> {
    let mut out: Kinematics<6, 7> = Kinematics::new();
    for j in 0..6 {
        out.h.set_column(j, &(-kin.h.column(5 - j)));
    }
    for i in 0..7 {
        out.p.set_column(i, &(-kin.p.column(6 - i)));
    }
    out
}

/// Zero the offsets of intersecting pairs and align parallel axes, FK-invariant.
pub(crate) fn canonicalise(
    kin: &Kinematics<6, 7>,
    pairs: &AxisPairs,
    tol: f64,
) -> (Kinematics<6, 7>, [f64; 6]) {
    let mut k = kin.clone();
    let mut sign = [1.0; 6];
    let scale = longest_offset(&kin.p);
    for i in 0..5 {
        let hi: Vector3<f64> = k.h.column(i).into();
        let hj: Vector3<f64> = k.h.column(i + 1).into();
        if pairs.parallel[i] {
            if hi.dot(&hj) < 0.0 {
                // Upstream parallel solvers want identical directions; flipping the axis negates the angle.
                k.h.set_column(i + 1, &hi);
                sign[i + 1] = -sign[i + 1];
            } else {
                k.h.set_column(i + 1, &hi);
            }
        } else if pairs.intersecting[i] {
            let p: Vector3<f64> = k.p.column(i + 1).into();
            let c = hi.dot(&hj);
            let det = 1.0 - c * c;
            let a = (p.dot(&hi) - c * p.dot(&hj)) / det;
            let b = (p.dot(&hj) - c * p.dot(&hi)) / det;
            // R_i h_i = h_i and R_{i+1} h_{i+1} = h_{i+1}, so both moves leave FK unchanged.
            let pi = k.p.column(i) + a * hi;
            let pk = k.p.column(i + 2) + b * hj;
            k.p.set_column(i, &pi);
            k.p.set_column(i + 1, &Vector3::zeros());
            k.p.set_column(i + 2, &pk);
        }
    }
    for i in 0..7 {
        if k.p.column(i).norm() < tol * scale {
            k.p.set_column(i, &Vector3::zeros());
        }
    }
    (k, sign)
}

fn family_of(kin_canonical: &Kinematics<6, 7>, pairs: &AxisPairs) -> Family {
    let (par, int) = (pairs.parallel, pairs.intersecting);
    let wrist_concurrent = int[3]
        && int[4]
        && kin_canonical.p.column(4).norm() == 0.0
        && kin_canonical.p.column(5).norm() == 0.0;
    if wrist_concurrent {
        if par[1] {
            Family::SphericalTwoParallel
        } else if int[0] {
            Family::SphericalTwoIntersecting
        } else {
            Family::Spherical
        }
    } else if par[1] && par[2] {
        if int[4] {
            Family::ThreeParallelTwoIntersecting
        } else if par[4] {
            Family::ThreeParallelTwoParallel
        } else {
            Family::ThreeParallel
        }
    } else if int[4] {
        if int[0] {
            Family::TwoIntersectingTwoIntersecting
        } else if par[1] {
            Family::TwoIntersectingTwoParallel
        } else {
            Family::TwoIntersecting
        }
    } else if par[1] {
        // Pair 5/6 parallel still solves with the 1D search; pair 4/5 parallel does not (see `Robot::new`).
        if par[3] || par[4] {
            Family::TwoParallelTwoParallel
        } else {
            Family::TwoParallel
        }
    } else {
        Family::General
    }
}

fn classify_direction(kin: &Kinematics<6, 7>, reversed: bool, tol: f64) -> CanonicalChain {
    let pairs = axis_pairs(kin, tol);
    let (canonical, axis_sign) = canonicalise(kin, &pairs, tol);
    let family = family_of(&canonical, &pairs);
    CanonicalChain {
        kin: canonical,
        axis_sign,
        reversed,
        family,
        pairs,
    }
}

/// Detect the family of a chain, trying the reversed chain when that gives a cheaper solver.
pub fn classify(kin: &Kinematics<6, 7>) -> CanonicalChain {
    classify_with_tolerance(kin, AXIS_TOL)
}

/// `classify` with an explicit axis tolerance (radians and fraction of the longest offset).
pub fn classify_with_tolerance(kin: &Kinematics<6, 7>, tol: f64) -> CanonicalChain {
    let forward = classify_direction(kin, false, tol);
    if forward.family.search_dimension() == 0 {
        return forward;
    }
    let backward = classify_direction(&reversed_kinematics(kin), true, tol);
    if backward.family.search_dimension() < forward.family.search_dimension() {
        backward
    } else {
        forward
    }
}

/// A robot: user conventions, the original chain for FK and frames, and the solver chain.
#[derive(Debug, Clone)]
pub struct Robot {
    pub model: KinematicModel,
    pub kin: Kinematics<6, 7>,
    pub chain: CanonicalChain,
    /// Largest per-joint move (radians) the tracker accepts before falling back to a cold solve.
    pub max_track_step: f64,
    /// Axis tolerance the classification used, see `AXIS_TOL`.
    pub axis_tolerance: f64,
}

impl Robot {
    /// Classify the chain and pick its solver. Chains whose solver cannot
    /// produce roots (axes 2/3 and 4/5 both parallel: Subproblems 5 and 6
    /// degenerate for the collinear wrist pair) are rejected here.
    pub fn new(model: KinematicModel) -> Result<Self, ModelError> {
        Self::with_tolerance(model, AXIS_TOL)
    }

    /// `new` with an explicit axis tolerance, for models whose axes carry float noise.
    pub fn with_tolerance(model: KinematicModel, axis_tolerance: f64) -> Result<Self, ModelError> {
        if !(axis_tolerance.is_finite() && axis_tolerance >= 0.0) {
            return Err(ModelError(format!(
                "axis_tolerance must be a non-negative number, got {axis_tolerance}"
            )));
        }
        model.validate()?;
        let kin = model.kinematics();
        let chain = classify_with_tolerance(&kin, axis_tolerance);
        if chain.family == Family::TwoParallelTwoParallel && chain.pairs.parallel[3] {
            return Err(ModelError(
                "axes 2/3 and 4/5 are both parallel: no vendored IK-Geo solver handles this chain"
                    .to_string(),
            ));
        }
        Ok(Robot {
            model,
            kin,
            chain,
            max_track_step: DEFAULT_MAX_TRACK_STEP,
            axis_tolerance,
        })
    }

    /// Measured deviation of every consecutive axis pair of the model's own chain.
    pub fn axis_report(&self) -> [AxisPairReport; 5] {
        axis_report(&self.kin, self.axis_tolerance)
    }

    pub fn set_max_track_step(&mut self, radians: f64) {
        self.max_track_step = radians;
    }

    pub fn family(&self) -> Family {
        self.chain.family
    }

    /// Flange rotation and position for internal angles.
    pub fn forward_internal(&self, q: &Vector6<f64>) -> (Matrix3<f64>, Vector3<f64>) {
        self.kin.forward_kinematics(q)
    }

    /// Flange pose for user angles in radians.
    pub fn forward(&self, joints: &[f64; 6]) -> Isometry3<f64> {
        let (r, t) = self.forward_rt(joints);
        iso(&r, &t)
    }

    /// Flange rotation and position for user angles in radians, without the quaternion
    /// conversion of `forward`; this is the batch FK hot path.
    pub fn forward_rt(&self, joints: &[f64; 6]) -> (Matrix3<f64>, Vector3<f64>) {
        let q = self.model.to_internal(joints);
        let mut r = Matrix3::identity();
        let mut t: Vector3<f64> = self.kin.p.column(0).into();
        for j in 0..6 {
            r *= rot_unit(&self.kin.h.column(j).into(), q[j]);
            t += r * self.kin.p.column(j + 1);
        }
        (r, t)
    }

    /// Frame j has its origin on axis j (at the model's own offset point) and rotation R_1..R_j.
    pub fn joint_frames(&self, joints: &[f64; 6]) -> [Isometry3<f64>; 6] {
        let q = self.model.to_internal(joints);
        let mut r = Matrix3::identity();
        let mut o: Vector3<f64> = self.kin.p.column(0).into();
        let mut out = [Isometry3::identity(); 6];
        for j in 0..6 {
            r *= rot(&self.kin.h.column(j).into(), q[j]);
            out[j] = iso(&r, &o);
            o += r * self.kin.p.column(j + 1);
        }
        out
    }

    /// Geometric Jacobian of the point `tcp_in_flange` (flange frame) with respect to the user
    /// angles in radians: rows 0..3 translation, rows 3..6 rotation, both in the base frame.
    pub fn jacobian(&self, joints: &[f64; 6], tcp_in_flange: &Vector3<f64>) -> Matrix6<f64> {
        let q = self.model.to_internal(joints);
        let mut r = Matrix3::identity();
        let mut o: Vector3<f64> = self.kin.p.column(0).into();
        let mut axes = [Vector3::zeros(); 6];
        let mut origins = [Vector3::zeros(); 6];
        for j in 0..6 {
            axes[j] = r * self.kin.h.column(j);
            origins[j] = o;
            r *= rot(&self.kin.h.column(j).into(), q[j]);
            o += r * self.kin.p.column(j + 1);
        }
        let tcp = o + r * tcp_in_flange;
        let mut jac = Matrix6::zeros();
        for j in 0..6 {
            // A flipped user axis spins the other way, so its column changes sign.
            let z = axes[j] * self.model.flip[j];
            let v = z.cross(&(tcp - origins[j]));
            jac.fixed_view_mut::<3, 1>(0, j).copy_from(&v);
            jac.fixed_view_mut::<3, 1>(3, j).copy_from(&z);
        }
        jac
    }
}

/// Smallest singular value of a 6x6 Jacobian.
pub fn sigma_min(jac: &Matrix6<f64>) -> f64 {
    let eig = (jac.transpose() * jac).symmetric_eigenvalues();
    eig.iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        .max(0.0)
        .sqrt()
}

/// Rodrigues rotation about a unit axis, written out so FK does not renormalise or
/// multiply skew matrices per joint.
pub fn rot_unit(k: &Vector3<f64>, theta: f64) -> Matrix3<f64> {
    let (s, c) = theta.sin_cos();
    let v = 1.0 - c;
    let (x, y, z) = (k[0], k[1], k[2]);
    Matrix3::new(
        c + x * x * v,
        x * y * v - z * s,
        x * z * v + y * s,
        y * x * v + z * s,
        c + y * y * v,
        y * z * v - x * s,
        z * x * v - y * s,
        z * y * v + x * s,
        c + z * z * v,
    )
}

pub fn iso(r: &Matrix3<f64>, t: &Vector3<f64>) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::from(*t),
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*r)),
    )
}

/// Test and example models shared by the solver steps.
pub mod fixtures {
    use super::{KinematicModel, Matrix3x6, Matrix3x7, Vector3};

    /// Comau NJ4-220-3.0 from drawing CR82275505, millimetres, zero pose at calibration
    /// (J2 = 0, J3 = -90 in Comau terms), base on the floor under axis 1, X forward, Z up.
    /// Axes 2/3 are parallel, 4/5 and 5/6 intersect at 60 degrees, 170 mm apart along axis 5.
    /// The 5/6 point sits 147.22 mm above the forearm axis at calibration, as sheet 1 draws it.
    pub fn comau_nj4_220() -> KinematicModel {
        let beta = 60.0_f64.to_radians();
        let h = Matrix3x6::from_columns(&[
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(beta.cos(), 0.0, beta.sin()),
            Vector3::new(1.0, 0.0, 0.0),
        ]);
        let p = Matrix3x7::from_columns(&[
            Vector3::zeros(),
            Vector3::new(460.0, 0.0, 1140.0),
            Vector3::new(0.0, 0.0, 1050.0),
            Vector3::new(1515.0, 0.0, 250.0),
            Vector3::zeros(),
            170.0 * Vector3::new(beta.cos(), 0.0, beta.sin()),
            Vector3::new(225.0, 0.0, 0.0),
        ]);
        KinematicModel::new(h, p, [0.0; 6], [false; 6])
    }

    /// Comau NJ165 as py-opw-kinematics' example parameterises it.
    pub fn comau_nj165_opw() -> KinematicModel {
        KinematicModel::from_opw(
            400.0,
            -250.0,
            0.0,
            830.0,
            1175.0,
            1444.0,
            230.0,
            [0.0; 6],
            [true, false, true, true, false, true],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;
    use crate::inverse_kinematics::hardcoded::setups::{
        Irb6640, KukaR800FixedQ3, RrcFixedQ6, SphericalBot, ThreeParallelBot, TwoParallelBot, Ur5,
    };
    use crate::subproblems::auxiliary::{random_angle, seed_random};
    use std::f64::consts::PI;

    fn random_q() -> Vector6<f64> {
        Vector6::from_fn(|_, _| random_angle())
    }

    #[test]
    fn invalid_models_are_rejected_before_classification() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for field in ["h", "p", "offsets", "flip"] {
                let mut model = comau_nj165_opw();
                match field {
                    "h" => model.h[(0, 0)] = value,
                    "p" => model.p[(0, 0)] = value,
                    "offsets" => model.offsets[0] = value,
                    _ => model.flip[0] = value,
                }
                assert!(Robot::new(model).is_err(), "accepted {field} = {value}");
            }
        }
        let mut model = comau_nj165_opw();
        model.h.set_column(0, &Vector3::zeros());
        assert!(Robot::new(model).is_err());
    }

    #[test]
    fn classifies_upstream_fixtures() {
        let cases: Vec<(Kinematics<6, 7>, Family)> = vec![
            (Irb6640::get_kin(), Family::SphericalTwoParallel),
            (Ur5::get_kin(), Family::ThreeParallelTwoIntersecting),
            (
                KukaR800FixedQ3::get_kin_partial().0,
                Family::SphericalTwoIntersecting,
            ),
            (RrcFixedQ6::get_kin_partial().0, Family::TwoIntersecting),
            (TwoParallelBot::get_kin(), Family::TwoParallel),
            (ThreeParallelBot::get_kin(), Family::ThreeParallel),
            (SphericalBot::get_kin(), Family::Spherical),
            (
                comau_nj4_220().kinematics(),
                Family::TwoIntersectingTwoParallel,
            ),
            (comau_nj165_opw().kinematics(), Family::SphericalTwoParallel),
        ];
        for (kin, expected) in cases {
            let chain = classify(&kin);
            assert_eq!(chain.family, expected, "{:?}", kin.h);
            assert!(!chain.reversed);
        }
        let opw_b =
            KinematicModel::from_opw(0.4, -0.25, 0.1, 0.8, 1.1, 1.4, 0.2, [0.0; 6], [false; 6]);
        let chain = classify(&opw_b.kinematics());
        assert_eq!(chain.family, Family::SphericalTwoParallel);
        assert!(!chain.pairs.intersecting[0]);
        let opw_a1 =
            KinematicModel::from_opw(0.0, -0.25, 0.0, 0.8, 1.1, 1.4, 0.2, [0.0; 6], [false; 6]);
        assert!(classify(&opw_a1.kinematics()).pairs.intersecting[0]);
    }

    #[test]
    fn reversed_chain_is_detected() {
        // A spherical wrist at the base and a parallel pair at the tool end.
        let kin = reversed_kinematics(&Irb6640::get_kin());
        let chain = classify(&kin);
        assert!(chain.reversed);
        assert_eq!(chain.family, Family::SphericalTwoParallel);
        assert!(classify(&Irb6640::get_kin()).pairs.parallel[1]);
    }

    #[test]
    fn reversed_chain_forward_is_inverse_pose() {
        seed_random(7);
        let kin = comau_nj4_220().kinematics();
        let rev = reversed_kinematics(&kin);
        for _ in 0..20 {
            let q = random_q();
            let (r, t) = kin.forward_kinematics(&q);
            let q_rev = Vector6::from_fn(|j, _| q[5 - j]);
            let (r2, t2) = rev.forward_kinematics(&q_rev);
            let inv = iso(&r, &t).inverse();
            assert!((inv.rotation.to_rotation_matrix().matrix() - r2).norm() < 1e-9);
            assert!((inv.translation.vector - t2).norm() < 1e-6);
        }
    }

    #[test]
    fn canonical_chain_preserves_fk() {
        seed_random(11);
        let models = [
            comau_nj4_220(),
            comau_nj165_opw(),
            KinematicModel::from_opw(0.4, -0.25, 0.1, 0.8, 1.1, 1.4, 0.2, [0.0; 6], [false; 6]),
        ];
        for model in models.iter() {
            let kin = model.kinematics();
            let chain = classify(&kin);
            let scale = longest_offset(&kin.p);
            for _ in 0..50 {
                let q = random_q();
                let (r, t) = kin.forward_kinematics(&q);
                let (r2, t2) = chain.kin.forward_kinematics(&chain.from_internal(&q));
                assert!((r - r2).norm() < 1e-12);
                assert!((t - t2).norm() < 1e-9 * scale);
                assert!((chain.to_internal(&chain.from_internal(&q)) - q).norm() < 1e-15);
            }
        }
        let nj4 = classify(&comau_nj4_220().kinematics());
        assert_eq!(nj4.kin.p.column(5).norm(), 0.0);
        let h5: Vector3<f64> = nj4.kin.h.column(4).into();
        assert!((nj4.kin.p.column(4) - 170.0 * h5).norm() < 1e-12);
        assert_eq!(nj4.axis_sign, [1.0, 1.0, -1.0, 1.0, 1.0, 1.0]);
        let opw = classify(&comau_nj165_opw().kinematics());
        assert_eq!(opw.kin.p.column(4).norm(), 0.0);
        assert_eq!(opw.kin.p.column(5).norm(), 0.0);
        assert!((opw.kin.p.column(6) - Vector3::new(0.0, 0.0, 230.0)).norm() < 1e-12);
    }

    #[test]
    fn opw_user_conventions_round_trip() {
        let robot = Robot::new(comau_nj165_opw()).unwrap();
        let joints = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6];
        let q = robot.model.to_internal(&joints);
        assert_eq!(robot.model.to_user(&q), joints);
        // Comau calibration pose J3 = -90 with flipped J3 puts the forearm horizontal.
        let pose = robot.forward(&[0.0, 0.0, -PI / 2.0, 0.0, 0.0, 0.0]);
        let t = pose.translation.vector;
        assert!(
            (t - Vector3::new(400.0 + 1444.0 + 230.0, 0.0, 830.0 + 1175.0 + 250.0)).norm() < 1e-9
        );
    }

    #[test]
    fn joint_frames_match_chain() {
        seed_random(3);
        let robot = Robot::new(comau_nj4_220()).unwrap();
        for _ in 0..20 {
            let joints: [f64; 6] = std::array::from_fn(|_| random_angle());
            let frames = robot.joint_frames(&joints);
            let flange = robot.forward(&joints);
            let last = frames[5] * Translation3::from(Vector3::new(225.0, 0.0, 0.0));
            assert!((last.translation.vector - flange.translation.vector).norm() < 1e-9);
            assert!((last.rotation.angle_to(&flange.rotation)).abs() < 1e-12);
            // Frame 1 sits on the base point, frame 2 at the shoulder of the original p.
            assert!((frames[0].translation.vector).norm() < 1e-12);
            let shoulder = frames[0].rotation * Vector3::new(460.0, 0.0, 1140.0);
            assert!((frames[1].translation.vector - shoulder).norm() < 1e-9);
        }
    }

    #[test]
    fn jacobian_matches_finite_differences() {
        seed_random(5);
        for model in [comau_nj4_220(), comau_nj165_opw()] {
            let robot = Robot::new(model).unwrap();
            let tcp = Vector3::new(10.0, -20.0, 30.0);
            for _ in 0..10 {
                let joints: [f64; 6] = std::array::from_fn(|_| random_angle());
                let jac = robot.jacobian(&joints, &tcp);
                let pose = robot.forward(&joints) * Translation3::from(tcp);
                let eps = 1e-6;
                for j in 0..6 {
                    let mut jp = joints;
                    jp[j] += eps;
                    let mut jm = joints;
                    jm[j] -= eps;
                    let pp = robot.forward(&jp) * Translation3::from(tcp);
                    let pm = robot.forward(&jm) * Translation3::from(tcp);
                    let dv = (pp.translation.vector - pm.translation.vector) / (2.0 * eps);
                    let dr = (pp.rotation.to_rotation_matrix().matrix()
                        - pm.rotation.to_rotation_matrix().matrix())
                        / (2.0 * eps);
                    let w_hat = dr * pose.rotation.to_rotation_matrix().matrix().transpose();
                    let w = Vector3::new(w_hat[(2, 1)], w_hat[(0, 2)], w_hat[(1, 0)]);
                    let col = jac.column(j);
                    assert!((col.fixed_rows::<3>(0) - dv).norm() < 1e-4, "{dv} vs {col}");
                    assert!((col.fixed_rows::<3>(3) - w).norm() < 1e-6, "{w} vs {col}");
                }
            }
        }
    }

    #[test]
    fn sigma_min_is_zero_at_wrist_singularity() {
        let robot = Robot::new(comau_nj165_opw()).unwrap();
        let singular = robot.jacobian(&[0.3, 0.2, -1.2, 0.4, 0.0, 0.1], &Vector3::zeros());
        assert!(sigma_min(&singular) < 1e-9);
        let regular = robot.jacobian(&[0.3, 0.2, -1.2, 0.4, 0.7, 0.1], &Vector3::zeros());
        assert!(sigma_min(&regular) > 1e-3);
    }
}
