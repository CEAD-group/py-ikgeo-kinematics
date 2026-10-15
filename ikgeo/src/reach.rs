//! Reach kernel: every inverse-kinematics branch of a pose with joint-limit,
//! singularity and wrist-bend margins. Angles are radians in the user frame
//! unless stated otherwise; a NaN joint row means the column is empty.

use crate::inverse::{wrap_to_pi, MAX_ROOTS};
use crate::inverse_kinematics::auxiliary::Kinematics;
use crate::robot::{axis_pairs, canonicalise, sigma_min, Robot};
use crate::subproblems::auxiliary::rot;
use nalgebra::{Isometry3, Matrix3, Vector3, Vector6};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::f64::consts::PI;

/// Per-pose reach output. Columns are ordered by `BranchKey`; the order is
/// deterministic per pose but a column is not the same physical branch from
/// one pose to the next, because a non-spherical wrist is cuspidal.
#[derive(Debug, Clone)]
pub struct PoseReach {
    pub joints: [[f64; 6]; MAX_ROOTS],
    pub limit_margin: [f64; MAX_ROOTS],
    pub extension: f64,
    pub sigma_min: [f64; MAX_ROOTS],
    pub wrist: [f64; MAX_ROOTS],
    pub n_solutions: usize,
}

impl PoseReach {
    pub fn nan() -> Self {
        PoseReach {
            joints: [[f64::NAN; 6]; MAX_ROOTS],
            limit_margin: [f64::NAN; MAX_ROOTS],
            extension: f64::NAN,
            sigma_min: [f64::NAN; MAX_ROOTS],
            wrist: [f64::NAN; MAX_ROOTS],
            n_solutions: 0,
        }
    }
}

/// Sort key of a branch: shoulder side, elbow side, sign of J5, then J4 ascending.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BranchKey {
    pub shoulder: u8,
    pub elbow: u8,
    pub wrist_flip: u8,
    pub q4: f64,
}

/// Geometry the margins need, derived once per robot from the forward (never
/// reversed) canonical chain. W is that chain's joint-4 point: the 4/5
/// intersection when axes 4 and 5 meet, the wrist centre for a spherical
/// wrist, and in every case a point of axis 4 that is fixed in link 3.
///
/// `front` is the unit direction, perpendicular to axis 1, in which a positive
/// internal J2 swings W away from the zero pose. Shoulder 0 has W on that side
/// of axis 1 in the J1 frame; for every OPW model this is py-opw's slot rule.
#[derive(Debug, Clone)]
pub struct ReachGeometry {
    kin: Kinematics<6, 7>,
    axis_sign: [f64; 6],
    parallel_23: bool,
    front: Vector3<f64>,
    beta1: f64,
    beta2: f64,
    phi_min: f64,
    phi_max: f64,
}

fn perp(v: &Vector3<f64>, axis: &Vector3<f64>) -> Vector3<f64> {
    v - axis * v.dot(axis)
}

fn angle_between(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    a.cross(b).norm().atan2(a.dot(b))
}

impl ReachGeometry {
    pub fn new(robot: &Robot) -> Self {
        let pairs = axis_pairs(&robot.kin, robot.axis_tolerance);
        let (kin, axis_sign) = canonicalise(&robot.kin, &pairs, robot.axis_tolerance);
        let h1: Vector3<f64> = kin.h.column(0).into();
        let h2: Vector3<f64> = kin.h.column(1).into();
        let o1: Vector3<f64> = kin.p.column(0).into();
        let o2 = o1 + kin.p.column(1);
        let w0 = o2 + kin.p.column(2) + kin.p.column(3);
        let tiny = 1e-9 * robot.link_length();
        // Canonicalisation may have flipped h2; the swing direction belongs to the model's J2.
        let h2_model = h2 * axis_sign[1];
        let swing = perp(&h2_model.cross(&(w0 - o2)), &h1);
        // Fallbacks keep the key defined when the zero pose has W on axis 2 or on axis 1.
        let arm_plane = perp(&h2, &h1);
        let lateral = if arm_plane.norm() > tiny {
            arm_plane.normalize().cross(&h1)
        } else {
            Vector3::zeros()
        };
        let radial = perp(&(w0 - o1), &h1);
        let front = if swing.norm() > tiny {
            swing.normalize()
        } else if lateral.norm() > tiny && radial.dot(&lateral) < 0.0 {
            -lateral
        } else if lateral.norm() > tiny {
            lateral
        } else {
            Vector3::x()
        };
        let h4: Vector3<f64> = robot.kin.h.column(3).into();
        let h5: Vector3<f64> = robot.kin.h.column(4).into();
        let h6: Vector3<f64> = robot.kin.h.column(5).into();
        let beta1 = angle_between(&h4, &h5);
        let beta2 = angle_between(&h5, &h6);
        ReachGeometry {
            kin,
            axis_sign,
            parallel_23: pairs.parallel[1],
            front,
            beta1,
            beta2,
            phi_min: (beta1 - beta2).abs(),
            // Axes are lines, so when beta1 + beta2 > pi the far fold is the reflex side.
            phi_max: (beta1 + beta2).min(2.0 * PI - beta1 - beta2),
        }
    }

    /// Angles between axes 4 and 5, and between 5 and 6.
    pub fn beta(&self) -> (f64, f64) {
        (self.beta1, self.beta2)
    }

    /// Joint points o1, o2, o3 and W (the canonical joint-4 point) with R1, for
    /// internal joints `q`; the canonical chain may have flipped parallel axes.
    fn arm_points(&self, q: &Vector6<f64>) -> ([Vector3<f64>; 4], Matrix3<f64>) {
        let k = &self.kin;
        let h1: Vector3<f64> = k.h.column(0).into();
        let h2: Vector3<f64> = k.h.column(1).into();
        let h3: Vector3<f64> = k.h.column(2).into();
        let s = &self.axis_sign;
        let r1 = rot(&h1, s[0] * q[0]);
        let r12 = r1 * rot(&h2, s[1] * q[1]);
        let r123 = r12 * rot(&h3, s[2] * q[2]);
        let o1: Vector3<f64> = k.p.column(0).into();
        let o2 = o1 + r1 * k.p.column(1);
        let o3 = o2 + r12 * k.p.column(2);
        let w = o3 + r123 * k.p.column(3);
        ([o1, o2, o3, w], r1)
    }

    /// W (the canonical joint-4 point) for internal joints `q`.
    pub fn w_point(&self, q: &Vector6<f64>) -> Vector3<f64> {
        self.arm_points(q).0[3]
    }

    /// W from the flange pose alone, used when a pose has no root. Exact for a
    /// spherical wrist, where W is fixed in the flange; otherwise the zero-wrist
    /// estimate, which keeps `extension` finite and continuous.
    pub fn w_from_pose(&self, r: &Matrix3<f64>, t: &Vector3<f64>) -> Vector3<f64> {
        let k = &self.kin;
        let tail = k.p.column(4) + k.p.column(5) + k.p.column(6);
        t - r * tail
    }

    /// Branch key of internal joints `q` (model frame, radians).
    pub fn key(&self, q: &Vector6<f64>) -> BranchKey {
        let h2: Vector3<f64> = self.kin.h.column(1).into();
        let ([o1, o2, o3, w], r1) = self.arm_points(q);
        let shoulder = if self.front.dot(&(r1.transpose() * (w - o1))) >= 0.0 {
            0
        } else {
            1
        };
        let elbow = if self.parallel_23 {
            // Sign of the bend between upper arm and forearm about the parallel axes.
            let bend = (r1 * h2).dot(&(o3 - o2).cross(&(w - o3)));
            if bend >= 0.0 {
                0
            } else {
                1
            }
        } else if q[2] >= 0.0 {
            0
        } else {
            1
        };
        BranchKey {
            shoulder,
            elbow,
            wrist_flip: if q[4] >= 0.0 { 0 } else { 1 },
            q4: q[3],
        }
    }

    /// Wrist bend margin in radians: distance of the angle between axes 4 and 6
    /// from the nearest fold of the wrist. Zero at a wrist singularity.
    pub fn wrist(&self, robot: &Robot, q5: f64) -> f64 {
        let h4: Vector3<f64> = robot.kin.h.column(3).into();
        let h5: Vector3<f64> = robot.kin.h.column(4).into();
        let h6: Vector3<f64> = robot.kin.h.column(5).into();
        let h6_now = rot(&h5, q5) * h6;
        let phi = angle_between(&h4, &h6_now);
        (phi - self.phi_min).min(self.phi_max - phi)
    }

    /// Signed distance of one W to the annulus the shoulder can reach, best over
    /// both shoulder sides, py-opw semantics. NaN unless axes 2 and 3 are parallel.
    /// `reach_pose` takes the maximum over the roots of a pose.
    pub fn extension(&self, w: &Vector3<f64>) -> f64 {
        if !self.parallel_23 {
            return f64::NAN;
        }
        let k = &self.kin;
        let h1: Vector3<f64> = k.h.column(0).into();
        let h2: Vector3<f64> = k.h.column(1).into();
        let p1: Vector3<f64> = k.p.column(1).into();
        let p2: Vector3<f64> = k.p.column(2).into();
        let p3: Vector3<f64> = k.p.column(3).into();
        let o1: Vector3<f64> = k.p.column(0).into();
        let p_1w = w - o1;
        // After J1, h2 . W' must equal the arm plane's fixed offset along h2.
        let d = h2.dot(&(p1 + p2 + p3));
        let c = h1.dot(&h2);
        let n = perp(&h2, &h1);
        let s = n.norm();
        if s < 1e-12 {
            return f64::NAN;
        }
        let n = n / s;
        let w_par = h1.dot(&p_1w);
        let rho = perp(&p_1w, &h1).norm();
        let b_eff = (d - c * w_par) / s;
        if rho < b_eff.abs() {
            // Inside the lateral-offset cylinder no branch exists; report the distance to it.
            return rho - b_eff.abs();
        }
        let l2 = perp(&p2, &h2).norm();
        let l3 = perp(&p3, &h2).norm();
        let annulus = |dist: f64| (l2 + l3 - dist).min(dist - (l2 - l3).abs());
        // The two q1 solutions put W' at lateral offset b_eff on either side of axis 1.
        let m = h1.cross(&n);
        let along = (rho * rho - b_eff * b_eff).max(0.0).sqrt();
        let mut best = f64::NEG_INFINITY;
        for sign in [1.0, -1.0] {
            let w_frame = h1 * w_par + n * b_eff + m * (sign * along);
            let in_plane = perp(&(w_frame - p1), &h2).norm();
            best = best.max(annulus(in_plane));
        }
        best
    }
}

fn key_order(a: &(BranchKey, Vector6<f64>), b: &(BranchKey, Vector6<f64>)) -> Ordering {
    a.0.shoulder
        .cmp(&b.0.shoulder)
        .then(a.0.elbow.cmp(&b.0.elbow))
        .then(a.0.wrist_flip.cmp(&b.0.wrist_flip))
        .then(a.0.q4.partial_cmp(&b.0.q4).unwrap_or(Ordering::Equal))
        .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
}

impl Robot {
    pub fn reach_geometry(&self) -> ReachGeometry {
        ReachGeometry::new(self)
    }

    /// Every branch of a flange pose with margins. `limits` are user radians;
    /// `tcp_in_flange` places the Jacobian at the requested TCP.
    pub fn reach_pose(
        &self,
        pose_flange: &Isometry3<f64>,
        limits: Option<&[(f64, f64); 6]>,
        tcp_in_flange: &Vector3<f64>,
    ) -> PoseReach {
        self.reach_pose_with(&self.reach_geometry(), pose_flange, limits, tcp_in_flange)
    }

    pub fn reach_pose_with(
        &self,
        geo: &ReachGeometry,
        pose_flange: &Isometry3<f64>,
        limits: Option<&[(f64, f64); 6]>,
        tcp_in_flange: &Vector3<f64>,
    ) -> PoseReach {
        let r: Matrix3<f64> = *pose_flange.rotation.to_rotation_matrix().matrix();
        let t = pose_flange.translation.vector;
        let mut roots: Vec<(BranchKey, Vector6<f64>)> = self
            .solve_cold(&r, &t)
            .into_iter()
            .map(|q| (geo.key(&q), q))
            .collect();
        roots.sort_by(key_order);
        roots.truncate(MAX_ROOTS);

        let mut out = PoseReach::nan();
        out.n_solutions = roots.len();
        // W differs per root on a non-spherical wrist, so take the best branch's margin.
        out.extension = if roots.is_empty() {
            geo.extension(&geo.w_from_pose(&r, &t))
        } else {
            roots
                .iter()
                .map(|(_, q)| geo.extension(&geo.w_point(q)))
                .fold(f64::NEG_INFINITY, f64::max)
        };
        for (s, (_, q)) in roots.iter().enumerate() {
            let user = self.model.to_user(q).map(wrap_to_pi);
            out.joints[s] = user;
            out.wrist[s] = geo.wrist(self, q[4]);
            out.limit_margin[s] = match limits {
                Some(lim) => (0..6)
                    .map(|j| (user[j] - lim[j].0).min(lim[j].1 - user[j]))
                    .fold(f64::INFINITY, f64::min),
                None => f64::INFINITY,
            };
            // An out-of-limit branch is unusable downstream, so its eigen solve is skipped.
            if out.limit_margin[s] < 0.0 {
                continue;
            }
            out.sigma_min[s] = sigma_min(&self.jacobian(&user, tcp_in_flange));
        }
        out
    }

    /// Reach for many flange poses with a rayon pool per call: `threads` 0 is one
    /// per core, 1 runs in the calling thread.
    pub fn reach(
        &self,
        poses: &[Isometry3<f64>],
        limits: Option<&[(f64, f64); 6]>,
        tcp_in_flange: &Vector3<f64>,
        threads: usize,
    ) -> Vec<PoseReach> {
        let geo = self.reach_geometry();
        let one = |pose: &Isometry3<f64>| self.reach_pose_with(&geo, pose, limits, tcp_in_flange);
        match threads {
            1 => poses.iter().map(one).collect(),
            0 => poses.par_iter().map(one).collect(),
            n => rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map(|pool| pool.install(|| poses.par_iter().map(one).collect()))
                .unwrap_or_else(|_| poses.iter().map(one).collect()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::fixtures::{comau_nj165_opw, comau_nj4_220};
    use crate::robot::KinematicModel;
    use crate::subproblems::auxiliary::{random_angle, seed_random};
    use std::time::Instant;

    #[allow(clippy::too_many_arguments)]
    fn opw_extension(a1: f64, a2: f64, b: f64, c1: f64, c2: f64, c3: f64, c: &Vector3<f64>) -> f64 {
        let nx1 = ((c.x * c.x + c.y * c.y) - b * b).max(0.0).sqrt() - a1;
        let tmp3 = c.z - c1;
        let s1 = (nx1 * nx1 + tmp3 * tmp3).sqrt();
        let tmp4 = nx1 + 2.0 * a1;
        let s2 = (tmp4 * tmp4 + tmp3 * tmp3).sqrt();
        let kappa = (a2 * a2 + c3 * c3).sqrt();
        let annulus = |d: f64| (c2 + kappa - d).min(d - (c2 - kappa).abs());
        let radius_xy = (c.x * c.x + c.y * c.y).sqrt();
        if radius_xy < b.abs() {
            radius_xy - b.abs()
        } else {
            annulus(s1).max(annulus(s2))
        }
    }

    #[test]
    fn extension_matches_opw_formula() {
        seed_random(31);
        let (a1, a2, b, c1, c2, c3, c4) = (400.0, -250.0, 120.0, 830.0, 1175.0, 1444.0, 230.0);
        let model = KinematicModel::from_opw(a1, a2, b, c1, c2, c3, c4, [0.0; 6], [false; 6]);
        let robot = Robot::new(model).unwrap();
        let geo = robot.reach_geometry();
        for _ in 0..500 {
            // Wrist centres in a box: inside the b cylinder, in reach and out of reach.
            let c = Vector3::new(
                random_angle() * 1200.0,
                random_angle() * 1200.0,
                random_angle() * 2000.0 + 1000.0,
            );
            let expected = opw_extension(a1, a2, b, c1, c2, c3, &c);
            let got = geo.extension(&c);
            assert!((got - expected).abs() < 1e-9, "{c}: {got} vs {expected}");
        }
    }

    #[test]
    fn column_order_is_deterministic_and_sorted_by_key() {
        seed_random(17);
        let robot = Robot::new(comau_nj4_220()).unwrap();
        let geo = robot.reach_geometry();
        for _ in 0..50 {
            let joints: [f64; 6] = std::array::from_fn(|_| random_angle());
            let pose = robot.forward(&joints);
            let a = robot.reach_pose(&pose, None, &Vector3::zeros());
            let b = robot.reach_pose(&pose, None, &Vector3::zeros());
            assert_eq!(a.n_solutions, b.n_solutions);
            assert!(a.n_solutions >= 1);
            for s in 0..a.n_solutions {
                assert_eq!(a.joints[s], b.joints[s]);
                assert!(a.wrist[s] >= -1e-12 && a.sigma_min[s] >= 0.0);
                assert!(a.limit_margin[s].is_infinite());
            }
            for s in a.n_solutions..MAX_ROOTS {
                assert!(a.joints[s][0].is_nan() && a.sigma_min[s].is_nan());
            }
            let keys: Vec<BranchKey> = a.joints[..a.n_solutions]
                .iter()
                .map(|j| geo.key(&robot.model.to_internal(j)))
                .collect();
            for w in keys.windows(2) {
                let lhs = (w[0].shoulder, w[0].elbow, w[0].wrist_flip);
                let rhs = (w[1].shoulder, w[1].elbow, w[1].wrist_flip);
                assert!(lhs <= rhs);
                if lhs == rhs {
                    assert!(w[0].q4 <= w[1].q4);
                }
            }
            assert!(a.extension.is_finite() && a.extension > 0.0);
        }
    }

    #[test]
    fn nj4_wrist_margin_is_zero_at_both_folds() {
        let robot = Robot::new(comau_nj4_220()).unwrap();
        let geo = robot.reach_geometry();
        let (b1, b2) = geo.beta();
        assert!((b1 - PI / 3.0).abs() < 1e-12 && (b2 - PI / 3.0).abs() < 1e-12);
        assert!(geo.wrist(&robot, 0.0).abs() < 1e-12);
        assert!(geo.wrist(&robot, PI).abs() < 1e-12);
        assert!(geo.wrist(&robot, -PI).abs() < 1e-12);
        assert!(geo.wrist(&robot, PI / 2.0) > 0.3);
        let opw = Robot::new(comau_nj165_opw()).unwrap();
        let g = opw.reach_geometry();
        for q5 in [-2.0, -0.4, 0.0, 0.7, std::f64::consts::FRAC_PI_2, 3.0] {
            let w = g.wrist(&opw, q5);
            assert!((w.sin() - q5.sin().abs()).abs() < 1e-12, "{q5}: {w}");
        }
    }

    #[test]
    fn wrist_margin_is_invariant_to_axis_direction() {
        // Negating h5 and q5 is the same physical motion with beta1 + beta2 > pi.
        let robot = Robot::new(comau_nj4_220()).unwrap();
        let mut flipped = comau_nj4_220();
        let h5 = -flipped.h.column(4).clone_owned();
        flipped.h.set_column(4, &h5);
        let flipped = Robot::new(flipped).unwrap();
        let (geo, geo_f) = (robot.reach_geometry(), flipped.reach_geometry());
        let (b1, b2) = geo_f.beta();
        assert!(b1 + b2 > PI);
        for q5 in [0.0, 0.3, 1.0, PI / 2.0, 2.5, PI] {
            let (w, wf) = (geo.wrist(&robot, q5), geo_f.wrist(&flipped, -q5));
            assert!((w - wf).abs() < 1e-12, "{q5}: {w} vs {wf}");
        }
        assert!(geo_f.wrist(&flipped, PI).abs() < 1e-12);
    }

    #[test]
    fn opw_branch_keys_follow_py_opw_slots() {
        seed_random(23);
        // (a1, a2, b): the NJ165, a short-link arm with a1 + a2 < 0, no offsets, and a b offset.
        let cases = [
            (400.333, -251.449, 0.0),
            (25.0, -35.0, 0.0),
            (0.0, 0.0, 0.0),
            (400.333, -251.449, 120.0),
        ];
        for (a1, a2, b) in cases {
            let model = KinematicModel::from_opw(
                a1, a2, b, 830.0, 1177.556, 1443.593, 230.0, [0.0; 6], [false; 6],
            );
            let robot = Robot::new(model).unwrap();
            let geo = robot.reach_geometry();
            let psi3 = a2.atan2(1443.593);
            let mut mismatches = 0;
            for _ in 0..2000 {
                let q = Vector6::from_fn(|_, _| random_angle());
                let key = geo.key(&q);
                // py-opw: elbow 0 has sin(theta3 + psi3) >= 0, wrist_flip 0 has theta5 >= 0.
                let elbow = if (q[2] + psi3).sin() >= 0.0 { 0 } else { 1 };
                assert_eq!(key.elbow, elbow, "{q}");
                assert_eq!(key.wrist_flip, if q[4] >= 0.0 { 0 } else { 1 });
                // py-opw shoulder 0 has the wrist centre at x >= 0 in the J1 frame.
                let frames = robot.joint_frames(&robot.model.to_user(&q));
                let local = frames[0].rotation.inverse() * frames[4].translation.vector;
                if key.shoulder != u8::from(local.x < 0.0) {
                    mismatches += 1;
                }
            }
            assert_eq!(
                mismatches, 0,
                "a1 {a1} a2 {a2} b {b}: {mismatches} shoulder mismatches"
            );
        }
    }

    #[test]
    fn nj4_extension_is_the_best_root_and_home_is_shoulder_zero() {
        seed_random(29);
        let robot = Robot::new(comau_nj4_220()).unwrap();
        let geo = robot.reach_geometry();
        assert_eq!(geo.key(&Vector6::zeros()).shoulder, 0);
        let mut spread_seen = false;
        for _ in 0..200 {
            let joints: [f64; 6] = std::array::from_fn(|_| random_angle());
            let out = robot.reach_pose(&robot.forward(&joints), None, &Vector3::zeros());
            let per_root: Vec<f64> = out.joints[..out.n_solutions]
                .iter()
                .map(|j| geo.extension(&geo.w_point(&robot.model.to_internal(j))))
                .collect();
            let best = per_root.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert!((out.extension - best).abs() < 1e-9);
            let worst = per_root.iter().cloned().fold(f64::INFINITY, f64::min);
            // W moves with the root on this wrist, so the per-root values must differ somewhere.
            if best - worst > 1.0 {
                spread_seen = true;
            }
        }
        assert!(spread_seen);
    }

    #[test]
    #[ignore]
    fn timing_reach() {
        seed_random(5);
        for (name, robot) in [
            ("NJ4-220", Robot::new(comau_nj4_220()).unwrap()),
            ("NJ165", Robot::new(comau_nj165_opw()).unwrap()),
        ] {
            let n = 1000;
            let poses: Vec<Isometry3<f64>> = (0..n)
                .map(|_| robot.forward(&std::array::from_fn(|_| random_angle())))
                .collect();
            let limits = [(-PI, PI); 6];
            let start = Instant::now();
            let out = robot.reach(&poses, Some(&limits), &Vector3::zeros(), 1);
            let us = start.elapsed().as_secs_f64() * 1e6 / n as f64;
            let branches: usize = out.iter().map(|p| p.n_solutions).sum();
            println!(
                "{name}: {us:.1} us per pose single thread, {:.2} branches on average",
                branches as f64 / n as f64
            );
        }
    }
}
