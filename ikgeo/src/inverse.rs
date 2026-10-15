//! Cold inverse kinematics: every root of the family solver, polished with
//! Gauss-Newton, validated against forward kinematics, deduplicated and
//! ordered. Angles are internal radians unless stated otherwise.

use crate::inverse_kinematics as ik;
use crate::robot::{Robot, Solver};
use crate::subproblems::auxiliary::rot;
use nalgebra::{Isometry3, Matrix3, Matrix6, Vector3, Vector6};
use std::f64::consts::{PI, TAU};

/// Most roots `inverse_all` returns; the 1D search can in principle find more.
pub const MAX_ROOTS: usize = 16;
/// Relative position tolerance (times max(1, total link length)) for a valid root.
pub const POSITION_TOL: f64 = 1e-9;
/// Rotation tolerance for a valid root, 1e-9 degrees in radians.
pub const ROTATION_TOL: f64 = 1e-9 * PI / 180.0;
/// Two roots whose joints all agree within this (modulo 2 pi) are the same root.
pub const DEDUPE_TOL: f64 = 1e-6;
const POLISH_STEPS: usize = 40;

/// Pose residual of a root: translation norm and rotation angle.
#[derive(Debug, Clone, Copy)]
pub struct Residual {
    pub translation: f64,
    pub rotation: f64,
}

/// Smallest signed difference, mapped into (-pi, pi].
pub fn wrap_to_pi(theta: f64) -> f64 {
    let w = (theta + PI).rem_euclid(TAU) - PI;
    if w == -PI {
        PI
    } else {
        w
    }
}

/// rs-opw-kinematics `normalize_near`: the representative of `now` closest to `near`.
pub fn normalize_near(now: f64, near: f64) -> f64 {
    near + (now - near + PI).rem_euclid(TAU) - PI
}

/// Rotation vector of `r`, exact enough near the identity, where the polish needs it.
fn log_so3(r: &Matrix3<f64>) -> Vector3<f64> {
    let v = Vector3::new(
        r[(2, 1)] - r[(1, 2)],
        r[(0, 2)] - r[(2, 0)],
        r[(1, 0)] - r[(0, 1)],
    );
    let s = 0.5 * v.norm();
    let c = 0.5 * (r.trace() - 1.0);
    let angle = s.atan2(c);
    if s < 1e-12 {
        if c > 0.0 {
            // Near the identity the rotation vector is half the skew part.
            0.5 * v
        } else {
            // Half turn: take the axis from the symmetric part.
            let m = (r + Matrix3::identity()) * 0.5;
            let col = (0..3)
                .max_by(|&a, &b| m[(a, a)].partial_cmp(&m[(b, b)]).unwrap())
                .unwrap();
            let axis = m.column(col).normalize();
            axis * PI
        }
    } else {
        v * (0.5 * angle / s)
    }
}

/// A seed is usable only when every joint is a number.
pub fn finite_seed(seed: &[f64; 6]) -> bool {
    seed.iter().all(|v| v.is_finite())
}

impl Robot {
    /// Sum of the offset lengths, the scale for relative position tolerances.
    pub fn link_length(&self) -> f64 {
        (0..7)
            .map(|i| self.kin.p.column(i).norm())
            .sum::<f64>()
            .max(1.0)
    }

    /// All roots the family solver finds, unpolished, in internal joint space.
    pub fn solve_raw(&self, r: &Matrix3<f64>, t: &Vector3<f64>) -> Vec<Vector6<f64>> {
        let chain = &self.chain;
        let (r_s, t_s) = if chain.reversed {
            // The reversed chain's forward map is the inverse pose.
            (r.transpose(), -(r.transpose() * t))
        } else {
            (*r, *t)
        };
        let kin = &chain.kin;
        let (q, _is_ls) = match chain.family.solver() {
            Solver::Spherical => ik::spherical(&r_s, &t_s, kin),
            Solver::SphericalTwoIntersecting => ik::spherical_two_intersecting(&r_s, &t_s, kin),
            Solver::SphericalTwoParallel => ik::spherical_two_parallel(&r_s, &t_s, kin),
            Solver::ThreeParallel => ik::three_parallel(&r_s, &t_s, kin),
            Solver::ThreeParallelTwoIntersecting => {
                ik::three_parallel_two_intersecting(&r_s, &t_s, kin)
            }
            Solver::TwoIntersecting => ik::two_intersecting(&r_s, &t_s, kin),
            Solver::TwoIntersectingTwoParallel => {
                ik::two_intersecting_two_parallel(&r_s, &t_s, kin)
            }
            Solver::TwoParallel => ik::two_parallel(&r_s, &t_s, kin),
            Solver::GenSixDof => ik::gen_six_dof(&r_s, &t_s, kin),
        };
        q.iter()
            .filter(|q| q.iter().all(|v| v.is_finite()))
            .map(|q| chain.to_internal(q))
            .collect()
    }

    /// Geometric Jacobian of the flange in internal joint space, base frame, translation first.
    fn jacobian_internal(&self, q: &Vector6<f64>) -> (Matrix3<f64>, Vector3<f64>, Matrix6<f64>) {
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
        let mut jac = Matrix6::zeros();
        for j in 0..6 {
            let v = axes[j].cross(&(o - origins[j]));
            jac.fixed_view_mut::<3, 1>(0, j).copy_from(&v);
            jac.fixed_view_mut::<3, 1>(3, j).copy_from(&axes[j]);
        }
        (r, o, jac)
    }

    /// Pose residual of internal joints `q` against the target.
    pub fn residual(&self, q: &Vector6<f64>, r: &Matrix3<f64>, t: &Vector3<f64>) -> Residual {
        let (rq, tq) = self.kin.forward_kinematics(q);
        Residual {
            translation: (t - tq).norm(),
            rotation: log_so3(&(r * rq.transpose())).norm(),
        }
    }

    /// Gauss-Newton on the se(3) error, falling back to Levenberg-Marquardt damping when a
    /// step does not reduce the residual, so roots at singular configurations (double roots
    /// of the search) still converge. Returns the polished joints and their residual.
    pub fn polish(
        &self,
        q0: &Vector6<f64>,
        r: &Matrix3<f64>,
        t: &Vector3<f64>,
    ) -> (Vector6<f64>, Residual) {
        let scale = self.link_length();
        let noise_t = 1e-13 * scale;
        let noise_r = 1e-14;
        let cost = |res: &Residual| res.translation / scale + res.rotation;
        let mut q = *q0;
        let mut best = self.residual(&q, r, t);
        let mut lambda = 0.0;
        for _ in 0..POLISH_STEPS {
            if best.translation < noise_t && best.rotation < noise_r {
                break;
            }
            let (rq, tq, jac) = self.jacobian_internal(&q);
            let mut e = Vector6::zeros();
            e.fixed_view_mut::<3, 1>(0, 0).copy_from(&(t - tq));
            e.fixed_view_mut::<3, 1>(3, 0)
                .copy_from(&log_so3(&(r * rq.transpose())));
            let jt = jac.transpose();
            let jtj = jt * jac;
            let jte = jt * e;
            let mut improved = false;
            for _ in 0..8 {
                let step = if lambda == 0.0 {
                    jac.lu().solve(&e)
                } else {
                    (jtj + Matrix6::identity() * lambda * jtj.trace() / 6.0)
                        .cholesky()
                        .map(|ch| ch.solve(&jte))
                };
                let step = match step {
                    Some(s) if s.iter().all(|v| v.is_finite()) => s,
                    _ => {
                        lambda = if lambda == 0.0 { 1e-9 } else { lambda * 10.0 };
                        continue;
                    }
                };
                let candidate = q + step;
                let res = self.residual(&candidate, r, t);
                if cost(&res) < cost(&best) {
                    q = candidate;
                    best = res;
                    improved = true;
                    if lambda > 0.0 {
                        lambda *= 0.1;
                        if lambda < 1e-12 {
                            lambda = 0.0;
                        }
                    }
                    break;
                }
                // The step overshot: damp harder and try again from the same point.
                lambda = if lambda == 0.0 { 1e-9 } else { lambda * 10.0 };
                if lambda > 1e3 {
                    break;
                }
            }
            if !improved {
                break;
            }
        }
        (q, best)
    }

    /// Validated, deduplicated roots in internal joint space, wrapped to (-pi, pi], unordered.
    pub fn solve_cold(&self, r: &Matrix3<f64>, t: &Vector3<f64>) -> Vec<Vector6<f64>> {
        let scale = self.link_length();
        let mut roots: Vec<(Vector6<f64>, Residual)> = Vec::with_capacity(MAX_ROOTS);
        for raw in self.solve_raw(r, t) {
            let (q, res) = self.polish(&raw, r, t);
            if !(res.translation < POSITION_TOL * scale && res.rotation < ROTATION_TOL) {
                continue;
            }
            let q = q.map(wrap_to_pi);
            let same = roots.iter().position(|(other, _)| {
                (0..6).all(|j| wrap_to_pi(q[j] - other[j]).abs() < DEDUPE_TOL)
            });
            match same {
                Some(i) => {
                    if res.translation + res.rotation * scale
                        < roots[i].1.translation + roots[i].1.rotation * scale
                    {
                        roots[i] = (q, res);
                    }
                }
                None => roots.push((q, res)),
            }
        }
        roots.into_iter().map(|(q, _)| q).collect()
    }

    /// Every solution for a flange pose, in user radians. Each joint is normalised near
    /// the seed and the list is ordered by the summed absolute joint distance to it, so
    /// the first element is the root closest to the seed. Without a (finite) seed the
    /// zero joint vector (home) is the seed, which keeps the first root a natural
    /// posture instead of an arbitrary one.
    pub fn inverse_all(&self, pose: &Isometry3<f64>, seed: Option<&[f64; 6]>) -> Vec<[f64; 6]> {
        // A NaN seed would poison every distance; py-opw callers pass NaN rows for "unknown".
        let seed = seed.filter(|s| finite_seed(s)).copied().unwrap_or([0.0; 6]);
        let r = *pose.rotation.to_rotation_matrix().matrix();
        let t = pose.translation.vector;
        let mut sols: Vec<[f64; 6]> = self
            .solve_cold(&r, &t)
            .iter()
            .map(|q| self.model.to_user(q).map(wrap_to_pi))
            .collect();
        for s in sols.iter_mut() {
            for j in 0..6 {
                s[j] = normalize_near(s[j], seed[j]);
            }
        }
        let dist =
            |s: &[f64; 6]| -> f64 { s.iter().zip(seed.iter()).map(|(a, b)| (a - b).abs()).sum() };
        // Ties (symmetric roots) break on the joint vector so the order is reproducible.
        sols.sort_by(|a, b| {
            dist(a).total_cmp(&dist(b)).then_with(|| {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| x.total_cmp(y))
                    .find(|o| o.is_ne())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        sols.truncate(MAX_ROOTS);
        sols
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse_kinematics::auxiliary::Kinematics;
    use crate::inverse_kinematics::hardcoded::setups::{
        Irb6640, KukaR800FixedQ3, RrcFixedQ6, SphericalBot, ThreeParallelBot, TwoParallelBot, Ur5,
    };
    use crate::robot::fixtures::{comau_nj165_opw, comau_nj4_220};
    use crate::robot::KinematicModel;
    use crate::subproblems::auxiliary::{random_angle, seed_random};
    use std::time::Instant;

    const N: usize = 1000;

    fn robot_from_kin(kin: Kinematics<6, 7>) -> Robot {
        Robot::new(KinematicModel::new(kin.h, kin.p, [0.0; 6], [false; 6])).unwrap()
    }

    fn fixtures() -> Vec<(&'static str, Robot, f64)> {
        vec![
            ("IRB6640", robot_from_kin(Irb6640::get_kin()), 1.0),
            ("UR5", robot_from_kin(Ur5::get_kin()), 1.0),
            (
                "KUKA R800 fixed q3",
                robot_from_kin(KukaR800FixedQ3::get_kin_partial().0),
                1.0,
            ),
            (
                "RRC fixed q6",
                robot_from_kin(RrcFixedQ6::get_kin_partial().0),
                0.95,
            ),
            (
                "three_parallel_bot",
                robot_from_kin(ThreeParallelBot::get_kin()),
                1.0,
            ),
            (
                "two_parallel_bot",
                robot_from_kin(TwoParallelBot::get_kin()),
                0.95,
            ),
            (
                "spherical_bot",
                robot_from_kin(SphericalBot::get_kin()),
                0.99,
            ),
            ("NJ4-220", Robot::new(comau_nj4_220()).unwrap(), 0.99),
            ("NJ165", Robot::new(comau_nj165_opw()).unwrap(), 1.0),
        ]
    }

    fn contains(sols: &[[f64; 6]], q: &[f64; 6]) -> bool {
        sols.iter()
            .any(|s| (0..6).all(|j| wrap_to_pi(s[j] - q[j]).abs() < 1e-9))
    }

    #[test]
    fn roots_round_trip_and_cover_generating_joints() {
        seed_random(2024);
        for (name, robot, min_hit) in fixtures() {
            let scale = robot.link_length();
            let mut hits = 0;
            let mut empty = 0;
            let mut max_count = 0;
            let mut worst = Residual {
                translation: 0.0,
                rotation: 0.0,
            };
            for _ in 0..N {
                let joints: [f64; 6] = std::array::from_fn(|_| random_angle());
                let pose = robot.forward(&joints);
                let sols = robot.inverse_all(&pose, None);
                max_count = max_count.max(sols.len());
                if sols.is_empty() {
                    empty += 1;
                }
                if contains(&sols, &joints) {
                    hits += 1;
                }
                let r = *pose.rotation.to_rotation_matrix().matrix();
                let t = pose.translation.vector;
                for s in &sols {
                    let res = robot.residual(&robot.model.to_internal(s), &r, &t);
                    assert!(
                        res.translation < POSITION_TOL * scale && res.rotation < ROTATION_TOL,
                        "{name}: root does not round-trip: {res:?}"
                    );
                    worst.translation = worst.translation.max(res.translation);
                    worst.rotation = worst.rotation.max(res.rotation);
                }
            }
            let hit_rate = hits as f64 / N as f64;
            println!(
                "{name}: hit {hit_rate:.3}, empty {empty}, max roots {max_count}, worst residual {:.2e} length / {:.2e} rad",
                worst.translation, worst.rotation
            );
            assert!(hit_rate >= min_hit, "{name}: hit rate {hit_rate}");
            assert!(
                max_count <= MAX_ROOTS,
                "{name}: more than {MAX_ROOTS} roots"
            );
            if name == "NJ4-220" || name == "NJ165" {
                assert_eq!(empty, 0, "{name}: {empty} poses without any root");
            }
        }
    }

    #[test]
    fn seed_orders_and_unwraps() {
        seed_random(99);
        let robot = Robot::new(comau_nj165_opw()).unwrap();
        for _ in 0..50 {
            let joints: [f64; 6] = std::array::from_fn(|_| random_angle());
            let pose = robot.forward(&joints);
            let mut seed = joints;
            seed[0] += 4.0 * PI + 0.01;
            seed[5] -= 2.0 * PI - 0.02;
            let sols = robot.inverse_all(&pose, Some(&seed));
            let first = sols[0];
            for j in 0..6 {
                assert!(
                    (first[j] - seed[j]).abs() < 0.1,
                    "joint {j}: {first:?} vs {seed:?}"
                );
            }
            let dist = |s: &[f64; 6]| -> f64 {
                s.iter().zip(seed.iter()).map(|(a, b)| (a - b).abs()).sum()
            };
            for w in sols.windows(2) {
                assert!(dist(&w[0]) <= dist(&w[1]) + 1e-12);
            }
            let unseeded = robot.inverse_all(&pose, None);
            assert_eq!(unseeded.len(), sols.len());
            assert_eq!(unseeded, robot.inverse_all(&pose, Some(&[0.0; 6])));
            let home_dist = |s: &[f64; 6]| -> f64 { s.iter().map(|v| v.abs()).sum() };
            for w in unseeded.windows(2) {
                assert!(home_dist(&w[0]) <= home_dist(&w[1]) + 1e-12);
            }
        }
    }

    #[test]
    fn non_finite_seed_is_ignored() {
        let robot = Robot::new(comau_nj4_220()).unwrap();
        let pose = robot.forward(&[0.3, -0.4, 0.5, 1.0, -1.2, 0.7]);
        let unseeded = robot.inverse_all(&pose, None);
        assert!(!unseeded.is_empty());
        assert_eq!(robot.inverse_all(&pose, Some(&[f64::NAN; 6])), unseeded);
        let one_nan = [0.3, f64::NAN, 0.5, 1.0, -1.2, 0.7];
        assert_eq!(robot.inverse_all(&pose, Some(&one_nan)), unseeded);
    }

    #[test]
    fn two_parallel_search_is_scale_free() {
        let mut rates = Vec::new();
        for scale in [1.0, 1000.0] {
            seed_random(77);
            let mut kin = TwoParallelBot::get_kin();
            kin.p *= scale;
            let robot = robot_from_kin(kin);
            let hits = (0..500)
                .filter(|_| {
                    let joints: [f64; 6] = std::array::from_fn(|_| random_angle());
                    contains(&robot.inverse_all(&robot.forward(&joints), None), &joints)
                })
                .count();
            rates.push(hits as f64 / 500.0);
        }
        println!("two_parallel_bot hit rate at scale 1 and 1000: {rates:?}");
        assert!(rates.iter().all(|&r| r >= 0.95), "{rates:?}");
        assert!((rates[0] - rates[1]).abs() <= 0.01, "{rates:?}");
    }

    #[test]
    fn reversed_chain_solves() {
        seed_random(5);
        let kin = crate::robot::reversed_kinematics(&Irb6640::get_kin());
        let robot = robot_from_kin(kin);
        assert!(robot.chain.reversed);
        for _ in 0..200 {
            let joints: [f64; 6] = std::array::from_fn(|_| random_angle());
            let pose = robot.forward(&joints);
            let sols = robot.inverse_all(&pose, None);
            assert!(contains(&sols, &joints), "{joints:?} not in {sols:?}");
        }
    }

    #[test]
    #[ignore]
    fn timing_cold_inverse() {
        seed_random(7);
        for (name, robot, _) in fixtures() {
            let poses: Vec<Isometry3<f64>> = (0..N)
                .map(|_| robot.forward(&std::array::from_fn(|_| random_angle())))
                .collect();
            let start = Instant::now();
            let mut total = 0;
            for pose in &poses {
                total += robot.inverse_all(pose, None).len();
            }
            let us = start.elapsed().as_secs_f64() * 1e6 / N as f64;
            println!(
                "{name}: {us:.1} us per cold inverse, {:.2} roots on average",
                total as f64 / N as f64
            );
        }
    }
}
