//! Tracked inverse kinematics: each pose is warm-started from the previous
//! solution with Gauss-Newton and falls back to a cold solve when that fails.
//! Angles are internal radians unless stated otherwise.

use crate::inverse::{finite_seed, normalize_near, POSITION_TOL, ROTATION_TOL};
use crate::robot::Robot;
use nalgebra::{Isometry3, Matrix3, Vector3, Vector6};

/// How often `batch_inverse` had to leave the tracked path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BatchStats {
    /// Poses solved cold because tracking from the previous joints failed.
    pub fallbacks: usize,
    /// Poses with no solution at all (NaN rows), NaN input rows included.
    pub failures: usize,
}

impl Robot {
    /// Newton from `prev` to the pose (r, t). `Some` only when the result meets the cold
    /// solve's residual tolerances and no joint moved more than `max_track_step`, so a root
    /// that drifted onto another branch, or a stall at a singular fold, is reported as a
    /// failure instead of a solution.
    pub fn track_step(
        &self,
        prev: &Vector6<f64>,
        r: &Matrix3<f64>,
        t: &Vector3<f64>,
    ) -> Option<Vector6<f64>> {
        let (q, res) = self.polish(prev, r, t);
        let scale = self.link_length();
        if !(res.translation < POSITION_TOL * scale && res.rotation < ROTATION_TOL) {
            return None;
        }
        if (0..6).any(|j| !q[j].is_finite() || (q[j] - prev[j]).abs() > self.max_track_step) {
            return None;
        }
        Some(q)
    }

    /// Joint path for a sequence of flange poses, in user radians, with py-opw's
    /// `batch_inverse` contract: every row continues the previous one (joints unwrapped near
    /// it, never re-wrapped), a `None` pose or an unsolvable pose gives a NaN row and leaves
    /// the chain seed untouched. Without a seed the first pose takes the first cold root.
    /// Tracking follows the previous row's branch by Newton continuation; only the fallback
    /// picks the nearest root, so near a fold the two can be different valid roots.
    pub fn batch_inverse(
        &self,
        poses: &[Option<Isometry3<f64>>],
        seed: Option<&[f64; 6]>,
    ) -> (Vec<[f64; 6]>, BatchStats) {
        let mut out = Vec::with_capacity(poses.len());
        let mut stats = BatchStats::default();
        let mut prev: Option<[f64; 6]> = seed.copied().filter(finite_seed);
        for pose in poses {
            let Some(pose) = pose else {
                out.push([f64::NAN; 6]);
                stats.failures += 1;
                continue;
            };
            let r = *pose.rotation.to_rotation_matrix().matrix();
            let t = pose.translation.vector;
            let tracked = prev.and_then(|p| {
                self.track_step(&self.model.to_internal(&p), &r, &t)
                    .map(|q| self.continue_from(&q, &p))
            });
            let joints = match tracked {
                Some(j) => Some(j),
                None => {
                    if prev.is_some() {
                        stats.fallbacks += 1;
                    }
                    self.inverse_all(pose, prev.as_ref()).first().copied()
                }
            };
            match joints {
                Some(j) => {
                    out.push(j);
                    prev = Some(j);
                }
                None => {
                    out.push([f64::NAN; 6]);
                    stats.failures += 1;
                }
            }
        }
        (out, stats)
    }

    /// User-space joints for internal `q`, each unwrapped to the turn nearest `prev`.
    fn continue_from(&self, q: &Vector6<f64>, prev: &[f64; 6]) -> [f64; 6] {
        let user = self.model.to_user(q);
        std::array::from_fn(|j| normalize_near(user[j], prev[j]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::fixtures::{comau_nj165_opw, comau_nj4_220};
    use crate::subproblems::auxiliary::{random_angle, seed_random};
    use nalgebra::{Rotation3, Translation3, UnitQuaternion};
    use std::f64::consts::FRAC_PI_2;
    use std::time::Instant;

    const RETRACTS: usize = 5;

    /// Layered serpentine on the NJ4 with the tool pointing down: 50 layers of two 1 m
    /// lines at 1 mm pitch, plus `RETRACTS` 300 mm up-and-back jumps, 100k poses in all.
    /// Returns the poses and the indices of the retract moves.
    fn printing_path() -> (Vec<Option<Isometry3<f64>>>, Vec<usize>) {
        let down = UnitQuaternion::from_rotation_matrix(&Rotation3::from_axis_angle(
            &Vector3::y_axis(),
            FRAC_PI_2,
        ));
        let at =
            |x: f64, y: f64, z: f64| Some(Isometry3::from_parts(Translation3::new(x, y, z), down));
        let mut poses = Vec::with_capacity(100_010);
        let mut retracts = Vec::new();
        for layer in 0..50 {
            let z = 600.0 + 2.0 * layer as f64;
            let y0 = -200.0 + 8.0 * layer as f64;
            for i in 0..1000 {
                poses.push(at(1300.0 + i as f64, y0, z));
            }
            for i in 0..1000 {
                poses.push(at(2299.0 - i as f64, y0 + 1.0, z));
            }
            if layer % 10 == 5 {
                let back = poses.len() - 1;
                let p = poses[back].unwrap();
                let v = p.translation.vector;
                retracts.push(poses.len());
                poses.push(at(v.x, v.y, v.z + 300.0));
                retracts.push(poses.len());
                poses.push(at(v.x, v.y, v.z));
            }
        }
        assert_eq!(poses.len(), 100_000 + 2 * RETRACTS);
        (poses, retracts)
    }

    /// Home seed: J1 forward, elbow up, as a caller's `current_joints` would be.
    const HOME: [f64; 6] = [0.0; 6];

    fn assert_rows_equal(a: &[[f64; 6]], b: &[[f64; 6]], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            for j in 0..6 {
                assert!(
                    (x[j] - y[j]).abs() < tol || (x[j].is_nan() && y[j].is_nan()),
                    "row {i} joint {j}: {x:?} vs {y:?}"
                );
            }
        }
    }

    #[test]
    fn printing_path_tracks_without_fallbacks() {
        let robot = Robot::new(comau_nj4_220()).unwrap();
        let (poses, retracts) = printing_path();
        let start = Instant::now();
        let (joints, stats) = robot.batch_inverse(&poses, Some(&HOME));
        let us = start.elapsed().as_secs_f64() * 1e6 / poses.len() as f64;
        println!("NJ4 printing path: {us:.2} us per pose, {stats:?}");
        assert_eq!(stats.failures, 0);
        assert!(
            stats.fallbacks <= 2 * RETRACTS + 1,
            "{} fallbacks on the printing path",
            stats.fallbacks
        );
        let mut max_print_step: f64 = 0.0;
        let mut max_retract_step: f64 = 0.0;
        for (i, w) in joints.windows(2).enumerate() {
            let pose = poses[i + 1].unwrap();
            let fk = robot.forward(&w[1]);
            assert!((fk.translation.vector - pose.translation.vector).norm() < 1e-6);
            assert!(fk.rotation.angle_to(&pose.rotation) < 1e-9);
            let step = (0..6)
                .map(|j| (w[1][j] - w[0][j]).abs())
                .fold(0.0, f64::max);
            if retracts.contains(&(i + 1)) {
                max_retract_step = max_retract_step.max(step);
            } else {
                max_print_step = max_print_step.max(step);
            }
        }
        println!("largest joint move: printing {max_print_step:.4} rad, retract {max_retract_step:.4} rad");
        // 1 mm moves on a 2 m arm stay far below a degree; the branch never switches.
        assert!(
            max_print_step < 0.02,
            "joint jump {max_print_step} rad on a 1 mm move"
        );
        assert!(
            max_retract_step < 1.0,
            "retract jump {max_retract_step} rad"
        );
    }

    #[test]
    fn printing_path_tracks_unseeded() {
        // Without a seed the first root is the one nearest home, so the chain must not
        // start on a backwards-folded branch that dies at the reach boundary.
        let robot = Robot::new(comau_nj4_220()).unwrap();
        let (poses, _) = printing_path();
        let (joints, stats) = robot.batch_inverse(&poses, None);
        println!("NJ4 printing path unseeded: {stats:?}");
        assert_eq!(stats.failures, 0);
        assert!(joints.iter().all(|row| row.iter().all(|v| v.is_finite())));
        assert!(
            stats.fallbacks <= 2 * RETRACTS + 1,
            "{} fallbacks on the unseeded printing path",
            stats.fallbacks
        );
        let (seeded, _) = robot.batch_inverse(&poses, Some(&HOME));
        assert_rows_equal(&joints, &seeded, 1e-9);
    }

    #[test]
    fn chunked_calls_match_one_call() {
        let robot = Robot::new(comau_nj4_220()).unwrap();
        let (poses, _) = printing_path();
        let (whole, _) = robot.batch_inverse(&poses, Some(&HOME));
        let mut chunked: Vec<[f64; 6]> = Vec::with_capacity(poses.len());
        for chunk in poses.chunks(1000) {
            let seed = chunked.last().copied().or(Some(HOME));
            let (part, _) = robot.batch_inverse(chunk, seed.as_ref());
            chunked.extend(part);
        }
        assert_rows_equal(&whole, &chunked, 1e-9);
    }

    #[test]
    fn random_poses_fall_back_and_match_cold_nearest() {
        seed_random(31);
        let robot = Robot::new(comau_nj165_opw()).unwrap();
        let poses: Vec<Option<Isometry3<f64>>> = (0..300)
            .map(|i| {
                if i % 50 == 7 {
                    None
                } else {
                    Some(robot.forward(&std::array::from_fn(|_| random_angle())))
                }
            })
            .collect();
        let seed = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6];
        let (joints, stats) = robot.batch_inverse(&poses, Some(&seed));
        assert_eq!(stats.failures, 6);
        let mut prev = seed;
        for (pose, row) in poses.iter().zip(&joints) {
            match pose {
                None => assert!(row.iter().all(|v| v.is_nan())),
                Some(p) => {
                    // py-opw's contract: the row is the cold root nearest the previous row.
                    let expect = robot.inverse_all(p, Some(&prev))[0];
                    assert_rows_equal(&[*row], &[expect], 1e-9);
                    prev = *row;
                }
            }
        }
    }

    #[test]
    fn fold_stall_falls_back_instead_of_drifting() {
        // Walking q5 through pi on the NJ4 crosses the 120 degree fold where the Jacobian is
        // singular; every row must still round-trip exactly.
        let robot = Robot::new(comau_nj4_220()).unwrap();
        let poses: Vec<Option<Isometry3<f64>>> = (0..400)
            .map(|i| {
                let q5 = 2.6 + 1.2 * i as f64 / 399.0;
                Some(robot.forward(&[0.3, 0.2, 0.4, 0.5, q5, -0.3]))
            })
            .collect();
        let (joints, stats) = robot.batch_inverse(&poses, Some(&[0.3, 0.2, 0.4, 0.5, 2.6, -0.3]));
        assert_eq!(stats.failures, 0);
        println!("fold walk: {stats:?}");
        for (pose, row) in poses.iter().zip(&joints) {
            let fk = robot.forward(row);
            let p = pose.unwrap();
            assert!((fk.translation.vector - p.translation.vector).norm() < 1e-6);
            assert!(fk.rotation.angle_to(&p.rotation) < 1e-9);
        }
    }

    #[test]
    #[ignore]
    fn timing_tracked_inverse() {
        let nj4 = Robot::new(comau_nj4_220()).unwrap();
        let (poses, _) = printing_path();
        let start = Instant::now();
        let (_, stats) = nj4.batch_inverse(&poses, Some(&HOME));
        let us_nj4 = start.elapsed().as_secs_f64() * 1e6 / poses.len() as f64;
        println!("NJ4-220: {us_nj4:.2} us per tracked pose, {stats:?}");

        let nj165 = Robot::new(comau_nj165_opw()).unwrap();
        let poses: Vec<Option<Isometry3<f64>>> = (0..100_000)
            .map(|i| {
                let s = i as f64 / 100_000.0;
                Some(nj165.forward(&[
                    0.5 * s,
                    0.2 + 0.3 * s,
                    -1.2 + 0.4 * s,
                    0.1 * s,
                    0.8 - 0.3 * s,
                    0.2 * s,
                ]))
            })
            .collect();
        let start = Instant::now();
        let (_, stats) = nj165.batch_inverse(&poses, None);
        let us_nj165 = start.elapsed().as_secs_f64() * 1e6 / poses.len() as f64;
        println!("NJ165: {us_nj165:.2} us per tracked pose, {stats:?}");
        assert!(us_nj4 < 10.0, "NJ4 tracked {us_nj4} us per pose");
    }

    #[test]
    fn non_finite_seed_equals_unseeded_batch() {
        seed_random(13);
        let robot = Robot::new(comau_nj4_220()).unwrap();
        let poses: Vec<Option<Isometry3<f64>>> = (0..50)
            .map(|i| Some(robot.forward(&[0.2 + 0.01 * i as f64, -0.3, 0.4, 0.5, -1.0, 0.6])))
            .collect();
        let (plain, _) = robot.batch_inverse(&poses, None);
        let (nan_seeded, _) = robot.batch_inverse(&poses, Some(&[f64::NAN; 6]));
        assert_eq!(plain, nan_seeded);
        assert!(plain.iter().all(|r| r.iter().all(|v| v.is_finite())));
    }
}
