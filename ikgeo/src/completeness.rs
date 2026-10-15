//! Completeness of the cold solver against a brute-force multi-start: every root the
//! dense multi-start finds must be in `solve_cold`'s set.

use crate::inverse::{wrap_to_pi, DEDUPE_TOL, POSITION_TOL, ROTATION_TOL};
use crate::inverse_kinematics::auxiliary::Kinematics;
use crate::inverse_kinematics::hardcoded::setups::{RrcFixedQ6, TwoParallelBot};
use crate::robot::fixtures::{comau_nj165_opw, comau_nj4_220};
use crate::robot::{KinematicModel, Robot};
use crate::subproblems::auxiliary::{random_angle, random_unit, seed_random};
use nalgebra::{Matrix3, Vector3, Vector6};
use std::collections::BTreeMap;
use std::f64::consts::PI;

fn same_root(a: &Vector6<f64>, b: &Vector6<f64>) -> bool {
    (0..6).all(|j| wrap_to_pi(a[j] - b[j]).abs() < DEDUPE_TOL)
}

/// Validated, deduplicated roots reached by polishing `starts` uniform random joint vectors.
fn brute_force_roots(
    robot: &Robot,
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
    starts: usize,
) -> Vec<Vector6<f64>> {
    let scale = robot.link_length();
    let mut roots: Vec<Vector6<f64>> = Vec::new();
    for _ in 0..starts {
        let q0 = Vector6::from_fn(|_, _| random_angle());
        let (q, res) = robot.polish(&q0, r, t);
        if !(res.translation < POSITION_TOL * scale && res.rotation < ROTATION_TOL) {
            continue;
        }
        let q = q.map(wrap_to_pi);
        if !roots.iter().any(|other| same_root(&q, other)) {
            roots.push(q);
        }
    }
    roots
}

struct Outcome {
    poses: usize,
    missing: usize,
    missing_poses: Vec<(Vector6<f64>, Vector6<f64>)>,
    brute_total: usize,
    extra: usize,
    histogram: BTreeMap<usize, usize>,
}

/// Compare `solve_cold` with the multi-start on poses from `joints`; the generating joints
/// are seeded into the multi-start so a brute-force miss never hides a solver miss there.
fn compare(robot: &Robot, joints: &[Vector6<f64>], starts: usize) -> Outcome {
    let mut out = Outcome {
        poses: joints.len(),
        missing: 0,
        missing_poses: Vec::new(),
        brute_total: 0,
        extra: 0,
        histogram: BTreeMap::new(),
    };
    for q_gen in joints {
        let (r, t) = robot.forward_internal(q_gen);
        let cold = robot.solve_cold(&r, &t);
        let mut brute = brute_force_roots(robot, &r, &t, starts);
        let q_wrapped = q_gen.map(wrap_to_pi);
        if !brute.iter().any(|b| same_root(b, &q_wrapped)) {
            brute.push(q_wrapped);
        }
        *out.histogram.entry(cold.len()).or_insert(0) += 1;
        out.brute_total += brute.len();
        for b in &brute {
            if !cold.iter().any(|c| same_root(c, b)) {
                out.missing += 1;
                out.missing_poses.push((*q_gen, *b));
            }
        }
        for c in &cold {
            if !brute.iter().any(|b| same_root(c, b)) {
                out.extra += 1;
            }
        }
    }
    out
}

fn report(name: &str, out: &Outcome) {
    println!(
        "{name}: {} poses, {} of {} brute-force roots missed by solve_cold, {} solver roots the multi-start did not reach, root count histogram {:?}",
        out.poses, out.missing, out.brute_total, out.extra, out.histogram
    );
    for (q_gen, missed) in out.missing_poses.iter().take(10) {
        println!(
            "  generating {:?}\n  missed     {:?}",
            q_gen.as_slice(),
            missed.as_slice()
        );
    }
}

fn random_joints(n: usize) -> Vec<Vector6<f64>> {
    (0..n)
        .map(|_| Vector6::from_fn(|_, _| random_angle()))
        .collect()
}

fn robot_from_kin(kin: Kinematics<6, 7>) -> Robot {
    Robot::new(KinematicModel::new(kin.h, kin.p, [0.0; 6], [false; 6])).unwrap()
}

fn run_random(name: &str, robot: &Robot, poses: usize, starts: usize) -> Outcome {
    let out = compare(robot, &random_joints(poses), starts);
    report(name, &out);
    out
}

/// Joints whose wrist point W (the 4/5 intersection) lies within `max_radius` of axis 1,
/// found by scanning q2, q3 on a grid and keeping the folded postures.
fn nj4_joints_with_w_near_axis1(robot: &Robot, n: usize, max_radius: f64) -> Vec<Vector6<f64>> {
    let kin = &robot.kin;
    let w_radius = |q2: f64, q3: f64| -> f64 {
        let p1: Vector3<f64> = kin.p.column(1).into();
        let p2: Vector3<f64> = kin.p.column(2).into();
        let p3: Vector3<f64> = kin.p.column(3).into();
        let h2: Vector3<f64> = kin.h.column(1).into();
        let h3: Vector3<f64> = kin.h.column(2).into();
        let w = p1 + crate::robot::rot_unit(&h2, q2) * (p2 + crate::robot::rot_unit(&h3, q3) * p3);
        w.fixed_rows::<2>(0).norm()
    };
    let mut out = Vec::new();
    let mut tries = 0;
    while out.len() < n && tries < 2_000_000 {
        tries += 1;
        let (q2, q3) = (random_angle(), random_angle());
        if w_radius(q2, q3) < max_radius {
            out.push(Vector6::new(
                random_angle(),
                q2,
                q3,
                random_angle(),
                random_angle(),
                random_angle(),
            ));
        }
    }
    assert_eq!(out.len(), n, "could not find enough folded postures");
    out
}

/// q3 that maximises the shoulder-to-wrist distance (the stretched elbow).
fn nj4_stretched_q3(robot: &Robot) -> f64 {
    let kin = &robot.kin;
    let p2: Vector3<f64> = kin.p.column(2).into();
    let p3: Vector3<f64> = kin.p.column(3).into();
    let h3: Vector3<f64> = kin.h.column(2).into();
    let reach = |q3: f64| (p2 + crate::robot::rot_unit(&h3, q3) * p3).norm();
    let mut best = (0.0, 0.0);
    for k in 0..100_000 {
        let q3 = -PI + 2.0 * PI * k as f64 / 100_000.0;
        let d = reach(q3);
        if d > best.1 {
            best = (q3, d);
        }
    }
    best.0
}

fn nj4_hard_joints(robot: &Robot, per_case: usize) -> Vec<(&'static str, Vec<Vector6<f64>>)> {
    let jitter = |width: f64| (random_unit() * 2.0 - 1.0) * width;
    let q5_near = |centre: f64| -> Vec<Vector6<f64>> {
        (0..per_case)
            .map(|_| {
                let mut q = Vector6::from_fn(|_, _| random_angle());
                q[4] = centre + jitter(0.05);
                q
            })
            .collect()
    };
    let q3_stretch = nj4_stretched_q3(robot);
    let stretched: Vec<Vector6<f64>> = (0..per_case)
        .map(|_| {
            let mut q = Vector6::from_fn(|_, _| random_angle());
            q[2] = q3_stretch + jitter(0.02);
            q
        })
        .collect();
    vec![
        ("q5 near 0", q5_near(0.0)),
        ("q5 near +pi", q5_near(PI)),
        ("q5 near -pi", q5_near(-PI)),
        ("stretched elbow", stretched),
        (
            "W near axis 1",
            nj4_joints_with_w_near_axis1(robot, per_case, 50.0),
        ),
    ]
}

#[test]
fn nj4_cold_solver_is_complete() {
    seed_random(4242);
    let robot = Robot::new(comau_nj4_220()).unwrap();
    let out = run_random("NJ4-220", &robot, 30, 1500);
    assert_eq!(
        out.missing, 0,
        "NJ4: solve_cold misses roots the multi-start finds"
    );
    for (case, joints) in nj4_hard_joints(&robot, 10) {
        let out = compare(&robot, &joints, 1500);
        report(&format!("NJ4-220 {case}"), &out);
        assert_eq!(out.missing, 0, "NJ4 {case}: solve_cold misses roots");
    }
}

#[test]
fn nj165_cold_solver_is_complete() {
    seed_random(4243);
    let robot = Robot::new(comau_nj165_opw()).unwrap();
    let out = run_random("NJ165", &robot, 30, 1500);
    assert_eq!(out.missing, 0, "NJ165: solve_cold misses roots");
    assert!(out.histogram.keys().all(|&k| k <= 8));
}

/// Full-size run from the brief: 300 poses x 4000 starts, plus 50 per hard case, and the
/// upstream fixtures for reference. Minutes, so ignored by default.
#[test]
#[ignore]
fn cold_solver_completeness_full() {
    seed_random(4244);
    let nj4 = Robot::new(comau_nj4_220()).unwrap();
    let out = run_random("NJ4-220", &nj4, 300, 4000);
    assert_eq!(out.missing, 0);
    for (case, joints) in nj4_hard_joints(&nj4, 50) {
        let out = compare(&nj4, &joints, 4000);
        report(&format!("NJ4-220 {case}"), &out);
        assert_eq!(out.missing, 0, "NJ4 {case}");
    }
    let nj165 = Robot::new(comau_nj165_opw()).unwrap();
    let out = run_random("NJ165", &nj165, 300, 4000);
    assert_eq!(out.missing, 0);
    // Generic 1D-search families, not shipped for. Upstream's two_intersecting feeds
    // Subproblem 5, whose slot order jumps along q4, so a few percent of roots are missed;
    // the bar is 10% of all roots.
    let rrc = robot_from_kin(RrcFixedQ6::get_kin_partial().0);
    let out = run_random("RRC fixed q6", &rrc, 100, 2000);
    assert!(
        out.missing * 10 <= out.brute_total,
        "RRC misses {}",
        out.missing
    );
    let tpb = robot_from_kin(TwoParallelBot::get_kin());
    let out = run_random("two_parallel_bot", &tpb, 100, 2000);
    assert!(
        out.missing * 10 <= out.brute_total,
        "two_parallel_bot misses {}",
        out.missing
    );
}
