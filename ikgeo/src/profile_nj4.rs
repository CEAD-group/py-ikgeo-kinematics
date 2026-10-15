//! Phase breakdown of the NJ4 cold solve, for tuning the 1D search. Ignored by default.

use crate::inverse::{wrap_to_pi, POSITION_TOL, ROTATION_TOL};
use crate::inverse_kinematics::auxiliary::search_1d_bracketed;
use crate::robot::fixtures::comau_nj4_220;
use crate::robot::Robot;
use crate::subproblems::auxiliary::{random_angle, seed_random};
use crate::subproblems::{subproblem1_sc, subproblem3_sc, subproblem4_sc};
use nalgebra::{Matrix3, Vector3, Vector4, Vector6};
use std::cell::Cell;
use std::f64::consts::PI;
use std::time::Instant;

const N: usize = 1000;

/// The NJ4 alignment error of `two_intersecting_two_parallel`, with an evaluation counter.
fn alignment_error(
    robot: &Robot,
    r_06: &Matrix3<f64>,
    p_0t: &Vector3<f64>,
    counter: &Cell<usize>,
) -> impl Fn(f64) -> Vector4<f64> {
    use crate::subproblems::auxiliary::rot_vec_sc;
    let kin = robot.chain.kin.clone();
    let h1: Vector3<f64> = kin.h.column(0).into();
    let h2: Vector3<f64> = kin.h.column(1).into();
    let h3: Vector3<f64> = kin.h.column(2).into();
    let h4: Vector3<f64> = kin.h.column(3).into();
    let h5: Vector3<f64> = kin.h.column(4).into();
    let h6: Vector3<f64> = kin.h.column(5).into();
    let p1: Vector3<f64> = kin.p.column(1).into();
    let p2: Vector3<f64> = kin.p.column(2).into();
    let p3: Vector3<f64> = kin.p.column(3).into();
    let p4: Vector3<f64> = kin.p.column(4).into();
    let p_16 = p_0t - kin.p.column(0) - r_06 * kin.p.column(6);
    let u6 = r_06 * h6;
    let d_12 = h2.dot(&(p1 + p2));
    let h5h6 = h5.dot(&h6);
    let counter = counter as *const Cell<usize>;
    move |q4: f64| -> Vector4<f64> {
        // SAFETY: the counter outlives the closure within one test body.
        unsafe { (*counter).set((*counter).get() + 1) };
        let mut error = Vector4::from_element(f64::INFINITY);
        let (s4, c4) = q4.sin_cos();
        let p_35 = p3 + rot_vec_sc(&h4, s4, c4, &p4);
        let (t1, q1_ls) = subproblem4_sc(&h2, &p_16, &-h1, d_12 + h2.dot(&p_35));
        if q1_ls {
            return error;
        }
        for (i1, (s1, c1)) in t1.iter().enumerate() {
            let v = rot_vec_sc(&h1, -s1, c1, &p_16) - p1;
            let (t3, q3_ls) = subproblem3_sc(&-p_35, &p2, &h3, v.norm());
            if q3_ls {
                continue;
            }
            for (i3, (s3, c3)) in t3.iter().enumerate() {
                let ((s2, c2), _) = subproblem1_sc(&(p2 + rot_vec_sc(&h3, s3, c3, &p_35)), &v, &h2);
                let w = rot_vec_sc(&h1, -s1, c1, &u6);
                let w = rot_vec_sc(&h2, -s2, c2, &w);
                let w = rot_vec_sc(&h3, -s3, c3, &w);
                let w = rot_vec_sc(&h4, -s4, c4, &w);
                error[2 * i1 + i3] = h5.dot(&w) - h5h6;
            }
        }
        error
    }
}

#[test]
#[ignore]
fn nj4_cold_phase_breakdown() {
    seed_random(7);
    let robot = Robot::new(comau_nj4_220()).unwrap();
    let scale = robot.link_length();
    let poses: Vec<(Matrix3<f64>, Vector3<f64>)> = (0..N)
        .map(|_| robot.forward_internal(&Vector6::from_fn(|_, _| random_angle())))
        .collect();

    // Whole phases.
    let start = Instant::now();
    let mut raw_count = 0;
    for (r, t) in &poses {
        raw_count += robot.solve_raw(r, t).len();
    }
    let t_raw = start.elapsed().as_secs_f64() * 1e6 / N as f64;

    let start = Instant::now();
    let mut root_count = 0;
    for (r, t) in &poses {
        root_count += robot.solve_cold(r, t).len();
    }
    let t_cold = start.elapsed().as_secs_f64() * 1e6 / N as f64;

    // Polish alone on the raw candidates, and how many of them already meet tolerance.
    let raws: Vec<Vec<Vector6<f64>>> = poses.iter().map(|(r, t)| robot.solve_raw(r, t)).collect();
    let start = Instant::now();
    let mut already_ok = 0;
    let mut rejected = 0;
    for ((r, t), raw) in poses.iter().zip(&raws) {
        for q in raw {
            let res0 = robot.residual(q, r, t);
            if res0.translation < POSITION_TOL * scale && res0.rotation < ROTATION_TOL {
                already_ok += 1;
            }
            let (_, res) = robot.polish(q, r, t);
            if !(res.translation < POSITION_TOL * scale && res.rotation < ROTATION_TOL) {
                rejected += 1;
            }
        }
    }
    let t_polish = start.elapsed().as_secs_f64() * 1e6 / N as f64;

    // Inside the search: evaluations and their cost.
    let counter = Cell::new(0usize);
    let start = Instant::now();
    let mut found = 0;
    for (r, t) in &poses {
        let f = alignment_error(&robot, r, t, &counter);
        found += search_1d_bracketed(f, -PI, PI, 200, 0.5).len();
    }
    let t_search = start.elapsed().as_secs_f64() * 1e6 / N as f64;
    let evals = counter.get() as f64 / N as f64;

    let counter2 = Cell::new(0usize);
    let start = Instant::now();
    let mut sink = 0.0;
    for (r, t) in &poses {
        let f = alignment_error(&robot, r, t, &counter2);
        for k in 0..201 {
            sink += f(-PI + k as f64 * 2.0 * PI / 200.0)[0].min(1.0);
        }
    }
    let t_eval = start.elapsed().as_secs_f64() * 1e6 / (N as f64 * 201.0);
    assert!(sink.is_finite() || sink.is_nan());

    // Duplicates among raw candidates.
    let mut dupes = 0;
    for raw in &raws {
        for (i, a) in raw.iter().enumerate() {
            if raw[..i]
                .iter()
                .any(|b| (0..6).all(|j| wrap_to_pi(a[j] - b[j]).abs() < 1e-6))
            {
                dupes += 1;
            }
        }
    }

    println!("NJ4 cold phase breakdown, {N} random poses, release:");
    println!("  solve_raw (search + wrist subproblems): {t_raw:.1} us");
    println!("  search_1d_bracketed alone:              {t_search:.1} us, {evals:.0} evaluations per pose at {t_eval:.3} us each, {:.2} candidates per pose", found as f64 / N as f64);
    println!("  polish + validate of raw candidates:    {t_polish:.1} us");
    println!("  solve_cold total:                       {t_cold:.1} us");
    println!(
        "  raw candidates {:.2} per pose, roots {:.2} per pose, raw already within tolerance {already_ok}, rejected after polish {rejected}, raw duplicates {dupes}",
        raw_count as f64 / N as f64,
        root_count as f64 / N as f64
    );
}
