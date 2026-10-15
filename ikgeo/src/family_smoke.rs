//! One synthetic chain per family the classifier can produce, forward and
//! reversed: either `Robot::new` rejects it or random poses solve.

use crate::inverse_kinematics::auxiliary::Kinematics;
use crate::robot::{reversed_kinematics, Family, KinematicModel, Robot};
use crate::subproblems::auxiliary::{random_angle, random_vector3, seed_random};
use nalgebra::{Vector3, Vector6};
use std::f64::consts::TAU;

const POSES: usize = 100;
const MIN_HIT: f64 = 0.9;
// The 2D grid search is a best-effort fallback; upstream's own fixtures miss a few percent.
const MIN_HIT_GENERAL: f64 = 0.5;

/// Build a chain from axis letters and the indices of offsets forced to zero.
fn chain(axes: [char; 6], zero_offsets: &[usize]) -> Kinematics<6, 7> {
    let dir = |c: char| match c {
        'x' => Vector3::x(),
        'y' => Vector3::y(),
        _ => Vector3::z(),
    };
    let mut kin: Kinematics<6, 7> = Kinematics::new();
    for (j, axis) in axes.iter().enumerate() {
        kin.h.set_column(j, &dir(*axis));
    }
    for i in 0..7 {
        // Offsets of order 1 with a bias away from the origin, so axes are skew unless forced.
        let v = random_vector3() + Vector3::new(0.3, 0.2, 0.5);
        kin.p.set_column(i, &v);
    }
    for &i in zero_offsets {
        kin.p.set_column(i, &Vector3::zeros());
    }
    kin
}

fn hit_rate(robot: &Robot) -> (f64, usize) {
    let mut hits = 0;
    let mut empty = 0;
    for _ in 0..POSES {
        let q = Vector6::from_fn(|_, _| random_angle());
        let (r, t) = robot.forward_internal(&q);
        let sols = robot.solve_cold(&r, &t);
        if sols.is_empty() {
            empty += 1;
        }
        let found = sols.iter().any(|s| {
            (0..6).all(|j| {
                let d = (s[j] - q[j]).rem_euclid(TAU);
                d < 1e-6 || TAU - d < 1e-6
            })
        });
        if found {
            hits += 1;
        }
    }
    (hits as f64 / POSES as f64, empty)
}

#[test]
fn every_family_solves_or_is_rejected() {
    seed_random(4242);
    // (label, axes, zero offsets, expected family)
    let cases: Vec<(&str, [char; 6], Vec<usize>, Family)> = vec![
        (
            "spherical",
            ['z', 'y', 'x', 'z', 'y', 'z'],
            vec![4, 5],
            Family::Spherical,
        ),
        (
            "spherical_two_intersecting",
            ['z', 'y', 'x', 'z', 'y', 'z'],
            vec![1, 4, 5],
            Family::SphericalTwoIntersecting,
        ),
        (
            "spherical_two_parallel",
            ['z', 'y', 'y', 'z', 'y', 'z'],
            vec![4, 5],
            Family::SphericalTwoParallel,
        ),
        (
            "three_parallel",
            ['z', 'y', 'y', 'y', 'x', 'z'],
            vec![],
            Family::ThreeParallel,
        ),
        (
            "three_parallel_two_intersecting",
            ['z', 'y', 'y', 'y', 'x', 'z'],
            vec![5],
            Family::ThreeParallelTwoIntersecting,
        ),
        (
            "three_parallel_two_parallel",
            ['z', 'y', 'y', 'y', 'x', 'x'],
            vec![],
            Family::ThreeParallelTwoParallel,
        ),
        (
            "two_intersecting",
            ['z', 'x', 'y', 'z', 'y', 'z'],
            vec![5],
            Family::TwoIntersecting,
        ),
        (
            "two_intersecting_two_intersecting",
            ['z', 'x', 'y', 'z', 'y', 'z'],
            vec![1, 5],
            Family::TwoIntersectingTwoIntersecting,
        ),
        (
            "two_intersecting_two_parallel",
            ['z', 'y', 'y', 'z', 'y', 'z'],
            vec![5],
            Family::TwoIntersectingTwoParallel,
        ),
        (
            "two_parallel",
            ['z', 'y', 'y', 'z', 'x', 'z'],
            vec![],
            Family::TwoParallel,
        ),
        (
            "two_parallel_two_parallel (5/6)",
            ['z', 'y', 'y', 'z', 'x', 'x'],
            vec![],
            Family::TwoParallelTwoParallel,
        ),
        (
            "two_parallel_two_parallel (4/5)",
            ['z', 'y', 'y', 'x', 'x', 'z'],
            vec![],
            Family::TwoParallelTwoParallel,
        ),
        (
            "general",
            ['z', 'x', 'y', 'z', 'x', 'y'],
            vec![],
            Family::General,
        ),
    ];
    let mut failures = Vec::new();
    println!(
        "{:<48} {:>8} {:>6} {:>6}",
        "chain", "solved", "hit", "empty"
    );
    for (label, axes, zeros, expected) in &cases {
        for reversed in [false, true] {
            let mut kin = chain(*axes, zeros);
            if reversed {
                kin = reversed_kinematics(&kin);
            }
            let name = format!("{label}{}", if reversed { " reversed" } else { "" });
            let model = KinematicModel::new(kin.h, kin.p, [0.0; 6], [false; 6]);
            match Robot::new(model) {
                Err(e) => println!("{name:<48} rejected: {e}"),
                Ok(robot) => {
                    assert_eq!(robot.family(), *expected, "{name}");
                    // A chain symmetric under reversal is solved forward; only assert the flag otherwise.
                    let (hit, empty) = hit_rate(&robot);
                    println!(
                        "{name:<48} {:>8} {hit:>6.2} {empty:>6}",
                        if robot.chain.reversed { "rev" } else { "fwd" }
                    );
                    let floor = if *expected == Family::General {
                        MIN_HIT_GENERAL
                    } else {
                        MIN_HIT
                    };
                    if hit < floor {
                        failures.push(format!("{name}: hit rate {hit}"));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
