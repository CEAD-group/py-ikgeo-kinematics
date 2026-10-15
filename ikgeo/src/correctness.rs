//! Correctness tests, adapted from upstream. Upstream only printed statistics;
//! here every family asserts a maximum miss rate, seeded so a run is reproducible.

use crate::{
    inverse_kinematics::{
        hardcoded::setups::{
            Irb6640, KukaR800FixedQ3, RrcFixedQ6, SphericalBot, ThreeParallelBot, TwoParallelBot,
            Ur5, YumiFixedQ3,
        },
        setups::{
            GenSixDofSetup, SetupIk, SphericalSetup, SphericalTwoIntersectingSetup,
            SphericalTwoParallelSetup, ThreeParallelSetup, ThreeParallelTwoIntersectingSetup,
            TwoIntersectingSetup, TwoParallelSetup,
        },
    },
    subproblems::{
        auxiliary::seed_random,
        setups::{
            SetupDynamic, SetupStatic, Subproblem1Setup, Subproblem2ExtendedSetup,
            Subproblem2Setup, Subproblem3Setup, Subproblem4Setup, Subproblem5Setup,
            Subproblem6Setup,
        },
    },
};

const TEST_ITERATIONS: usize = 1000;
const EXACT_ERROR: f64 = 1e-6;

struct IkStats {
    name: &'static str,
    none: usize,
    inexact: usize,
    ls_solutions: usize,
    solutions: usize,
    sum_exact_error: f64,
}

impl IkStats {
    fn miss_rate(&self) -> f64 {
        (self.none + self.inexact) as f64 / TEST_ITERATIONS as f64
    }

    fn print(&self) {
        let exact = TEST_ITERATIONS - self.none - self.inexact;
        println!(
            "{}\n\tmiss rate {:.2}% (no solution {}, none exact {})\n\tLS solutions {:.2}%\n\tavg exact error {:.2e}",
            self.name,
            100.0 * self.miss_rate(),
            self.none,
            self.inexact,
            100.0 * self.ls_solutions as f64 / self.solutions.max(1) as f64,
            self.sum_exact_error / exact.max(1) as f64,
        );
    }
}

fn ik_stats<S: SetupStatic + SetupIk>() -> IkStats {
    let mut setup = S::new();
    let mut stats = IkStats {
        name: <S as SetupStatic>::name(),
        none: 0,
        inexact: 0,
        ls_solutions: 0,
        solutions: 0,
        sum_exact_error: 0.0,
    };

    for _ in 0..TEST_ITERATIONS {
        setup.setup();
        setup.run();

        let n = setup.solution_count();
        stats.solutions += n;
        stats.ls_solutions += setup.ls_count();

        // error() is the minimum FK error over all returned solutions, NaN when there are none.
        let error = setup.error();
        if n == 0 {
            stats.none += 1;
        } else if error.is_nan() || error >= EXACT_ERROR {
            stats.inexact += 1;
        } else {
            stats.sum_exact_error += error;
        }
    }

    stats.print();
    stats
}

fn assert_ik<S: SetupStatic + SetupIk>(max_miss_rate: f64) {
    let stats = ik_stats::<S>();
    assert!(
        stats.miss_rate() <= max_miss_rate,
        "{}: miss rate {:.2}% exceeds {:.2}%",
        stats.name,
        100.0 * stats.miss_rate(),
        100.0 * max_miss_rate
    );
}

fn assert_subproblem<S: SetupStatic + SetupDynamic>(max_avg_error: f64, max_nan: usize) {
    let mut setup = S::new();
    let mut total_error = 0.0;
    let mut ls_count = 0;
    let mut nan_count = 0;

    for _ in 0..TEST_ITERATIONS {
        setup.setup();
        if setup.run_report_info() {
            ls_count += 1;
        }
        let error = setup.error();
        if error.is_finite() {
            total_error += error;
        } else {
            nan_count += 1;
        }
    }

    let avg = total_error / (TEST_ITERATIONS - nan_count) as f64;
    println!(
        "{}\n\tavg error {avg:.2e}\n\tLS {ls_count}\n\tNaN {nan_count}",
        <S as SetupStatic>::name()
    );
    assert_eq!(
        ls_count,
        0,
        "{}: exact setup flagged LS",
        <S as SetupStatic>::name()
    );
    assert!(
        nan_count <= max_nan,
        "{}: {nan_count} NaN results",
        <S as SetupStatic>::name()
    );
    assert!(
        avg <= max_avg_error,
        "{}: avg error {avg:.2e}",
        <S as SetupStatic>::name()
    );
}

#[test]
fn subproblems_exact() {
    println!();
    seed_random(0xC0FF_EE01);
    assert_subproblem::<Subproblem1Setup>(1e-12, 0);
    assert_subproblem::<Subproblem2Setup>(1e-12, 0);
    assert_subproblem::<Subproblem2ExtendedSetup>(1e-12, 0);
    assert_subproblem::<Subproblem3Setup>(1e-12, 0);
    assert_subproblem::<Subproblem4Setup>(1e-12, 0);
    // Subproblems 5 and 6 go through the quartic solver, which is less precise
    // and on rare degenerate random inputs returns no real root.
    assert_subproblem::<Subproblem5Setup>(1e-6, 10);
    assert_subproblem::<Subproblem6Setup>(1e-8, 10);
}

#[test]
fn subproblems_least_squares_are_local_minima() {
    seed_random(0xC0FF_EE02);
    let setups: Vec<Box<dyn SetupDynamic>> = vec![
        Box::new(Subproblem1Setup::new()),
        Box::new(Subproblem2Setup::new()),
        Box::new(Subproblem3Setup::new()),
        Box::new(Subproblem4Setup::new()),
    ];

    for mut setup in setups {
        for _ in 0..TEST_ITERATIONS {
            setup.setup_ls();
            setup.run();
            assert!(
                setup.is_at_local_min(),
                "{} solution was not at local minimum",
                setup.name()
            );
        }
    }
}

#[test]
fn closed_form_families_random_kinematics() {
    println!();
    seed_random(0xC0FF_EE03);
    assert_ik::<SphericalTwoParallelSetup>(0.0);
    assert_ik::<SphericalTwoIntersectingSetup>(0.0);
    assert_ik::<ThreeParallelTwoIntersectingSetup>(0.0);
    assert_ik::<ThreeParallelSetup>(0.0);
    // The quartic in Subproblem 5 occasionally loses a real root on random geometry.
    assert_ik::<SphericalSetup>(0.02);
}

#[test]
fn search_families_random_kinematics() {
    println!();
    seed_random(0xC0FF_EE04);
    // Miss rates come from the upstream implementation on the same random fixtures;
    // see tmp/upstream_correctness_baseline.txt for the run that set them.
    assert_ik::<TwoIntersectingSetup>(0.05);
    assert_ik::<TwoParallelSetup>(0.12);
    assert_ik::<GenSixDofSetup>(0.05);
}

#[test]
fn hardcoded_robots() {
    println!();
    seed_random(0xC0FF_EE05);
    assert_ik::<Irb6640>(0.0);
    assert_ik::<KukaR800FixedQ3>(0.0);
    assert_ik::<Ur5>(0.0);
    assert_ik::<ThreeParallelBot>(0.0);
    assert_ik::<SphericalBot>(0.0);
    assert_ik::<TwoParallelBot>(0.15);
    assert_ik::<RrcFixedQ6>(0.25);
    assert_ik::<YumiFixedQ3>(0.45);
}
