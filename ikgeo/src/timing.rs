//! Timing tests, adapted from upstream: inputs come from `test_cases/` next to
//! the manifest instead of a `data/` directory, no output files are written,
//! and a randomised run is added because the vendored CSVs hold 10 rows each.
//!
//! Run with `cargo test -p ikgeo --release timing -- --nocapture --test-threads 1`.

use {
    crate::{
        inverse_kinematics::{
            hardcoded::setups::{
                Irb6640, KukaR800FixedQ3, RrcFixedQ6, SphericalBot, ThreeParallelBot,
                TwoParallelBot, Ur5, YumiFixedQ3,
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
    },
    std::{fs, path::PathBuf, time::Instant},
};

const RANDOM_ITERATIONS: usize = 1000;

fn csv_rows(file: &str) -> Vec<String> {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "test_cases", file]
        .iter()
        .collect();
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(str::to_owned)
        .collect()
}

#[test]
fn time_subproblems_csv() {
    println!();
    time_subproblem_csv::<Subproblem1Setup>("sp_1.csv");
    time_subproblem_csv::<Subproblem2Setup>("sp_2.csv");
    time_subproblem_csv::<Subproblem2ExtendedSetup>("sp_2E.csv");
    time_subproblem_csv::<Subproblem3Setup>("sp_3.csv");
    time_subproblem_csv::<Subproblem4Setup>("sp_4.csv");
    time_subproblem_csv::<Subproblem5Setup>("sp_5.csv");
    time_subproblem_csv::<Subproblem6Setup>("sp_6.csv");
}

#[test]
fn time_ik_csv() {
    println!();
    time_ik_setup_csv::<SphericalSetup>("IK_spherical.csv");
    time_ik_setup_csv::<SphericalTwoIntersectingSetup>("IK_spherical_2_intersecting.csv");
    time_ik_setup_csv::<SphericalTwoParallelSetup>("IK_spherical_2_parallel.csv");
    time_ik_setup_csv::<ThreeParallelSetup>("IK_3_parallel.csv");
    time_ik_setup_csv::<ThreeParallelTwoIntersectingSetup>("IK_3_parallel_2_intersecting.csv");
    time_ik_setup_csv::<TwoParallelSetup>("IK_2_parallel.csv");
    time_ik_setup_csv::<TwoIntersectingSetup>("IK_2_intersecting.csv");
    time_ik_setup_csv::<GenSixDofSetup>("IK_gen_6_dof.csv");
}

#[test]
fn time_hardcoded_csv() {
    println!();
    time_ik_setup_csv::<Irb6640>("IRB_6640.csv");
    time_ik_setup_csv::<KukaR800FixedQ3>("KUKA_R800_fixed_q3.csv");
    time_ik_setup_csv::<RrcFixedQ6>("RRC_fixed_q6.csv");
    time_ik_setup_csv::<YumiFixedQ3>("yumi_fixed_q3.csv");
    time_ik_setup_csv::<Ur5>("ur5.csv");
    time_ik_setup_csv::<ThreeParallelBot>("three_parallel_bot.csv");
    time_ik_setup_csv::<TwoParallelBot>("two_parallel_bot.csv");
    time_ik_setup_csv::<SphericalBot>("spherical_bot.csv");
}

#[test]
fn time_ik_random() {
    println!();
    seed_random(0x5EED_7111);
    time_ik_setup_random::<SphericalSetup>();
    time_ik_setup_random::<SphericalTwoIntersectingSetup>();
    time_ik_setup_random::<SphericalTwoParallelSetup>();
    time_ik_setup_random::<ThreeParallelSetup>();
    time_ik_setup_random::<ThreeParallelTwoIntersectingSetup>();
    time_ik_setup_random::<TwoParallelSetup>();
    time_ik_setup_random::<TwoIntersectingSetup>();
    time_ik_setup_random::<GenSixDofSetup>();
}

#[test]
fn time_hardcoded_random() {
    println!();
    seed_random(0x5EED_7112);
    time_ik_setup_random::<Irb6640>();
    time_ik_setup_random::<KukaR800FixedQ3>();
    time_ik_setup_random::<RrcFixedQ6>();
    time_ik_setup_random::<YumiFixedQ3>();
    time_ik_setup_random::<Ur5>();
    time_ik_setup_random::<ThreeParallelBot>();
    time_ik_setup_random::<TwoParallelBot>();
    time_ik_setup_random::<SphericalBot>();
}

fn time_subproblem_csv<S: SetupStatic + SetupDynamic>(file: &str) {
    let mut setups: Vec<S> = csv_rows(file)
        .iter()
        .map(|row| {
            let mut s = S::new();
            s.setup_from_str(row);
            s
        })
        .collect();

    let start = Instant::now();
    for setup in setups.iter_mut() {
        setup.run();
    }
    let ns = start.elapsed().as_nanos() / setups.len() as u128;

    // The CSV rows are exact cases, so a parse mismatch shows up as a large error.
    let worst = setups.iter().map(SetupDynamic::error).fold(0.0, f64::max);
    println!(
        "csv     \t{}\t{ns} ns\tworst error {worst:.1e}",
        <S as SetupStatic>::name()
    );
    assert!(
        worst < 1e-6,
        "{}: CSV case error {worst:.2e}",
        <S as SetupStatic>::name()
    );
}

fn time_ik_setup_csv<S: SetupStatic + SetupIk>(file: &str) {
    let mut setups: Vec<S> = csv_rows(file)
        .iter()
        .map(|row| {
            let mut s = S::new();
            s.setup_from_str(row);
            s
        })
        .collect();

    let start = Instant::now();
    for setup in setups.iter_mut() {
        setup.run();
    }
    let ns = start.elapsed().as_nanos() / setups.len() as u128;

    let misses = setups
        .iter()
        .filter(|s| s.error().is_nan() || s.error() >= 1e-6)
        .count();
    println!(
        "csv     \t{}\t{ns} ns\t{misses}/{} rows without an exact solution",
        <S as SetupStatic>::name(),
        setups.len()
    );
}

fn time_ik_setup_random<S: SetupStatic + SetupIk>() {
    let mut setups: Vec<S> = (0..RANDOM_ITERATIONS)
        .map(|_| {
            let mut s = S::new();
            s.setup();
            s
        })
        .collect();

    let start = Instant::now();
    for setup in setups.iter_mut() {
        setup.run();
    }
    let ns = start.elapsed().as_nanos() / setups.len() as u128;

    let misses = setups
        .iter()
        .filter(|s| s.error().is_nan() || s.error() >= 1e-6)
        .count();
    println!(
        "random  \t{}\t{ns} ns\t{misses}/{} poses without an exact solution",
        <S as SetupStatic>::name(),
        setups.len()
    );
}
