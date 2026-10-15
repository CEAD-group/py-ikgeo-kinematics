pub mod auxiliary;
/// Hardcoded robots from upstream, test fixtures only.
pub mod hardcoded;
/// Randomised test fixtures, kept public so the parity suites in later steps can reuse them.
pub mod setups;

use {
    crate::{
        solutionset::SolutionSet4,
        subproblems::{
            auxiliary::{rot, rot_vec_sc},
            subproblem1, subproblem1_sc, subproblem2, subproblem3, subproblem3_sc, subproblem4,
            subproblem4_sc, subproblem5, subproblem6,
        },
    },
    auxiliary::{search_1d_bracketed, search_2d, wrap_to_pi, Kinematics},
    nalgebra::{Matrix3, Matrix4, Vector3, Vector4, Vector6},
    std::f64::consts::PI,
};

pub fn spherical_two_parallel(
    r_0t: &Matrix3<f64>,
    p_0t: &Vector3<f64>,
    kin: &Kinematics<6, 7>,
) -> (Vec<Vector6<f64>>, Vec<bool>) {
    let mut q = Vec::with_capacity(8);
    let mut is_ls = Vec::with_capacity(8);

    let (t1, q1_is_ls) = subproblem4(
        &(kin.h.column(1).into()),
        &(p_0t - r_0t * kin.p.column(6) - kin.p.column(0)),
        &(-kin.h.column(0)),
        (kin.h.column(1).transpose() * (kin.p.column(1) + kin.p.column(2) + kin.p.column(3)))[0],
    );

    for q1 in t1.get_all() {
        let (t3, q3_is_ls) = subproblem3(
            &(-kin.p.column(3)),
            &(kin.p.column(2).into()),
            &(kin.h.column(2).into()),
            (rot(&(-kin.h.column(0)), q1) * (-p_0t + r_0t * kin.p.column(6) + kin.p.column(0))
                + kin.p.column(1))
            .norm(),
        );

        for q3 in t3.get_all() {
            let (q2, q2_is_ls) = subproblem1(
                &(-kin.p.column(2) - rot(&kin.h.column(2).into(), q3) * kin.p.column(3)),
                &(rot(&(-kin.h.column(0)), q1)
                    * (-p_0t + r_0t * kin.p.column(6) + kin.p.column(0))
                    + kin.p.column(1)),
                &(kin.h.column(1).into()),
            );

            let r_36 = rot(&(-kin.h.column(2)), q3)
                * rot(&(-kin.h.column(1)), q2)
                * rot(&(-kin.h.column(0)), q1)
                * r_0t;

            let (t5, q5_is_ls) = subproblem4(
                &(kin.h.column(3).into()),
                &(kin.h.column(5).into()),
                &(kin.h.column(4).into()),
                (kin.h.column(3).transpose() * r_36 * kin.h.column(5))[0],
            );

            for q5 in t5.get_all() {
                let (q4, q4_is_ls) = subproblem1(
                    &(rot(&(kin.h.column(4).into()), q5) * kin.h.column(5)),
                    &(r_36 * kin.h.column(5)),
                    &(kin.h.column(3).into()),
                );

                let (q6, q6_is_ls) = subproblem1(
                    &(rot(&(-kin.h.column(4)), q5) * kin.h.column(3)),
                    &(r_36.transpose() * kin.h.column(3)),
                    &(-kin.h.column(5)),
                );

                q.push(Vector6::new(q1, q2, q3, q4, q5, q6));
                is_ls.push(q1_is_ls || q2_is_ls || q3_is_ls || q4_is_ls || q5_is_ls || q6_is_ls)
            }
        }
    }

    (q, is_ls)
}

pub fn spherical_two_intersecting(
    r_0t: &Matrix3<f64>,
    p_0t: &Vector3<f64>,
    kin: &Kinematics<6, 7>,
) -> (Vec<Vector6<f64>>, Vec<bool>) {
    let mut q = Vec::with_capacity(8);
    let mut is_ls = Vec::with_capacity(8);

    let p_16 = p_0t - r_0t * kin.p.column(6) - kin.p.column(0);

    let (t3, t3_is_ls) = subproblem3(
        &kin.p.column(3).into(),
        &-kin.p.column(2),
        &kin.h.column(2).into(),
        p_16.norm(),
    );

    for q3 in t3.get_all() {
        let (t12, t12_is_ls) = subproblem2(
            &p_16,
            &(kin.p.column(2) + rot(&kin.h.column(2).into(), q3) * kin.p.column(3)),
            &-kin.h.column(0),
            &kin.h.column(1).into(),
        );

        for (q1, q2) in t12.get_all() {
            let r_36 = rot(&-kin.h.column(2), q3)
                * rot(&-kin.h.column(1), q2)
                * rot(&-kin.h.column(0), q1)
                * r_0t;

            let (t5, q5_is_ls) = subproblem4(
                &kin.h.column(3).into(),
                &kin.h.column(5).into(),
                &kin.h.column(4).into(),
                (kin.h.column(3).transpose() * r_36 * kin.h.column(5))[0],
            );

            for q5 in t5.get_all() {
                let (q4, q4_is_ls) = subproblem1(
                    &(rot(&kin.h.column(4).into(), q5) * kin.h.column(5)),
                    &(r_36 * kin.h.column(5)),
                    &kin.h.column(3).into(),
                );

                let (q6, q6_is_ls) = subproblem1(
                    &(rot(&-kin.h.column(4), q5) * kin.h.column(3)),
                    &(r_36.transpose() * kin.h.column(3)),
                    &-kin.h.column(5),
                );

                q.push(Vector6::new(q1, q2, q3, q4, q5, q6));
                is_ls.push(t3_is_ls || t12_is_ls || q5_is_ls || q4_is_ls || q6_is_ls);
            }
        }
    }

    (q, is_ls)
}

pub fn spherical(
    r_06: &Matrix3<f64>,
    p_0t: &Vector3<f64>,
    kin: &Kinematics<6, 7>,
) -> (Vec<Vector6<f64>>, Vec<bool>) {
    let mut q = Vec::with_capacity(8);
    let mut is_ls = Vec::with_capacity(8);

    let p_16 = p_0t - kin.p.column(0) - r_06 * kin.p.column(6);

    let t123 = subproblem5(
        &-kin.p.column(1),
        &p_16,
        &kin.p.column(2).into(),
        &kin.p.column(3).into(),
        &-kin.h.column(0),
        &kin.h.column(1).into(),
        &kin.h.column(2).into(),
    );

    for (q1, q2, q3) in t123.get_all() {
        let r_36 = rot(&-kin.h.column(2), q3)
            * rot(&-kin.h.column(1), q2)
            * rot(&-kin.h.column(0), q1)
            * r_06;

        let (t5, q5_is_ls) = subproblem4(
            &kin.h.column(3).into(),
            &kin.h.column(5).into(),
            &kin.h.column(4).into(),
            (kin.h.column(3).transpose() * r_36 * kin.h.column(5))[0],
        );

        for q5 in t5.get_all() {
            let (q4, q4_is_ls) = subproblem1(
                &(rot(&kin.h.column(4).into(), q5) * kin.h.column(5)),
                &(r_36 * kin.h.column(5)),
                &kin.h.column(3).into(),
            );

            let (q6, q6_is_ls) = subproblem1(
                &(rot(&-kin.h.column(4), q5) * kin.h.column(3)),
                &(r_36.transpose() * kin.h.column(3)),
                &-kin.h.column(5),
            );

            q.push(Vector6::new(q1, q2, q3, q4, q5, q6));
            is_ls.push(q5_is_ls || q4_is_ls || q6_is_ls);
        }
    }

    (q, is_ls)
}

pub fn three_parallel_two_intersecting(
    r_06: &Matrix3<f64>,
    p_0t: &Vector3<f64>,
    kin: &Kinematics<6, 7>,
) -> (Vec<Vector6<f64>>, Vec<bool>) {
    let mut q = Vec::with_capacity(8);
    let mut is_ls = Vec::with_capacity(8);

    let sum_p_2_5 = kin.p.fixed_columns::<4>(1).column_sum();
    let p_16 = p_0t - kin.p.column(0) - r_06 * kin.p.column(6);

    let (theta1, theta1_is_ls) = subproblem4(
        &kin.h.column(1).into(),
        &p_16,
        &-kin.h.column(0),
        (kin.h.column(1).transpose() * sum_p_2_5)[0],
    );

    for q1 in theta1.get_all() {
        let r_01 = rot(&kin.h.column(0).into(), q1);

        let (theta5, theta5_is_ls) = subproblem4(
            &kin.h.column(1).into(),
            &kin.h.column(5).into(),
            &kin.h.column(4).into(),
            (kin.h.column(1).transpose() * r_01.transpose() * r_06 * kin.h.column(5))[0],
        );

        for q5 in theta5.get_all() {
            let (theta14, theta_14_is_ls) = subproblem1(
                &(rot(&kin.h.column(4).into(), q5) * kin.h.column(5)),
                &(r_01.transpose() * r_06 * kin.h.column(5)),
                &kin.h.column(1).into(),
            );

            let r_01 = rot(&kin.h.column(0).into(), q1);
            let r_45 = rot(&kin.h.column(4).into(), q5);
            let r_14 = rot(&kin.h.column(1).into(), theta14);

            let p_12: &Vector3<f64> = &kin.p.column(1).into();
            let p_23: &Vector3<f64> = &kin.p.column(2).into();
            let p_34: &Vector3<f64> = &kin.p.column(3).into();
            let p_45: &Vector3<f64> = &(kin.p.column(4) + kin.p.column(5));

            let d_inner = r_01.transpose() * p_16 - p_12 - r_14 * p_45;
            let d = d_inner.norm();

            let (theta_3, theta_3_is_ls) = subproblem3(&-p_34, p_23, &kin.h.column(1).into(), d);

            for q3 in theta_3.get_all() {
                let (q2, q2_is_ls) = subproblem1(
                    &(p_23 + rot(&kin.h.column(1).into(), q3) * p_34),
                    &d_inner,
                    &kin.h.column(1).into(),
                );

                let q4 = wrap_to_pi(theta14 - q2 - q3);

                let (q6, q6_is_ls) = subproblem1(
                    &kin.h.column(4).into(),
                    &(r_45.transpose()
                        * r_14.transpose()
                        * r_01.transpose()
                        * r_06
                        * kin.h.column(4)),
                    &kin.h.column(5).into(),
                );

                q.push(Vector6::new(q1, q2, q3, q4, q5, q6));
                is_ls.push(
                    theta1_is_ls
                        || theta5_is_ls
                        || theta_14_is_ls
                        || theta_3_is_ls
                        || q2_is_ls
                        || q6_is_ls,
                );
            }
        }
    }

    (q, is_ls)
}

pub fn three_parallel(
    r_06: &Matrix3<f64>,
    p_0t: &Vector3<f64>,
    kin: &Kinematics<6, 7>,
) -> (Vec<Vector6<f64>>, Vec<bool>) {
    let mut q = Vec::with_capacity(8);
    let mut is_ls = Vec::with_capacity(8);

    let p_16 = p_0t - kin.p.column(0) - r_06 * kin.p.column(6);

    let h_sp: [Vector3<f64>; 4] = [
        kin.h.column(1).into(),
        kin.h.column(1).into(),
        kin.h.column(1).into(),
        kin.h.column(1).into(),
    ];
    let k_sp: [Vector3<f64>; 4] = [
        -kin.h.column(0),
        kin.h.column(4).into(),
        -kin.h.column(0),
        kin.h.column(4).into(),
    ];
    let p_sp: [Vector3<f64>; 4] = [
        p_16,
        -kin.p.column(5),
        r_06 * kin.h.column(5),
        -kin.h.column(5),
    ];
    let d1 = (kin.h.column(1).transpose()
        * (kin.p.column(2) + kin.p.column(3) + kin.p.column(4) + kin.p.column(1)))[0];
    let d2 = 0.0;

    let theta15 = subproblem6(&h_sp, &k_sp, &p_sp, d1, d2);

    for (q1, q5) in theta15.get_all() {
        let r_01 = rot(&kin.h.column(0).into(), q1);
        let r_45 = rot(&kin.h.column(4).into(), q5);

        let (theta14, theta14_is_ls) = subproblem1(
            &(r_45 * kin.h.column(5)),
            &(r_01.transpose() * r_06 * kin.h.column(5)),
            &kin.h.column(1).into(),
        );
        let (q6, q6_is_ls) = subproblem1(
            &(r_45.transpose() * kin.h.column(1)),
            &(r_06.transpose() * r_01 * kin.h.column(1)),
            &-kin.h.column(5),
        );

        let r_14 = rot(&kin.h.column(1).into(), theta14);
        let d_inner = r_01.transpose() * p_16
            - kin.p.column(1)
            - r_14 * r_45 * kin.p.column(5)
            - r_14 * kin.p.column(4);
        let d = d_inner.norm();
        let (theta3, theta3_is_ls) = subproblem3(
            &-kin.p.column(3),
            &kin.p.column(2).into(),
            &kin.h.column(1).into(),
            d,
        );

        for q3 in theta3.get_all() {
            let (q2, q2_is_ls) = subproblem1(
                &(kin.p.column(2) + rot(&kin.h.column(1).into(), q3) * kin.p.column(3)),
                &d_inner,
                &kin.h.column(1).into(),
            );
            let q4 = wrap_to_pi(theta14 - q2 - q3);

            q.push(Vector6::new(q1, q2, q3, q4, q5, q6));
            is_ls.push(theta14_is_ls || theta3_is_ls || q2_is_ls || q6_is_ls);
        }
    }

    (q, is_ls)
}

pub fn two_parallel(
    r_06: &Matrix3<f64>,
    p_0t: &Vector3<f64>,
    kin: &Kinematics<6, 7>,
) -> (Vec<Vector6<f64>>, Vec<bool>) {
    let mut q = Vec::with_capacity(8);
    let mut is_ls = Vec::with_capacity(8);

    let p_16 = p_0t - kin.p.column(0) - r_06 * kin.p.column(6);

    fn t64_given_q1(
        r_06: &Matrix3<f64>,
        kin: &Kinematics<6, 7>,
        q1: f64,
        p_16: &Vector3<f64>,
    ) -> SolutionSet4<(f64, f64)> {
        let r_01 = rot(&kin.h.column(0).into(), q1);

        let h1: Vector3<f64> = (kin.h.column(1).transpose() * r_01.transpose() * r_06).transpose();
        let h2: Vector3<f64> = kin.h.column(1).into();
        let h3: Vector3<f64> = h1;
        let h4: Vector3<f64> = h2;
        let sp_h: [Vector3<f64>; 4] = [h1, h2, h3, h4];

        let sp_k: [Vector3<f64>; 4] = [
            -kin.h.column(5),
            kin.h.column(3).into(),
            -kin.h.column(5),
            kin.h.column(3).into(),
        ];

        let sp_p: [Vector3<f64>; 4] = [
            kin.p.column(5).into(),
            kin.p.column(4).into(),
            kin.h.column(4).into(),
            -kin.h.column(4),
        ];

        let d1 = (kin.h.column(1).transpose()
            * (r_01.transpose() * p_16 - kin.p.column(1) - kin.p.column(2) - kin.p.column(3)))[0];
        let d2 = 0.0;

        subproblem6(&sp_h, &sp_k, &sp_p, d1, d2)
    }

    let error_given_q1 = |q1: f64| {
        let mut error = Vector4::from_element(f64::INFINITY);
        let r_01 = rot(&kin.h.column(0).into(), q1);

        let h1: Vector3<f64> = (kin.h.column(1).transpose() * r_01.transpose() * r_06).transpose();
        let h2: Vector3<f64> = kin.h.column(1).into();
        let h3: Vector3<f64> = h1;
        let h4: Vector3<f64> = h2;
        let sp_h: [Vector3<f64>; 4] = [h1, h2, h3, h4];

        let sp_k: [Vector3<f64>; 4] = [
            -kin.h.column(5),
            kin.h.column(3).into(),
            -kin.h.column(5),
            kin.h.column(3).into(),
        ];

        let sp_p: [Vector3<f64>; 4] = [
            kin.p.column(5).into(),
            kin.p.column(4).into(),
            kin.h.column(4).into(),
            -kin.h.column(4),
        ];

        let d1 = (kin.h.column(1).transpose()
            * (r_01.transpose() * p_16 - kin.p.column(1) - kin.p.column(2) - kin.p.column(3)))[0];
        let d2 = 0.0;

        let t64 = subproblem6(&sp_h, &sp_k, &sp_p, d1, d2);

        for (i, (t6, t4)) in t64.get_all().into_iter().enumerate() {
            let r_34 = rot(&kin.h.column(3).into(), t4);
            let r_56 = rot(&kin.h.column(5).into(), t6);
            let (t23, _) = subproblem1(
                &(r_34 * kin.h.column(4)),
                &(r_01.transpose() * r_06 * r_56.transpose() * kin.h.column(4)),
                &kin.h.column(1).into(),
            );
            let r_13 = rot(&kin.h.column(1).into(), t23);

            // Relative to |p2| so the search's touch threshold is independent of the length unit.
            let l2 = kin.p.column(2).norm().max(f64::MIN_POSITIVE);
            error[i] = ((r_01.transpose() * p_16
                - kin.p.column(1)
                - r_13 * kin.p.column(3)
                - r_13 * r_34 * kin.p.column(4)
                - r_01.transpose() * r_06 * r_56.transpose() * kin.p.column(5))
            .norm()
                - l2)
                / l2;
        }

        error
    };

    let q1_vec = search_1d_bracketed(error_given_q1, -PI, PI, 200, 0.5);

    for (q1, i) in q1_vec {
        let t64 = t64_given_q1(r_06, kin, q1, &p_16).get_all();
        let q6 = t64[i].0;
        let q4 = t64[i].1;
        let r_01 = rot(&kin.h.column(0).into(), q1);
        let r_34 = rot(&kin.h.column(3).into(), q4);
        let r_56 = rot(&kin.h.column(5).into(), q6);

        let (t23, t23_is_ls) = subproblem1(
            &(r_34 * kin.h.column(4)),
            &(r_01.transpose() * r_06 * r_56.transpose() * kin.h.column(4)),
            &kin.h.column(1).into(),
        );
        let r_13 = rot(&kin.h.column(1).into(), t23);

        let (q2, q2_is_ls) = subproblem1(
            &kin.p.column(2).into(),
            &(r_01.transpose() * p_16
                - kin.p.column(1)
                - r_13 * kin.p.column(3)
                - r_13 * r_34 * kin.p.column(4)
                - r_01.transpose() * r_06 * r_56.transpose() * kin.p.column(5)),
            &kin.h.column(1).into(),
        );

        let (q5, q5_is_ls) = subproblem1(
            &(r_34.transpose() * kin.h.column(1)),
            &(r_56 * r_06.transpose() * r_01 * kin.h.column(1)),
            &-kin.h.column(4),
        );

        let q3 = wrap_to_pi(t23 - q2);

        q.push(Vector6::new(q1, q2, q3, q4, q5, q6));
        is_ls.push(t23_is_ls || q2_is_ls || q5_is_ls);
    }

    (q, is_ls)
}

pub fn two_intersecting(
    r_06: &Matrix3<f64>,
    p_0t: &Vector3<f64>,
    kin: &Kinematics<6, 7>,
) -> (Vec<Vector6<f64>>, Vec<bool>) {
    let mut q = Vec::with_capacity(8);
    let mut is_ls = Vec::with_capacity(8);

    let p_16 = p_0t - kin.p.column(0) - r_06 * kin.p.column(6);

    fn q_partial_given_q4(q4: f64, kin: &Kinematics<6, 7>, p_16: &Vector3<f64>) -> Matrix4<f64> {
        let mut q_partial = Matrix4::from_element(f64::NAN);

        let p_35_3 = kin.p.column(3) + rot(&kin.h.column(3).into(), q4) * kin.p.column(4);

        let t123 = subproblem5(
            &-kin.p.column(1),
            p_16,
            &kin.p.column(2).into(),
            &p_35_3,
            &-kin.h.column(0),
            &kin.h.column(1).into(),
            &kin.h.column(2).into(),
        );

        for (i, (q1, q2, q3)) in t123.get_all().into_iter().enumerate() {
            q_partial.set_column(i, &Vector4::new(q1, q2, q3, q4));
        }

        q_partial
    }

    let alignment_error_given_q4 = |q4: f64| {
        let mut error = Vector4::from_element(f64::INFINITY);

        let p_35_3 = kin.p.column(3) + rot(&kin.h.column(3).into(), q4) * kin.p.column(4);

        let t123 = subproblem5(
            &-kin.p.column(1),
            &p_16,
            &kin.p.column(2).into(),
            &p_35_3,
            &-kin.h.column(0),
            &kin.h.column(1).into(),
            &kin.h.column(2).into(),
        );

        for (i, (q1, q2, q3)) in t123.get_all().into_iter().enumerate() {
            let r_04 = rot(&kin.h.column(0).into(), q1)
                * rot(&kin.h.column(1).into(), q2)
                * rot(&kin.h.column(2).into(), q3)
                * rot(&kin.h.column(3).into(), q4);

            error[i] = (kin.h.column(4).transpose() * r_04.transpose() * r_06 * kin.h.column(5)
                - kin.h.column(4).transpose() * kin.h.column(5))[0];
        }

        error
    };

    let q4_vec = search_1d_bracketed(alignment_error_given_q4, -PI, PI, 200, 0.5);

    for (q4, i) in q4_vec {
        let q_partial: Vector4<f64> = q_partial_given_q4(q4, kin, &p_16).column(i).into();

        let r_04 = rot(&kin.h.column(0).into(), q_partial[0])
            * rot(&kin.h.column(1).into(), q_partial[1])
            * rot(&kin.h.column(2).into(), q_partial[2])
            * rot(&kin.h.column(3).into(), q_partial[3]);

        let (q5, q5_is_ls) = subproblem1(
            &kin.h.column(5).into(),
            &(r_04.transpose() * r_06 * kin.h.column(5)),
            &kin.h.column(4).into(),
        );
        let (q6, q6_is_ls) = subproblem1(
            &kin.h.column(4).into(),
            &(r_06.transpose() * r_04 * kin.h.column(4)),
            &-kin.h.column(5),
        );

        q.push(Vector6::new(
            q_partial[0],
            q_partial[1],
            q_partial[2],
            q_partial[3],
            q5,
            q6,
        ));

        is_ls.push(q5_is_ls || q6_is_ls);
    }

    (q, is_ls)
}

pub fn gen_six_dof(
    r_06: &Matrix3<f64>,
    p_0t: &Vector3<f64>,
    kin: &Kinematics<6, 7>,
) -> (Vec<Vector6<f64>>, Vec<bool>) {
    fn q_given_q12_k(
        q1: f64,
        q2: f64,
        k: usize,
        p16: &Vector3<f64>,
        r_06: &Matrix3<f64>,
        kin: &Kinematics<6, 7>,
    ) -> (Vector6<f64>, bool) {
        let p63 = rot(&-kin.h.column(1), q2) * (rot(&-kin.h.column(0), q1) * p16 - kin.p.column(1))
            - kin.p.column(2);
        let p = Vector3::z();

        let t345 = subproblem5(
            &-kin.p.column(3),
            &p63,
            &kin.p.column(4).into(),
            &kin.p.column(5).into(),
            &-kin.h.column(2),
            &kin.h.column(3).into(),
            &kin.h.column(4).into(),
        )
        .get_all();

        let (q3, q4, q5) = t345[k];

        let r05 = rot(&kin.h.column(0).into(), q1)
            * rot(&kin.h.column(1).into(), q2)
            * rot(&kin.h.column(2).into(), q3)
            * rot(&kin.h.column(3).into(), q4)
            * rot(&kin.h.column(4).into(), q5);

        let (q6, q6_is_ls) =
            subproblem1(&p, &(r05.transpose() * r_06 * p), &kin.h.column(5).into());

        (Vector6::new(q1, q2, q3, q4, q5, q6), q6_is_ls)
    }

    let mut q = Vec::with_capacity(8);
    let mut is_ls = Vec::with_capacity(8);

    let p16 = p_0t - kin.p.column(0) - r_06 * kin.p.column(6);

    let alignment_error_given_q12 = |q1: f64, q2: f64| {
        let mut error = Vector4::from_element(f64::INFINITY);

        let p63 = rot(&-kin.h.column(1), q2) * (rot(&-kin.h.column(0), q1) * p16 - kin.p.column(1))
            - kin.p.column(2);

        let t345 = subproblem5(
            &-kin.p.column(3),
            &p63,
            &kin.p.column(4).into(),
            &kin.p.column(5).into(),
            &-kin.h.column(2),
            &kin.h.column(3).into(),
            &kin.h.column(4).into(),
        );

        for (i, (q3, q4, q5)) in t345.get_all().into_iter().enumerate() {
            let r05 = rot(&kin.h.column(0).into(), q1)
                * rot(&kin.h.column(1).into(), q2)
                * rot(&kin.h.column(2).into(), q3)
                * rot(&kin.h.column(3).into(), q4)
                * rot(&kin.h.column(4).into(), q5);

            error[i] = (r05 * kin.h.column(5) - r_06 * kin.h.column(5)).norm();
        }

        error
    };

    let minima = search_2d(alignment_error_given_q12, (-PI, -PI), (PI, PI), 100);

    for (x0, x1, k) in minima {
        let (q_i, q_is_ls) = q_given_q12_k(x0, x1, k, &p16, r_06, kin);

        q.push(q_i);
        is_ls.push(q_is_ls);
    }

    (q, is_ls)
}

/// Two intersecting axes (5, 6) and two parallel axes (2, 3): the 1D search over q4 of
/// `two_intersecting`, with Subproblem 5 replaced by Subproblems 4, 3 and 1 because the
/// parallel pair makes the position problem planar (paper Table II footnote). Requires
/// `h2 == h3` exactly and `p5 == 0`; the branch index is `2 * shoulder + elbow`, which
/// stays continuous in q4 so sign changes are tracked per physical branch.
/// Initial q4 samples of the two_intersecting_two_parallel search.
const TITP_SAMPLES: usize = 128;

pub fn two_intersecting_two_parallel(
    r_06: &Matrix3<f64>,
    p_0t: &Vector3<f64>,
    kin: &Kinematics<6, 7>,
) -> (Vec<Vector6<f64>>, Vec<bool>) {
    let mut q = Vec::with_capacity(8);
    let mut is_ls = Vec::with_capacity(8);

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
    // Constants of the search: h6 in the base frame and the parts of the J1 lateral offset.
    let u6 = r_06 * h6;
    let d_12 = h2.dot(&(p1 + p2));
    let h5h6 = h5.dot(&h6);

    // Up to four (q1, q2, q3) for a given q4 as (sin, cos) pairs, slot 2 * shoulder + elbow,
    // None where absent or least squares. No transcendental call: the subproblems hand over
    // sin and cos directly and the rotations use them.
    type Sc3 = ((f64, f64), (f64, f64), (f64, f64));
    let q123_given_q4 = |s4: f64, c4: f64| -> [Option<Sc3>; 4] {
        let mut out = [None; 4];
        let p_35 = p3 + rot_vec_sc(&h4, s4, c4, &p4);
        let (t1, q1_ls) = subproblem4_sc(&h2, &p_16, &-h1, d_12 + h2.dot(&p_35));
        if q1_ls {
            return out;
        }
        for (i1, (s1, c1)) in t1.iter().enumerate() {
            let v = rot_vec_sc(&h1, -s1, c1, &p_16) - p1;
            let (t3, q3_ls) = subproblem3_sc(&-p_35, &p2, &h3, v.norm());
            if q3_ls {
                continue;
            }
            for (i3, (s3, c3)) in t3.iter().enumerate() {
                let ((s2, c2), _) = subproblem1_sc(&(p2 + rot_vec_sc(&h3, s3, c3, &p_35)), &v, &h2);
                out[2 * i1 + i3] = Some(((s1, c1), (s2, c2), (s3, c3)));
            }
        }
        out
    };

    // h5 . R_04^T R_06 h6 - h5 . h6, with R_04^T applied to the vector one axis at a time.
    let alignment_error_given_q4 = |q4: f64| -> Vector4<f64> {
        let mut error = Vector4::from_element(f64::INFINITY);
        let (s4, c4) = q4.sin_cos();
        for (i, sol) in q123_given_q4(s4, c4).iter().enumerate() {
            if let Some(((s1, c1), (s2, c2), (s3, c3))) = sol {
                let w = rot_vec_sc(&h1, -s1, *c1, &u6);
                let w = rot_vec_sc(&h2, -s2, *c2, &w);
                let w = rot_vec_sc(&h3, -s3, *c3, &w);
                let w = rot_vec_sc(&h4, -s4, c4, &w);
                error[i] = h5.dot(&w) - h5h6;
            }
        }
        error
    };

    for (q4, i) in search_1d_bracketed(alignment_error_given_q4, -PI, PI, TITP_SAMPLES, 0.5) {
        let (s4, c4) = q4.sin_cos();
        let Some(((s1, c1), (s2, c2), (s3, c3))) = q123_given_q4(s4, c4)[i] else {
            continue;
        };
        let (q1, q2, q3) = (s1.atan2(c1), s2.atan2(c2), s3.atan2(c3));
        let r_04 = rot(&h1, q1) * rot(&h2, q2) * rot(&h3, q3) * rot(&h4, q4);
        let (q5, q5_is_ls) = subproblem1(&h6, &(r_04.transpose() * r_06 * h6), &h5);
        let (q6, q6_is_ls) = subproblem1(&h5, &(r_06.transpose() * r_04 * h5), &-h6);
        q.push(Vector6::new(q1, q2, q3, q4, q5, q6));
        is_ls.push(q5_is_ls || q6_is_ls);
    }

    (q, is_ls)
}
