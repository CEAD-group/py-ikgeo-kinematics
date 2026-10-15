use {
    crate::subproblems::auxiliary::rot,
    nalgebra::{ArrayStorage, Const, Matrix, Matrix3, Vector2, Vector3, U1, U3, U7, U8},
    std::f64::consts::{PI, TAU},
};

pub type Matrix3x7<T> = Matrix<T, U3, U7, ArrayStorage<T, 3, 7>>;
pub type Matrix3x8<T> = Matrix<T, U3, U8, ArrayStorage<T, 3, 8>>;

pub type Vector<T, const N: usize> = Matrix<T, Const<N>, U1, ArrayStorage<f64, N, 1>>;

#[derive(Debug, Clone)]
pub struct Kinematics<const C1: usize, const C2: usize> {
    // TODO: somehow statically ensure that C2 - C1 = 1
    pub h: Matrix<f64, U3, Const<C1>, ArrayStorage<f64, 3, C1>>,
    pub p: Matrix<f64, U3, Const<C2>, ArrayStorage<f64, 3, C2>>,
}

impl<const C1: usize, const C2: usize> Default for Kinematics<C1, C2> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const C1: usize, const C2: usize> Kinematics<C1, C2> {
    pub fn new() -> Self {
        Self {
            h: Matrix::<f64, U3, Const<C1>, ArrayStorage<f64, 3, C1>>::zeros(),
            p: Matrix::<f64, U3, Const<C2>, ArrayStorage<f64, 3, C2>>::zeros(),
        }
    }

    pub fn forward_kinematics(
        &self,
        theta: &Matrix<f64, Const<C1>, U1, ArrayStorage<f64, C1, 1>>,
    ) -> (Matrix3<f64>, Vector3<f64>) {
        let mut p: Vector3<f64> = self.p.column(0).into();
        let mut r = Matrix3::identity();

        for (i, &t) in theta.iter().enumerate() {
            r *= rot(&self.h.column(i).into(), t);
            p += r * self.p.column(i + 1);
        }

        (r, p)
    }
}

impl Kinematics<7, 8> {
    pub fn forward_kinematics_partial(
        &self,
        q_n: f64,
        n: usize,
        r_6t: &Matrix3<f64>,
    ) -> (Kinematics<6, 7>, Matrix3<f64>) {
        let mut kin_new: Kinematics<6, 7> = Kinematics::new();
        let r_n = rot(&self.h.column(n).into(), q_n);

        for i in 0..self.h.ncols() {
            if i > n {
                kin_new.h.set_column(i - 1, &(r_n * self.h.column(i)));
            } else {
                kin_new.h.set_column(i, &self.h.column(i));
            }
        }

        for i in 0..self.p.ncols() {
            if i == n {
                kin_new
                    .p
                    .set_column(i, &(self.p.column(i) + r_n * self.p.column(i + 1)));
            } else if i > n + 1 {
                kin_new.p.set_column(i - 1, &(r_n * self.p.column(i)));
            } else {
                kin_new.p.set_column(i, &self.p.column(i));
            }
        }

        (kin_new, r_n * r_6t)
    }
}

pub fn wrap_to_pi(theta: f64) -> f64 {
    (theta + PI).rem_euclid(TAU) - PI
}

fn find_zero<const N: usize, F: Fn(f64) -> Vector<f64, N>>(
    f: F,
    left: f64,
    right: f64,
    i: usize,
) -> Option<f64> {
    const ITERATIONS: usize = 100;
    const EPSILON: f64 = 1e-5;

    let mut x_left = left;
    let mut x_right = right;

    let mut y_left = f(x_left)[i];
    let mut y_right = f(x_right)[i];

    for _ in 0..ITERATIONS {
        let delta = y_right - y_left;

        if delta.abs() < EPSILON {
            break;
        }

        let x_0 = x_left - y_left * (x_right - x_left) / delta;
        let y_0 = f(x_0)[i];

        if !y_0.is_finite() {
            return None;
        }

        if (y_left < 0.0) != (y_0 < 0.0) {
            x_left = x_0;
            y_left = y_0;
        } else {
            x_right = x_0;
            y_right = y_0;
        }
    }

    if left <= x_left && x_left <= right {
        Some(x_left)
    } else {
        None
    }
}

pub fn search_1d<const N: usize, F: Fn(f64) -> Vector<f64, N>>(
    f: F,
    left: f64,
    right: f64,
    initial_samples: usize,
) -> Vec<(f64, usize)> {
    const CROSS_THRESHOLD: f64 = 0.1;

    let delta = (right - left) / initial_samples as f64;

    let mut last_v = f(left);
    let mut x = left + delta;

    let mut zeros = Vec::with_capacity(8);

    for _ in 0..initial_samples {
        let v = f(x);

        for (i, (&y, &last_y)) in v.iter().zip(last_v.iter()).enumerate() {
            if (y < 0.0) != (last_y < 0.0)
                && y.abs() < CROSS_THRESHOLD
                && last_y.abs() < CROSS_THRESHOLD
            {
                if let Some(z) = find_zero(&f, x - delta, x, i) {
                    zeros.push((z, i));
                }
            }
        }

        last_v = v;
        x += delta;
    }

    zeros
}

/// Local minimum of `f` near `x0` by Nelder-Mead on two parameters.
///
/// Replaces the `argmin` solver upstream used so the crate has no optimiser
/// dependency. Same parameters as upstream: reflection 1, expansion 2,
/// contraction 0.5, shrink 0.5, stop when the simplex values agree to `tol`.
fn nelder_mead_2d<F: Fn(f64, f64) -> f64>(
    f: F,
    x0: Vector2<f64>,
    step: (f64, f64),
    tol: f64,
    max_iters: usize,
) -> Vector2<f64> {
    let mut simplex = [
        x0,
        Vector2::new(x0.x + step.0, x0.y),
        Vector2::new(x0.x, x0.y - step.1),
    ];
    let mut values = simplex.map(|v| f(v.x, v.y));

    for _ in 0..max_iters {
        // Order so that index 0 is best and index 2 is worst.
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        simplex = order.map(|i| simplex[i]);
        values = order.map(|i| values[i]);

        let mean = (values[0] + values[1] + values[2]) / 3.0;
        let sd = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 3.0).sqrt();
        if sd < tol || !values[2].is_finite() {
            break;
        }

        let centroid = (simplex[0] + simplex[1]) / 2.0;
        let reflected = centroid + (centroid - simplex[2]);
        let f_reflected = f(reflected.x, reflected.y);

        if f_reflected < values[0] {
            let expanded = centroid + 2.0 * (centroid - simplex[2]);
            let f_expanded = f(expanded.x, expanded.y);
            if f_expanded < f_reflected {
                simplex[2] = expanded;
                values[2] = f_expanded;
            } else {
                simplex[2] = reflected;
                values[2] = f_reflected;
            }
            continue;
        }

        if f_reflected < values[1] {
            simplex[2] = reflected;
            values[2] = f_reflected;
            continue;
        }

        let contracted = if f_reflected < values[2] {
            centroid + 0.5 * (reflected - centroid)
        } else {
            centroid + 0.5 * (simplex[2] - centroid)
        };
        let f_contracted = f(contracted.x, contracted.y);
        if f_contracted < values[2].min(f_reflected) {
            simplex[2] = contracted;
            values[2] = f_contracted;
            continue;
        }

        for i in 1..3 {
            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0]);
            values[i] = f(simplex[i].x, simplex[i].y);
        }
    }

    let mut best = 0;
    for i in 1..3 {
        if values[i] < values[best] {
            best = i;
        }
    }
    simplex[best]
}

/// Grid-then-minimise 2D search (paper section V-E): sample `n` by `n`, take
/// the smallest sample below the threshold, clear its connected blob, repeat,
/// then refine every seed with Nelder-Mead.
pub fn search_2d<const N: usize, F: Fn(f64, f64) -> Vector<f64, N>>(
    f: F,
    min: (f64, f64),
    max: (f64, f64),
    n: usize,
) -> Vec<(f64, f64, usize)> {
    fn minimum<const N: usize>(mesh: &[Vector<f64, N>], n: usize) -> Option<(usize, usize, usize)> {
        let mut min_v = f64::INFINITY;
        let mut min_i = 0;
        let mut min_j = 0;
        let mut min_k = 0;

        for i in 0..n {
            for j in 0..n {
                for (k, &v) in mesh[i + j * n].iter().enumerate() {
                    if v < min_v {
                        min_v = v;
                        min_i = i;
                        min_j = j;
                        min_k = k;
                    }
                }
            }
        }

        if min_v.is_finite() {
            Some((min_i, min_j, min_k))
        } else {
            None
        }
    }

    // Iterative flood fill on the torus; the recursive upstream version can overflow the stack.
    fn clear_blob<const N: usize>(
        mesh: &mut [Vector<f64, N>],
        i: usize,
        j: usize,
        k: usize,
        n: usize,
    ) {
        let mut stack = vec![(i as isize, j as isize)];

        while let Some((i, j)) = stack.pop() {
            let i = i.rem_euclid(n as isize);
            let j = j.rem_euclid(n as isize);
            let cell = &mut mesh[i as usize + j as usize * n][k];

            if cell.is_nan() {
                continue;
            }
            *cell = f64::NAN;

            stack.push((i + 1, j));
            stack.push((i - 1, j));
            stack.push((i, j + 1));
            stack.push((i, j - 1));
        }
    }

    const N_MAX_MINIMA: usize = 1000;
    const MIN_THRESHOLD: f64 = 1e-1;

    let delta0 = (max.0 - min.0) / n as f64;
    let delta1 = (max.1 - min.1) / n as f64;

    let x0_vals: Vec<f64> = (0..n).map(|m| m as f64 * delta0 + min.0).collect();
    let x1_vals: Vec<f64> = (0..n).map(|m| m as f64 * delta1 + min.1).collect();

    let mut mesh = vec![Vector::zeros(); n * n];
    let mut minima: Vec<(f64, f64, usize)> = Vec::new();

    for i in 0..n {
        for j in 0..n {
            mesh[i + j * n] =
                f(x0_vals[i], x1_vals[j]).map(|v| if v > MIN_THRESHOLD { f64::NAN } else { v });
        }
    }

    while minima.len() < N_MAX_MINIMA {
        if let Some((i, j, k)) = minimum(&mesh, n) {
            minima.push((x0_vals[i], x1_vals[j], k));
            clear_blob(&mut mesh, i, j, k, n);
        } else {
            break;
        }
    }

    assert!(minima.len() < N_MAX_MINIMA, "Too many minima found");

    for (x0, x1, k) in &mut minima {
        let k_fixed = *k;
        let best = nelder_mead_2d(
            |a, b| f(a, b)[k_fixed],
            Vector2::new(*x0, *x1),
            (delta0 / 2.0, delta1 / 2.0),
            1e-12,
            10_000,
        );
        *x0 = best[0];
        *x1 = best[1];
    }

    minima
}

/// Illinois false position on a sign-changing bracket, for one component of `f`.
fn refine_bracket<const N: usize, F: Fn(f64) -> Vector<f64, N>>(
    f: &F,
    i: usize,
    mut a: f64,
    mut b: f64,
    mut fa: f64,
    mut fb: f64,
) -> f64 {
    const ITERATIONS: usize = 60;
    let mut side = 0;
    let mut x = a;
    for _ in 0..ITERATIONS {
        let next = (a * fb - b * fa) / (fb - fa);
        if !next.is_finite() {
            break;
        }
        let fx = f(next)[i];
        if !fx.is_finite() {
            break;
        }
        x = next;
        if fx.abs() < 1e-15 || (b - a).abs() < 1e-14 {
            break;
        }
        if (fx < 0.0) == (fa < 0.0) {
            a = x;
            fa = fx;
            if side == -1 {
                // Illinois: halve the stale end so the bracket keeps shrinking.
                fb *= 0.5;
            }
            side = -1;
        } else {
            b = x;
            fb = fx;
            if side == 1 {
                fa *= 0.5;
            }
            side = 1;
        }
    }
    x
}

/// Golden-section search for the point of `f[i]` closest to zero on [a, b], where both ends
/// have the same sign. A minimum that crosses zero yields two brackets and two roots; one
/// that does not is returned as a tangent candidate for the polish step.
fn minimise_dip<const N: usize, F: Fn(f64) -> Vector<f64, N>>(
    f: &F,
    i: usize,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    zeros: &mut Vec<(f64, usize)>,
) {
    const ITERATIONS: usize = 16;
    const INV_PHI: f64 = 0.618_033_988_749_895;
    let sign = if fa < 0.0 { -1.0 } else { 1.0 };
    let g = |x: f64| -> f64 {
        let y = f(x)[i];
        if y.is_finite() {
            sign * y
        } else {
            f64::INFINITY
        }
    };
    let (mut lo, mut hi) = (a, b);
    let mut x1 = hi - INV_PHI * (hi - lo);
    let mut x2 = lo + INV_PHI * (hi - lo);
    let (mut g1, mut g2) = (g(x1), g(x2));
    let (mut best_x, mut best_g) = if g1 < g2 { (x1, g1) } else { (x2, g2) };
    for _ in 0..ITERATIONS {
        if g1 < g2 {
            hi = x2;
            x2 = x1;
            g2 = g1;
            x1 = hi - INV_PHI * (hi - lo);
            g1 = g(x1);
        } else {
            lo = x1;
            x1 = x2;
            g1 = g2;
            x2 = lo + INV_PHI * (hi - lo);
            g2 = g(x2);
        }
        let (x, gx) = if g1 < g2 { (x1, g1) } else { (x2, g2) };
        if gx < best_g {
            best_x = x;
            best_g = gx;
        }
        if best_g < 0.0 {
            break;
        }
    }
    if best_g < 0.0 {
        let fm = sign * best_g;
        zeros.push((refine_bracket(f, i, a, best_x, fa, fm), i));
        zeros.push((refine_bracket(f, i, best_x, b, fm, fb), i));
    } else if best_g.is_finite() {
        zeros.push((best_x, i));
    }
}

/// Sign changes and shallow dips along finite samples `pts` (sorted by x) of component `i`.
fn scan_samples<const N: usize, F: Fn(f64) -> Vector<f64, N>>(
    f: &F,
    i: usize,
    pts: &[(f64, f64)],
    touch: f64,
    zeros: &mut Vec<(f64, usize)>,
) {
    let n = pts.len();
    for k in 1..n {
        let ((x0, y0), (x1, y1)) = (pts[k - 1], pts[k]);
        if (y0 < 0.0) != (y1 < 0.0) {
            zeros.push((refine_bracket(f, i, x0, x1, y0, y1), i));
        } else if k + 1 < n {
            let (x2, y2) = pts[k + 1];
            if y1.abs() < touch
                && y1.abs() <= y0.abs()
                && y1.abs() < y2.abs()
                && (y0 < 0.0) == (y2 < 0.0)
            {
                // A dip that does not cross at the samples: a close root pair or a tangent root.
                minimise_dip(f, i, x0, x2, y0, y2, zeros);
            }
        }
    }
    // A run end that is still falling may hide a dip between the last two samples.
    if n >= 2 {
        let ((x0, y0), (x1, y1)) = (pts[n - 2], pts[n - 1]);
        if (y0 < 0.0) == (y1 < 0.0) && y1.abs() < touch && y1.abs() <= y0.abs() {
            minimise_dip(f, i, x0, x1, y0, y1, zeros);
        }
        let ((x0, y0), (x1, y1)) = (pts[0], pts[1]);
        if (y0 < 0.0) == (y1 < 0.0) && y0.abs() < touch && y0.abs() <= y1.abs() {
            minimise_dip(f, i, x0, x1, y0, y1, zeros);
        }
    }
}

/// Bracketing 1D search for solvers whose error components are continuous per branch
/// and infinite where the branch does not exist. Every sign change between finite
/// neighbouring samples is refined with Illinois false position, and a sample where
/// |f| dips to a local minimum below `touch` is returned too so tangent roots reach the
/// polish step. Where a branch ends between two samples its end is located by bisection
/// and the last stretch is resampled densely towards the end, because the two branches
/// born at a fold separate like a square root and their roots cluster there.
/// Returns (x, component).
#[allow(clippy::needless_range_loop)]
pub fn search_1d_bracketed<const N: usize, F: Fn(f64) -> Vector<f64, N>>(
    f: F,
    left: f64,
    right: f64,
    initial_samples: usize,
    touch: f64,
) -> Vec<(f64, usize)> {
    const EDGE_BISECTIONS: usize = 16;
    const EDGE_SUBSAMPLES: usize = 8;
    let delta = (right - left) / initial_samples as f64;
    let xs: Vec<f64> = (0..=initial_samples)
        .map(|k| left + k as f64 * delta)
        .collect();
    let ys: Vec<Vector<f64, N>> = xs.iter().map(|&x| f(x)).collect();
    let mut zeros = Vec::with_capacity(8);

    for i in 0..N {
        let mut run: Vec<(f64, f64)> = Vec::with_capacity(initial_samples + 1);
        for k in 0..=initial_samples {
            let y = ys[k][i];
            if y.is_finite() {
                run.push((xs[k], y));
            }
            // A run of finite samples ends here, or at the far end of the grid.
            if y.is_finite() && k != initial_samples {
                continue;
            }
            if run.is_empty() {
                continue;
            }
            scan_samples(&f, i, &run, touch, &mut zeros);
            run.clear();
        }
        for k in 1..=initial_samples {
            let (y0, y1) = (ys[k - 1][i], ys[k][i]);
            if y0.is_finite() == y1.is_finite() {
                continue;
            }
            let (x_in, y_in, x_out) = if y0.is_finite() {
                (xs[k - 1], y0, xs[k])
            } else {
                (xs[k], y1, xs[k - 1])
            };
            let (mut lo, mut y_lo, mut hi) = (x_in, y_in, x_out);
            for _ in 0..EDGE_BISECTIONS {
                let mid = 0.5 * (lo + hi);
                let y_mid = f(mid)[i];
                if y_mid.is_finite() {
                    lo = mid;
                    y_lo = y_mid;
                } else {
                    hi = mid;
                }
            }
            // Quadratic spacing from the branch end so the sqrt-like start is resolved.
            let mut pts: Vec<(f64, f64)> = Vec::with_capacity(EDGE_SUBSAMPLES + 1);
            pts.push((lo, y_lo));
            for j in 1..EDGE_SUBSAMPLES {
                let s = (j as f64 / EDGE_SUBSAMPLES as f64).powi(2);
                let x = lo + (x_in - lo) * s;
                let y = f(x)[i];
                if y.is_finite() {
                    pts.push((x, y));
                }
            }
            pts.push((x_in, y_in));
            if x_in < lo {
                pts.reverse();
            }
            scan_samples(&f, i, &pts, touch, &mut zeros);
        }
    }

    zeros
}
