// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for Legendre polynomials.

/// Evaluates the Legendre polynomial Pm(n,m) at x. x must satisfy |x| <= 1.
///
/// # Arguments
///
/// * `n` - the maximum first index of the Legendre function, which must be at
///   least 0.
/// * `m` - the second index of the Legendre function, which must be at least 0,
///   and no greater than N.
/// * `x` - the points at which the function is to be evaluated.
///
// This code is a re-write of the C code here:
// https://people.sc.fsu.edu/~jburkardt/c_src/legendre_polynomial/legendre_polynomial.html
// The C code distributed under the GNU LGPL license, and thus this function is
// also licensed under the LGPL. A copy of the LGPL license can be found at
// https://www.gnu.org/licenses/lgpl-3.0.en.html
fn _legendre_values(n: usize, m: usize, x: &[f64]) -> Vec<f64> {
    let mm = x.len();
    let mut v = vec![0.0; mm * (n + 1)];

    // J = M is the first non-zero function.
    if m <= n {
        v[mm * m..mm * (m + 1)].fill(1.0);

        let mut fact = 1.0;
        for _ in 0..m {
            v.iter_mut()
                .skip(m * mm)
                .take(mm)
                .zip(x.iter().take(mm).copied())
                .for_each(|(v, x)| {
                    *v *= -fact * (1.0 - x * x).sqrt();
                });
            fact += 2.0;
        }
    }

    // J = M + 1 is the second nonzero function.
    if m < n {
        for (i, x) in (0..mm).zip(x.iter().copied()) {
            v[i + (m + 1) * mm] = x * (2 * m + 1) as f64 * v[i + m * mm];
        }
    }

    for j in (m + 2)..=n {
        for (i, x) in (0..mm).zip(x.iter().copied()) {
            let ji = j as isize;
            let mi = m as isize;
            v[i + j * mm] = ((2 * j - 1) as f64 * x * v[i + (j - 1) * mm]
                + (-ji - mi + 1) as f64 * v[i + (j - 2) * mm])
                / (ji - mi) as f64;
        }
    }

    v
}

/// The same as `legendre_values`, but only takes a single `x`.
// This code is a re-write of the C code here:
// https://people.sc.fsu.edu/~jburkardt/c_src/legendre_polynomial/legendre_polynomial.html
// The C code distributed under the GNU LGPL license, and thus this function is
// also licensed under the LGPL. A copy of the LGPL license can be found at
// https://www.gnu.org/licenses/lgpl-3.0.en.html
pub(crate) fn legendre_single(n: usize, m: usize, x: f64) -> Vec<f64> {
    let mut v = vec![0.0; n + 1];

    // J = M is the first non-zero function.
    if m <= n {
        v[m] = 1.0;

        let mut fact = 1.0;
        for _ in 0..m {
            v[m] *= -fact * (1.0 - x * x).sqrt();
            fact += 2.0;
        }
    }

    // J = M + 1 is the second nonzero function.
    if m < n {
        v[m + 1] = x * (2 * m + 1) as f64 * v[m];
    }

    for j in (m + 2)..=n {
        let ji = j as isize;
        let mi = m as isize;
        v[j] = ((2 * j - 1) as f64 * x * v[j - 1] + (-ji - mi + 1) as f64 * v[j - 2])
            / (ji - mi) as f64;
    }

    v
}

/// Returns list of Legendre polynomial values calculated up to order n_max.
// This function is a re-write of P1SIN within the RTS file mwa_tile.c.
pub(crate) fn p1sin(n_max: u8, theta: f64) -> (Vec<f64>, Vec<f64>) {
    let n_max = usize::from(n_max);
    let mut all_vals = vec![0.0; (n_max + 1) * (n_max + 2) / 2];

    let size = n_max * n_max + 2 * n_max;
    let mut p1sin_out = vec![0.0; size];
    let mut p1_out = vec![0.0; size];

    let (s_theta, u) = theta.sin_cos();
    let delta_u = 1e-6;

    let mut pm_in = u;
    let mut m_incr = 0;
    for m in 0..=n_max {
        let pm_vals = legendre_single(n_max, m, pm_in);
        for i in m..=n_max {
            if !(i == 0 && m == 0) {
                all_vals[(i - m) + m_incr] = pm_vals[i];
            }
        }
        m_incr += n_max - m + 1;
    }

    let mut p = vec![0.0; n_max + 1];
    let mut pm1 = vec![0.0; n_max + 1];
    let mut pm_sin = vec![0.0; n_max + 1];
    for n in 1..=n_max {
        m_incr = 0;
        for order in 0..=n {
            let index = n + m_incr;
            p[order] = all_vals[index];
            if order > 0 {
                pm1[order - 1] = all_vals[index];
            }
            if order == n {
                pm1[order] = 0.0;
            }
            pm_sin[order] = 0.0;
            m_incr += n_max - order;
        }

        // Floating point comparisons are pretty awful, but this is what the C++
        // code does, so...
        #[allow(clippy::float_cmp)]
        if u == 1.0 {
            pm_in = u - delta_u;
            let pm_vals = legendre_single(n, 0, pm_in);
            pm_sin[1] = -(p[0] - pm_vals[n]) / delta_u;
        } else if u == -1.0 {
            pm_in = u - delta_u;
            let pm_vals = legendre_single(n, 0, pm_in);
            pm_sin[1] = -(pm_vals[n] - p[0]) / delta_u;
        } else {
            pm_sin
                .iter_mut()
                .zip(p.iter().copied())
                .for_each(|(pm_sin, p)| *pm_sin = p / s_theta);
        }

        let ind_start = (n - 1) * (n - 1) + 2 * (n - 1);
        let ind_stop = n * n + 2 * n;
        for i in ind_start..ind_stop {
            let index = i - ind_start;
            let j = n.abs_diff(index);
            p1sin_out[i] = pm_sin[j];
            p1_out[i] = pm1[j];
        }
    }

    (p1sin_out, p1_out)
}

/// `sqrt(0..=2*n_max)`, the precomputed table `legendre_norm` indexes into.
pub(crate) fn root_n_table(n_max: usize) -> Vec<f64> {
    (0..=2 * n_max).map(|i| (i as f64).sqrt()).collect()
}

/// Computes fully-normalised associated Legendre functions, following the
/// MATLAB `legendre` implementation with `normalize = "norm"`.
///
/// # Arguments
///
/// * `n` - the degree, which must be at least 1.
/// * `cos_theta` - the point (x) at which to evaluate the Legendre function.
/// * `sin_theta` - the complementary value, `sqrt(1 - x^2)`.
/// * `root_n` - `sqrt(0..=2*n)`, from [`root_n_table`].
/// * `y` - the output, of length at least `n + 1`; `y[m]` is the normalised
///   `P_n^m(cos_theta)`.
///
// This function is a re-write of sdp_legendre_norm within
// src/ska-sdp-func/math/sdp_legendre_polynomial.h of ska-sdp-func, which is
// distributed under the BSD-3-Clause license by the SKA Organisation and the
// University of Oxford. The strided (num_points, idx_point) output of the
// original is dropped; here `y` is contiguous.
#[allow(clippy::float_cmp)]
pub(crate) fn legendre_norm(
    n: usize,
    cos_theta: f64,
    sin_theta: f64,
    root_n: &[f64],
    y: &mut [f64],
) {
    let tol = f64::from(f32::MIN_POSITIVE).sqrt();
    let root_n_plus_half = (n as f64 + 0.5).sqrt();
    let twocot = if sin_theta > 0.0 {
        -2.0 * cos_theta / sin_theta
    } else {
        0.0
    };
    let sn = if sin_theta == 0.0 {
        0.0
    } else {
        let s = sin_theta.powi(n as i32);
        if n % 2 == 0 {
            s
        } else {
            -s
        }
    };

    y[..=n].fill(0.0);

    // Underflow branch: used only when cos_theta is very very close to 1.
    if sin_theta > 0.0 && sn.abs() <= tol {
        let tstart = f64::from(f32::EPSILON);
        let v = 9.2 - tol.ln() / (n as f64 * sin_theta);
        let w = 1.0 / v.ln();
        let m1d = 1.0 + n as f64 * sin_theta * v * w * (1.0058 + w * (3.819 - w * 12.173));
        let m1 = (m1d.floor() as isize).clamp(1, n as isize) as usize;
        let sgn_m1 = if m1 % 2 == 1 { 1.0 } else { -1.0 };
        let sgn_n1 = if (n + 1) % 2 == 1 { 1.0 } else { -1.0 };
        y[m1 - 1] = if cos_theta < 0.0 {
            sgn_n1 * tstart
        } else {
            sgn_m1 * tstart
        };
        let mut sumsq = tol;
        for m in (0..m1 - 1).rev() {
            let a = (m + 1) as f64 * twocot * y[m + 1];
            let b = root_n[n + m + 2] * root_n[n - m - 1] * y[m + 2];
            y[m] = (a - b) / (root_n[n + m + 1] * root_n[n - m]);
            sumsq += y[m] * y[m];
        }
        let scale = 1.0 / (2.0 * sumsq - y[0] * y[0]).sqrt();
        y[..=m1].iter_mut().for_each(|y| *y *= scale);
    }

    // Non-underflow branch.
    if cos_theta != 1.0 && sn.abs() >= tol {
        let mut c = 1.0;
        for k in 1..=n {
            c *= 1.0 - 1.0 / (2.0 * k as f64);
        }
        y[n] = c.sqrt() * sn;
        y[n - 1] = y[n] * twocot * n as f64 / root_n[2 * n];

        for m in (0..n.saturating_sub(1)).rev() {
            let a = (m + 1) as f64 * twocot * y[m + 1];
            let b = root_n[n + m + 2] * root_n[n - m - 1] * y[m + 2];
            y[m] = (a - b) / (root_n[n + m + 1] * root_n[n - m]);
        }
    }

    // Polar case (cos_theta == +/- 1).
    if sin_theta == 0.0 {
        y[0] = cos_theta.powi(n as i32);
    }

    y[..=n].iter_mut().enumerate().for_each(|(m, y)| {
        let sgn = if m % 2 == 0 { 1.0 } else { -1.0 };
        *y *= root_n_plus_half * sgn;
    });
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4};

    use super::*;
    use approx::*;
    use ndarray::prelude::*;

    #[test]
    fn test_legendre_single() {
        let result = Array1::from(legendre_single(5, 0, -1.0));
        let expected = array![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        assert_abs_diff_eq!(result, expected, epsilon = 1e-6);

        let result = Array1::from(legendre_single(5, 0, -0.25));
        let expected = array![
            1.0,
            -0.25,
            -0.40625,
            0.3359375,
            0.15771484375,
            -0.3397216796875
        ];
        assert_abs_diff_eq!(result, expected, epsilon = 1e-6);

        let result = Array1::from(legendre_single(5, 0, 0.25));
        let expected = array![
            1.0,
            0.25,
            -0.40625,
            -0.3359375,
            0.15771484375,
            0.3397216796875
        ];
        assert_abs_diff_eq!(result, expected, epsilon = 1e-6);

        let result = Array1::from(legendre_single(5, 0, 1.0));
        let expected = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        assert_abs_diff_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn legendre_values_n5m0() {
        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let result = Array1::from(_legendre_values(5, 0, &x));
        let expected = array![
            1.000000, 1.000000, 1.000000, 1.000000, 1.000000, -1.000000, -0.500000, 0.000000,
            0.500000, 1.000000, 1.000000, -0.125000, -0.500000, -0.125000, 1.000000, -1.000000,
            0.437500, -0.000000, -0.437500, 1.000000, 1.000000, -0.289062, 0.375000, -0.289062,
            1.000000, -1.000000, -0.089844, 0.000000, 0.089844, 1.000000,
        ];
        assert_abs_diff_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn legendre_values_n5m1() {
        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let result = Array1::from(_legendre_values(5, 1, &x));
        let expected = array![
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -0.000000, -0.866025, -1.000000,
            -0.866025, -0.000000, 0.000000, 1.299038, -0.000000, -1.299038, -0.000000, 0.000000,
            -0.324760, 1.500000, -0.324760, 0.000000, -0.000000, -1.353165, 0.000000, 1.353165,
            0.000000, 0.000000, 1.928260, -1.875000, 1.928260, 0.000000,
        ];
        assert_abs_diff_eq!(result, expected, epsilon = 1e-6);
    }

    // Expected values from sdp_legendre_norm in ska-sdp-func, compiled and run
    // over the same inputs.
    fn check_legendre_norm(n: usize, theta: f64, expected: &[f64]) {
        let mut y = vec![0.0; n + 1];
        let (sin_theta, cos_theta) = theta.sin_cos();
        legendre_norm(n, cos_theta, sin_theta, &root_n_table(n), &mut y);
        assert_abs_diff_eq!(
            Array1::from(y),
            Array1::from(expected.to_vec()),
            epsilon = 1e-12
        );
    }

    #[test]
    fn legendre_norm_n1() {
        check_legendre_norm(1, 0.7, &[0.936_736_546_300_602_4, 0.557_908_882_715_098_6]);
    }

    #[test]
    fn legendre_norm_n4() {
        check_legendre_norm(
            4,
            0.7,
            &[
                -0.682_085_334_814_557_4,
                0.639_741_100_234_545_7,
                1.0770257203098461,
                0.641_578_874_944_802,
                0.191_058_283_520_412_2,
            ],
        );
    }

    #[test]
    fn legendre_norm_n8() {
        check_legendre_norm(
            8,
            1.234,
            &[
                -0.789_955_190_511_052_8,
                -0.23884092441762633,
                0.781_170_135_463_941_7,
                0.380_642_086_730_243_9,
                -0.716_062_501_501_500_4,
                -0.687_023_551_522_400_1,
                0.42558107842215626,
                1.139364905679676,
                0.8135150420097278,
            ],
        );
    }

    #[test]
    fn legendre_norm_obtuse() {
        check_legendre_norm(
            8,
            3.0,
            &[
                1.9537801460512567,
                -1.456_266_908_426_77,
                0.460_615_529_002_849_5,
                -0.091_251_902_181_680_49,
                0.012764396511760562,
                -0.0013215075762835816,
                0.00010210587746311471,
                -5.702_572_474_257_961e-6,
                2.032204982090008e-07,
            ],
        );
    }

    // sin_theta is tiny but non-zero, so sin_theta^n underflows.
    #[test]
    fn legendre_norm_underflow_branch() {
        let mut expected = vec![0.0; 17];
        expected[0] = 4.062_019_202_315_218;
        expected[1] = 3.349_626_844_887_709e-6;
        expected[2] = 1.375999636627549e-12;
        expected[3] = 3.740312486944613e-19;
        check_legendre_norm(16, 1e-7, &expected);
    }

    // theta = pi doesn't give sin_theta == 0, so this is the underflow branch
    // with cos_theta < 0.
    #[test]
    fn legendre_norm_pi() {
        let mut expected = vec![0.0; 9];
        expected[0] = 2.9154759474226504;
        check_legendre_norm(8, std::f64::consts::PI, &expected);
    }

    #[test]
    fn legendre_norm_polar() {
        let mut expected = vec![0.0; 9];
        expected[0] = 2.9154759474226504;
        check_legendre_norm(8, 0.0, &expected);
    }

    #[test]
    fn p1sin_16_0() {
        let (p1sin_out, p1_out) = p1sin(16, 0.0);
        for (i, &p) in p1sin_out.iter().enumerate() {
            match i {
                0 | 2 => assert_abs_diff_eq!(p, -1.000000, epsilon = 1e-6),
                4 | 6 => assert_abs_diff_eq!(p, -2.999999, epsilon = 1e-6),
                10 | 12 => assert_abs_diff_eq!(p, -5.999993, epsilon = 1e-6),
                18 | 20 => assert_abs_diff_eq!(p, -9.999978, epsilon = 1e-6),
                28 | 30 => assert_abs_diff_eq!(p, -14.999948, epsilon = 1e-6),
                40 | 42 => assert_abs_diff_eq!(p, -20.999895, epsilon = 1e-6),
                // Because we've matched specific indices above, this range will
                // only match indices that have not already been matched. Use
                // the range of the biggest index above, as not all indices
                // after it will be 0.
                0..=42 => assert_abs_diff_eq!(p, 0.000000, epsilon = 1e-6),
                _ => (),
            }
        }
        for (i, &p) in p1_out.iter().enumerate() {
            match i {
                1 | 5 | 9 | 13 | 17 | 21 | 25 | 33 | 37 | 45 => {
                    assert_abs_diff_eq!(p, -0.000000, epsilon = 1e-6)
                }
                0..=45 => assert_abs_diff_eq!(p, 0.000000, epsilon = 1e-6),
                _ => (),
            }
        }
    }

    #[test]
    fn p1sin_5_0() {
        let (p1sin_out, p1_out) = p1sin(5, 0.0);
        for (i, &p) in p1sin_out.iter().enumerate() {
            match i {
                0 | 2 => assert_abs_diff_eq!(p, -1.000000, epsilon = 1e-6),
                4 | 6 => assert_abs_diff_eq!(p, -2.999999, epsilon = 1e-6),
                10 | 12 => assert_abs_diff_eq!(p, -5.999993, epsilon = 1e-6),
                18 | 20 => assert_abs_diff_eq!(p, -9.999978, epsilon = 1e-6),
                28 | 30 => assert_abs_diff_eq!(p, -14.999948, epsilon = 1e-6),
                0..=30 => assert_abs_diff_eq!(p, 0.000000, epsilon = 1e-6),
                _ => (),
            }
        }
        for (i, &p) in p1_out.iter().enumerate() {
            match i {
                1 | 5 | 9 | 13 | 17 | 21 | 25 | 33 => {
                    assert_abs_diff_eq!(p, -0.000000, epsilon = 1e-6)
                }
                0..=33 => assert_abs_diff_eq!(p, 0.000000, epsilon = 1e-6),
                _ => (),
            }
        }
    }

    #[test]
    fn p1sin_5_0785398160() {
        let (p1sin_out, p1_out) = p1sin(5, FRAC_PI_4);
        for (i, &p) in p1sin_out.iter().enumerate() {
            match i {
                0 | 2 => assert_abs_diff_eq!(p, -1.000000, epsilon = 1e-6),
                1 => assert_abs_diff_eq!(p, 1.000000, epsilon = 1e-6),
                3 | 7 => assert_abs_diff_eq!(p, 2.121320, epsilon = 1e-6),
                4 | 6 => assert_abs_diff_eq!(p, -2.121320, epsilon = 1e-6),
                5 => assert_abs_diff_eq!(p, 0.353553, epsilon = 1e-6),
                8 | 14 => assert_abs_diff_eq!(p, -7.500000, epsilon = 1e-6),
                9 | 13 => assert_abs_diff_eq!(p, 7.500000, epsilon = 1e-6),
                _ => (),
            }
        }
        for (i, &p) in p1_out.iter().enumerate() {
            match i {
                1 => assert_abs_diff_eq!(p, -FRAC_1_SQRT_2, epsilon = 1e-6),
                4 | 6 => assert_abs_diff_eq!(p, 1.500000, epsilon = 1e-6),
                5 => assert_abs_diff_eq!(p, -1.500000, epsilon = 1e-6),
                9 | 13 => assert_abs_diff_eq!(p, -5.303301, epsilon = 1e-6),
                10 | 12 => assert_abs_diff_eq!(p, 5.303301, epsilon = 1e-6),
                11 => assert_abs_diff_eq!(p, -1.590990, epsilon = 1e-6),
                0..=13 => assert_abs_diff_eq!(p, 0.000000, epsilon = 1e-6),
                _ => (),
            }
        }
    }
}
