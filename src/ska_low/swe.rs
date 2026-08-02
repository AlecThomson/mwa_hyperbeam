// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The FEKO spherical-wave expansion kernel.

use std::f64::consts::PI;

use ndarray::prelude::*;
use num_complex::Complex64 as c64;

use crate::{
    constants::J_POWER_TABLE,
    legendre::legendre_norm,
    ska_low::{ElementPattern, N_FEEDS},
};

/// Free-space impedance \[ohms\], post-2019 definition.
const ETA_0: f64 = 376.730313668;

/// Evaluate the spherical-wave expansion of one element at one direction.
///
/// `coeffs` is `(l_max * (l_max + 2), 4)` for a single element, its second axis
/// being `[X_te, X_tm, Y_te, Y_tm]`. `root_n` is
/// [`crate::legendre::root_n_table`] for `n_max`, and `workspace` must hold at
/// least `n_max + 2` elements.
///
/// `use_ticra_convention` conjugates the coefficients, scales them by
/// `sqrt(8 pi)` and mirrors m; it is a property of how the coefficients were
/// generated.
//
// This function is a re-write of sdp_spherical_wave_pattern_feko within
// src/ska-sdp-func/station_beam/sdp_element_spherical_wave_feko.cpp of
// ska-sdp-func, which is distributed under the BSD-3-Clause license by the SKA
// Organisation and the University of Oxford. The method there is credited to
// Matlab code by David Davidson and Adrian Sutinjo.
//
// The original's NaN fast-path is dropped; NaN inputs propagate through the
// arithmetic to the same result.
pub(super) fn spherical_wave_pattern_feko(
    theta: f64,
    phi: [f64; N_FEEDS],
    n_max: usize,
    root_n: &[f64],
    coeffs: ArrayView2<c64>,
    use_ticra_convention: bool,
    workspace: &mut [f64],
) -> ElementPattern {
    let coeff_scale = if use_ticra_convention {
        (8.0 * PI).sqrt()
    } else {
        1.0
    };

    // Avoid divide-by-zero.
    let theta = theta.max(1e-5);
    let (sin_theta, cos_theta) = theta.sin_cos();
    let cos_phi = phi.map(f64::cos);
    let sin_phi = phi.map(f64::sin);

    let mut e_theta = [c64::default(); N_FEEDS];
    let mut e_phi = [c64::default(); N_FEEDS];

    // Loop over degree of spherical wave.
    for n in 1..=n_max {
        legendre_norm(n, cos_theta, sin_theta, root_n, workspace);
        workspace[n + 1] = 0.0;
        let sqrt_n = (n as f64 * (n + 1) as f64).sqrt();
        let i_n = J_POWER_TABLE[n % 4];
        let i_np1 = J_POWER_TABLE[(n + 1) % 4];

        // cos(m phi) and sin(m phi) for m = 0, advanced by recurrence below.
        let mut cos_m_phi = [1.0; N_FEEDS];
        let mut sin_m_phi = [0.0; N_FEEDS];

        for abs_m in 0..=n {
            // Constants that depend on abs_m only.
            let cmn_constant = ((n + abs_m + 1) as f64 * (n - abs_m) as f64).sqrt();
            let np_1 = cmn_constant * workspace[abs_m + 1];
            let np_ds = workspace[abs_m] / sin_theta;
            let np_diff = np_ds * abs_m as f64 * cos_theta - np_1;

            // Do both +m and -m, but abs_m == 0 only once.
            let signed_ms: &[i32] = if abs_m == 0 {
                &[0]
            } else {
                &[abs_m as i32, -(abs_m as i32)]
            };
            for &m in signed_ms {
                let sign = if m >= 0 || abs_m % 2 == 0 { 1.0 } else { -1.0 };
                let np_ds_m = np_ds * f64::from(m);
                let m_indx = if use_ticra_convention { -m } else { m };
                let mode = (n * (n + 1)) as i32 + m_indx - 1;
                let mode = coeffs.index_axis(Axis(0), mode as usize);

                for a in 0..N_FEEDS {
                    let (te, tm) = if use_ticra_convention {
                        (
                            coeff_scale * mode[2 * a].conj(),
                            coeff_scale * mode[2 * a + 1].conj(),
                        )
                    } else {
                        (coeff_scale * mode[2 * a], coeff_scale * mode[2 * a + 1])
                    };

                    // e^(i m phi), scaled.
                    let sin_m_phi_m = if m >= 0 { sin_m_phi[a] } else { -sin_m_phi[a] };
                    let phi_const = c64::new(cos_m_phi[a], sin_m_phi_m) * (sign / sqrt_n);

                    e_theta[a] += phi_const * i_n * (tm * np_diff - np_ds_m * te);
                    e_phi[a] += phi_const * i_np1 * (np_ds_m * tm - te * np_diff);
                }
            }

            // Advance to cos((m + 1) phi), sin((m + 1) phi).
            for a in 0..N_FEEDS {
                let (c, s) = (cos_m_phi[a], sin_m_phi[a]);
                cos_m_phi[a] = cos_phi[a] * c - sin_phi[a] * s;
                sin_m_phi[a] = sin_phi[a] * c + cos_phi[a] * s;
            }
        }
    }

    let sqrt_fac = (ETA_0 / (2.0 * PI)).sqrt();
    ElementPattern {
        e_theta: e_theta.map(|e| sqrt_fac * e),
        e_phi: e_phi.map(|e| sqrt_fac * e),
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::legendre::root_n_table;

    /// Deterministic stand-in coefficients, so that the kernel can be checked
    /// against the reference implementation without a beam file.
    fn fake_coeffs(n_max: usize) -> Array2<c64> {
        Array2::from_shape_fn((n_max * (n_max + 2), 4), |(mode, comp)| {
            let k = (4 * mode + comp) as f64;
            c64::new((0.7 * k + 0.3).sin(), (1.3 * k - 0.9).cos())
        })
    }

    // Expected values from sdp_spherical_wave_pattern_feko in ska-sdp-func, fed
    // the same coefficients. Ordered [X e_theta, X e_phi, Y e_theta, Y e_phi],
    // which is the reference implementation's output order.
    fn check(theta: f64, phi_x: f64, n_max: usize, ticra: bool, expected: [c64; 4]) {
        let mut workspace = vec![0.0; n_max + 2];
        let p = spherical_wave_pattern_feko(
            theta,
            [phi_x, phi_x + std::f64::consts::FRAC_PI_2],
            n_max,
            &root_n_table(n_max),
            fake_coeffs(n_max).view(),
            ticra,
            &mut workspace,
        );
        let got = [p.e_theta[0], p.e_phi[0], p.e_theta[1], p.e_phi[1]];
        for (got, expected) in got.into_iter().zip(expected) {
            assert_abs_diff_eq!(got.re, expected.re, epsilon = 1e-10);
            assert_abs_diff_eq!(got.im, expected.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn feko_n_max_3() {
        check(
            0.4,
            1.1,
            3,
            false,
            [
                c64::new(-9.341_471_692_874_16, 17.023_178_843_028_2),
                c64::new(-4.590180942879852, 11.253_735_888_818_13),
                c64::new(0.2228738140280896, 18.997699479415108),
                c64::new(33.205721525667926, -5.094144959892966),
            ],
        );
    }

    #[test]
    fn feko_n_max_8() {
        check(
            0.4,
            1.1,
            8,
            false,
            [
                c64::new(56.434760499739184, -0.301713648929257),
                c64::new(-17.048707746820515, -16.132057617849714),
                c64::new(18.206_310_641_161_53, -42.587184140215655),
                c64::new(-52.228_590_512_106_16, 0.673229413036714),
            ],
        );
    }

    #[test]
    fn feko_ticra_convention() {
        check(
            0.4,
            1.1,
            8,
            true,
            [
                c64::new(64.432_043_352_315_16, -181.49901642858725),
                c64::new(-117.619_020_035_755_6, 31.109555082151886),
                c64::new(-104.473_639_058_658_6, 272.35344529497587),
                c64::new(201.01786121585434, 268.804_667_950_316),
            ],
        );
    }

    #[test]
    fn feko_obtuse_theta() {
        check(
            1.5,
            -2.3,
            12,
            false,
            [
                c64::new(105.587_335_328_083_1, 49.735991717152544),
                c64::new(69.245_839_660_080_53, -93.055_933_528_373_05),
                c64::new(13.934889366854378, 29.541657544117154),
                c64::new(27.663390498709028, 11.419891939569421),
            ],
        );
    }

    // theta == 0 is clamped to 1e-5 to avoid dividing by sin(theta).
    #[test]
    fn feko_zenith() {
        check(
            0.0,
            0.0,
            5,
            false,
            [
                c64::new(-6.512950703474875, -12.173_874_089_256_84),
                c64::new(-10.754594752454212, -1.617757963885789),
                c64::new(-18.893_131_608_105_57, 17.207091662845986),
                c64::new(18.217244076078245, -4.229623284018038),
            ],
        );
    }
}
