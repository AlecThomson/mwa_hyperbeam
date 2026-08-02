// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests for the SKA-Low beam.
//!
//! These build their own beam files rather than depending on a data file, so
//! that the station summation can be checked in isolation from any particular
//! set of coefficients.

use std::path::PathBuf;

use approx::assert_abs_diff_eq;
use hdf5_metno::H5Type;
use marlu::constants::VEL_C;

use super::*;

const L_MAX: usize = 4;
pub(super) const N_ELEMENTS: usize = 3;

#[derive(H5Type, Clone, Copy)]
#[repr(C)]
struct H5Complex {
    r: f64,
    i: f64,
}

/// The same deterministic stand-in coefficients that the kernel tests use, made
/// to differ per element.
fn fake_coeff(element: usize, mode: usize, comp: usize) -> c64 {
    let k = (4 * mode + comp + 17 * element) as f64;
    c64::new((0.7 * k + 0.3).sin(), (1.3 * k - 0.9).cos())
}

/// A directory holding one synthetic beam file, removed on drop.
pub(super) struct TempBeamDir(pub(super) PathBuf);

impl Drop for TempBeamDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(super) fn write_beam_dir(name: &str) -> TempBeamDir {
    let dir = std::env::temp_dir().join(format!("hyperbeam_ska_low_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let file = dir.join("SKA-Low_Synthetic_Vogel_3_elem_FEKO_100.00000_MHz.h5");
    let h5 = hdf5_metno::File::create(&file).unwrap();

    // Only the last element of /modes is read, and it must be l_max.
    let n_modes = L_MAX * (L_MAX + 2);
    let mut modes = Array2::<i64>::zeros((3, n_modes));
    modes[[2, n_modes - 1]] = L_MAX as i64;
    h5.new_dataset::<i64>()
        .shape((3, n_modes))
        .create("modes")
        .unwrap()
        .write(&modes)
        .unwrap();

    let coeffs = Array3::from_shape_fn((N_ELEMENTS, n_modes, 4), |(e, mode, comp)| {
        let z = fake_coeff(e, mode, comp);
        H5Complex { r: z.re, i: z.im }
    });
    h5.new_dataset::<H5Complex>()
        .shape((N_ELEMENTS, n_modes, 4))
        .create("coefficients")
        .unwrap()
        .write(&coeffs)
        .unwrap();

    TempBeamDir(dir)
}

fn offsets(cols: [[f64; 3]; N_ELEMENTS]) -> Array2<f64> {
    let mut a = Array2::zeros((3, N_ELEMENTS));
    for (i, col) in cols.iter().enumerate() {
        for j in 0..3 {
            a[[j, i]] = col[j];
        }
    }
    a
}

/// One element with a unit weight and no offset must reproduce the bare
/// spherical-wave kernel, phi components negated.
#[test]
fn single_element_matches_the_kernel() {
    let dir = write_beam_dir("single");
    let beam =
        SkaLowBeam::new(&dir.0, offsets([[0.0; 3]; N_ELEMENTS]).view(), None, false).unwrap();

    let azel = AzEl::from_radians(1.1, 0.9);
    let weights = [c64::new(1.0, 0.0), c64::default(), c64::default()];
    let jones = beam.calc_jones(azel, 100e6, &weights, None, false).unwrap();

    let phi_x = FRAC_PI_2 - azel.az;
    let coeffs = Array2::from_shape_fn((L_MAX * (L_MAX + 2), 4), |(mode, comp)| {
        fake_coeff(0, mode, comp)
    });
    let mut workspace = vec![0.0; L_MAX + 2];
    let expected = swe::spherical_wave_pattern_feko(
        azel.za(),
        [phi_x, phi_x + FRAC_PI_2],
        L_MAX,
        &crate::legendre::root_n_table(L_MAX),
        coeffs.view(),
        false,
        &mut workspace,
    );

    assert_abs_diff_eq!(jones[0], expected.e_theta[0], epsilon = 1e-12);
    assert_abs_diff_eq!(jones[1], -expected.e_phi[0], epsilon = 1e-12);
    assert_abs_diff_eq!(jones[2], expected.e_theta[1], epsilon = 1e-12);
    assert_abs_diff_eq!(jones[3], -expected.e_phi[1], epsilon = 1e-12);
}

/// Two elements with identical coefficients but different offsets differ only
/// by the geometric phase, so their sum is one pattern times `1 + e^(i k.d)`.
#[test]
fn geometric_phase() {
    let dir = write_beam_dir("phase");
    // 5 m East, 0, 0. Elements 0 and 1 have different coefficients, so use
    // element 0 twice by giving element 1 a zero weight and comparing the two
    // single-element responses instead.
    let at_origin = SkaLowBeam::new(
        &dir.0,
        offsets([[0.0; 3], [0.0; 3], [0.0; 3]]).view(),
        None,
        false,
    )
    .unwrap();
    let displaced = SkaLowBeam::new(
        &dir.0,
        offsets([[5.0, 0.0, 0.0], [0.0; 3], [0.0; 3]]).view(),
        None,
        false,
    )
    .unwrap();

    let azel = AzEl::from_radians(1.1, 0.9);
    let weights = [c64::new(1.0, 0.0), c64::default(), c64::default()];
    let a = at_origin
        .calc_jones(azel, 100e6, &weights, None, false)
        .unwrap();
    let b = displaced
        .calc_jones(azel, 100e6, &weights, None, false)
        .unwrap();

    // The East component of the direction unit vector.
    let east = azel.az.sin() * azel.el.cos();
    let phase = c64::cis(TAU * 100e6 / VEL_C * 5.0 * east);
    for i in 0..4 {
        assert_abs_diff_eq!(b[i], a[i] * phase, epsilon = 1e-12);
    }
}

/// The station response is the weighted sum over elements.
#[test]
fn station_sum_is_linear_in_the_weights() {
    let dir = write_beam_dir("sum");
    let beam = SkaLowBeam::new(
        &dir.0,
        offsets([[1.0, 2.0, 0.0], [-3.0, 0.5, 0.1], [0.0, -4.0, 0.0]]).view(),
        None,
        false,
    )
    .unwrap();

    let azel = AzEl::from_radians(-0.4, 0.3);
    let w = [c64::new(0.5, -0.2), c64::new(-1.0, 0.7), c64::new(0.1, 0.1)];
    let all = beam.calc_jones(azel, 100e6, &w, None, false).unwrap();

    let mut sum = Jones::default();
    for i in 0..N_ELEMENTS {
        let mut one = [c64::default(); N_ELEMENTS];
        one[i] = w[i];
        sum += beam.calc_jones(azel, 100e6, &one, None, false).unwrap();
    }
    for i in 0..4 {
        assert_abs_diff_eq!(all[i], sum[i], epsilon = 1e-12);
    }
}

/// A quarter-turn of the station's coordinate axes about Up must send an
/// offset along p to one along q.
#[test]
fn coordinate_axes_rotate_the_offsets() {
    let dir = write_beam_dir("axes");
    // p -> North, q -> West, r -> Up.
    let axes = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let rotated = SkaLowBeam::new(
        &dir.0,
        offsets([[5.0, 0.0, 0.0]; N_ELEMENTS]).view(),
        Some(axes),
        false,
    )
    .unwrap();
    // The same station described directly in East/North/Up.
    let direct = SkaLowBeam::new(
        &dir.0,
        offsets([[0.0, 5.0, 0.0]; N_ELEMENTS]).view(),
        None,
        false,
    )
    .unwrap();

    let azel = AzEl::from_radians(1.1, 0.9);
    let w = [c64::new(1.0, 0.0); N_ELEMENTS];
    let a = rotated.calc_jones(azel, 100e6, &w, None, false).unwrap();
    let b = direct.calc_jones(azel, 100e6, &w, None, false).unwrap();
    for i in 0..4 {
        assert_abs_diff_eq!(a[i], b[i], epsilon = 1e-12);
    }
}

#[test]
fn wrong_weight_count_is_an_error() {
    let dir = write_beam_dir("weights");
    let beam =
        SkaLowBeam::new(&dir.0, offsets([[0.0; 3]; N_ELEMENTS]).view(), None, false).unwrap();
    let azel = AzEl::from_radians(0.0, 1.0);
    assert!(beam
        .calc_jones(azel, 100e6, &[c64::new(1.0, 0.0)], None, false)
        .is_err());
}

#[test]
fn freqs_and_closest() {
    let dir = write_beam_dir("freqs");
    let beam =
        SkaLowBeam::new(&dir.0, offsets([[0.0; 3]; N_ELEMENTS]).view(), None, false).unwrap();
    assert_eq!(beam.get_freqs(), vec![100e6]);
    assert_abs_diff_eq!(beam.find_closest_freq(50e6), 100e6);
    assert_eq!(beam.num_elements(), N_ELEMENTS);
}
