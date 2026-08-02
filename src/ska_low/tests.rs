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

/// Tests against the real EveryBeam coefficient files, which are not committed;
/// point `SKA_LOW_BEAM_DIR` at a directory of
/// `SKA-Low_Perturbed_Vogel_256_elem_FEKO_*.h5` to run them.
mod real_files {
    use super::*;
    use serial_test::serial;

    const REAL_L_MAX: usize = 5;
    const REAL_N_ELEMENTS: usize = 256;

    fn beam_dir() -> Option<String> {
        std::env::var("SKA_LOW_BEAM_DIR").ok()
    }

    fn beam(dir: &str) -> SkaLowBeam {
        // The station layout isn't in the beam files; these tests only need a
        // consistent one, so put every element at the origin and let the
        // element patterns speak for themselves.
        let offsets = Array2::zeros((3, REAL_N_ELEMENTS));
        SkaLowBeam::new(dir, offsets.view(), None, false).unwrap()
    }

    /// The file's schema is what the loader expects: /modes is (3, 35) of
    /// int32, giving l_max = 5, and /coefficients is (256, 35, 4) compound.
    #[test]
    #[serial]
    fn reads_the_everybeam_schema() {
        let Some(dir) = beam_dir() else { return };
        let beam = beam(&dir);
        assert_eq!(beam.num_elements(), REAL_N_ELEMENTS);
        assert!(!beam.get_freqs().is_empty());
        assert_abs_diff_eq!(beam.find_closest_freq(101e6), 100e6, epsilon = 1.0);

        let prepared = beam
            .prepare(100e6, &vec![c64::default(); REAL_N_ELEMENTS])
            .unwrap();
        match prepared {
            Prepared::Swe { coeffs, .. } => {
                assert_eq!(coeffs.l_max, REAL_L_MAX);
                assert_eq!(coeffs.coeffs.dim(), (REAL_N_ELEMENTS, 35, 4));
            }
            Prepared::Grid { .. } => panic!("expected a spherical-wave source"),
        }
    }

    /// The coefficients as this crate lays them out must produce what the
    /// reference kernel produces from the same file. This is the check that the
    /// (element, mode, component) axis order matches the flat
    /// `4 * (n * (n + 1) + m - 1)` indexing the kernel does.
    //
    // Expected values from a C program that reads the same file with the HDF5
    // C API and calls sdp_spherical_wave_pattern_feko from ska-sdp-func.
    #[test]
    #[serial]
    fn element_patterns_match_the_reference() {
        let Some(dir) = beam_dir() else { return };
        let beam = beam(&dir);
        let prepared = beam
            .prepare(100e6, &vec![c64::default(); REAL_N_ELEMENTS])
            .unwrap();
        let Prepared::Swe { coeffs, .. } = prepared else {
            panic!("expected a spherical-wave source")
        };

        let cases: [(usize, f64, f64, [c64; 4]); 4] = [
            (
                0,
                0.4,
                1.1,
                [
                    c64::new(0.30529999830290161, -0.26176162873213316),
                    c64::new(-0.47268519583593221, 0.646198630437258),
                    c64::new(0.1744977833984144, -0.22853724365298478),
                    c64::new(-0.33095124044588531, 0.65795940881489512),
                ],
            ),
            (
                1,
                0.4,
                1.1,
                [
                    c64::new(0.23465885348319929, -0.31153981262260899),
                    c64::new(-0.47948409276452481, 0.66619976605116815),
                    c64::new(0.17649833030723994, -0.31450511028730954),
                    c64::new(-0.35227071264967941, 0.55199859513848115),
                ],
            ),
            (
                255,
                1.2,
                -2.3,
                [
                    c64::new(-0.032028292714747719, 0.086761584350798338),
                    c64::new(-0.10826804460798332, -0.36075552076197842),
                    c64::new(-0.12354336405118108, 0.12071188484113851),
                    c64::new(0.046348894411093182, -0.41453121253883596),
                ],
            ),
            (
                128,
                0.0,
                0.0,
                [
                    c64::new(0.64866091637639678, -0.66256351938788594),
                    c64::new(-0.029630102870054954, 0.0080780906408140785),
                    c64::new(0.60751075492056672, -0.63903642072990496),
                    c64::new(0.025155482454602317, -0.016230465606812759),
                ],
            ),
        ];

        let root_n = crate::legendre::root_n_table(REAL_L_MAX);
        let mut workspace = vec![0.0; REAL_L_MAX + 2];
        for (element, theta, phi_x, expected) in cases {
            let p = swe::spherical_wave_pattern_feko(
                theta,
                [phi_x, phi_x + FRAC_PI_2],
                REAL_L_MAX,
                &root_n,
                coeffs.element(element),
                false,
                &mut workspace,
            );
            let got = [p.e_theta[0], p.e_phi[0], p.e_theta[1], p.e_phi[1]];
            for (got, expected) in got.into_iter().zip(expected) {
                assert_abs_diff_eq!(got.re, expected.re, epsilon = 1e-12);
                assert_abs_diff_eq!(got.im, expected.im, epsilon = 1e-12);
            }
        }
    }

    /// A 256-element station with the elements spread out: the beam must be
    /// finite, and steering the weights to a direction must maximise the
    /// response there.
    #[test]
    #[serial]
    fn station_beam_peaks_where_it_is_steered() {
        let Some(dir) = beam_dir() else { return };
        // Elements on a 20 m ring, which is enough structure for the geometric
        // phase to matter.
        let mut offsets = Array2::zeros((3, REAL_N_ELEMENTS));
        for i in 0..REAL_N_ELEMENTS {
            let angle = TAU * i as f64 / REAL_N_ELEMENTS as f64;
            offsets[[0, i]] = 20.0 * angle.cos();
            offsets[[1, i]] = 20.0 * angle.sin();
        }
        let beam = SkaLowBeam::new(&dir, offsets.view(), None, false).unwrap();

        let freq_hz = 100e6;
        let target = AzEl::from_radians(1.0, 1.0);
        // Conjugate the geometric phase towards the target to steer there.
        let k = [
            target.az.sin() * target.el.cos(),
            target.az.cos() * target.el.cos(),
            target.el.sin(),
        ];
        let wavenumber = TAU * freq_hz / VEL_C;
        let weights: Vec<c64> = (0..REAL_N_ELEMENTS)
            .map(|i| {
                let path: f64 = (0..3).map(|j| offsets[[j, i]] * k[j]).sum();
                c64::cis(-wavenumber * path)
            })
            .collect();

        let on = beam
            .calc_jones(target, freq_hz, &weights, None, false)
            .unwrap();
        let on_power: f64 = on.iter().map(|j| j.norm_sqr()).sum();
        assert!(on_power.is_finite() && on_power > 0.0);

        // Every other direction in a spread should be weaker.
        for (az, el) in [(0.0, 0.5), (2.0, 0.3), (-1.5, 1.2), (3.0, 0.8)] {
            let off = beam
                .calc_jones(AzEl::from_radians(az, el), freq_hz, &weights, None, false)
                .unwrap();
            let off_power: f64 = off.iter().map(|j| j.norm_sqr()).sum();
            assert!(
                off_power < on_power,
                "steered power {on_power} was not the maximum; ({az}, {el}) gave {off_power}"
            );
        }
    }

    /// Files written by numpy itself, rather than by the hand-built buffers
    /// the grid unit tests use, so that a bug shared between this crate's
    /// reader and its test writer can't hide. Point `SKA_LOW_GRID_DIR` at a
    /// directory holding both the .npy and .npz forms of the same data.
    #[test]
    #[serial]
    fn gridded_files_from_numpy() {
        let Some(dir) = std::env::var("SKA_LOW_GRID_DIR").ok() else {
            return;
        };
        let filebase = "HARP_SKALA41_randvogel_1m_4_elem_50ohm_";
        let n = 4;
        let offsets = Array2::zeros((3, n));
        let mut weights = vec![c64::default(); n];
        weights[0] = c64::new(1.0, 0.0);
        // az 90 deg (East) is EEP az 0, el 45 deg is za 45.
        let azel = AzEl::from_radians(90f64.to_radians(), 45f64.to_radians());

        // The four values numpy stored at that pixel, conjugated on load, with
        // the Y polarisation negated, then assembled theta-first with the phi
        // components negated.
        let expected = [
            c64::new(-0.06335825, 0.14406662),
            c64::new(0.20452249, -1.4110916),
            c64::new(0.13131122, 0.5075359),
            c64::new(0.3747996, 0.657074),
        ];

        for format in [GridFormat::Npy, GridFormat::Npz] {
            let beam =
                SkaLowBeam::new_from_grid(&dir, filebase, format, offsets.view(), None, 0.0, false)
                    .unwrap();
            assert_eq!(beam.get_freqs(), &[99e6]);
            let jones = beam.calc_jones(azel, 99e6, &weights, None, false).unwrap();
            for i in 0..4 {
                assert_abs_diff_eq!(jones[i].re, expected[i].re, epsilon = 1e-7);
                assert_abs_diff_eq!(jones[i].im, expected[i].im, epsilon = 1e-7);
            }
        }
    }

    /// Loading is cached per frequency, and different frequencies give
    /// different beams.
    #[test]
    #[serial]
    fn frequencies_are_distinct_and_cached() {
        let Some(dir) = beam_dir() else { return };
        let beam = beam(&dir);
        let freqs = beam.get_freqs().to_vec();
        if freqs.len() < 2 {
            return;
        }

        let mut weights = vec![c64::default(); REAL_N_ELEMENTS];
        weights[0] = c64::new(1.0, 0.0);
        let azel = AzEl::from_radians(1.1, 0.9);
        let a = beam
            .calc_jones(azel, freqs[0], &weights, None, false)
            .unwrap();
        let b = beam
            .calc_jones(azel, freqs[1], &weights, None, false)
            .unwrap();
        assert!(
            (0..4).any(|i| (a[i] - b[i]).norm() > 1e-9),
            "two different frequencies gave the same element pattern"
        );

        // The second call for the same frequency comes out of the cache.
        let a2 = beam
            .calc_jones(azel, freqs[0], &weights, None, false)
            .unwrap();
        assert_abs_diff_eq!(a, a2, epsilon = 1e-15);
        beam.empty_cache();
        let a3 = beam
            .calc_jones(azel, freqs[0], &weights, None, false)
            .unwrap();
        assert_abs_diff_eq!(a, a3, epsilon = 1e-15);
    }
}
