// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests for the SKA-Low beam's C API.
//!
//! These check that the C API agrees with the native API on the same inputs, so
//! that argument marshalling (interleaved complex weights, the row-major
//! coordinate axes, the optional latitude pointer) can't silently drift.

use std::ffi::CString;

use approx::assert_abs_diff_eq;
use marlu::Jones;

use super::*;
use crate::ska_low::tests::{write_beam_dir, N_ELEMENTS};

fn jones_from_buf(buf: &[f64; 8]) -> Jones<f64> {
    Jones::from([
        c64::new(buf[0], buf[1]),
        c64::new(buf[2], buf[3]),
        c64::new(buf[4], buf[5]),
        c64::new(buf[6], buf[7]),
    ])
}

/// Element offsets in the (3, n) layout the C API expects, flattened
/// row-major.
const OFFSETS: [f64; 3 * N_ELEMENTS] = [
    1.0, -3.0, 0.0, // East
    2.0, 0.5, -4.0, // North
    0.0, 0.1, 0.0, // Up
];

const WEIGHTS: [f64; 2 * N_ELEMENTS] = [0.5, -0.2, -1.0, 0.7, 0.1, 0.1];

fn native_weights() -> Vec<c64> {
    WEIGHTS
        .chunks_exact(2)
        .map(|c| c64::new(c[0], c[1]))
        .collect()
}

#[test]
fn ffi_new_and_calc_jones_match_the_native_api() {
    let dir = write_beam_dir("ffi_calc");
    let dir_c = CString::new(dir.0.to_str().unwrap()).unwrap();

    let mut beam = std::ptr::null_mut();
    let ret = unsafe {
        new_ska_low_beam(
            dir_c.as_ptr(),
            N_ELEMENTS as c_uint,
            OFFSETS.as_ptr(),
            std::ptr::null(),
            0,
            &mut beam,
        )
    };
    assert_eq!(ret, 0);
    assert!(!beam.is_null());

    let (az, za) = (1.1, 0.6);
    let latitude = -0.4660608448386394;
    let mut buf = [0.0; 8];
    let ret = unsafe {
        ska_low_calc_jones(
            beam,
            az,
            za,
            100e6,
            WEIGHTS.as_ptr(),
            &latitude,
            1,
            buf.as_mut_ptr(),
        )
    };
    assert_eq!(ret, 0);

    let offsets = Array2::from_shape_vec((3, N_ELEMENTS), OFFSETS.to_vec()).unwrap();
    let native = crate::ska_low::SkaLowBeam::new(&dir.0, offsets.view(), None, false).unwrap();
    let expected = native
        .calc_jones(
            AzEl::from_radians(az, std::f64::consts::FRAC_PI_2 - za),
            100e6,
            &native_weights(),
            Some(latitude),
            true,
        )
        .unwrap();
    assert_abs_diff_eq!(jones_from_buf(&buf), expected, epsilon = 1e-12);

    unsafe { free_ska_low_beam(beam) };
}

/// A null latitude pointer must mean "no parallactic correction", not a
/// dereference of nothing.
#[test]
fn ffi_null_latitude_skips_the_correction() {
    let dir = write_beam_dir("ffi_nolat");
    let dir_c = CString::new(dir.0.to_str().unwrap()).unwrap();
    let mut beam = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            new_ska_low_beam(
                dir_c.as_ptr(),
                N_ELEMENTS as c_uint,
                OFFSETS.as_ptr(),
                std::ptr::null(),
                0,
                &mut beam,
            )
        },
        0
    );

    let mut buf = [0.0; 8];
    assert_eq!(
        unsafe {
            ska_low_calc_jones(
                beam,
                1.1,
                0.6,
                100e6,
                WEIGHTS.as_ptr(),
                std::ptr::null(),
                1,
                buf.as_mut_ptr(),
            )
        },
        0
    );

    let offsets = Array2::from_shape_vec((3, N_ELEMENTS), OFFSETS.to_vec()).unwrap();
    let native = crate::ska_low::SkaLowBeam::new(&dir.0, offsets.view(), None, false).unwrap();
    let expected = native
        .calc_jones(
            AzEl::from_radians(1.1, std::f64::consts::FRAC_PI_2 - 0.6),
            100e6,
            &native_weights(),
            None,
            true,
        )
        .unwrap();
    assert_abs_diff_eq!(jones_from_buf(&buf), expected, epsilon = 1e-12);

    unsafe { free_ska_low_beam(beam) };
}

/// The array entry point must agree with the single-direction one.
#[test]
fn ffi_calc_jones_array_matches_single() {
    let dir = write_beam_dir("ffi_array");
    let dir_c = CString::new(dir.0.to_str().unwrap()).unwrap();
    let mut beam = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            new_ska_low_beam(
                dir_c.as_ptr(),
                N_ELEMENTS as c_uint,
                OFFSETS.as_ptr(),
                std::ptr::null(),
                0,
                &mut beam,
            )
        },
        0
    );

    let az = [0.1, 1.1, -2.0];
    let za = [0.2, 0.6, 1.4];
    let mut many = [0.0; 24];
    assert_eq!(
        unsafe {
            ska_low_calc_jones_array(
                beam,
                3,
                az.as_ptr(),
                za.as_ptr(),
                100e6,
                WEIGHTS.as_ptr(),
                std::ptr::null(),
                0,
                many.as_mut_ptr(),
            )
        },
        0
    );

    for i in 0..3 {
        let mut one = [0.0; 8];
        assert_eq!(
            unsafe {
                ska_low_calc_jones(
                    beam,
                    az[i],
                    za[i],
                    100e6,
                    WEIGHTS.as_ptr(),
                    std::ptr::null(),
                    0,
                    one.as_mut_ptr(),
                )
            },
            0
        );
        let slice: [f64; 8] = many[i * 8..(i + 1) * 8].try_into().unwrap();
        assert_abs_diff_eq!(
            jones_from_buf(&slice),
            jones_from_buf(&one),
            epsilon = 1e-12
        );
    }

    unsafe { free_ska_low_beam(beam) };
}

/// The coordinate axes arrive row-major; a quarter turn about Up must land on
/// the same beam as offsets given directly in East/North/Up.
#[test]
fn ffi_coordinate_axes_are_row_major() {
    let dir = write_beam_dir("ffi_axes");
    let dir_c = CString::new(dir.0.to_str().unwrap()).unwrap();
    // p -> North, q -> West, r -> Up.
    let axes = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    let mut beam = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            new_ska_low_beam(
                dir_c.as_ptr(),
                N_ELEMENTS as c_uint,
                OFFSETS.as_ptr(),
                axes.as_ptr(),
                0,
                &mut beam,
            )
        },
        0
    );
    let mut buf = [0.0; 8];
    assert_eq!(
        unsafe {
            ska_low_calc_jones(
                beam,
                1.1,
                0.6,
                100e6,
                WEIGHTS.as_ptr(),
                std::ptr::null(),
                0,
                buf.as_mut_ptr(),
            )
        },
        0
    );

    let offsets = Array2::from_shape_vec((3, N_ELEMENTS), OFFSETS.to_vec()).unwrap();
    let native = crate::ska_low::SkaLowBeam::new(
        &dir.0,
        offsets.view(),
        Some([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        false,
    )
    .unwrap();
    let expected = native
        .calc_jones(
            AzEl::from_radians(1.1, std::f64::consts::FRAC_PI_2 - 0.6),
            100e6,
            &native_weights(),
            None,
            false,
        )
        .unwrap();
    assert_abs_diff_eq!(jones_from_buf(&buf), expected, epsilon = 1e-12);

    unsafe { free_ska_low_beam(beam) };
}

#[test]
fn ffi_freq_and_element_functions() {
    let dir = write_beam_dir("ffi_freqs");
    let dir_c = CString::new(dir.0.to_str().unwrap()).unwrap();
    let mut beam = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            new_ska_low_beam(
                dir_c.as_ptr(),
                N_ELEMENTS as c_uint,
                OFFSETS.as_ptr(),
                std::ptr::null(),
                0,
                &mut beam,
            )
        },
        0
    );

    let mut freqs = std::ptr::null();
    let mut num_freqs = 0;
    unsafe { ska_low_get_freqs(beam, &mut freqs, &mut num_freqs) };
    assert_eq!(num_freqs, 1);
    assert_abs_diff_eq!(unsafe { *freqs }, 100e6);
    assert_abs_diff_eq!(unsafe { ska_low_closest_freq(beam, 50e6) }, 100e6);
    assert_eq!(unsafe { ska_low_get_num_elements(beam) }, N_ELEMENTS);

    unsafe { free_ska_low_beam(beam) };
}

/// A bad directory must set the error string rather than panic.
#[test]
fn ffi_reports_a_missing_directory() {
    let dir_c = CString::new("/definitely/not/a/beam/directory").unwrap();
    let mut beam = std::ptr::null_mut();
    let ret = unsafe {
        new_ska_low_beam(
            dir_c.as_ptr(),
            N_ELEMENTS as c_uint,
            OFFSETS.as_ptr(),
            std::ptr::null(),
            0,
            &mut beam,
        )
    };
    assert_ne!(ret, 0);
    assert!(crate::ffi::hb_last_error_length() > 0);
}
