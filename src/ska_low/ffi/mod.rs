// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for allowing other languages to talk to this Rust library's SKA-Low
//! beam code. See the examples directory for usage.

#[cfg(test)]
mod tests;

use std::{
    ffi::CStr,
    os::raw::{c_char, c_uint},
    slice,
};

use marlu::AzEl;
use ndarray::prelude::*;
use num_complex::Complex64 as c64;

use super::{GridFormat, SkaLowBeam};
use crate::ffi::update_last_error;

/// Record an error and return the non-zero exit code the C API expects.
fn fail(msg: String) -> i32 {
    update_last_error(msg);
    1
}

/// Complex element weights are passed as interleaved (real, imaginary) doubles,
/// as complex numbers can't cross the FFI boundary.
unsafe fn weights_from_ffi(weights: *const f64, num_elements: usize) -> Vec<c64> {
    slice::from_raw_parts(weights, 2 * num_elements)
        .chunks_exact(2)
        .map(|c| c64::new(c[0], c[1]))
        .collect()
}

unsafe fn offsets_from_ffi(
    element_offsets: *const f64,
    num_elements: usize,
) -> Result<Array2<f64>, String> {
    let s = slice::from_raw_parts(element_offsets, 3 * num_elements);
    Array2::from_shape_vec((3, num_elements), s.to_vec())
        .map_err(|e| format!("Couldn't use the element offsets: {e}"))
}

unsafe fn str_from_ffi(s: *const c_char, what: &str) -> Result<String, String> {
    CStr::from_ptr(s)
        .to_str()
        .map(str::to_string)
        .map_err(|_| format!("{what} isn't valid UTF-8"))
}

/// Create a new SKA-Low beam from a directory of FEKO spherical-wave
/// coefficient files.
///
/// Element offsets, station orientation and element flags are not in the beam
/// files; the caller supplies them, e.g. from the MSv3 `PHASED_ARRAY` subtable.
/// A flagged element is one with a weight of zero.
///
/// # Arguments
///
/// * `beam_dir` - a null-terminated path to a directory of
///   `..._FEKO_<freq>_MHz.h5` files.
/// * `num_elements` - the number of elements in the station.
/// * `element_offsets` - a pointer to a `3 * num_elements` array of element
///   offsets in metres, three rows of (East, North, Up), unless
///   `coordinate_axes` is given.
/// * `coordinate_axes` - an optional pointer to a 9-element array; the station
///   axes (p, q, r) expressed in (East, North, Up), row-major. If null, the
///   offsets are taken to already be in (East, North, Up).
/// * `use_ticra_convention` - a boolean indicating whether the coefficients
///   follow the TICRA convention.
/// * `ska_low_beam` - a double pointer to the `SkaLowBeam` struct which is set
///   by this function. This struct must be freed by calling
///   `free_ska_low_beam`.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
///
/// # Safety
///
/// The pointers must be valid for the lengths described above.
#[no_mangle]
pub unsafe extern "C" fn new_ska_low_beam(
    beam_dir: *const c_char,
    num_elements: c_uint,
    element_offsets: *const f64,
    coordinate_axes: *const f64,
    use_ticra_convention: u8,
    ska_low_beam: *mut *mut SkaLowBeam,
) -> i32 {
    let num_elements = num_elements as usize;
    let dir = match str_from_ffi(beam_dir, "beam_dir") {
        Ok(d) => d,
        Err(e) => return fail(e),
    };
    let offsets = match offsets_from_ffi(element_offsets, num_elements) {
        Ok(o) => o,
        Err(e) => return fail(e),
    };
    let axes = axes_from_ffi(coordinate_axes);

    match SkaLowBeam::new(&dir, offsets.view(), axes, use_ticra_convention != 0) {
        Ok(b) => {
            *ska_low_beam = Box::into_raw(Box::new(b));
            0
        }
        Err(e) => fail(e.to_string()),
    }
}

/// Create a new SKA-Low beam from a directory of gridded EEP files.
///
/// # Arguments
///
/// * `beam_dir` - a null-terminated path to a directory of EEP files.
/// * `filebase` - a null-terminated string; the part of the file names before
///   the frequency, e.g. `"HARP_SKALA41_randvogel_1m_256_elem_50ohm_"`. There
///   is deliberately no default, because any default goes stale.
/// * `is_npz` - a boolean; 1 for `.npz` files, 0 for `.npy`.
/// * `num_elements`, `element_offsets`, `coordinate_axes` - as for
///   `new_ska_low_beam`.
/// * `rotation_deg` - the rotation of the station, and therefore of the EEP
///   azimuth axis, in degrees.
/// * `normalise` - a boolean indicating whether to divide each element by its
///   power integral over the grid.
/// * `ska_low_beam` - a double pointer to the `SkaLowBeam` struct which is set
///   by this function. This struct must be freed by calling
///   `free_ska_low_beam`.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
///
/// # Safety
///
/// The pointers must be valid for the lengths described above.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn new_ska_low_beam_from_grid(
    beam_dir: *const c_char,
    filebase: *const c_char,
    is_npz: u8,
    num_elements: c_uint,
    element_offsets: *const f64,
    coordinate_axes: *const f64,
    rotation_deg: f64,
    normalise: u8,
    ska_low_beam: *mut *mut SkaLowBeam,
) -> i32 {
    let num_elements = num_elements as usize;
    let dir = match str_from_ffi(beam_dir, "beam_dir") {
        Ok(d) => d,
        Err(e) => return fail(e),
    };
    let filebase = match str_from_ffi(filebase, "filebase") {
        Ok(f) => f,
        Err(e) => return fail(e),
    };
    let offsets = match offsets_from_ffi(element_offsets, num_elements) {
        Ok(o) => o,
        Err(e) => return fail(e),
    };
    let format = if is_npz != 0 {
        GridFormat::Npz
    } else {
        GridFormat::Npy
    };

    match SkaLowBeam::new_from_grid(
        &dir,
        &filebase,
        format,
        offsets.view(),
        axes_from_ffi(coordinate_axes),
        rotation_deg,
        normalise != 0,
    ) {
        Ok(b) => {
            *ska_low_beam = Box::into_raw(Box::new(b));
            0
        }
        Err(e) => fail(e.to_string()),
    }
}

/// Get the beam response Jones matrix for the given direction.
///
/// Note the type of `jones` (`*double`); we can't pass complex numbers across
/// the FFI boundary, so the real and imaginary components are unpacked into
/// doubles. The output contains 8 doubles, where j00 is the first pair, j01 is
/// the second pair, etc.
///
/// # Arguments
///
/// * `ska_low_beam` - a pointer to a `SkaLowBeam` struct.
/// * `az_rad` - the azimuth direction to get the beam response (radians).
/// * `za_rad` - the zenith angle direction to get the beam response (radians).
/// * `freq_hz` - the frequency used for the beam response, in Hertz.
/// * `weights` - a pointer to a `2 * num_elements` array of interleaved (real,
///   imaginary) element weights. Flagged elements have a weight of zero.
/// * `latitude_rad` - an optional pointer to the telescope latitude. If null,
///   no parallactic-angle correction is applied and `iau_order` does nothing.
/// * `iau_order` - a boolean; if true, the beam response is [NS-NS NS-EW EW-NS
///   EW-EW], otherwise [EW-EW EW-NS NS-EW NS-NS].
/// * `jones` - a pointer to a buffer with at least `8 * sizeof(double)`
///   allocated. The Jones matrix beam response is written here.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
///
/// # Safety
///
/// The pointers must be valid for the lengths described above.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ska_low_calc_jones(
    ska_low_beam: *mut SkaLowBeam,
    az_rad: f64,
    za_rad: f64,
    freq_hz: f64,
    weights: *const f64,
    latitude_rad: *const f64,
    iau_order: u8,
    jones: *mut f64,
) -> i32 {
    let beam = &*ska_low_beam;
    let w = weights_from_ffi(weights, beam.num_elements());
    let azel = AzEl::from_radians(az_rad, std::f64::consts::FRAC_PI_2 - za_rad);

    match beam.calc_jones(
        azel,
        freq_hz,
        &w,
        latitude_rad.as_ref().copied(),
        iau_order != 0,
    ) {
        Ok(j) => {
            let buf = slice::from_raw_parts_mut(jones, 8);
            buf.copy_from_slice(&[
                j[0].re, j[0].im, j[1].re, j[1].im, j[2].re, j[2].im, j[3].re, j[3].im,
            ]);
            0
        }
        Err(e) => fail(e.to_string()),
    }
}

/// Get the beam response Jones matrices for several az/za directions. The Jones
/// matrix elements for each direction are put into a single array.
///
/// As there are 8 doubles per Jones matrix, there must be at least `8 *
/// num_azza * sizeof(double)` allocated in the output array. Rust calculates
/// the Jones matrices in parallel. See the documentation for
/// `ska_low_calc_jones` for more info.
///
/// # Arguments
///
/// * `ska_low_beam` - a pointer to a `SkaLowBeam` struct.
/// * `num_azza` - the number of directions in `az_rad` and `za_rad`.
/// * `az_rad` - a pointer to `num_azza` azimuth directions (radians).
/// * `za_rad` - a pointer to `num_azza` zenith angle directions (radians).
/// * `freq_hz` - the frequency used for the beam responses, in Hertz.
/// * `weights` - a pointer to a `2 * num_elements` array of interleaved (real,
///   imaginary) element weights.
/// * `latitude_rad` - an optional pointer to the telescope latitude.
/// * `iau_order` - a boolean; see `ska_low_calc_jones`.
/// * `jones` - a pointer to a buffer with at least `8 * num_azza *
///   sizeof(double)` allocated.
///
/// # Returns
///
/// * An exit code integer. If this is non-zero then an error occurred; the
///   details can be obtained by (1) getting the length of the error string by
///   calling `hb_last_error_length` and (2) calling `hb_last_error_message`
///   with a string buffer with a length at least equal to the error length.
///
/// # Safety
///
/// The pointers must be valid for the lengths described above.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ska_low_calc_jones_array(
    ska_low_beam: *mut SkaLowBeam,
    num_azza: c_uint,
    az_rad: *const f64,
    za_rad: *const f64,
    freq_hz: f64,
    weights: *const f64,
    latitude_rad: *const f64,
    iau_order: u8,
    jones: *mut f64,
) -> i32 {
    let beam = &*ska_low_beam;
    let num_azza = num_azza as usize;
    let w = weights_from_ffi(weights, beam.num_elements());
    let az = slice::from_raw_parts(az_rad, num_azza);
    let za = slice::from_raw_parts(za_rad, num_azza);
    let azels: Vec<AzEl> = az
        .iter()
        .zip(za)
        .map(|(&az, &za)| AzEl::from_radians(az, std::f64::consts::FRAC_PI_2 - za))
        .collect();

    match beam.calc_jones_array(
        &azels,
        freq_hz,
        &w,
        latitude_rad.as_ref().copied(),
        iau_order != 0,
    ) {
        Ok(js) => {
            let buf = slice::from_raw_parts_mut(jones, 8 * num_azza);
            for (out, j) in buf.chunks_exact_mut(8).zip(js) {
                out.copy_from_slice(&[
                    j[0].re, j[0].im, j[1].re, j[1].im, j[2].re, j[2].im, j[3].re, j[3].im,
                ]);
            }
            0
        }
        Err(e) => fail(e.to_string()),
    }
}

/// Get the available frequencies inside the beam directory.
///
/// # Arguments
///
/// * `ska_low_beam` - a pointer to a `SkaLowBeam` struct.
/// * `freqs_hz` - a double pointer to the FFI frequencies array. The array is
///   owned by the beam and must not be freed by the caller.
/// * `num_freqs` - a pointer to a `size_t` where the length of `freqs_hz` is
///   written.
///
/// # Safety
///
/// The pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn ska_low_get_freqs(
    ska_low_beam: *mut SkaLowBeam,
    freqs_hz: *mut *const f64,
    num_freqs: *mut usize,
) {
    let beam = &*ska_low_beam;
    let freqs = beam.get_freqs();
    *freqs_hz = freqs.as_ptr();
    *num_freqs = freqs.len();
}

/// The number of elements in the station.
///
/// # Safety
///
/// The pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn ska_low_get_num_elements(ska_low_beam: *mut SkaLowBeam) -> usize {
    (*ska_low_beam).num_elements()
}

/// Given a frequency in Hz, get the closest frequency that the beam directory
/// provides.
///
/// # Safety
///
/// The pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn ska_low_closest_freq(ska_low_beam: *mut SkaLowBeam, freq_hz: f64) -> f64 {
    (*ska_low_beam).find_closest_freq(freq_hz)
}

/// Free the memory associated with a `SkaLowBeam`.
///
/// # Safety
///
/// The pointer must have come from `new_ska_low_beam` or
/// `new_ska_low_beam_from_grid`, and must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn free_ska_low_beam(ska_low_beam: *mut SkaLowBeam) {
    drop(Box::from_raw(ska_low_beam));
}

unsafe fn axes_from_ffi(coordinate_axes: *const f64) -> Option<[[f64; 3]; 3]> {
    if coordinate_axes.is_null() {
        return None;
    }
    let s = slice::from_raw_parts(coordinate_axes, 9);
    Some([[s[0], s[1], s[2]], [s[3], s[4], s[5]], [s[6], s[7], s[8]]])
}
