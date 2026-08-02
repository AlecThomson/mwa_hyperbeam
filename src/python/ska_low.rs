// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Python interface to hyperbeam SKA-Low beam code.

use marlu::{c64, AzEl};
use ndarray::prelude::*;
use numpy::*;
use pyo3::prelude::*;

use crate::ska_low::{GridFormat, SkaLowBeam as SkaLowBeamRust};

/// A Python class interfacing with the hyperbeam SKA-Low beam code written in
/// Rust.
#[pyclass]
pub(super) struct SkaLowBeam {
    beam: SkaLowBeamRust,
}

/// The station geometry, which is common to both constructors.
///
/// `element_offsets` is `(3, n_elements)` in metres. If `coordinate_axes` is
/// given, it is `(3, 3)`, each row being one of the station's axes `(p, q, r)`
/// expressed in (East, North, Up), and the offsets are taken to be in the
/// station frame; otherwise the offsets are already (East, North, Up). These
/// mirror the `ELEMENT_OFFSET` and `COORDINATE_AXES` columns of the MSv3
/// `PHASED_ARRAY` subtable, which the caller reads.
fn geometry(
    element_offsets: PyReadonlyArray2<f64>,
    coordinate_axes: Option<PyReadonlyArray2<f64>>,
) -> PyResult<(Array2<f64>, Option<[[f64; 3]; 3]>)> {
    let offsets = element_offsets.as_array().to_owned();
    let axes = match coordinate_axes {
        None => None,
        Some(a) => {
            let a = a.as_array();
            if a.shape() != [3, 3] {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "coordinate_axes must have shape (3, 3), not {:?}",
                    a.shape()
                )));
            }
            let mut axes = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    axes[i][j] = a[[i, j]];
                }
            }
            Some(axes)
        }
    };
    Ok((offsets, axes))
}

#[pymethods]
impl SkaLowBeam {
    /// Create a new `SkaLowBeam` from a directory of FEKO spherical-wave
    /// coefficient files (`..._FEKO_<freq>_MHz.h5`, one per frequency).
    ///
    /// See the class documentation for `element_offsets` and
    /// `coordinate_axes`. A flagged element is one with a weight of zero;
    /// there is no separate flag argument.
    #[new]
    #[pyo3(signature = (beam_dir, element_offsets, coordinate_axes=None, use_ticra_convention=None))]
    fn new(
        beam_dir: std::path::PathBuf,
        element_offsets: PyReadonlyArray2<f64>,
        coordinate_axes: Option<PyReadonlyArray2<f64>>,
        use_ticra_convention: Option<bool>,
    ) -> PyResult<SkaLowBeam> {
        let (offsets, axes) = geometry(element_offsets, coordinate_axes)?;
        Ok(SkaLowBeam {
            beam: SkaLowBeamRust::new(
                beam_dir,
                offsets.view(),
                axes,
                use_ticra_convention.unwrap_or(false),
            )?,
        })
    }

    /// Create a new `SkaLowBeam` from a directory of gridded EEP files.
    ///
    /// `filebase` is the part of the file names before the frequency, e.g.
    /// `"HARP_SKALA41_randvogel_1m_256_elem_50ohm_"`. It is required rather
    /// than defaulted, because it encodes the station, layout and impedance,
    /// and any baked-in default goes stale.
    ///
    /// `npz` selects the container: `True` for
    /// `<filebase><freq>MHz_<X|Y>pol.npz`, `False` for
    /// `<filebase><freq>MHz_<X|Y>pol_<phi|theta>.npy`.
    ///
    /// `rotation_deg` rotates the EEP azimuth axis with the station.
    /// `normalise` divides each element by its power integral over the grid.
    #[staticmethod]
    #[pyo3(signature = (beam_dir, filebase, element_offsets, npz=None, coordinate_axes=None, rotation_deg=None, normalise=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_grid(
        beam_dir: std::path::PathBuf,
        filebase: &str,
        element_offsets: PyReadonlyArray2<f64>,
        npz: Option<bool>,
        coordinate_axes: Option<PyReadonlyArray2<f64>>,
        rotation_deg: Option<f64>,
        normalise: Option<bool>,
    ) -> PyResult<SkaLowBeam> {
        let (offsets, axes) = geometry(element_offsets, coordinate_axes)?;
        let format = if npz.unwrap_or(false) {
            GridFormat::Npz
        } else {
            GridFormat::Npy
        };
        Ok(SkaLowBeam {
            beam: SkaLowBeamRust::new_from_grid(
                beam_dir,
                filebase,
                format,
                offsets.view(),
                axes,
                rotation_deg.unwrap_or(0.0),
                normalise.unwrap_or(false),
            )?,
        })
    }

    /// Calculate the station Jones matrix for a single direction.
    ///
    /// `weights` is the complex excitation of each element, one per element,
    /// with any element flags folded in as a weight of zero.
    ///
    /// If `latitude_rad` is not supplied, the result is in the beam's own
    /// (theta, phi) frame and `iau_order` does nothing. Astronomers are more
    /// likely to want to supply it, applying the parallactic-angle correction.
    #[pyo3(signature = (az_rad, za_rad, freq_hz, weights, latitude_rad=None, iau_order=None))]
    #[allow(clippy::too_many_arguments)]
    fn calc_jones<'py>(
        &self,
        py: Python<'py>,
        az_rad: f64,
        za_rad: f64,
        freq_hz: f64,
        weights: PyReadonlyArray1<c64>,
        latitude_rad: Option<f64>,
        iau_order: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray1<c64>>> {
        let azel = AzEl::from_radians(az_rad, std::f64::consts::FRAC_PI_2 - za_rad);
        let weights = weights.as_array();
        let weights = weights.as_slice().expect("a 1-D numpy array is contiguous");
        let jones = self.beam.calc_jones(
            azel,
            freq_hz,
            weights,
            latitude_rad,
            iau_order.unwrap_or(false),
        )?;
        Ok(PyArray1::from_vec(py, jones.to_vec()))
    }

    /// Calculate the station Jones matrices for multiple directions. Each
    /// direction is calculated in parallel by Rust; the number of threads used
    /// can be controlled by setting RAYON_NUM_THREADS.
    #[pyo3(signature = (az_rad, za_rad, freq_hz, weights, latitude_rad=None, iau_order=None))]
    #[allow(clippy::too_many_arguments)]
    fn calc_jones_array<'py>(
        &self,
        py: Python<'py>,
        az_rad: Vec<f64>,
        za_rad: Vec<f64>,
        freq_hz: f64,
        weights: PyReadonlyArray1<c64>,
        latitude_rad: Option<f64>,
        iau_order: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray2<c64>>> {
        if az_rad.len() != za_rad.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "az_rad has {} directions but za_rad has {}",
                az_rad.len(),
                za_rad.len()
            )));
        }
        let azels: Vec<AzEl> = az_rad
            .iter()
            .zip(&za_rad)
            .map(|(&az, &za)| AzEl::from_radians(az, std::f64::consts::FRAC_PI_2 - za))
            .collect();
        let weights = weights.as_array();
        let weights = weights.as_slice().expect("a 1-D numpy array is contiguous");
        let jones = self.beam.calc_jones_array(
            &azels,
            freq_hz,
            weights,
            latitude_rad,
            iau_order.unwrap_or(false),
        )?;
        let flat: Vec<c64> = jones.iter().flat_map(|j| j.to_vec()).collect();
        let arr =
            Array2::from_shape_vec((azels.len(), 4), flat).expect("four components per direction");
        Ok(PyArray2::from_owned_array(py, arr))
    }

    /// The frequencies \[Hz\] of the available beam files, ascendingly sorted.
    fn get_freqs(&self) -> Vec<f64> {
        self.beam.get_freqs().to_vec()
    }

    /// Given a frequency in Hz, get the closest frequency that the beam
    /// directory provides.
    fn find_closest_freq(&self, freq_hz: f64) -> f64 {
        self.beam.find_closest_freq(freq_hz)
    }

    /// The number of elements in the station.
    fn num_elements(&self) -> usize {
        self.beam.num_elements()
    }

    /// Empty the cached beam data to recover memory.
    fn empty_cache(&self) {
        self.beam.empty_cache();
    }
}
