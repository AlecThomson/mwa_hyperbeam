// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Primary beam code for SKA-Low stations.
//!
//! Unlike the MWA beams, the beam files here describe a single *element*; a
//! station response is the weighted, phase-steered sum over the station's
//! elements. Element positions, station orientation and element flags are not
//! in the beam files, so they are constructor arguments - hyperbeam does not
//! read MeasurementSets.
//!
//! Two file formats describe the same thing. FEKO spherical-wave coefficients
//! (HDF5, one file per frequency) are evaluated analytically; gridded EEPs
//! (`.npy`/`.npz`, four arrays per frequency) are sampled. They differ only in
//! how one element is evaluated at one direction, so they are an enum rather
//! than two beam types - everything downstream is shared.

mod coeffs;
mod error;
mod ffi;
mod grid;
mod swe;
#[cfg(test)]
mod tests;

pub use error::{InitSkaLowBeamError, SkaLowBeamError};
pub use grid::GridFormat;

use std::{
    collections::HashMap,
    f64::consts::{FRAC_PI_2, TAU},
    path::Path,
    sync::Arc,
};

use marlu::{constants::VEL_C, AzEl, Jones};
use ndarray::prelude::*;
use num_complex::Complex64 as c64;
use parking_lot::RwLock;
use rayon::prelude::*;

use crate::types::apply_parallactic_correction;
use coeffs::{CoeffFiles, SweCoefficients};
use grid::{GridFiles, GriddedEeps};

/// The number of feeds per element; X then Y.
const N_FEEDS: usize = 2;

/// The default tolerance for matching a gridded EEP file to a frequency,
/// mirroring `maximum_offset_mhz` in `eep.py`.
const DEFAULT_GRID_TOLERANCE_HZ: f64 = 3e6;

/// The electric field of one element in one direction, per feed.
///
/// This is where the two element sources converge: the spherical-wave kernel
/// and the gridded EEPs both produce exactly these four numbers, and everything
/// downstream is shared.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ElementPattern {
    pub(crate) e_theta: [c64; N_FEEDS],
    pub(crate) e_phi: [c64; N_FEEDS],
}

/// Where per-element patterns come from.
enum ElementSource {
    Swe {
        files: CoeffFiles,
        use_ticra_convention: bool,
        cache: RwLock<HashMap<u64, Arc<SweCoefficients>>>,
    },
    Grid {
        files: GridFiles,
        /// Rotation of the station, and therefore of the EEP azimuth axis
        /// \[degrees\]. `eep_rotation_deg` in `eep.py`.
        rotation_deg: f64,
        normalise: bool,
        tolerance_hz: f64,
        cache: RwLock<HashMap<u64, Arc<GriddedEeps>>>,
    },
}

/// One frequency's element source, ready to evaluate.
enum Prepared {
    Swe {
        coeffs: Arc<SweCoefficients>,
        use_ticra_convention: bool,
    },
    Grid {
        eeps: Arc<GriddedEeps>,
        rotation_deg: f64,
    },
}

impl Prepared {
    fn num_elements(&self) -> usize {
        match self {
            Prepared::Swe { coeffs, .. } => coeffs.num_elements(),
            Prepared::Grid { eeps, .. } => eeps.num_elements(),
        }
    }

    /// Scratch space for one thread's worth of evaluation.
    fn scratch(&self) -> (Vec<f64>, Vec<f64>) {
        match self {
            Prepared::Swe { coeffs, .. } => (
                crate::legendre::root_n_table(coeffs.l_max),
                vec![0.0; coeffs.l_max + 2],
            ),
            Prepared::Grid { .. } => (vec![], vec![]),
        }
    }
}

/// The main struct to be used for calculating SKA-Low station Jones matrices.
pub struct SkaLowBeam {
    source: ElementSource,
    /// The frequencies \[Hz\] of the available beam files, ascendingly sorted.
    /// Held rather than recomputed so that the C API can hand out a pointer.
    freqs_hz: Vec<f64>,
    /// Element offsets in the station's local (East, North, Up) frame
    /// \[metres\], shape `(3, n_elements)`. Already rotated by the station's
    /// coordinate axes, if any were given.
    element_offsets: Array2<f64>,
}

impl SkaLowBeam {
    /// Create a new [`SkaLowBeam`] from a directory of per-frequency FEKO
    /// spherical-wave coefficient files.
    ///
    /// `element_offsets` is `(3, n_elements)` in metres. If `coordinate_axes`
    /// is `None` the offsets are taken to already be in the station's local
    /// (East, North, Up) frame; otherwise they are in the station frame and
    /// each row of `coordinate_axes` is one of the station's axes `(p, q, r)`
    /// expressed in (East, North, Up). This mirrors the `ELEMENT_OFFSET` and
    /// `COORDINATE_AXES` columns of the MSv3 `PHASED_ARRAY` subtable, which the
    /// caller is expected to read.
    ///
    /// Flagged elements are simply elements with a weight of zero; there is no
    /// separate flag concept here.
    pub fn new<T: AsRef<Path>>(
        beam_dir: T,
        element_offsets: ArrayView2<f64>,
        coordinate_axes: Option<[[f64; 3]; 3]>,
        use_ticra_convention: bool,
    ) -> Result<Self, InitSkaLowBeamError> {
        // so that libhdf5 doesn't print errors to stdout
        hdf5_metno::silence_errors(true);
        let files = CoeffFiles::new(beam_dir.as_ref())?;
        Ok(Self {
            freqs_hz: files.freqs_hz().collect(),
            source: ElementSource::Swe {
                files,
                use_ticra_convention,
                cache: RwLock::new(HashMap::new()),
            },
            element_offsets: place_elements(element_offsets, coordinate_axes)?,
        })
    }

    /// Create a new [`SkaLowBeam`] from a directory of gridded EEP files.
    ///
    /// `filebase` is the part of the file names before the frequency; it is
    /// required rather than defaulted, because it encodes the station, layout
    /// and impedance and any baked-in default goes stale. For
    /// `HARP_SKALA41_randvogel_1m_256_elem_50ohm_99MHz_Ypol.npz` it is
    /// `"HARP_SKALA41_randvogel_1m_256_elem_50ohm_"`.
    ///
    /// `rotation_deg` rotates the EEP azimuth axis with the station, as
    /// `eep_rotation_deg` does in `eep.py`. `normalise` divides each element by
    /// its power integral over the whole grid; see [`GriddedEeps::normalise`].
    ///
    /// `element_offsets` and `coordinate_axes` are as for [`SkaLowBeam::new`].
    #[allow(clippy::too_many_arguments)]
    pub fn new_from_grid<T: AsRef<Path>>(
        beam_dir: T,
        filebase: &str,
        format: GridFormat,
        element_offsets: ArrayView2<f64>,
        coordinate_axes: Option<[[f64; 3]; 3]>,
        rotation_deg: f64,
        normalise: bool,
    ) -> Result<Self, InitSkaLowBeamError> {
        let files = GridFiles::new(beam_dir.as_ref(), filebase, format)?;
        Ok(Self {
            freqs_hz: files.freqs_hz().collect(),
            source: ElementSource::Grid {
                files,
                rotation_deg,
                normalise,
                tolerance_hz: DEFAULT_GRID_TOLERANCE_HZ,
                cache: RwLock::new(HashMap::new()),
            },
            element_offsets: place_elements(element_offsets, coordinate_axes)?,
        })
    }

    /// Create a new spherical-wave [`SkaLowBeam`] from the `SKA_LOW_BEAM_DIR`
    /// environment variable.
    pub fn new_from_env(
        element_offsets: ArrayView2<f64>,
        coordinate_axes: Option<[[f64; 3]; 3]>,
        use_ticra_convention: bool,
    ) -> Result<Self, InitSkaLowBeamError> {
        let dir = std::env::var("SKA_LOW_BEAM_DIR")?;
        Self::new(dir, element_offsets, coordinate_axes, use_ticra_convention)
    }

    /// The frequencies \[Hz\] of the available beam files. Ascendingly sorted.
    pub fn get_freqs(&self) -> &[f64] {
        &self.freqs_hz
    }

    /// Given a frequency in Hz, find the closest frequency that has a beam
    /// file.
    pub fn find_closest_freq(&self, desired_freq_hz: f64) -> f64 {
        match &self.source {
            ElementSource::Swe { files, .. } => files.find_closest(desired_freq_hz).0,
            ElementSource::Grid { files, .. } => files.find_closest(desired_freq_hz),
        }
    }

    /// The number of elements in the station.
    pub fn num_elements(&self) -> usize {
        self.element_offsets.len_of(Axis(1))
    }

    /// Calculate the beam-response Jones matrix for a given direction.
    ///
    /// `weights` is the complex excitation of each element, one per element,
    /// with any element flags folded in as a weight of zero.
    ///
    /// If `latitude_rad` is *not* supplied, the result is in the beam's own
    /// (theta, phi) frame. Astronomers are more likely to want to specify
    /// `latitude_rad` (which applies the parallactic-angle correction) and
    /// `iau_order`; if `latitude_rad` is `None`, `iau_order` does nothing. See
    /// <https://github.com/MWATelescope/mwa_hyperbeam/blob/main/fee_pols.pdf>.
    pub fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        weights: &[c64],
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Jones<f64>, SkaLowBeamError> {
        let prepared = self.prepare(freq_hz, weights)?;
        Ok(self.calc_jones_direct(azel, freq_hz, weights, &prepared, latitude_rad, iau_order))
    }

    /// Calculate the beam-response Jones matrices for many directions. This is
    /// a wrapper around `calc_jones` that calculates them in parallel; the
    /// number of threads used can be controlled by setting `RAYON_NUM_THREADS`.
    pub fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        weights: &[c64],
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Result<Vec<Jones<f64>>, SkaLowBeamError> {
        let mut results = vec![Jones::default(); azels.len()];
        self.calc_jones_array_inner(
            azels,
            freq_hz,
            weights,
            latitude_rad,
            iau_order,
            &mut results,
        )?;
        Ok(results)
    }

    /// The same as `calc_jones_array`, but uses pre-allocated memory.
    pub fn calc_jones_array_inner(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        weights: &[c64],
        latitude_rad: Option<f64>,
        iau_order: bool,
        results: &mut [Jones<f64>],
    ) -> Result<(), SkaLowBeamError> {
        let prepared = self.prepare(freq_hz, weights)?;
        azels
            .par_iter()
            .zip(results.par_iter_mut())
            .for_each(|(&azel, result)| {
                *result = self.calc_jones_direct(
                    azel,
                    freq_hz,
                    weights,
                    &prepared,
                    latitude_rad,
                    iau_order,
                );
            });
        Ok(())
    }

    /// Empty the cached beam data to recover memory.
    pub fn empty_cache(&self) {
        match &self.source {
            ElementSource::Swe { cache, .. } => cache.write().clear(),
            ElementSource::Grid { cache, .. } => cache.write().clear(),
        }
    }

    /// Actually calculate a Jones matrix. All other "calc" functions use this
    /// function.
    fn calc_jones_direct(
        &self,
        azel: AzEl,
        freq_hz: f64,
        weights: &[c64],
        prepared: &Prepared,
        latitude_rad: Option<f64>,
        iau_order: bool,
    ) -> Jones<f64> {
        let az_rad = azel.az;
        let za_rad = azel.za();

        // Convert azimuth to FEKO phi (East through North), as the FEE beam
        // does. The Y feed is the X feed rotated by a quarter turn.
        let phi_x = FRAC_PI_2 - az_rad;
        let phi = [phi_x, phi_x + FRAC_PI_2];
        let (root_n, mut workspace) = prepared.scratch();

        // The direction's unit vector in the station's (East, North, Up) frame.
        let (s_az, c_az) = az_rad.sin_cos();
        let (s_el, c_el) = azel.el.sin_cos();
        let k = [s_az * c_el, c_az * c_el, s_el];
        let wavenumber = TAU * freq_hz / VEL_C;

        let mut sum = ElementPattern::default();
        for (element, &weight) in weights.iter().enumerate() {
            if weight == c64::default() {
                continue;
            }
            let pattern = match prepared {
                Prepared::Swe {
                    coeffs,
                    use_ticra_convention,
                } => swe::spherical_wave_pattern_feko(
                    za_rad,
                    phi,
                    coeffs.l_max,
                    &root_n,
                    coeffs.element(element),
                    *use_ticra_convention,
                    &mut workspace,
                ),
                Prepared::Grid { eeps, rotation_deg } => {
                    eeps.element_pattern(element, az_rad, azel.el, *rotation_deg)
                }
            };
            let offset = self.element_offsets.index_axis(Axis(1), element);
            let path = (0..3).map(|i| offset[i] * k[i]).sum::<f64>();
            let excitation = weight * c64::cis(wavenumber * path);
            for a in 0..N_FEEDS {
                sum.e_theta[a] += excitation * pattern.e_theta[a];
                sum.e_phi[a] += excitation * pattern.e_phi[a];
            }
        }

        // The phi components are negated to put this in the same convention as
        // the FEE beam, which is what `apply_parallactic_correction` and
        // hyperdrive expect. Neither EveryBeam nor eep.py negates, and eep.py
        // additionally orders its Jones phi-first.
        let mut jones = Jones::from([sum.e_theta[0], -sum.e_phi[0], sum.e_theta[1], -sum.e_phi[1]]);
        if let Some(latitude_rad) = latitude_rad {
            apply_parallactic_correction(az_rad, za_rad, latitude_rad, iau_order, &mut jones);
        }
        jones
    }

    /// Get the beam data for a frequency, populating the cache if needed.
    fn prepare(&self, freq_hz: f64, weights: &[c64]) -> Result<Prepared, SkaLowBeamError> {
        if weights.len() != self.num_elements() {
            return Err(SkaLowBeamError::WeightCountMismatch {
                expected: self.num_elements(),
                got: weights.len(),
            });
        }

        let prepared = match &self.source {
            ElementSource::Swe {
                files,
                use_ticra_convention,
                cache,
            } => {
                let (file_freq, file) = files.find_closest(freq_hz);
                let key = file_freq.to_bits();
                let coeffs = match cache.read().get(&key) {
                    Some(c) => Some(Arc::clone(c)),
                    None => None,
                };
                let coeffs = match coeffs {
                    Some(c) => c,
                    None => {
                        let c = Arc::new(SweCoefficients::from_file(file)?);
                        self.check_element_count(c.num_elements(), &file.display().to_string())?;
                        cache.write().insert(key, Arc::clone(&c));
                        c
                    }
                };
                Prepared::Swe {
                    coeffs,
                    use_ticra_convention: *use_ticra_convention,
                }
            }

            ElementSource::Grid {
                files,
                rotation_deg,
                normalise,
                tolerance_hz,
                cache,
            } => {
                // Before the cache, so that a nearby frequency having already
                // populated it can't smuggle an out-of-tolerance request past.
                files.check_tolerance(freq_hz, *tolerance_hz)?;
                let key = files.find_closest(freq_hz).to_bits();
                let eeps = match cache.read().get(&key) {
                    Some(e) => Some(Arc::clone(e)),
                    None => None,
                };
                let eeps = match eeps {
                    Some(e) => e,
                    None => {
                        let mut e = files.load(freq_hz)?;
                        if *normalise {
                            e.normalise();
                        }
                        let e = Arc::new(e);
                        self.check_element_count(e.num_elements(), "the gridded EEPs")?;
                        cache.write().insert(key, Arc::clone(&e));
                        e
                    }
                };
                Prepared::Grid {
                    eeps,
                    rotation_deg: *rotation_deg,
                }
            }
        };

        debug_assert_eq!(prepared.num_elements(), self.num_elements());
        Ok(prepared)
    }

    fn check_element_count(&self, got: usize, what: &str) -> Result<(), SkaLowBeamError> {
        if got == self.num_elements() {
            Ok(())
        } else {
            Err(SkaLowBeamError::ElementCountMismatch {
                expected: self.num_elements(),
                got,
                file: what.to_string(),
            })
        }
    }
}

/// Put element offsets into the station's local (East, North, Up) frame.
fn place_elements(
    element_offsets: ArrayView2<f64>,
    coordinate_axes: Option<[[f64; 3]; 3]>,
) -> Result<Array2<f64>, InitSkaLowBeamError> {
    if element_offsets.len_of(Axis(0)) != 3 {
        return Err(InitSkaLowBeamError::ElementOffsetsShape(
            element_offsets.shape().to_vec(),
        ));
    }
    Ok(match coordinate_axes {
        None => element_offsets.to_owned(),
        // An offset (a, b, c) in the station frame is a*p + b*q + c*r.
        Some(axes) => {
            let mut rotated = Array2::zeros(element_offsets.raw_dim());
            for (i, offset) in element_offsets.axis_iter(Axis(1)).enumerate() {
                for j in 0..3 {
                    rotated[[j, i]] = (0..3).map(|k| offset[k] * axes[k][j]).sum();
                }
            }
            rotated
        }
    })
}
