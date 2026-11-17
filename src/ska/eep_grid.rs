// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The SKA-Low gridded embedded-element-pattern model ("ska-low-eep-grid"),
//! equivalent to EveryBeam's `gridded_eep` element response model.

use std::{f64::consts::TAU, path::Path};

use marlu::{c64, Jones};

use super::{InitSkaBeamError, SkaBeamError, SkaElementResponse};

/// The environment variable that [`SkaLowEepGrid::new_from_env`] reads first.
pub const EEP_GRID_FILE_VAR: &str = "SKA_LOW_EEP_GRID_FILE";
/// The environment variable [`SkaLowEepGrid::new_from_env`] falls back to; the
/// file `oskar_eep_grid.h5` in this directory is used (as EveryBeam does for
/// OSKAR telescopes).
pub const EVERYBEAM_DATADIR_VAR: &str = "EVERYBEAM_DATADIR";

/// The SKA-Low gridded, station-averaged embedded-element-pattern (EEP) model
/// ("ska-low-eep-grid").
///
/// This is equivalent to EveryBeam's `gridded_eep` element response model
/// (`kGriddedEep`), as of EveryBeam master (November 2026, after 0.9.1). The HDF5 file must contain a group
/// `averageEEP` with the datasets:
///
/// - `Freq`: frequencies \[Hz\], ascending;
/// - `Phi`: phi values \[radians\] in `[0, 2pi]`, ascending;
/// - `Theta`: theta values \[radians\] in `[0, pi]`, ascending;
/// - `averageEEP`: a `Phi x Theta x Freq x 4 x 2` dataset of doubles, where
///   the last dimension holds real and imaginary parts.
///
/// (The one-dimensional datasets may also be `N x 1` or `1 x N`, as written
/// by MATLAB.)
///
/// As in EveryBeam, the four correlations are stored in the order `[J01, J00,
/// J11, J10]` relative to the returned Jones matrix, and values are
/// tri-linearly interpolated in phi, theta and frequency. Values outside of
/// the grid are an error (no extrapolation is done). EveryBeam notes that this
/// file format "is not final and will be adjusted in the future".
///
/// By default the SKA-Low (OSKAR) response convention is applied, i.e. all
/// four values are conjugated and the Y-dipole values negated, as EveryBeam
/// does for OSKAR telescopes. Use [`SkaLowEepGrid::new_with_convention`] to
/// disable this.
pub struct SkaLowEepGrid {
    phis: Vec<f64>,
    thetas: Vec<f64>,
    freqs: Vec<f64>,
    /// Dimensions: phi x theta x freq, flattened. Each element is
    /// `[J00, J01, J10, J11]` (already re-ordered from the file order).
    grid: Vec<[c64; 4]>,
    ska_low_convention: bool,
}

impl SkaLowEepGrid {
    /// Load an EEP grid file, applying the SKA-Low (OSKAR) response
    /// convention, as EveryBeam does for OSKAR telescopes.
    pub fn new<P: AsRef<Path>>(file: P) -> Result<SkaLowEepGrid, InitSkaBeamError> {
        Self::new_with_convention(file, true)
    }

    /// Load an EEP grid file. If `ska_low_convention` is `true`, values are
    /// conjugated and the Y-dipole (second row) values negated after
    /// interpolation (EveryBeam's behaviour for OSKAR telescopes); otherwise
    /// the interpolated values are returned as they are (EveryBeam's
    /// behaviour for other telescopes).
    pub fn new_with_convention<P: AsRef<Path>>(
        file: P,
        ska_low_convention: bool,
    ) -> Result<SkaLowEepGrid, InitSkaBeamError> {
        let file = file.as_ref();
        hdf5_metno::silence_errors(true);
        if !file.exists() {
            return Err(InitSkaBeamError::PathDoesntExist(
                file.display().to_string(),
            ));
        }
        let file_str = file.display().to_string();
        let h5 = hdf5_metno::File::open(file)?;
        let group = h5.group("averageEEP")?;

        let read_vector = |name: &str| -> Result<Vec<f64>, InitSkaBeamError> {
            let dataset = group.dataset(name)?;
            let shape = dataset.shape();
            let ok = match shape.as_slice() {
                [n] => *n >= 2,
                [1, n] | [n, 1] => *n >= 2,
                _ => false,
            };
            if !ok {
                return Err(InitSkaBeamError::BadShape {
                    file: file_str.clone(),
                    dataset: name.to_string(),
                    got: shape,
                    expected: "expected a vector with at least two values".to_string(),
                });
            }
            let values: Vec<f64> = dataset.read_raw()?;
            if values.windows(2).any(|w| w[0] > w[1]) {
                return Err(InitSkaBeamError::NotSorted {
                    file: file_str.clone(),
                    dataset: name.to_string(),
                });
            }
            Ok(values)
        };
        let phis = read_vector("Phi")?;
        let thetas = read_vector("Theta")?;
        let freqs = read_vector("Freq")?;

        let dataset = group.dataset("averageEEP")?;
        let shape = dataset.shape();
        let expected_shape = [phis.len(), thetas.len(), freqs.len(), 4, 2];
        if shape.as_slice() != expected_shape {
            return Err(InitSkaBeamError::BadShape {
                file: file_str,
                dataset: "averageEEP".to_string(),
                got: shape,
                expected: format!("expected {expected_shape:?} (Phi x Theta x Freq x 4 x 2)"),
            });
        }
        let buffer: Vec<f64> = dataset.read_raw()?;
        let grid = buffer
            .chunks_exact(8)
            .map(|b| {
                // The file stores [J01, J00, J11, J10] (EveryBeam's phi/theta
                // swap when reading the grid).
                [
                    c64::new(b[2], b[3]),
                    c64::new(b[0], b[1]),
                    c64::new(b[6], b[7]),
                    c64::new(b[4], b[5]),
                ]
            })
            .collect();

        Ok(SkaLowEepGrid {
            phis,
            thetas,
            freqs,
            grid,
            ska_low_convention,
        })
    }

    /// Load the EEP grid file named by the `SKA_LOW_EEP_GRID_FILE` environment
    /// variable or, if that isn't set, `$EVERYBEAM_DATADIR/oskar_eep_grid.h5`
    /// (EveryBeam's default for OSKAR telescopes). The SKA-Low convention is
    /// applied.
    pub fn new_from_env() -> Result<SkaLowEepGrid, InitSkaBeamError> {
        if let Ok(f) = std::env::var(EEP_GRID_FILE_VAR) {
            return Self::new(f);
        }
        match std::env::var(EVERYBEAM_DATADIR_VAR) {
            Ok(d) => Self::new(Path::new(&d).join("oskar_eep_grid.h5")),
            Err(_) => Err(InitSkaBeamError::EnvVar {
                var: EEP_GRID_FILE_VAR,
            }),
        }
    }

    /// The phi values of the grid \[radians\].
    pub fn get_phis(&self) -> &[f64] {
        &self.phis
    }

    /// The theta values of the grid \[radians\].
    pub fn get_thetas(&self) -> &[f64] {
        &self.thetas
    }

    /// The frequencies of the grid \[Hz\].
    pub fn get_freqs(&self) -> &[f64] {
        &self.freqs
    }

    /// Tri-linearly interpolate the grid (without applying any sign
    /// convention).
    pub fn interpolate(
        &self,
        phi_rad: f64,
        theta_rad: f64,
        freq_hz: f64,
    ) -> Result<Jones<f64>, SkaBeamError> {
        // Map negative phi values into [0, 2pi). Unlike EveryBeam, which only
        // adds 2pi once, wrap fully, so that e.g. phi = pi/2 - az works for any
        // az.
        let phi = if (0.0..=TAU).contains(&phi_rad) {
            phi_rad
        } else {
            phi_rad.rem_euclid(TAU)
        };

        let (ip, wp) = locate("phi", &self.phis, phi)?;
        let (it, wt) = locate("theta", &self.thetas, theta_rad)?;
        let (i_f, wf) = locate("frequency", &self.freqs, freq_hz)?;

        let nt = self.thetas.len();
        let nf = self.freqs.len();
        let mut out = [c64::default(); 4];
        for (dp, w_p) in [(0, 1.0 - wp), (1, wp)] {
            for (dt, w_t) in [(0, 1.0 - wt), (1, wt)] {
                for (df, w_f) in [(0, 1.0 - wf), (1, wf)] {
                    let w = w_p * w_t * w_f;
                    let g = &self.grid[((ip + dp) * nt + (it + dt)) * nf + (i_f + df)];
                    out.iter_mut().zip(g).for_each(|(o, g)| *o += g * w);
                }
            }
        }
        Ok(Jones::from(out))
    }
}

/// Find the lower index `i` of the grid interval containing `value`, and the
/// weight of the upper value (`values[i + 1]`).
fn locate(what: &'static str, values: &[f64], value: f64) -> Result<(usize, f64), SkaBeamError> {
    let (min, max) = (values[0], values[values.len() - 1]);
    // `!(a <= b)` rather than `a > b` so that NaN is caught.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(value >= min && value <= max) {
        return Err(SkaBeamError::OutOfGrid {
            what,
            value,
            min,
            max,
        });
    }
    // Equivalent to EveryBeam's `std::lower_bound` followed by a decrement.
    let i = values
        .partition_point(|&v| v < value)
        .saturating_sub(1)
        .min(values.len() - 2);
    let (lo, hi) = (values[i], values[i + 1]);
    Ok((i, (value - lo) / (hi - lo)))
}

impl SkaElementResponse for SkaLowEepGrid {
    fn element_response(
        &self,
        _element_id: usize,
        freq_hz: f64,
        theta_rad: f64,
        phi_rad: f64,
    ) -> Result<Jones<f64>, SkaBeamError> {
        let j = self.interpolate(phi_rad, theta_rad, freq_hz)?;
        Ok(if self.ska_low_convention {
            Jones::from([j[0].conj(), j[1].conj(), -j[2].conj(), -j[3].conj()])
        } else {
            j
        })
    }
}
