// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Reading SKA-Low FEKO spherical-wave coefficients out of HDF5 files.
//!
//! One file per frequency, following the EveryBeam schema:
//!
//! * `/modes` - `(3, n_modes)`; rows are mode type, m and n. `l_max` is the
//!   last n.
//! * `/coefficients` - `(n_elements, l_max * (l_max + 2), 4)` of an HDF5
//!   compound type with members "r" and "i". The last axis is
//!   `[X_te, X_tm, Y_te, Y_tm]`, which is the order the spherical-wave kernel
//!   indexes.

use std::path::{Path, PathBuf};

use hdf5_metno::{Conversion, H5Type};
use ndarray::prelude::*;
use num_complex::Complex64 as c64;

use super::error::InitSkaLowBeamError;

/// The HDF5 compound type that `/coefficients` is stored as.
#[derive(H5Type, Clone, Copy)]
#[repr(C)]
struct H5Complex {
    r: f64,
    i: f64,
}

/// The beam files found in a directory, ascendingly sorted by frequency.
pub(super) struct CoeffFiles {
    /// (frequency \[Hz\], file).
    files: Vec<(f64, PathBuf)>,
}

impl CoeffFiles {
    /// Scan a directory for `.h5` beam files, keyed by the frequency in their
    /// names.
    pub(super) fn new(dir: &Path) -> Result<Self, InitSkaLowBeamError> {
        if !dir.is_dir() {
            return Err(InitSkaLowBeamError::BeamDirDoesntExist(
                dir.display().to_string(),
            ));
        }

        let mut files = vec![];
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("h5") {
                continue;
            }
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            let freq_hz = parse_freq_hz(&name)
                .ok_or_else(|| InitSkaLowBeamError::ParseFreq(name.to_string()))?;
            files.push((freq_hz, path.clone()));
        }

        if files.is_empty() {
            return Err(InitSkaLowBeamError::NoBeamFiles(dir.display().to_string()));
        }
        files.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        Ok(Self { files })
    }

    /// The frequencies \[Hz\] of the beam files. Ascendingly sorted.
    pub(super) fn freqs_hz(&self) -> impl Iterator<Item = f64> + '_ {
        self.files.iter().map(|(f, _)| *f)
    }

    /// Given a frequency in Hz, find the closest beam file. Ties go to the
    /// lower frequency, as they do for the FEE beam.
    pub(super) fn find_closest(&self, freq_hz: f64) -> (f64, &Path) {
        let (f, p) = self
            .files
            .iter()
            .min_by(|a, b| (a.0 - freq_hz).abs().total_cmp(&(b.0 - freq_hz).abs()))
            .expect("self.files is never empty");
        (*f, p.as_path())
    }
}

/// One frequency's worth of spherical-wave coefficients.
pub(super) struct SweCoefficients {
    /// The maximum spherical-wave degree.
    pub(super) l_max: usize,
    /// `(n_elements, l_max * (l_max + 2), 4)`.
    pub(super) coeffs: Array3<c64>,
}

impl SweCoefficients {
    pub(super) fn from_file(file: &Path) -> Result<Self, InitSkaLowBeamError> {
        let h5 = hdf5_metno::File::open(file)?;
        let l_max = read_l_max(&h5, file)?;
        let n_coeffs = l_max * (l_max + 2);

        let ds = dataset(&h5, file, "coefficients")?;
        let shape = ds.shape();
        if shape.len() != 3 || shape[1] < n_coeffs || shape[2] != 4 {
            return Err(InitSkaLowBeamError::DatasetShape {
                file: file.display().to_string(),
                dataset: "coefficients".to_string(),
                expected: "(n_elements, >= l_max * (l_max + 2), 4)",
                got: shape,
            });
        }

        let raw: Vec<H5Complex> = ds.read_raw()?;
        let coeffs = Array3::from_shape_vec(
            (shape[0], shape[1], shape[2]),
            raw.into_iter().map(|z| c64::new(z.r, z.i)).collect(),
        )
        .expect("the shape came from the dataset itself");
        // A file may carry more modes than /modes declares; only l_max of them
        // are meant to be used.
        let coeffs = if shape[1] > n_coeffs {
            coeffs.slice(s![.., ..n_coeffs, ..]).to_owned()
        } else {
            coeffs
        };

        Ok(Self { l_max, coeffs })
    }

    pub(super) fn num_elements(&self) -> usize {
        self.coeffs.len_of(Axis(0))
    }

    /// The flat, `[X_te, X_tm, Y_te, Y_tm]`-ordered coefficients of one
    /// element, as the spherical-wave kernel wants them.
    pub(super) fn element(&self, element: usize) -> ArrayView2<'_, c64> {
        self.coeffs.index_axis(Axis(0), element)
    }
}

/// Parse a frequency \[Hz\] out of a beam file name.
///
/// The SWE files spell it `..._FEKO_100.00000_MHz.h5` and the gridded ones
/// `...99MHz_Ypol.npz`, so key off the MHz token rather than a field position;
/// the number of fields before it varies with station, layout and impedance.
pub(super) fn parse_freq_hz(file_name: &str) -> Option<f64> {
    parse_freq_field(file_name).map(|(_, hz)| hz)
}

/// As [`parse_freq_hz`], but also returns the MHz field exactly as it was
/// spelled. The gridded loader needs it to rebuild sibling file names, whose
/// spelling ("99" vs "99.00") it cannot otherwise guess.
pub(super) fn parse_freq_field(file_name: &str) -> Option<(&str, f64)> {
    let stem = file_name.rsplit_once('.').map_or(file_name, |(s, _)| s);
    let mut prev = None;
    for field in stem.split('_') {
        if let Some(mhz) = field.strip_suffix("MHz") {
            let mhz = if mhz.is_empty() { prev? } else { mhz };
            return mhz.parse::<f64>().ok().map(|hz| (mhz, hz * 1e6));
        }
        prev = Some(field);
    }
    None
}

fn dataset(
    h5: &hdf5_metno::File,
    file: &Path,
    name: &str,
) -> Result<hdf5_metno::Dataset, InitSkaLowBeamError> {
    // The OSKAR/HARP files are also named "..._FEKO_....h5" but hold entirely
    // different datasets, so say what we wanted rather than letting HDF5
    // complain about a missing link.
    h5.dataset(name)
        .map_err(|_| InitSkaLowBeamError::MissingDataset {
            file: file.display().to_string(),
            dataset: name.to_string(),
        })
}

fn read_l_max(h5: &hdf5_metno::File, file: &Path) -> Result<usize, InitSkaLowBeamError> {
    let ds = dataset(h5, file, "modes")?;
    let shape = ds.shape();
    if shape.len() != 2 || shape[0] != 3 || shape[1] == 0 {
        return Err(InitSkaLowBeamError::DatasetShape {
            file: file.display().to_string(),
            dataset: "modes".to_string(),
            expected: "(3, n_modes)",
            got: shape,
        });
    }

    // l_max is row 2's last column, which in C order is the last element of the
    // whole dataset. The on-disk integer width varies, so let HDF5 convert.
    let modes: Vec<i64> = ds.as_reader().conversion(Conversion::Soft).read_raw()?;
    match modes.last() {
        Some(&l_max) if l_max >= 1 => Ok(l_max as usize),
        _ => Err(InitSkaLowBeamError::DatasetShape {
            file: file.display().to_string(),
            dataset: "modes".to_string(),
            expected: "a final n of at least 1",
            got: shape,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn parse_swe_freq() {
        let f = parse_freq_hz("SKA-Low_Perturbed_Vogel_256_elem_FEKO_100.00000_MHz.h5");
        assert_abs_diff_eq!(f.unwrap(), 100e6);
    }

    #[test]
    fn parse_grid_freq() {
        let f = parse_freq_hz("HARP_SKALA41_randvogel_1m_256_elem_50ohm_99MHz_Ypol.npz");
        assert_abs_diff_eq!(f.unwrap(), 99e6);

        let f = parse_freq_hz("FEKO_AAVS3_vogel_256_elem_50ohm_50MHz_Xpol_theta.npy");
        assert_abs_diff_eq!(f.unwrap(), 50e6);
    }

    #[test]
    fn parse_freq_rejects_nonsense() {
        assert!(parse_freq_hz("SKA-Low_Perturbed_Vogel_256_elem_FEKO.h5").is_none());
        assert!(parse_freq_hz("SKA-Low_FEKO_banana_MHz.h5").is_none());
    }

    #[test]
    fn closest_freq() {
        let files = CoeffFiles {
            files: vec![
                (50e6, "50.h5".into()),
                (100e6, "100.h5".into()),
                (106.25e6, "106.h5".into()),
            ],
        };
        assert_eq!(files.find_closest(0.0).1, Path::new("50.h5"));
        assert_eq!(files.find_closest(1e9).1, Path::new("106.h5"));
        assert_eq!(files.find_closest(102e6).1, Path::new("100.h5"));
        assert_eq!(files.find_closest(104e6).1, Path::new("106.h5"));
        // Exactly halfway; ties go to the lower frequency.
        assert_eq!(files.find_closest(75e6).1, Path::new("50.h5"));
    }
}
