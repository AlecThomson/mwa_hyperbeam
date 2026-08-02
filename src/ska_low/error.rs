// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Errors associated with the SKA-Low beam.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum InitSkaLowBeamError {
    #[error("Specified beam directory '{0}' doesn't exist")]
    BeamDirDoesntExist(String),

    #[error("No beam files were found in '{0}'")]
    NoBeamFiles(String),

    #[error("Couldn't parse a frequency out of the beam file name '{0}'")]
    ParseFreq(String),

    #[error("Expected element offsets with shape (3, n_elements), but got {0:?}")]
    ElementOffsetsShape(Vec<usize>),

    /// The OSKAR/HARP HDF5 schema is also named "..._FEKO_....h5"; say what we
    /// wanted rather than letting HDF5 report a missing link.
    #[error("HDF5 file '{file}' is missing the dataset '{dataset}'; expected the EveryBeam SKA-Low schema with '/modes' and '/coefficients'")]
    MissingDataset { file: String, dataset: String },

    #[error(
        "Unexpected shape {got:?} for HDF5 dataset '{dataset}' in '{file}'; expected {expected}"
    )]
    DatasetShape {
        file: String,
        dataset: String,
        expected: &'static str,
        got: Vec<usize>,
    },

    /// An error associated with the SKA_LOW_BEAM_DIR environment variable.
    #[error("Problem with the SKA_LOW_BEAM_DIR variable: {0}")]
    BeamDirVarError(#[from] std::env::VarError),

    /// An error associated with the hdf5_metno crate.
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5_metno::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Error, Debug)]
pub enum SkaLowBeamError {
    #[error("No beam data was found within {tolerance_hz} Hz of {freq_hz} Hz")]
    NoFreqMatch { freq_hz: f64, tolerance_hz: f64 },

    #[error("Got {got} element weights, but the beam has {expected} elements")]
    WeightCountMismatch { expected: usize, got: usize },

    #[error("'{file}' has {got} elements, but the station geometry has {expected}")]
    ElementCountMismatch {
        expected: usize,
        got: usize,
        file: String,
    },

    #[error("The four gridded EEP arrays do not all have the same shape")]
    GridShapeMismatch,

    #[error("Gridded EEPs need at least 2 azimuth and 2 zenith-angle pixels, but got {0:?}")]
    GridTooSmall(Vec<usize>),

    #[error("Gridded EEPs should have 2 or 3 dimensions, not {0}")]
    GridDimensions(usize),

    #[error("Couldn't read the .npy data: {0}")]
    Npy(String),

    #[error("Unsupported .npy dtype '{0}'; expected complex64 ('<c8') or complex128 ('<c16')")]
    NpyDtype(String),

    #[error("Couldn't read the .npz container: {0}")]
    Npz(String),

    #[error("'{0}' in the .npz is deflated; only numpy.savez (stored) output is supported, not numpy.savez_compressed")]
    NpzCompressed(String),

    #[error("The .npz has no member '{member}' (it has {found} members)")]
    NpzMissingMember { member: String, found: usize },

    /// Beam files are opened lazily, so initialisation errors can also surface
    /// when a beam response is requested.
    #[error(transparent)]
    Init(#[from] InitSkaLowBeamError),
}
