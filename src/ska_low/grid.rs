// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Gridded embedded element patterns: `[az, za, n_elements]` complex cubes,
//! sampled at the nearest pixel.
//!
//! The layout and every convention here comes from `eep.py` in
//! `ska-low-mccs-calibration`, which is the authoritative public spec:
//!
//! * On disk, four arrays per frequency, one per (instrument pol x sky pol).
//!   `.npy` spells all four out; `.npz` packs phi and theta together.
//! * The grid itself is implied by the array shape, not stored:
//!   `daz = 360 / (n_az - 1)` and `dza = 90 / (n_za - 1)`, both endpoints
//!   inclusive.
//! * EEP azimuth runs North-from-East, not the astronomical East-from-North.
//! * The data is conjugated on load, and the Y polarisation is negated to match
//!   real data (SKAO "Memo 002 - Using station EEPs and analytic models with
//!   real data").
//!
//! `eep.py` assembles its Jones matrices as `[Xphi, Xtheta, Yphi, Ytheta]`,
//! i.e. phi first. Here the four values are put into an [`ElementPattern`],
//! which is theta-first like the spherical-wave kernel, so that both element
//! sources feed identical downstream code.

use std::path::{Path, PathBuf};

use ndarray::prelude::*;
use num_complex::Complex64 as c64;

use super::{
    coeffs::parse_freq_field,
    error::{InitSkaLowBeamError, SkaLowBeamError},
    ElementPattern, N_FEEDS,
};

/// Which container the gridded EEPs are stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridFormat {
    /// Four files per frequency, `<filebase><freq>MHz_<X|Y>pol_<phi|theta>.npy`.
    Npy,
    /// Two files per frequency, `<filebase><freq>MHz_<X|Y>pol.npz`, each
    /// holding `Ephi` and `Etheta`.
    Npz,
}

impl GridFormat {
    fn extension(self) -> &'static str {
        match self {
            GridFormat::Npy => "npy",
            GridFormat::Npz => "npz",
        }
    }

    /// The file that a directory scan looks for, minus the frequency.
    fn probe_suffix(self) -> &'static str {
        match self {
            GridFormat::Npy => "MHz_Xpol_phi.npy",
            GridFormat::Npz => "MHz_Xpol.npz",
        }
    }
}

/// The gridded beam files found in a directory, ascendingly sorted by
/// frequency.
pub(super) struct GridFiles {
    dir: PathBuf,
    filebase: String,
    format: GridFormat,
    /// (frequency \[Hz\], the MHz field exactly as spelled in the file name).
    freqs: Vec<(f64, String)>,
}

impl GridFiles {
    pub(super) fn new(
        dir: &Path,
        filebase: &str,
        format: GridFormat,
    ) -> Result<Self, InitSkaLowBeamError> {
        if !dir.is_dir() {
            return Err(InitSkaLowBeamError::BeamDirDoesntExist(
                dir.display().to_string(),
            ));
        }

        let mut freqs = vec![];
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if !name.starts_with(filebase) || !name.ends_with(format.probe_suffix()) {
                continue;
            }
            let (field, freq_hz) = parse_freq_field(&name)
                .ok_or_else(|| InitSkaLowBeamError::ParseFreq(name.to_string()))?;
            freqs.push((freq_hz, field.to_string()));
        }

        if freqs.is_empty() {
            return Err(InitSkaLowBeamError::NoBeamFiles(format!(
                "{} (looking for '{filebase}*{}')",
                dir.display(),
                format.probe_suffix()
            )));
        }
        // Numerically, so that 50MHz precedes 100MHz.
        freqs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        Ok(Self {
            dir: dir.to_path_buf(),
            filebase: filebase.to_string(),
            format,
            freqs,
        })
    }

    pub(super) fn freqs_hz(&self) -> impl Iterator<Item = f64> + '_ {
        self.freqs.iter().map(|(f, _)| *f)
    }

    pub(super) fn find_closest(&self, freq_hz: f64) -> f64 {
        self.freqs
            .iter()
            .map(|(f, _)| *f)
            .min_by(|a, b| (a - freq_hz).abs().total_cmp(&(b - freq_hz).abs()))
            .expect("self.freqs is never empty")
    }

    /// Reject a frequency with no file within `tolerance_hz`, mirroring
    /// `get_nearest_file(..., maximum_offset_mhz=3)`. The spherical-wave path
    /// has no equivalent because those files are a dense frequency sweep.
    ///
    /// This is separate from `load` so that callers check it before consulting
    /// a cache, which a nearby frequency may already have populated.
    pub(super) fn check_tolerance(
        &self,
        freq_hz: f64,
        tolerance_hz: f64,
    ) -> Result<(), SkaLowBeamError> {
        if (self.find_closest(freq_hz) - freq_hz).abs() > tolerance_hz {
            return Err(SkaLowBeamError::NoFreqMatch {
                freq_hz,
                tolerance_hz,
            });
        }
        Ok(())
    }

    /// Load the four EEP cubes closest to `freq_hz`.
    pub(super) fn load(&self, freq_hz: f64) -> Result<GriddedEeps, SkaLowBeamError> {
        let (_, field) = self
            .freqs
            .iter()
            .min_by(|a, b| (a.0 - freq_hz).abs().total_cmp(&(b.0 - freq_hz).abs()))
            .expect("self.freqs is never empty");

        let path = |suffix: &str| {
            self.dir.join(format!(
                "{}{field}MHz_{suffix}.{}",
                self.filebase,
                self.format.extension()
            ))
        };

        // [X phi, X theta, Y phi, Y theta], as eep.py returns them.
        let mut eeps = match self.format {
            GridFormat::Npy => [
                read_npy_file(&path("Xpol_phi"))?,
                read_npy_file(&path("Xpol_theta"))?,
                read_npy_file(&path("Ypol_phi"))?,
                read_npy_file(&path("Ypol_theta"))?,
            ],
            GridFormat::Npz => {
                let x = std::fs::read(path("Xpol")).map_err(InitSkaLowBeamError::Io)?;
                let y = std::fs::read(path("Ypol")).map_err(InitSkaLowBeamError::Io)?;
                [
                    read_npy(npz_member(&x, "Ephi")?)?,
                    read_npy(npz_member(&x, "Etheta")?)?,
                    read_npy(npz_member(&y, "Ephi")?)?,
                    read_npy(npz_member(&y, "Etheta")?)?,
                ]
            }
        };

        let shape = eeps[0].raw_dim();
        if eeps.iter().any(|e| e.raw_dim() != shape) {
            return Err(SkaLowBeamError::GridShapeMismatch);
        }
        if shape[0] < 2 || shape[1] < 2 {
            return Err(SkaLowBeamError::GridTooSmall(shape.as_array_view().to_vec()));
        }

        // eep.py conjugates on load, and negates the Y polarisation to match
        // real data.
        for (i, eep) in eeps.iter_mut().enumerate() {
            let sign = if i < 2 { 1.0 } else { -1.0 };
            eep.mapv_inplace(|z| sign * z.conj());
        }

        Ok(GriddedEeps { eeps })
    }
}

/// One frequency's gridded EEPs.
pub(super) struct GriddedEeps {
    /// `[X phi, X theta, Y phi, Y theta]`, each `(n_az, n_za, n_elements)`.
    eeps: [Array3<c64>; 4],
}

impl GriddedEeps {
    pub(super) fn num_elements(&self) -> usize {
        self.eeps[0].len_of(Axis(2))
    }

    /// Per-antenna normalisation by the power integral, jointly across the two
    /// sky polarisations, as `normalise_eeps` does.
    ///
    /// `eep.py` applies this over a resampled pixel set; hyperbeam has no such
    /// set, so this integrates over the whole grid. That makes it the beam
    /// solid angle rather than a resampling correction, which is why it is
    /// opt-in.
    pub(super) fn normalise(&mut self) {
        let power = |eep: &Array3<c64>, element: usize| {
            eep.index_axis(Axis(2), element)
                .iter()
                .map(|z| z.norm_sqr())
                .sum::<f64>()
        };
        for element in 0..self.num_elements() {
            for (phi, theta) in [(0, 1), (2, 3)] {
                let norm = (power(&self.eeps[phi], element) + power(&self.eeps[theta], element))
                    .sqrt()
                    .recip();
                if norm.is_finite() {
                    for i in [phi, theta] {
                        self.eeps[i]
                            .index_axis_mut(Axis(2), element)
                            .mapv_inplace(|z| z * norm);
                    }
                }
            }
        }
    }

    /// The nearest-pixel EEP of one element.
    ///
    /// `az_rad` and `el_rad` are astronomical (East-from-North); the
    /// North-from-East conversion and the station rotation happen here.
    /// Directions below the horizon return zero, where `azel_to_eep_indices`
    /// raises.
    pub(super) fn element_pattern(
        &self,
        element: usize,
        az_rad: f64,
        el_rad: f64,
        eep_rotation_deg: f64,
    ) -> ElementPattern {
        if !(0.0..=std::f64::consts::FRAC_PI_2).contains(&el_rad) {
            return ElementPattern::default();
        }

        let (n_az, n_za) = (self.eeps[0].len_of(Axis(0)), self.eeps[0].len_of(Axis(1)));
        let daz = 360.0 / (n_az - 1) as f64;
        let dza = 90.0 / (n_za - 1) as f64;

        let az_eep = (90.0 - az_rad.to_degrees() + eep_rotation_deg).rem_euclid(360.0);
        let za_deg = 90.0 - el_rad.to_degrees();
        // ponytail: nearest pixel, matching the reference implementation.
        // Bilinear would need to avoid double-counting the duplicated az wrap
        // column; add it when the grid resolution is the limiting error.
        let iaz = (az_eep / daz).round() as usize % n_az;
        let iza = ((za_deg / dza).round() as usize).min(n_za - 1);

        let at = |i: usize| self.eeps[i][[iaz, iza, element]];
        ElementPattern {
            e_theta: [at(1), at(3)],
            e_phi: [at(0), at(2)],
        }
    }
}

/// Read a `.npy` file into an `[az, za, n_elements]` complex cube.
fn read_npy_file(path: &Path) -> Result<Array3<c64>, SkaLowBeamError> {
    let bytes = std::fs::read(path).map_err(InitSkaLowBeamError::Io)?;
    read_npy(&bytes)
}

/// Parse a `.npy` buffer: a short ASCII header giving `descr`, `fortran_order`
/// and `shape`, then the raw array.
fn read_npy(bytes: &[u8]) -> Result<Array3<c64>, SkaLowBeamError> {
    let bad = |what: &str| SkaLowBeamError::Npy(what.to_string());

    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(bad("not a .npy file (bad magic)"));
    }
    // Version 1 writes a 2-byte header length, later versions 4 bytes.
    let (header_len, header_start) = if bytes[6] == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        if bytes.len() < 12 {
            return Err(bad("truncated header"));
        }
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let data_start = header_start + header_len;
    if bytes.len() < data_start {
        return Err(bad("truncated header"));
    }
    let header = std::str::from_utf8(&bytes[header_start..data_start])
        .map_err(|_| bad("header is not UTF-8"))?;

    // The header is a Python dict literal. Pulling the three keys out by hand
    // beats a literal parser for a format this fixed.
    let value_after = |key: &str| -> Option<&str> {
        let rest = header.split_once(&format!("'{key}':"))?.1.trim_start();
        Some(rest)
    };

    let descr = value_after("descr")
        .and_then(|v| v.trim_start_matches(['\'', '"']).split(['\'', '"']).next())
        .ok_or_else(|| bad("no 'descr' in header"))?;
    let (is_f64, item_size) = match descr {
        "<c8" => (false, 8),
        "<c16" => (true, 16),
        d => return Err(SkaLowBeamError::NpyDtype(d.to_string())),
    };

    let fortran_order = value_after("fortran_order")
        .map(|v| v.starts_with("True"))
        .ok_or_else(|| bad("no 'fortran_order' in header"))?;

    let shape: Vec<usize> = value_after("shape")
        .and_then(|v| v.strip_prefix('('))
        .and_then(|v| v.split(')').next())
        .ok_or_else(|| bad("no 'shape' in header"))?
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|_| bad("bad shape")))
        .collect::<Result<_, _>>()?;
    // Antenna-averaged EEPs come as [az, za]; promote them the way load_eeps
    // does.
    let shape = match shape.len() {
        2 => (shape[0], shape[1], 1),
        3 => (shape[0], shape[1], shape[2]),
        n => return Err(SkaLowBeamError::GridDimensions(n)),
    };

    let n = shape.0 * shape.1 * shape.2;
    let raw = &bytes[data_start..];
    if raw.len() < n * item_size {
        return Err(bad("truncated data"));
    }
    // ponytail: everything is widened to complex128 so one type flows
    // downstream. A complex64 file therefore doubles in memory; keep the native
    // dtype (or memory-map) if the grids get big enough to matter.
    let data: Vec<c64> = raw
        .chunks_exact(item_size)
        .take(n)
        .map(|c| {
            if is_f64 {
                let f = |i: usize| f64::from_le_bytes(c[i..i + 8].try_into().unwrap());
                c64::new(f(0), f(8))
            } else {
                let f = |i: usize| f64::from(f32::from_le_bytes(c[i..i + 4].try_into().unwrap()));
                c64::new(f(0), f(4))
            }
        })
        .collect();

    let arr = if fortran_order {
        Array3::from_shape_vec(shape.f(), data)
    } else {
        Array3::from_shape_vec(shape, data)
    };
    arr.map_err(|_| bad("shape does not match the data"))
}

/// Find a member of a `.npz` (a ZIP of `.npy` files) and return its bytes.
///
/// `numpy.savez` stores members uncompressed, so no inflate is needed.
/// `numpy.savez_compressed` deflates them, which this reports rather than
/// silently mis-reading.
fn npz_member<'a>(bytes: &'a [u8], name: &str) -> Result<&'a [u8], SkaLowBeamError> {
    let bad = |what: &str| SkaLowBeamError::Npz(what.to_string());
    let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[i], b[i + 1]]) as usize;
    let u32_at =
        |b: &[u8], i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;

    // The end-of-central-directory record is last, after a comment of up to
    // 64 KiB.
    let eocd = bytes
        .len()
        .checked_sub(22)
        .map(|end| end.saturating_sub(u16::MAX as usize))
        .and_then(|from| {
            (from..=bytes.len() - 22)
                .rev()
                .find(|&i| bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        })
        .ok_or_else(|| bad("no end-of-central-directory record"))?;

    let n_entries = u16_at(bytes, eocd + 10);
    let mut pos = u32_at(bytes, eocd + 16);
    if pos == u32::MAX as usize {
        return Err(bad("zip64 containers are not supported"));
    }

    let wanted = format!("{name}.npy");
    for _ in 0..n_entries {
        if pos + 46 > bytes.len() || bytes[pos..pos + 4] != [0x50, 0x4b, 0x01, 0x02] {
            return Err(bad("malformed central directory"));
        }
        let method = u16_at(bytes, pos + 10);
        let compressed_size = u32_at(bytes, pos + 20);
        let name_len = u16_at(bytes, pos + 28);
        let extra_len = u16_at(bytes, pos + 30);
        let comment_len = u16_at(bytes, pos + 32);
        let local = u32_at(bytes, pos + 42);
        let entry_name = std::str::from_utf8(&bytes[pos + 46..pos + 46 + name_len])
            .map_err(|_| bad("member name is not UTF-8"))?;

        if entry_name == wanted {
            if method != 0 {
                return Err(SkaLowBeamError::NpzCompressed(entry_name.to_string()));
            }
            if local + 30 > bytes.len() || bytes[local..local + 4] != [0x50, 0x4b, 0x03, 0x04] {
                return Err(bad("malformed local file header"));
            }
            // The local header repeats the name and extra lengths, and its
            // extra field need not match the central directory's.
            let start = local + 30 + u16_at(bytes, local + 26) + u16_at(bytes, local + 28);
            let end = start + compressed_size;
            return bytes.get(start..end).ok_or_else(|| bad("truncated member"));
        }
        pos += 46 + name_len + extra_len + comment_len;
    }

    Err(SkaLowBeamError::NpzMissingMember {
        member: wanted,
        found: n_entries,
    })
}

/// The number of feeds is fixed; this keeps the constant honest if it changes.
const _: () = assert!(N_FEEDS == 2);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Build a `.npy` buffer the way numpy would.
    fn npy(shape: (usize, usize, usize), fortran: bool, f64_dtype: bool) -> Vec<u8> {
        let (descr, item) = if f64_dtype { ("<c16", 16) } else { ("<c8", 8) };
        let header = format!(
            "{{'descr': '{descr}', 'fortran_order': {}, 'shape': ({}, {}, {}), }}",
            if fortran { "True" } else { "False" },
            shape.0,
            shape.1,
            shape.2
        );
        // numpy pads the header so that the data starts on a 64-byte boundary.
        let mut header = header.into_bytes();
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(b' ');
        }
        header.push(b'\n');

        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(&header);

        // Element (i, j, k) is encoded as i*100 + j*10 + k, so that ordering
        // mistakes are visible.
        let n = shape.0 * shape.1 * shape.2;
        let value = |flat: usize| {
            let (i, j, k) = if fortran {
                (
                    flat % shape.0,
                    (flat / shape.0) % shape.1,
                    flat / (shape.0 * shape.1),
                )
            } else {
                (
                    flat / (shape.1 * shape.2),
                    (flat / shape.2) % shape.1,
                    flat % shape.2,
                )
            };
            (i * 100 + j * 10 + k) as f64
        };
        for flat in 0..n {
            let (re, im) = (value(flat), -value(flat));
            if f64_dtype {
                out.extend_from_slice(&re.to_le_bytes());
                out.extend_from_slice(&im.to_le_bytes());
            } else {
                out.extend_from_slice(&(re as f32).to_le_bytes());
                out.extend_from_slice(&(im as f32).to_le_bytes());
            }
        }
        assert_eq!(out.len(), 10 + header.len() + n * item);
        out
    }

    fn check_roundtrip(fortran: bool, f64_dtype: bool) {
        let shape = (3, 4, 2);
        let arr = read_npy(&npy(shape, fortran, f64_dtype)).unwrap();
        assert_eq!(arr.dim(), shape);
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                for k in 0..shape.2 {
                    let want = (i * 100 + j * 10 + k) as f64;
                    assert_abs_diff_eq!(arr[[i, j, k]].re, want);
                    assert_abs_diff_eq!(arr[[i, j, k]].im, -want);
                }
            }
        }
    }

    #[test]
    fn npy_c_order_complex64() {
        check_roundtrip(false, false);
    }

    #[test]
    fn npy_c_order_complex128() {
        check_roundtrip(false, true);
    }

    #[test]
    fn npy_fortran_order() {
        check_roundtrip(true, false);
    }

    #[test]
    fn npy_two_dimensional_is_promoted() {
        let mut buf = npy((3, 4, 1), false, false);
        // Rewrite the shape as 2-D, keeping the same data.
        let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
        let header = String::from_utf8(buf[10..10 + header_len].to_vec()).unwrap();
        let new = header.replace("(3, 4, 1)", "(3, 4)   ");
        buf.splice(10..10 + header_len, new.into_bytes());
        assert_eq!(read_npy(&buf).unwrap().dim(), (3, 4, 1));
    }

    #[test]
    fn npy_rejects_real_dtype() {
        let buf = npy((2, 2, 1), false, false);
        let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
        let header = String::from_utf8(buf[10..10 + header_len].to_vec()).unwrap();
        let mut bad = buf.clone();
        bad.splice(
            10..10 + header_len,
            header.replace("'<c8'", "'<f8'").into_bytes(),
        );
        assert!(matches!(read_npy(&bad), Err(SkaLowBeamError::NpyDtype(_))));
    }

    #[test]
    fn npy_rejects_garbage() {
        assert!(read_npy(b"not a numpy file at all").is_err());
    }

    /// Build a stored (uncompressed) ZIP holding the named members.
    fn zip(members: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = vec![];
        let mut directory = vec![];
        for (name, data) in members {
            let name = format!("{name}.npy");
            let local = out.len() as u32;
            out.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04]);
            out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // version..date
            out.extend_from_slice(&0u32.to_le_bytes()); // crc, unchecked
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // extra len
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(data);

            directory.extend_from_slice(&[0x50, 0x4b, 0x01, 0x02]);
            directory.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            directory.extend_from_slice(&0u32.to_le_bytes()); // crc
            directory.extend_from_slice(&(data.len() as u32).to_le_bytes());
            directory.extend_from_slice(&(data.len() as u32).to_le_bytes());
            directory.extend_from_slice(&(name.len() as u16).to_le_bytes());
            directory.extend_from_slice(&0u16.to_le_bytes()); // extra
            directory.extend_from_slice(&0u16.to_le_bytes()); // comment
            directory.extend_from_slice(&0u16.to_le_bytes()); // disk
            directory.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
            directory.extend_from_slice(&0u32.to_le_bytes()); // external attrs
            directory.extend_from_slice(&local.to_le_bytes());
            directory.extend_from_slice(name.as_bytes());
        }

        let cd_offset = out.len() as u32;
        let cd_size = directory.len() as u32;
        out.extend_from_slice(&directory);
        out.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06]);
        out.extend_from_slice(&[0, 0, 0, 0]); // disk numbers
        out.extend_from_slice(&(members.len() as u16).to_le_bytes());
        out.extend_from_slice(&(members.len() as u16).to_le_bytes());
        out.extend_from_slice(&cd_size.to_le_bytes());
        out.extend_from_slice(&cd_offset.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // comment len
        out
    }

    #[test]
    fn npz_finds_each_member() {
        let phi = npy((2, 3, 1), false, false);
        let theta = npy((3, 2, 1), false, true);
        let z = zip(&[("Ephi", phi.clone()), ("Etheta", theta.clone())]);
        assert_eq!(npz_member(&z, "Ephi").unwrap(), phi.as_slice());
        assert_eq!(npz_member(&z, "Etheta").unwrap(), theta.as_slice());
        assert_eq!(
            read_npy(npz_member(&z, "Etheta").unwrap()).unwrap().dim(),
            (3, 2, 1)
        );
    }

    #[test]
    fn npz_reports_a_missing_member() {
        let z = zip(&[("Ephi", npy((2, 2, 1), false, false))]);
        assert!(matches!(
            npz_member(&z, "Etheta"),
            Err(SkaLowBeamError::NpzMissingMember { .. })
        ));
    }

    #[test]
    fn npz_reports_deflated_members() {
        let mut z = zip(&[("Ephi", npy((2, 2, 1), false, false))]);
        // Set the compression method to deflate in both headers.
        let cd = z
            .windows(4)
            .position(|w| w == [0x50, 0x4b, 0x01, 0x02])
            .unwrap();
        z[cd + 10] = 8;
        assert!(matches!(
            npz_member(&z, "Ephi"),
            Err(SkaLowBeamError::NpzCompressed(_))
        ));
    }

    fn grid(n_az: usize, n_za: usize) -> GriddedEeps {
        // Each of the four cubes is tagged so they can't be confused.
        let cube = |tag: f64| {
            Array3::from_shape_fn((n_az, n_za, 1), |(a, z, _)| {
                c64::new(tag + a as f64, z as f64)
            })
        };
        GriddedEeps {
            eeps: [cube(0.0), cube(1000.0), cube(2000.0), cube(3000.0)],
        }
    }

    /// The EEP azimuth axis runs North-from-East: astronomical az 90 (East) is
    /// EEP az 0.
    #[test]
    fn azimuth_convention() {
        let g = grid(361, 91);
        let east = g.element_pattern(0, 90f64.to_radians(), 45f64.to_radians(), 0.0);
        assert_abs_diff_eq!(east.e_phi[0].re, 0.0);

        let north = g.element_pattern(0, 0.0, 45f64.to_radians(), 0.0);
        assert_abs_diff_eq!(north.e_phi[0].re, 90.0);

        // A station rotation shifts the sampled pixel by the same amount.
        let rotated = g.element_pattern(0, 90f64.to_radians(), 45f64.to_radians(), 30.0);
        assert_abs_diff_eq!(rotated.e_phi[0].re, 30.0);
    }

    #[test]
    fn zenith_angle_axis() {
        let g = grid(361, 91);
        let zenith = g.element_pattern(0, 0.0, std::f64::consts::FRAC_PI_2, 0.0);
        assert_abs_diff_eq!(zenith.e_phi[0].im, 0.0);
        let horizon = g.element_pattern(0, 0.0, 0.0, 0.0);
        assert_abs_diff_eq!(horizon.e_phi[0].im, 90.0);
    }

    /// theta and phi must not be swapped, and X must not be confused with Y.
    #[test]
    fn component_ordering() {
        let g = grid(361, 91);
        let p = g.element_pattern(0, 0.0, 45f64.to_radians(), 0.0);
        assert_abs_diff_eq!(p.e_phi[0].re, 90.0); // X phi, tag 0
        assert_abs_diff_eq!(p.e_theta[0].re, 1090.0); // X theta, tag 1000
        assert_abs_diff_eq!(p.e_phi[1].re, 2090.0); // Y phi, tag 2000
        assert_abs_diff_eq!(p.e_theta[1].re, 3090.0); // Y theta, tag 3000
    }

    #[test]
    fn below_the_horizon_is_zero_not_a_panic() {
        let g = grid(361, 91);
        let p = g.element_pattern(0, 0.0, -0.1, 0.0);
        assert_eq!(p.e_theta[0], c64::default());
        assert_eq!(p.e_phi[1], c64::default());
    }

    /// The whole gridded path through the public constructor: file discovery
    /// from `filebase`, the load-time conjugation and Y-pol negation, and the
    /// theta-first Jones assembly.
    #[test]
    fn end_to_end_from_npy_files() {
        let dir = std::env::temp_dir().join(format!("hyperbeam_grid_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let filebase = "HARP_SKALA41_randvogel_1m_1_elem_50ohm_";
        // A 3x3 grid: az 0/180/360, za 0/45/90. Element (i, j, 0) holds
        // i*100 + j*10, so the sampled pixel is identifiable.
        for suffix in ["Xpol_phi", "Xpol_theta", "Ypol_phi", "Ypol_theta"] {
            std::fs::write(
                dir.join(format!("{filebase}99MHz_{suffix}.npy")),
                npy((3, 3, 1), false, false),
            )
            .unwrap();
        }

        let offsets = Array2::zeros((3, 1));
        let beam = crate::ska_low::SkaLowBeam::new_from_grid(
            &dir,
            filebase,
            GridFormat::Npy,
            offsets.view(),
            None,
            0.0,
            false,
        )
        .unwrap();
        assert_eq!(beam.get_freqs(), vec![99e6]);

        // Astronomical az 90 (East) is EEP az 0, so iaz = 0; el 45 is za 45, so
        // iza = 1. The stored value is 0*100 + 1*10 = 10, conjugated on load.
        let azel = marlu::AzEl::from_radians(90f64.to_radians(), 45f64.to_radians());
        let jones = beam
            .calc_jones(azel, 99e6, &[c64::new(1.0, 0.0)], None, false)
            .unwrap();
        let stored = c64::new(10.0, -10.0);
        // X theta: conjugated only. X phi: conjugated, then negated by the
        // theta-first Jones assembly.
        assert_abs_diff_eq!(jones[0], stored.conj(), epsilon = 1e-12);
        assert_abs_diff_eq!(jones[1], -stored.conj(), epsilon = 1e-12);
        // Y: additionally negated on load, per SKAO Memo 002.
        assert_abs_diff_eq!(jones[2], -stored.conj(), epsilon = 1e-12);
        assert_abs_diff_eq!(jones[3], stored.conj(), epsilon = 1e-12);

        // Outside the 3 MHz tolerance.
        assert!(matches!(
            beam.calc_jones(azel, 120e6, &[c64::new(1.0, 0.0)], None, false),
            Err(SkaLowBeamError::NoFreqMatch { .. })
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn normalisation_is_per_antenna_and_joint_across_sky_pols() {
        let mut g = grid(5, 5);
        g.normalise();
        let power: f64 = [0, 1]
            .iter()
            .map(|&i| g.eeps[i].iter().map(|z| z.norm_sqr()).sum::<f64>())
            .sum();
        assert_abs_diff_eq!(power, 1.0, epsilon = 1e-12);
    }
}
