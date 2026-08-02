// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Example code using hyperbeam's SKA-Low beam with Rust.
//!
//! Build and run with something like:
//! `cargo run --release --example ska_low -- 10000 /path/to/skalow/coeffs`
//!
//! The beam files describe a single element; the station geometry comes from
//! the caller. Here the layout is a made-up ring, because hyperbeam does not
//! read MeasurementSets - in practice these come from the MSv3 `PHASED_ARRAY`
//! subtable.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use mwa_hyperbeam::{ska_low::SkaLowBeam, AzEl};
use ndarray::Array2;
use num_complex::Complex64 as c64;

fn main() {
    if let Err(e) = try_main() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn try_main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let num_directions: usize = args
        .next()
        .expect("number of directions supplied")
        .parse()
        .expect("number of directions is a number");
    let beam_dir = args.next();

    // A stand-in station layout: elements on a 20 m ring. Replace this with
    // ELEMENT_OFFSET from the PHASED_ARRAY subtable.
    let num_elements = 16;
    let mut element_offsets = Array2::zeros((3, num_elements));
    for i in 0..num_elements {
        let angle = TAU * i as f64 / num_elements as f64;
        element_offsets[[0, i]] = 20.0 * angle.cos(); // East
        element_offsets[[1, i]] = 20.0 * angle.sin(); // North
    }

    // If we were given a directory, use it. Otherwise fall back on
    // SKA_LOW_BEAM_DIR.
    let beam = match beam_dir {
        Some(d) => SkaLowBeam::new(d, element_offsets.view(), None, false)?,
        None => SkaLowBeam::new_from_env(element_offsets.view(), None, false)?,
    };
    println!(
        "{} elements, {} frequencies available",
        beam.num_elements(),
        beam.get_freqs().len()
    );

    // Set up the directions to test.
    let mut azels = vec![];
    for i in 0..num_directions {
        let az = 0.9 * PI * i as f64 / num_directions as f64;
        let za = 0.1 + 0.9 * PI / 2.0 * i as f64 / num_directions as f64;
        azels.push(AzEl::from_radians(az, FRAC_PI_2 - za));
    }

    let freq_hz = 100e6;
    // The complex excitation of each element. A flagged element is one with a
    // weight of zero; there is no separate flag argument.
    let weights = vec![c64::new(1.0, 0.0); beam.num_elements()];
    let latitude_rad = Some(-0.4646099); // SKA-Low
    let iau_order = true;

    let jones = beam.calc_jones_array(&azels, freq_hz, &weights, latitude_rad, iau_order)?;

    println!("The first Jones matrix:");
    let j = jones[0];
    println!(
        "[[{:+.8}, {:+.8}]\n [{:+.8}, {:+.8}]]",
        j[0], j[1], j[2], j[3]
    );

    Ok(())
}
