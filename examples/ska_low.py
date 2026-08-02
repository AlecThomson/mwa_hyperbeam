#!/usr/bin/env python

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Example code using hyperbeam's SKA-Low beam with Python.
#
#   python ska_low.py /path/to/skalow/coeffs
#
# The beam files describe a single element; the station geometry comes from the
# caller. The layout below is a made-up ring, because hyperbeam does not read
# MeasurementSets - in practice these come from the MSv3 PHASED_ARRAY subtable.

import sys
import time

import numpy as np

import mwa_hyperbeam

if len(sys.argv) < 2:
    sys.exit("Expected one argument - the path to the beam directory.")
beam_dir = sys.argv[1]

# A stand-in station layout: elements on a 20 m ring. Replace with
# ELEMENT_OFFSET from the PHASED_ARRAY subtable. The count has to match the
# number of elements in the beam files, which is 256 for an SKA-Low station.
num_elements = 256
angles = np.linspace(0, 2 * np.pi, num_elements, endpoint=False)
element_offsets = np.zeros((3, num_elements))
element_offsets[0] = 20.0 * np.cos(angles)  # East
element_offsets[1] = 20.0 * np.sin(angles)  # North

beam = mwa_hyperbeam.SkaLowBeam(beam_dir, element_offsets)
print(f"{beam.num_elements()} elements, {len(beam.get_freqs())} frequencies available")

# Make a lot of directions for hyperbeam to calculate in parallel.
n = 100000
az_rad = np.linspace(0, 0.9 * np.pi, n)
za_rad = np.linspace(0.1, 0.9 * np.pi / 2, n)
freq_hz = 100e6
# The complex excitation of each element. A flagged element is one with a
# weight of zero; there is no separate flag argument.
weights = np.ones(beam.num_elements(), dtype=np.complex128)
latitude_rad = -0.4646099  # SKA-Low
iau_order = True

# calc_jones is also available, but that makes a single Jones matrix at a time.
# calc_jones_array is done in parallel with Rust.
start_time = time.time()
jones = beam.calc_jones_array(az_rad, za_rad, freq_hz, weights, latitude_rad, iau_order)
duration = time.time() - start_time

print(f"Made {n} Jones matrices in {duration:.3f} seconds")
print("First Jones matrix:")
print(jones[0].reshape(2, 2))

# A gridded-EEP station is the same thing from the other file format; the
# filebase is required, because any baked-in default goes stale.
#
# beam = mwa_hyperbeam.SkaLowBeam.from_grid(
#     beam_dir,
#     "HARP_SKALA41_randvogel_1m_256_elem_50ohm_",
#     element_offsets,
#     npz=True,
# )
