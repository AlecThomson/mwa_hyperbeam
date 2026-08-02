// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Build and run with something like:
// gcc -O3 -I ../include/ -L ../target/release/ -l mwa_hyperbeam ./ska_low.c -o ska_low -lm
// LD_LIBRARY_PATH=../target/release ./ska_low /path/to/skalow/coeffs

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mwa_hyperbeam.h"

// Must match the number of elements in the beam files.
#define NUM_ELEMENTS 256

void handle_hyperbeam_error(char file[], int line_num, const char function_name[]) {
    int err_length = hb_last_error_length();
    char *err = malloc(err_length * sizeof(char));
    int err_status = hb_last_error_message(err, err_length);
    if (err_status == -1) {
        printf("Something really bad happened!\n");
        exit(EXIT_FAILURE);
    }
    printf("File %s:%d: hyperbeam error in %s: %s\n", file, line_num, function_name, err);

    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        fprintf(stderr, "Expected one argument - the path to the beam directory.\n");
        exit(EXIT_FAILURE);
    }

    // A stand-in station layout: elements on a 20 m ring, as three rows of
    // (East, North, Up). In practice these come from the ELEMENT_OFFSET column
    // of the MSv3 PHASED_ARRAY subtable; hyperbeam does not read
    // MeasurementSets.
    double element_offsets[3 * NUM_ELEMENTS];
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        double angle = 2.0 * M_PI * i / NUM_ELEMENTS;
        element_offsets[i] = 20.0 * cos(angle);
        element_offsets[NUM_ELEMENTS + i] = 20.0 * sin(angle);
        element_offsets[2 * NUM_ELEMENTS + i] = 0.0;
    }

    // Get a new beam object from hyperbeam. Passing NULL for the coordinate
    // axes means the offsets are already in (East, North, Up).
    SkaLowBeam *beam;
    if (new_ska_low_beam(argv[1], NUM_ELEMENTS, element_offsets, NULL, 0, &beam))
        handle_hyperbeam_error(__FILE__, __LINE__, "new_ska_low_beam");

    // Set up the direction to test.
    double az = 45.0 * M_PI / 180.0;
    double za = 30.0 * M_PI / 180.0;
    double freq_hz = 100e6;
    // The complex excitation of each element, as interleaved (real, imaginary)
    // doubles. A flagged element is one with a weight of zero.
    double weights[2 * NUM_ELEMENTS];
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        weights[2 * i] = 1.0;
        weights[2 * i + 1] = 0.0;
    }
    double latitude_rad = -0.4646099; // SKA-Low

    // hyperbeam can't pass complex numbers over the FFI boundary, so the Jones
    // matrix comes back as 8 doubles.
    double jones[8];
    if (ska_low_calc_jones(beam, az, za, freq_hz, weights, &latitude_rad, 1, jones))
        handle_hyperbeam_error(__FILE__, __LINE__, "ska_low_calc_jones");

    printf("The Jones matrix:\n");
    printf("[[%+.8f%+.8fi,", jones[0], jones[1]);
    printf("  %+.8f%+.8fi]\n", jones[2], jones[3]);
    printf(" [%+.8f%+.8fi,", jones[4], jones[5]);
    printf("  %+.8f%+.8fi]]\n", jones[6], jones[7]);

    // Freeing the beam is the caller's job.
    free_ska_low_beam(beam);

    return EXIT_SUCCESS;
}
