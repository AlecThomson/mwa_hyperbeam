"""Benchmark (and cross-check) hyperbeam's SKA-Low element models against
EveryBeam's Python bindings. Both are vectorised over directions.

Requirements: `pip install everybeam h5py` and hyperbeam's Python module
(`maturin develop --release --features python`).

Usage:
    python benches/ska_vs_everybeam.py make-grid GRID_FILE
    python benches/ska_vs_everybeam.py GRID_FILE FEKO_DIR N_DIRECTIONS

`make-grid` writes a synthetic EEP grid (1 deg in phi and theta, 31
frequencies) with random values. FEKO_DIR is a directory of SKA-Low FEKO
coefficient files, e.g. EveryBeam's `coeffs/skalow`. Set RAYON_NUM_THREADS to
control hyperbeam's threads (EveryBeam's Python calls are single-threaded).
"""
import os
import sys
import time

import numpy as np

if len(sys.argv) == 3 and sys.argv[1] == "make-grid":
    import h5py

    phi = np.deg2rad(np.arange(0, 361, 1.0))
    theta = np.deg2rad(np.arange(0, 91, 1.0))
    freq = np.linspace(50e6, 350e6, 31)
    g = np.random.default_rng(0).standard_normal((phi.size, theta.size, freq.size, 4, 2))
    with h5py.File(sys.argv[2], "w") as f:
        grp = f.create_group("averageEEP")
        grp["Phi"], grp["Theta"], grp["Freq"], grp["averageEEP"] = phi, theta, freq, g
    sys.exit(0)

import everybeam as eb  # noqa: E402
import mwa_hyperbeam as hb  # noqa: E402

grid, feko_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rng = np.random.default_rng(42)
az = rng.uniform(0, 2 * np.pi, n)
za = rng.uniform(0, np.deg2rad(89), n)
theta, phi = za, (np.pi / 2 - az) % (2 * np.pi)
# hyperbeam's calc_jones gives [X_theta, X_az, Y_theta, Y_az] with X_az = -X_phi.
FLIP = np.array([1, -1, 1, -1])


def best(f, reps=3):
    ts, out = [], None
    for _ in range(reps):
        t = time.perf_counter()
        out = f()
        ts.append(time.perf_counter() - t)
    return min(ts), out


def eb_model(name, path=None):
    o = eb.Options()
    o.element_response_model = getattr(eb.ElementResponseModel, name)
    if path:
        o.coeff_path = path
    return eb.ElementResponse.create(o)


def compare(label, eb_f, hb_f, perm=(0, 1, 2, 3), n_pts=n):
    t_eb, r_eb = best(eb_f)
    t_hb, r_hb = best(hb_f)
    r_eb = r_eb.reshape(-1, 4)[:, list(perm)]
    r_hb = r_hb * FLIP
    err = np.max(np.abs(r_eb - r_hb)) / np.max(np.abs(r_eb))
    print(f"| {label} | {n_pts:,} | {t_eb * 1e3:.1f} | {t_hb * 1e3:.1f} | "
          f"{t_eb / t_hb:.1f}x | {err:.1e} |", flush=True)


print(f"RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', 'all')}")
print("| model | directions | EveryBeam [ms] | hyperbeam [ms] | speed-up | max rel. diff |")
print("|---|---|---|---|---|---|")

freq = 150e6
f = np.full(n, freq)
m_eb = eb_model("oskar_dipole_cos")
m_hb = hb.SkaLowAnalytic()
compare("ska-low-analytic (oskar_dipole_cos)",
        lambda: m_eb.response(f, theta, phi),
        lambda: m_hb.calc_jones_array(az, za, freq))

# EveryBeam 0.9.1 (the newest wheel) predates master's [J01, J00, J11, J10]
# storage order and SKA-Low convention, so compare against the unconverted grid
# with the old order.
m_eb = eb_model("gridded_eep", grid)
m_hb = hb.SkaLowEepGrid(grid, ska_low_convention=False)
f = np.full(n, 123.4e6)
compare("ska-low-eep-grid (gridded_eep)",
        lambda: m_eb.response(f, theta, phi),
        lambda: m_hb.calc_jones_array(az, za, 123.4e6),
        perm=(1, 0, 3, 2))

m_eb = eb_model("skalow_feko", feko_dir)
m_hb = hb.SkaLowEepSph(feko_dir)
n_sph = min(n, 20_000)
f = np.full(n_sph, 100e6)
ids = np.zeros(n_sph, dtype=np.int32)
compare("ska-low-eep-sph (skalow_feko), 1 element",
        lambda: m_eb.response(ids, f, theta[:n_sph], phi[:n_sph]),
        lambda: m_hb.calc_jones_array(az[:n_sph], za[:n_sph], 100e6, element_id=0),
        n_pts=n_sph)

# A full 256-element station's worth of element responses for a few directions.
n_dirs = 200
ids = np.repeat(np.arange(256, dtype=np.int32), n_dirs)
f = np.full(ids.size, 100e6)
th = np.tile(theta[:n_dirs], 256)
ph = np.tile(phi[:n_dirs], 256)


def hb_all_elements():
    return np.concatenate([
        m_hb.calc_jones_array(az[:n_dirs], za[:n_dirs], 100e6, element_id=int(e))
        for e in range(256)
    ])


compare("ska-low-eep-sph (skalow_feko), 256 elements",
        lambda: m_eb.response(ids, f, th, ph),
        hb_all_elements,
        n_pts=ids.size)
