"""Issue #161 on real VEST data: shot 32308, camera and the paper's own probe.

The published reference shots are 26757, 26576, 29770 and 27134. Three of them
hold no camera data anywhere, and 27134 — the one that does — has no magnetics
at all, so it can only exercise the background subtraction. **32308** is the
shot this work is actually validated on: 1301 frames at 50 kFrames/s together
with legacy field 171, the probe the publications call B_theta,LFS, at its
native 250 kHz.

Everything here needs the HSDS source and the VEST SQL archive, so every test
skips when they are not reachable. They are not part of any CI gate; they are
the record of what was checked against real data, runnable on a machine that
has it.
"""

from __future__ import annotations

import numpy as np
import pytest

SHOT = 32308
CAMERA_SOURCE = "camera-visible-fluctuation"
#: Legacy field 171 -> `magnetics.b_field_pol_probe.36`, `MagneticFieldProbe_C2-05_Bz`
#: at R = 0.796 m, Z = 0.020 m: the probe the publications analyse.
LFS_MIRNOV_FIELD = 171
#: The camera window in which this shot's MHD activity and its IREs sit.
WINDOW = (0.3095, 0.3180)


@pytest.fixture(scope="module")
def camera():
    """The published camera cube, or a skip."""
    vaft = pytest.importorskip("vaft")
    try:
        ods = vaft.database.load(SHOT, source=CAMERA_SOURCE)
    except Exception as exc:  # noqa: BLE001 - any unreachable-source failure skips
        pytest.skip(f"{CAMERA_SOURCE} is not reachable: {exc}")
    prefix = "camera_visible.channel.0.detector.0.frame"
    count = len(ods[prefix])
    times = np.asarray([float(ods[f"{prefix}.{i}.time"]) for i in range(count)])
    keep = np.nonzero((times >= WINDOW[0]) & (times <= WINDOW[1]))[0]
    frames = np.stack(
        [np.asarray(ods[f"{prefix}.{i}.image_raw"], dtype=float) for i in keep]
    )
    return frames, times[keep]


@pytest.fixture(scope="module")
def lfs_mirnov():
    """The paper's probe on the shot clock, or a skip."""
    raw = pytest.importorskip("vaft.database.raw")
    try:
        time, data = raw.load_raw(SHOT, LFS_MIRNOV_FIELD)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the VEST SQL archive is not reachable: {exc}")
    if time is None:
        pytest.skip(f"field {LFS_MIRNOV_FIELD} is not archived for shot {SHOT}")
    return np.asarray(time, dtype=float), np.asarray(data, dtype=float)


def test_the_camera_is_the_published_acquisition_regime(camera):
    frames, times = camera
    assert frames.shape[1:] == (208, 208)
    rate = 1.0 / float(np.median(np.diff(times)))
    assert rate == pytest.approx(50_000.0, rel=1e-6)


def test_the_probe_and_the_camera_share_a_clock(lfs_mirnov, camera):
    """Not aligned by index: the camera window must lie inside the probe record.

    The raw archive stores this era's fast channels from zero; `load_raw`
    applies the shot-era trigger correction, which is the only reason the two
    records can be compared at all.
    """
    time, _data = lfs_mirnov
    _frames, camera_time = camera
    assert 1.0 / float(np.median(np.diff(time))) == pytest.approx(250_000.0, rel=1e-6)
    assert time[0] == pytest.approx(0.24, abs=1e-6)
    assert time[0] <= camera_time[0] and camera_time[-1] <= time[-1]


def test_the_background_subtraction_is_the_high_pass_it_claims_to_be(camera):
    """On real emission, not a synthetic tone: slow content goes, the MHD band stays.

    The measured response of the 15-frame window is what chose it over the seven
    the issue named, so the claim is checked where it matters -- on a real
    discharge, whose brightness rises, falls and bursts inside the window.
    """
    from vaft.process.camera_fluctuation import (
        subtract_temporal_background,
        summed_region_signal,
    )
    from vaft.process.fluctuation import compute_psd

    frames, times = camera
    fluctuation = subtract_temporal_background(frames)
    assert fluctuation.shape == frames.shape
    assert np.all(np.isfinite(fluctuation))

    before = compute_psd(times, summed_region_signal(frames), nperseg=128)
    after = compute_psd(times, summed_region_signal(fluctuation), nperseg=128)
    np.testing.assert_allclose(before.frequency, after.frequency)

    def band(spectrum, low, high):
        inside = (spectrum.frequency >= low) & (spectrum.frequency <= high)
        return float(spectrum.psd[inside].sum())

    slow = band(after, 0.0, 1_000.0) / band(before, 0.0, 1_000.0)
    mhd = band(after, 5_000.0, 8_000.0) / band(before, 5_000.0, 8_000.0)
    assert slow < 0.05, f"slow emission survived the subtraction ({slow:.3f} of it)"
    assert mhd > 0.5, f"the MHD band was attenuated to {mhd:.3f} of itself"


def test_the_probe_reports_a_mode_the_tracker_follows(lfs_mirnov):
    """This shot's mode chirps down through the search range before an IRE."""
    from vaft.process.camera_fluctuation import track_reference_frequency
    from vaft.process.magnetics import mirnov_spectrogram

    time, data = lfs_mirnov
    keep = (time >= WINDOW[0]) & (time <= WINDOW[1])
    spectrogram = mirnov_spectrogram(time[keep], data[keep], window_size=500, time_resolution=5)
    tracked = track_reference_frequency(spectrogram)
    usable = tracked[np.isfinite(tracked)]
    assert usable.size > 10
    assert usable.min() >= 3_000.0 and usable.max() <= 15_000.0
    # It falls rather than sitting still: this is a chirping mode, not a line.
    assert usable[: usable.size // 3].mean() > usable[-usable.size // 3 :].mean()


def test_the_filtered_image_is_structure_not_a_dark_pixel_artefact(camera, lfs_mirnov):
    """The whole chain, and the check that the result means something.

    Dividing by a local mean can manufacture a bright rim wherever the image is
    dark, so the brightest part of the filtered image is compared against the
    emission underneath it: if the structure were an artefact, it would sit where
    the emission is lowest.
    """
    from vaft.process.camera_fluctuation import (
        mhd_band_power,
        normalize_by_local_emission,
        pixelwise_spectrogram,
        subtract_temporal_background,
    )
    from vaft.process.magnetics import mirnov_spectrogram
    from vaft.process.camera_fluctuation import track_reference_frequency

    frames, times = camera
    time, data = lfs_mirnov
    keep = (time >= WINDOW[0] - 1e-3) & (time <= WINDOW[1] + 1e-3)

    fluctuation = subtract_temporal_background(frames)
    spectrogram = pixelwise_spectrogram(fluctuation, times, overlap=0.96)
    probe = mirnov_spectrogram(time[keep], data[keep], window_size=500, time_resolution=5)
    centres = np.interp(spectrogram.time, probe.time, track_reference_frequency(probe))
    power = mhd_band_power(spectrogram, centre_frequency=centres)
    normalised = normalize_by_local_emission(
        power, frames, frame_time=times, power_time=spectrogram.time
    )

    assert normalised.shape == (spectrogram.time.size,) + frames.shape[1:]
    assert np.all(np.isfinite(normalised))
    assert normalised.max() > 0.0

    step = normalised.shape[0] // 2
    image = normalised[step]
    nearest = int(np.argmin(np.abs(times - spectrogram.time[step])))
    emission = frames[max(nearest - 5, 0) : nearest + 5].mean(axis=0)
    brightest = image > np.percentile(image, 97)
    # An artefact would show the brightest filtered pixels over the darkest
    # emission; here the two are comparable.
    assert emission[brightest].mean() > 0.5 * emission[~brightest].mean()
    # And the structure is already in the un-normalised power.
    assert power[step][brightest].mean() > 2 * power[step][~brightest].mean()
