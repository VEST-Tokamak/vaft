"""Issue #161: the three camera-fluctuation views, on a synthetic camera ODS.

The views must consume analysis results rather than re-derive them, so these
tests check that what reaches the screen is what
:mod:`vaft.process.camera_fluctuation` computed, and that the options the
published method needs -- the background window, the summed region, the band --
actually reach it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from omas import ODS

import vaft
from vaft.machine_mapping.camera_visible import (
    vfit_camera_visible_dynamic,
    vfit_camera_visible_static,
)
from vaft.plot.backend.recipes import build_model
from vaft.plot.models import Image2D, Spectrogram
from vaft.process.camera_fluctuation import subtract_temporal_background

FRAME_RATE = 50_000.0
ROWS, COLS = 12, 16


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def camera_ods():
    """A bright, slowly drifting scene with a 6 kHz oscillation on its right half."""
    n_frames = 200
    time = np.arange(n_frames) / FRAME_RATE + 0.3
    background = 80.0 + 20.0 * np.sin(2 * np.pi * 300.0 * time)
    frames = np.broadcast_to(background[:, None, None], (n_frames, ROWS, COLS)).copy()
    oscillation = 15.0 * np.sin(2 * np.pi * 6_000.0 * time)
    frames[:, :, COLS // 2 :] += oscillation[:, None, None]

    ods = ODS()
    vfit_camera_visible_static(ods, lines_n=ROWS, columns_n=COLS, exposure_time_s=1.91e-5)
    vfit_camera_visible_dynamic(ods, images=list(frames.astype(int)), times_s=list(time))
    # `image_raw` is an INT_2D node, so what the views read back is the rounded
    # cube; comparisons must start from that, not from the float it was made of.
    prefix = "camera_visible.channel.0.detector.0.frame"
    stored = np.stack(
        [np.asarray(ods[f"{prefix}.{i}.image_raw"], dtype=float) for i in range(n_frames)]
    )
    return ods, time, stored


def _entries(ods):
    from vaft.omas.entries import normalize_entries

    return normalize_entries(ods)


class TestTheFluctuationFrame:
    def test_it_shows_what_the_process_layer_computed(self, camera_ods):
        """Not merely 'an image': the drawn values must be the subtraction's own."""
        ods, time, frames = camera_ods
        target = 100
        model = build_model(
            "camera_visible_image_fluctuation", _entries(ods), frame_index=target
        )
        assert isinstance(model, Image2D)
        expected = subtract_temporal_background(frames, window_frames=15)[target]
        np.testing.assert_allclose(model.values, expected, atol=1e-9)

    def test_reading_only_the_window_matches_reading_the_whole_record(self, camera_ods):
        """The builder reads a window of frames; that must not change the answer."""
        ods, _time, frames = camera_ods
        for target in (0, 7, 100, 199):
            model = build_model(
                "camera_visible_image_fluctuation", _entries(ods), frame_index=target
            )
            expected = subtract_temporal_background(frames, window_frames=15)[target]
            np.testing.assert_allclose(model.values, expected, atol=1e-9, err_msg=f"frame {target}")

    def test_a_record_shorter_than_the_window_still_gets_that_window(self):
        """Published shot 28928 holds fourteen frames; the preset is fifteen.

        Clamping the window to the record would substitute a narrower background
        than the one the title claims, and would make a short acquisition differ
        from a fourteen-frame stretch of a long one.
        """
        n_frames = 14
        time = np.arange(n_frames) / FRAME_RATE + 0.3
        frames = (np.arange(n_frames)[:, None, None] * np.ones((1, ROWS, COLS))).astype(int)
        ods = ODS()
        vfit_camera_visible_static(ods, lines_n=ROWS, columns_n=COLS)
        vfit_camera_visible_dynamic(ods, images=list(frames), times_s=list(time))

        model = build_model(
            "camera_visible_image_fluctuation", _entries(ods), frame_index=0
        )
        # The 15-frame window centred on frame 0 clips to frames 0..7.
        assert model.values[0, 0] == pytest.approx(0.0 - np.arange(0, 8).mean())
        expected = subtract_temporal_background(
            np.asarray(frames, dtype=float), window_frames=15
        )[0]
        np.testing.assert_allclose(model.values, expected, atol=1e-9)

    def test_the_background_window_reaches_the_analysis(self, camera_ods):
        ods, _time, frames = camera_ods
        narrow = build_model(
            "camera_visible_image_fluctuation", _entries(ods),
            frame_index=100, background_frames=3,
        )
        expected = subtract_temporal_background(frames, window_frames=3)[100]
        np.testing.assert_allclose(narrow.values, expected, atol=1e-9)

    def test_the_title_states_the_window_that_was_used(self, camera_ods):
        ods, _time, _frames = camera_ods
        model = build_model(
            "camera_visible_image_fluctuation", _entries(ods), frame_index=100, background_frames=7
        )
        assert "7-frame background removed" in model.title

    def test_it_renders(self, camera_ods):
        ods, _time, _frames = camera_ods
        figure, axes = vaft.omas.plot_camera_visible_image_fluctuation(ods, frame_index=100)
        assert figure is not None and axes.images


class TestTheCameraSpectrogram:
    def test_it_finds_the_injected_component(self, camera_ods):
        ods, _time, _frames = camera_ods
        model = build_model(
            "camera_visible_spectrogram", _entries(ods),
            region=(0, ROWS, COLS // 2, COLS), nperseg=50,
        )
        assert isinstance(model, Spectrogram)
        peak = model.frequency[np.argmax(model.magnitude.mean(axis=1))]
        assert peak == pytest.approx(6_000.0, abs=1_100.0)

    def test_the_region_decides_what_is_summed(self, camera_ods):
        """The oscillation is on the right half only, so the left half must be quiet."""
        ods, _time, _frames = camera_ods
        right = build_model(
            "camera_visible_spectrogram", _entries(ods),
            region=(0, ROWS, COLS // 2, COLS), nperseg=50,
        )
        left = build_model(
            "camera_visible_spectrogram", _entries(ods),
            region=(0, ROWS, 0, COLS // 2), nperseg=50,
        )
        band = (right.frequency >= 5_000.0) & (right.frequency <= 7_000.0)
        assert right.magnitude[band].mean() > 10 * left.magnitude[band].mean()

    def test_summing_before_subtracting_is_the_same_as_after(self, camera_ods):
        """The builder sums the region first; both operations are linear.

        If that shortcut were wrong the spectrum would differ from the honest
        order, which is what this compares against.
        """
        from vaft.process.camera_fluctuation import summed_region_signal
        from vaft.process.fluctuation import compute_spectrogram

        ods, time, frames = camera_ods
        region = (0, ROWS, COLS // 2, COLS)
        honest = summed_region_signal(
            subtract_temporal_background(frames, window_frames=15), region=region
        )
        reference = compute_spectrogram(time, honest, nperseg=50, overlap=0.5)
        model = build_model(
            "camera_visible_spectrogram", _entries(ods), region=region, nperseg=50
        )
        np.testing.assert_allclose(model.magnitude, reference.magnitude, rtol=1e-9, atol=1e-9)

    def test_the_time_range_selects_frames(self, camera_ods):
        ods, time, _frames = camera_ods
        model = build_model(
            "camera_visible_spectrogram", _entries(ods),
            time_range=(time[50], time[150]), nperseg=50,
        )
        assert model.time.min() >= time[50]
        assert model.time.max() <= time[150]

    def test_a_time_range_holding_no_frames_says_what_the_record_spans(self, camera_ods):
        ods, _time, _frames = camera_ods
        with pytest.raises(ValueError, match="the record spans"):
            build_model("camera_visible_spectrogram", _entries(ods), time_range=(0.9, 1.0))

    def test_it_renders(self, camera_ods):
        ods, _time, _frames = camera_ods
        figure, axes = vaft.omas.plot_camera_visible_spectrogram(ods, nperseg=50)
        assert figure is not None and axes.collections

    def test_it_gets_the_shared_default_display_band(self):
        """The content is at 6 kHz of a 25 kHz axis; #765's rule must apply here too.

        A quieter scene than the shared fixture: `image_raw` is an integer node,
        so a 15-count oscillation carries a rounding floor near 1% of its own
        peak, which is exactly the level at which the shared rule -- correctly --
        declines to zoom.
        """
        n_frames = 200
        time = np.arange(n_frames) / FRAME_RATE + 0.3
        frames = np.full((n_frames, ROWS, COLS), 2_000.0)
        frames[:, :, COLS // 2 :] += 400.0 * np.sin(2 * np.pi * 6_000.0 * time)[:, None, None]
        ods = ODS()
        vfit_camera_visible_static(ods, lines_n=ROWS, columns_n=COLS)
        vfit_camera_visible_dynamic(ods, images=list(frames.astype(int)), times_s=list(time))

        region = (0, ROWS, COLS // 2, COLS)
        model = build_model(
            "camera_visible_spectrogram", _entries(ods), region=region, nperseg=50
        )
        assert model.max_frequency is not None
        assert model.max_frequency < model.frequency[-1] / 2, (
            "a 6 kHz component on a 25 kHz axis must not be drawn on the whole band"
        )
        explicit = build_model(
            "camera_visible_spectrogram", _entries(ods), region=region,
            nperseg=50, max_frequency=20_000.0,
        )
        assert explicit.max_frequency == 20_000.0


class TestTheMhdPowerImage:
    def test_the_band_is_the_one_asked_for(self, camera_ods):
        ods, _time, _frames = camera_ods
        model = build_model(
            "camera_visible_image_mhd_power", _entries(ods),
            frame_index=100, centre_frequency=6_000.0,
        )
        assert isinstance(model, Image2D)
        assert "6.0 kHz band power" in model.title
        assert np.all(np.isfinite(model.values))

    def test_it_separates_the_oscillating_half_from_the_still_one(self, camera_ods):
        ods, _time, _frames = camera_ods
        model = build_model(
            "camera_visible_image_mhd_power", _entries(ods),
            frame_index=100, centre_frequency=6_000.0,
        )
        left = model.values[:, : COLS // 2].mean()
        right = model.values[:, COLS // 2 :].mean()
        assert right > 10 * left

    def test_without_a_stated_band_it_says_where_the_centre_came_from(self, camera_ods):
        """The published method takes the centre from magnetics; a camera-only
        product has none, and the plot must not quietly pretend otherwise."""
        ods, _time, _frames = camera_ods
        model = build_model(
            "camera_visible_image_mhd_power", _entries(ods), frame_index=100
        )
        assert "camera's own dominant component" in model.title
        assert "magnetic probe" in model.title

    def test_a_record_too_short_for_the_transform_is_refused(self, camera_ods):
        ods, _time, _frames = camera_ods
        with pytest.raises(ValueError, match="the record holds 200"):
            build_model(
                "camera_visible_image_mhd_power", _entries(ods),
                frame_index=100, window_frames=400,
            )

    @pytest.mark.parametrize("target", [0, 3, 100, 196, 199])
    def test_a_frame_near_either_end_still_gets_its_window(self, camera_ods, target):
        """The span slides inwards; a frame near the start is not centred but is
        perfectly analysable, and refusing it would make the first and last
        millisecond of every movie unreachable."""
        ods, _time, _frames = camera_ods
        model = build_model(
            "camera_visible_image_mhd_power", _entries(ods),
            frame_index=target, centre_frequency=6_000.0,
        )
        assert np.all(np.isfinite(model.values))
        assert model.values.shape == (ROWS, COLS)

    def test_it_renders(self, camera_ods):
        ods, _time, _frames = camera_ods
        figure, axes = vaft.omas.plot_camera_visible_image_mhd_power(
            ods, frame_index=100, centre_frequency=6_000.0
        )
        assert figure is not None and axes.images
