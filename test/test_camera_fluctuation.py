"""Issue #161: the published FAST-camera fluctuation chain, on synthetic cubes.

The published method is specified across a thesis, a paper and one surviving
MATLAB script that disagree in places, so these tests pin what was actually
chosen: the window that reproduces the paper's stated ~3 kHz behaviour, the
exponential window it names but does not parameterise, the closed band edges the
one-dimensional layer already uses, and the normalisation by local emission.

Real-data checks live in ``test_camera_fluctuation_real.py``; everything here
runs offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.camera_fluctuation import (
    BACKGROUND_FRAMES_2500FPS,
    BACKGROUND_FRAMES_50KFPS,
    EMISSION_NORMALISATION_FRAMES_50KFPS,
    MHD_BAND_HALF_WIDTH_HZ,
    REFERENCE_SEARCH_RANGE_HZ,
    SPECTRAL_WINDOW_FRAMES_50KFPS,
    mhd_band_power,
    normalize_by_local_emission,
    pixelwise_spectrogram,
    poisson_window,
    subtract_temporal_background,
    summed_region_signal,
    track_reference_frequency,
)

FRAME_RATE = 50_000.0


def _tone_cube(frequency, n_frames=400, shape=(4, 3), amplitude=1.0, offset=0.0):
    """Every pixel carrying the same tone, so an expected gain is exact."""
    time = np.arange(n_frames) / FRAME_RATE
    wave = offset + amplitude * np.sin(2 * np.pi * frequency * time)
    return time, np.broadcast_to(wave[:, None, None], (n_frames,) + shape).copy()


def _moving_average_highpass_gain(frequency, window):
    """|1 - D_n(f)|, the analytic response of subtracting a centred n-tap mean."""
    offsets = np.arange(window) - (window - 1) / 2
    kernel = np.exp(-2j * np.pi * frequency * offsets / FRAME_RATE).sum() / window
    return abs(1.0 - kernel)


class TestTemporalBackgroundSubtraction:
    def test_a_constant_background_is_removed_exactly(self):
        frames = np.full((60, 5, 4), 37.0)
        result = subtract_temporal_background(frames, window_frames=BACKGROUND_FRAMES_50KFPS)
        assert np.allclose(result, 0.0, atol=1e-12)

    def test_a_static_image_is_removed_pixel_by_pixel(self):
        """A still scene of varying brightness leaves nothing, not just a flat field."""
        scene = np.arange(20, dtype=float).reshape(4, 5)
        frames = np.broadcast_to(scene, (60, 4, 5)).copy()
        result = subtract_temporal_background(frames, window_frames=15)
        assert np.allclose(result, 0.0, atol=1e-12)

    @pytest.mark.parametrize(
        "window, frequency, expected",
        [
            (15, 3_000.0, 0.890),
            (15, 6_000.0, 1.107),
            (7, 3_000.0, 0.262),
            (7, 6_000.0, 0.813),
        ],
    )
    def test_the_window_has_the_response_the_preset_was_chosen_for(self, window, frequency, expected):
        """The 15-frame preset passes the MHD band; the 7-frame one attenuates it.

        This is why `BACKGROUND_FRAMES_50KFPS` is 15 and not 7: the paper says
        the subtraction removes components below about 3 kHz, and only the
        longer window does that while leaving a 6 kHz mode intact.
        """
        time, frames = _tone_cube(frequency)
        result = subtract_temporal_background(frames, window_frames=window)
        interior = result[window:-window, 0, 0]
        measured = interior.max() - interior.min()
        assert measured / 2.0 == pytest.approx(expected, rel=0.02)
        assert _moving_average_highpass_gain(frequency, window) == pytest.approx(expected, rel=0.02)

    def test_the_preset_keeps_more_of_the_mhd_band_than_the_shorter_one(self):
        time, frames = _tone_cube(6_000.0)
        long = subtract_temporal_background(frames, window_frames=BACKGROUND_FRAMES_50KFPS)
        short = subtract_temporal_background(frames, window_frames=7)
        assert np.ptp(long[40:-40, 0, 0]) > np.ptp(short[40:-40, 0, 0])

    def test_the_edges_shrink_the_window_rather_than_padding_with_zeros(self):
        """A zero-padded edge would leave the first frame nearly unchanged."""
        frames = np.full((60, 2, 2), 10.0)
        frames[:, 0, 0] = np.arange(60, dtype=float)
        result = subtract_temporal_background(frames, window_frames=15)
        # Frame 0 averages over frames 0..7 -> 3.5, so the residual is -3.5.
        assert result[0, 0, 0] == pytest.approx(0.0 - np.arange(0, 8).mean())
        assert result[-1, 0, 0] == pytest.approx(59.0 - np.arange(52, 60).mean())

    def test_the_2500_fps_preset_is_a_three_frame_mean(self):
        frames = np.zeros((9, 1, 1))
        frames[:, 0, 0] = [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = subtract_temporal_background(frames, window_frames=BACKGROUND_FRAMES_2500FPS)
        assert result[1, 0, 0] == pytest.approx(3.0 - 1.0)
        assert result[0, 0, 0] == pytest.approx(0.0 - 1.5)
        assert result[4, 0, 0] == pytest.approx(0.0)

    def test_frame_order_and_length_are_untouched(self):
        rng = np.random.default_rng(0)
        frames = rng.random((40, 6, 7))
        result = subtract_temporal_background(frames, window_frames=15)
        assert result.shape == frames.shape

    def test_a_one_frame_window_is_refused(self):
        with pytest.raises(ValueError, match="at least 2"):
            subtract_temporal_background(np.zeros((10, 2, 2)), window_frames=1)

    def test_a_window_longer_than_the_record_clips_the_same_way(self):
        """Allowed, and every frame is then measured against the whole record.

        Refusing it would make the answer depend on how many frames a caller
        passed rather than on the window asked for -- and the views read a span
        around one frame, so a short acquisition would silently get a different
        background from a long one.
        """
        frames = np.arange(10, dtype=float).reshape(10, 1, 1)
        result = subtract_temporal_background(frames, window_frames=101)
        np.testing.assert_allclose(result[:, 0, 0], np.arange(10) - 4.5)


class TestSummedRegionSignal:
    def test_the_whole_frame_sums_every_pixel(self):
        frames = np.ones((5, 4, 3))
        assert np.allclose(summed_region_signal(frames), 12.0)

    def test_a_region_is_row_then_column_and_half_open(self):
        frames = np.zeros((2, 4, 4))
        frames[:, 1, 2] = 5.0
        assert np.allclose(summed_region_signal(frames, region=(1, 2, 2, 3)), 5.0)
        assert np.allclose(summed_region_signal(frames, region=(0, 1, 2, 3)), 0.0)

    def test_empty_and_inverted_regions_are_refused(self):
        frames = np.zeros((2, 4, 4))
        with pytest.raises(ValueError, match="increasing"):
            summed_region_signal(frames, region=(2, 2, 0, 3))
        with pytest.raises(ValueError, match="row_start"):
            summed_region_signal(frames, region=(0, 1, 2))


class TestThePublishedWindow:
    def test_it_is_scipys_exponential_window(self):
        """The paper names an exponential peak-shaped window; substituting Hann is forbidden."""
        from scipy.signal import windows

        made = poisson_window(SPECTRAL_WINDOW_FRAMES_50KFPS)
        expected = windows.exponential(
            SPECTRAL_WINDOW_FRAMES_50KFPS, tau=SPECTRAL_WINDOW_FRAMES_50KFPS / 8.0, sym=False
        )
        np.testing.assert_allclose(made, expected)
        assert not np.allclose(made, windows.hann(SPECTRAL_WINDOW_FRAMES_50KFPS, sym=False))

    def test_it_peaks_in_the_middle_and_decays_to_the_edges(self):
        window = poisson_window(50)
        assert 20 <= int(np.argmax(window)) <= 30
        assert window[0] < 0.05 * window.max()
        assert window[-1] < 0.05 * window.max()

    def test_the_decay_is_a_stated_choice_the_caller_can_change(self):
        slow = poisson_window(50, decay_frames=25.0)
        fast = poisson_window(50, decay_frames=3.0)
        assert slow[0] > fast[0]

    def test_a_degenerate_window_is_refused(self):
        with pytest.raises(ValueError, match="at least 2"):
            poisson_window(1)
        with pytest.raises(ValueError, match="positive"):
            poisson_window(50, decay_frames=0.0)


class TestPixelwiseSpectrogram:
    def test_a_non_uniform_time_axis_is_refused(self):
        """The 1-D sibling rejects it; a median step would silently mis-scale the axis."""
        _, frames = _tone_cube(6_000.0, n_frames=300)
        time = np.arange(300) / FRAME_RATE
        time[150:] += 0.01
        with pytest.raises(ValueError, match="uniformly sampled"):
            pixelwise_spectrogram(frames, time, window_frames=50)

    def test_the_axes_lead_and_the_pixels_trail(self):
        time, frames = _tone_cube(6_000.0, n_frames=300, shape=(4, 5))
        result = pixelwise_spectrogram(frames, time, window_frames=50)
        assert result.magnitude.shape == (
            result.frequency.size,
            result.time.size,
            4,
            5,
        )
        assert result.pixel_shape == (4, 5)

    def test_the_time_axis_is_the_callers_own(self):
        time, frames = _tone_cube(6_000.0, n_frames=300)
        shifted = time + 0.307
        result = pixelwise_spectrogram(frames, shifted, window_frames=50)
        assert result.time[0] >= 0.307
        assert result.time[-1] <= shifted[-1]

    def test_each_pixel_gets_its_own_transform(self):
        """Two pixels carrying different tones must peak at different frequencies."""
        time = np.arange(300) / FRAME_RATE
        frames = np.zeros((300, 1, 2))
        frames[:, 0, 0] = np.sin(2 * np.pi * 6_000.0 * time)
        frames[:, 0, 1] = np.sin(2 * np.pi * 12_000.0 * time)
        result = pixelwise_spectrogram(frames, time, window_frames=50)
        mid = result.time.size // 2
        first = result.frequency[np.argmax(result.magnitude[:, mid, 0, 0])]
        second = result.frequency[np.argmax(result.magnitude[:, mid, 0, 1])]
        assert first == pytest.approx(6_000.0, abs=1_100.0)
        assert second == pytest.approx(12_000.0, abs=1_100.0)

    def test_a_mismatched_time_axis_is_refused(self):
        _, frames = _tone_cube(6_000.0, n_frames=300)
        with pytest.raises(ValueError, match="each frame needs its own time"):
            pixelwise_spectrogram(frames, np.arange(10) / FRAME_RATE)

    def test_a_window_longer_than_the_record_is_refused(self):
        time, frames = _tone_cube(6_000.0, n_frames=30)
        with pytest.raises(ValueError, match="exceeds"):
            pixelwise_spectrogram(frames, time, window_frames=50)


class TestMhdBandPower:
    @staticmethod
    def _spectrogram(frequency, n_frames=400):
        time, frames = _tone_cube(frequency, n_frames=n_frames, shape=(2, 2))
        return pixelwise_spectrogram(frames, time, window_frames=50)

    def test_an_in_band_tone_is_recovered_and_an_off_band_one_is_not(self):
        in_band = mhd_band_power(self._spectrogram(6_000.0), centre_frequency=6_000.0)
        off_band = mhd_band_power(self._spectrogram(20_000.0), centre_frequency=6_000.0)
        assert in_band.mean() > 20 * off_band.mean()

    def test_the_result_is_frame_first(self):
        spectrogram = self._spectrogram(6_000.0)
        power = mhd_band_power(spectrogram, centre_frequency=6_000.0)
        assert power.shape == (spectrogram.time.size, 2, 2)

    def test_at_the_published_settings_the_band_is_one_bin(self):
        """1 ms resolves 1 kHz, so `6 +/- 0.5 kHz` is the single 6 kHz bin.

        This is why the reduction is a sum over bins rather than the trapezoidal
        integral `vaft.process.fluctuation.compute_band_power` applies to a
        density: that rule returns zero for a band holding fewer than two bins,
        which at the published window is every band.
        """
        spectrogram = self._spectrogram(6_000.0)
        assert spectrogram.frequency[1] - spectrogram.frequency[0] == pytest.approx(1_000.0)
        power = mhd_band_power(
            spectrogram, centre_frequency=6_000.0, half_width=MHD_BAND_HALF_WIDTH_HZ
        )
        bin_index = int(np.argmin(np.abs(spectrogram.frequency - 6_000.0)))
        np.testing.assert_allclose(power[:, 1, 0], spectrogram.magnitude[bin_index, :, 1, 0])
        assert power[:, 1, 0].max() > 0.0

    def test_a_wider_band_sums_more_bins(self):
        spectrogram = self._spectrogram(6_000.0)
        one_bin = mhd_band_power(spectrogram, centre_frequency=6_000.0, half_width=500.0)
        three_bins = mhd_band_power(spectrogram, centre_frequency=6_000.0, half_width=1_500.0)
        window = 3
        expected = spectrogram.magnitude[5:8, window, 1, 0].sum()
        assert three_bins[window, 1, 0] == pytest.approx(expected)
        assert three_bins[window, 1, 0] > one_bin[window, 1, 0]

    def test_a_tracked_centre_may_move_between_windows(self):
        spectrogram = self._spectrogram(6_000.0)
        centres = np.full(spectrogram.time.size, 6_000.0)
        centres[0] = 20_000.0
        power = mhd_band_power(spectrogram, centre_frequency=centres)
        assert power[0].mean() < power[1].mean() / 10

    def test_a_window_with_no_tracked_frequency_is_nan_not_zero(self):
        spectrogram = self._spectrogram(6_000.0)
        centres = np.full(spectrogram.time.size, 6_000.0)
        centres[2] = np.nan
        power = mhd_band_power(spectrogram, centre_frequency=centres)
        assert np.all(np.isnan(power[2]))
        assert np.all(np.isfinite(power[3]))

    def test_a_band_holding_no_bin_at_all_reports_zero(self):
        spectrogram = self._spectrogram(6_000.0)
        power = mhd_band_power(spectrogram, centre_frequency=6_400.0, half_width=1.0)
        assert np.all(power == 0.0)

    def test_a_mismatched_centre_series_is_refused(self):
        spectrogram = self._spectrogram(6_000.0)
        with pytest.raises(ValueError, match="one value or one per window"):
            mhd_band_power(spectrogram, centre_frequency=np.zeros(3))


class TestTrackReferenceFrequency:
    class _Result:
        def __init__(self, frequency, magnitude):
            self.frequency = frequency
            self.magnitude = magnitude
            self.time = np.arange(magnitude.shape[1], dtype=float)

    def test_it_follows_a_moving_peak(self):
        frequency = np.arange(0.0, 25_000.0, 1_000.0)
        magnitude = np.zeros((frequency.size, 3))
        magnitude[6, 0] = 1.0     # 6 kHz
        magnitude[9, 1] = 1.0     # 9 kHz
        magnitude[7, 2] = 1.0     # 7 kHz
        tracked = track_reference_frequency(self._Result(frequency, magnitude))
        np.testing.assert_allclose(tracked, [6_000.0, 9_000.0, 7_000.0])

    def test_the_low_edge_rejects_the_slow_equilibrium_drift(self):
        """Without the lower bound the peak sits on the field's own slow change."""
        frequency = np.arange(0.0, 25_000.0, 1_000.0)
        magnitude = np.zeros((frequency.size, 1))
        magnitude[0, 0] = 10.0    # DC, far stronger
        magnitude[6, 0] = 1.0
        tracked = track_reference_frequency(self._Result(frequency, magnitude))
        assert tracked[0] == pytest.approx(6_000.0)

    def test_the_high_edge_keeps_the_search_in_the_mhd_range(self):
        frequency = np.arange(0.0, 25_000.0, 1_000.0)
        magnitude = np.zeros((frequency.size, 1))
        magnitude[20, 0] = 10.0
        magnitude[6, 0] = 1.0
        tracked = track_reference_frequency(self._Result(frequency, magnitude))
        assert tracked[0] == pytest.approx(6_000.0)

    def test_the_default_range_is_the_legacy_one_in_hertz(self):
        assert REFERENCE_SEARCH_RANGE_HZ == (3_000.0, 15_000.0)

    def test_an_empty_window_reports_no_frequency(self):
        frequency = np.arange(0.0, 25_000.0, 1_000.0)
        magnitude = np.zeros((frequency.size, 2))
        magnitude[6, 1] = 1.0
        tracked = track_reference_frequency(self._Result(frequency, magnitude))
        assert np.isnan(tracked[0])
        assert tracked[1] == pytest.approx(6_000.0)

    def test_a_pixel_cube_is_refused(self):
        frequency = np.arange(0.0, 25_000.0, 1_000.0)
        with pytest.raises(ValueError, match=r"\(frequency, time\)"):
            track_reference_frequency(self._Result(frequency, np.zeros((frequency.size, 2, 4))))


class TestNormalizeByLocalEmission:
    def test_it_divides_by_the_local_mean_of_the_unsubtracted_frames(self):
        frames = np.full((40, 2, 2), 4.0)
        frame_time = np.arange(40) / FRAME_RATE
        power = np.full((3, 2, 2), 8.0)
        power_time = frame_time[[5, 20, 30]]
        result = normalize_by_local_emission(
            power, frames, frame_time=frame_time, power_time=power_time
        )
        assert np.allclose(result, 2.0)

    def test_the_window_is_the_published_ten_frames(self):
        assert EMISSION_NORMALISATION_FRAMES_50KFPS == 10

    def test_brighter_pixels_are_damped_relative_to_dim_ones(self):
        """This is the point of the normalisation: structure, not brightness."""
        frames = np.ones((40, 1, 2))
        frames[:, 0, 1] = 10.0
        frame_time = np.arange(40) / FRAME_RATE
        power = np.ones((1, 1, 2))
        result = normalize_by_local_emission(
            power, frames, frame_time=frame_time, power_time=frame_time[[20]]
        )
        assert result[0, 0, 0] == pytest.approx(1.0)
        assert result[0, 0, 1] == pytest.approx(0.1)

    def test_a_dark_pixel_yields_zero_rather_than_infinity(self):
        frames = np.ones((40, 1, 2))
        frames[:, 0, 1] = 0.0
        frame_time = np.arange(40) / FRAME_RATE
        power = np.ones((1, 1, 2))
        result = normalize_by_local_emission(
            power, frames, frame_time=frame_time, power_time=frame_time[[20]]
        )
        assert np.all(np.isfinite(result))
        assert result[0, 0, 1] == 0.0

    def test_a_raised_floor_refuses_more(self):
        frames = np.ones((40, 1, 2))
        frames[:, 0, 1] = 0.5
        frame_time = np.arange(40) / FRAME_RATE
        power = np.ones((1, 1, 2))
        result = normalize_by_local_emission(
            power, frames, frame_time=frame_time, power_time=frame_time[[20]], floor=0.75
        )
        assert result[0, 0, 0] == pytest.approx(1.0)
        assert result[0, 0, 1] == 0.0

    def test_the_two_time_bases_are_matched_by_time_not_by_index(self):
        """Power window k is not frame k; a shifted base must still find its frame."""
        frames = np.ones((40, 1, 1))
        frames[30:, 0, 0] = 4.0
        frame_time = np.arange(40) / FRAME_RATE
        power = np.ones((2, 1, 1))
        result = normalize_by_local_emission(
            power,
            frames,
            frame_time=frame_time,
            power_time=frame_time[[1, 36]],
            window_frames=4,
        )
        assert result[0, 0, 0] == pytest.approx(1.0)
        assert result[1, 0, 0] < 0.3

    def test_mismatched_shapes_are_refused(self):
        frames = np.ones((40, 2, 2))
        frame_time = np.arange(40) / FRAME_RATE
        with pytest.raises(ValueError, match="pixel shape"):
            normalize_by_local_emission(
                np.ones((2, 3, 3)), frames, frame_time=frame_time, power_time=frame_time[[0, 1]]
            )
        with pytest.raises(ValueError, match="power_time"):
            normalize_by_local_emission(
                np.ones((2, 2, 2)), frames, frame_time=frame_time, power_time=frame_time[[0]]
            )


class TestTheChainHoldsTogether:
    def test_a_rotating_structure_survives_the_whole_chain(self):
        """Background removal, per-pixel transform, band power, normalisation."""
        n_frames = 400
        time = np.arange(n_frames) / FRAME_RATE
        rows, cols = 6, 8
        column = np.arange(cols)
        # A 6 kHz oscillation whose phase varies across the image, on a bright,
        # slowly drifting background that the subtraction must remove.
        phase = 2 * np.pi * column / cols
        oscillation = np.sin(2 * np.pi * 6_000.0 * time[:, None] + phase[None, :])
        background = 50.0 + 10.0 * np.sin(2 * np.pi * 200.0 * time)[:, None]
        frames = np.broadcast_to(
            (background + oscillation)[:, None, :], (n_frames, rows, cols)
        ).copy()

        fluctuation = subtract_temporal_background(frames)
        spectrogram = pixelwise_spectrogram(fluctuation, time)
        centres = np.full(spectrogram.time.size, 6_000.0)
        power = mhd_band_power(spectrogram, centre_frequency=centres, half_width=MHD_BAND_HALF_WIDTH_HZ)
        normalised = normalize_by_local_emission(
            power, frames, frame_time=time, power_time=spectrogram.time
        )

        assert normalised.shape == (spectrogram.time.size, rows, cols)
        assert np.all(np.isfinite(normalised))
        assert normalised.max() > 0.0
        # The oscillation is uniform in amplitude, so after normalising by the
        # (uniform) emission every pixel should carry comparable power.
        spread = normalised[1:-1].std(axis=0).mean()
        assert spread < normalised[1:-1].mean()
