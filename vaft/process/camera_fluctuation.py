"""Fluctuation analysis of FAST-camera image sequences.

Reproduces the image-processing chain published for VEST by E. C. Jung: temporal
background removal, per-pixel short-time Fourier analysis at the MHD frequency the
magnetics report, and the normalised band power that those papers use directly as
the pixel intensity of the processed image.  Everything here is array in, array
out -- no ODS, no file I/O, no plotting -- so the same operations serve any
camera whose frames arrive as a time-ordered cube.

This is the first image-cube consumer in :mod:`vaft.process`; every other module
there works one channel at a time.  The one-dimensional definitions in
:mod:`vaft.process.fluctuation` remain the reference: the transform, the closed
band edges and the trapezoidal integration are the same, applied over an extra
pair of pixel axes rather than redefined.

Notation
--------
========================  ====================================================
``frames``                ``(frame, row, column)`` intensity cube, frame first
``I_fluc``                ``frames`` minus its local temporal mean
``f_MHD``                 centre frequency, taken from a magnetic probe
``P_MHD``                 band power about ``f_MHD``, divided by local emission
========================  ====================================================

Conventions
-----------
The frame axis is first, matching how the camera mapping stores a sequence and
how :class:`vaft.plot.models.ImageSequence` consumes one.  Spectrogram magnitudes
carry ``(frequency, time)`` first and the pixel axes last, so integrating the
frequency axis away returns a cube in the original frame-first layout.

Optical intensity is not density.  VEST visible emission is dominated by line
radiation, so every quantity here is an emission fluctuation measure; nothing in
this module may be renamed to suggest a calibrated density perturbation.

Provenance
----------
.. [Jung2022] E. C. Jung *et al.*, "Observation of MHD-correlated blobs during
   internal reconnection events in VEST", *Nucl. Fusion* **62** (2022) 126029.
.. [JungThesis] E. C. Jung, "MHD-Coherent External Filaments and Internal
   Reconnection Event in VEST", M.S. thesis, Seoul National University, 2022.
.. [CameraFFT] ``CameraFFT.m`` in the VEST_Fast Camera_Diagnostics repository,
   the surviving legacy implementation of the per-pixel Fourier path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "BACKGROUND_FRAMES_2500FPS",
    "BACKGROUND_FRAMES_50KFPS",
    "EMISSION_NORMALISATION_FRAMES_50KFPS",
    "MHD_BAND_HALF_WIDTH_HZ",
    "PixelSpectrogram",
    "REFERENCE_SEARCH_RANGE_HZ",
    "SPECTRAL_WINDOW_FRAMES_50KFPS",
    "mhd_band_power",
    "normalize_by_local_emission",
    "pixelwise_spectrogram",
    "poisson_window",
    "subtract_temporal_background",
    "summed_region_signal",
    "track_reference_frequency",
]

#: Background window for 50 kFrames/s acquisitions: the analysed frame plus the
#: seven before and the seven after it [JungThesis]_.
BACKGROUND_FRAMES_50KFPS = 15

#: Background window for the 2.5 kFrames/s wide view [JungThesis]_.
BACKGROUND_FRAMES_2500FPS = 3

#: Short-time window for the per-pixel transform at 50 kFrames/s: 1 ms
#: [Jung2022]_ section 4.1.
SPECTRAL_WINDOW_FRAMES_50KFPS = 50

#: Local-emission window the band power is divided by at 50 kFrames/s: ten
#: images, 0.2 ms [Jung2022]_ section 4.1.
EMISSION_NORMALISATION_FRAMES_50KFPS = 10

#: Half-width of the filtered band; the published images are labelled
#: ``6 +/- 0.5 kHz`` [Jung2022]_ figure 3.
MHD_BAND_HALF_WIDTH_HZ = 500.0

#: Where the reference frequency is looked for, in hertz.
REFERENCE_SEARCH_RANGE_HZ = (3000.0, 15000.0)

#: Fractional departure from the median frame interval still counted as uniform,
#: the same tolerance :mod:`vaft.process.fluctuation` applies to a waveform.
NONUNIFORM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class PixelSpectrogram:
    """A short-time Fourier magnitude map held for every pixel of a frame.

    ``time`` and ``frequency`` name the axes exactly as
    :class:`vaft.process.fluctuation.FluctuationSpectrogram` does; ``magnitude``
    carries them first and the camera's own axes last, so
    ``magnitude[:, :, row, column]`` is one pixel's ordinary spectrogram and
    integrating the frequency axis away leaves a frame-first image cube.
    """

    time: np.ndarray
    frequency: np.ndarray
    magnitude: np.ndarray

    @property
    def pixel_shape(self) -> tuple[int, ...]:
        """The image shape each spectrogram belongs to."""
        return tuple(self.magnitude.shape[2:])


def _as_cube(frames: Any) -> np.ndarray:
    values = np.asarray(frames, dtype=float)
    if values.ndim < 2:
        raise ValueError(
            "frames must be a time-ordered stack of images, at least "
            f"(frame, pixel); got shape {values.shape}"
        )
    return values


def _rolling_mean(cube: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean along the frame axis, edges shrinking rather than padded."""
    from vaft.process.signal_processing import detrend_moving_average

    flipped = np.moveaxis(cube, 0, -1)
    # ``detrend_moving_average`` returns the residual; the trend is what is left.
    trend = flipped - detrend_moving_average(flipped, window)
    return np.moveaxis(trend, -1, 0)


def subtract_temporal_background(frames, *, window_frames: int = BACKGROUND_FRAMES_50KFPS) -> np.ndarray:
    """Remove each pixel's slowly varying emission by subtracting its local temporal mean.

    Parameters
    ----------
    frames : array_like
        Time-ordered image cube, frame axis first [any].
    window_frames : int, optional
        Width of the local mean, in frames [-].

    Returns
    -------
    np.ndarray
        ``frames`` minus its local temporal mean, same shape, float [any].

    Processing steps
    ----------------
    1. Take the centred rolling mean of every pixel over ``window_frames``.
    2. Subtract it from the frame at the centre of that window.

    Convention
    ----------
    The window is centred on the analysed frame and spans
    ``[i - w//2, i + (w-1)//2]``, clipped to the record, so the first and last
    frames average over a shrinking window rather than over zeros and the output
    keeps the input's length.  For an odd window -- both published presets are
    odd -- that span is symmetric; an even window takes one more frame from
    before than after.

    A window wider than the record is allowed and clips the same way, so every
    frame is measured against the whole record.  Refusing it would make the
    result depend on how many frames a caller happened to pass rather than on
    the window it asked for: a fourteen-frame acquisition and a fourteen-frame
    slice of a long one would come out differently.

    Frame values and their order are otherwise untouched: this subtracts, it does
    not filter, reorder or resample.

    Defaults
    --------
    ``BACKGROUND_FRAMES_50KFPS = 15`` and ``BACKGROUND_FRAMES_2500FPS = 3`` are
    literature values.  The 50 kFrames/s preset is the analysed frame plus seven
    before and seven after, which is what reproduces the published filter
    behaviour: measured at 50 kFrames/s this window passes 0.89 of a 3 kHz
    component and 1.11 of a 6 kHz one, matching the statement that it removes
    components below about 3 kHz while enhancing the several-kHz structures.  A
    15-frame window is therefore not interchangeable with a 7-frame one, which
    passes only 0.26 at 3 kHz and 0.81 at 6 kHz -- it would attenuate the MHD
    band this analysis exists to see.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The subtraction is zero-phase but the shrinking edge windows bias the first
    and last ``window_frames / 2`` frames toward the local level; a burst there is
    partly removed along with the background.  A single very bright frame pulls
    the mean toward itself and appears, inverted, in its neighbours.

    Provenance
    ----------
    .. [1] [Jung2022]_ section 3.1: "subtracts the average value of each pixel in
       a series of seven images before and after the analysed image ... removes
       slowly varying components (less than about 3 kHz)".
    .. [2] [JungThesis]_, the background-removal figure.
    """
    cube = _as_cube(frames)
    window = int(window_frames)
    if window < 2:
        raise ValueError(
            "window_frames must be at least 2; a one-frame mean is the frame "
            f"itself, so subtracting it yields exactly zero. Got {window_frames!r}"
        )
    return cube - _rolling_mean(cube, window)


def summed_region_signal(frames, *, region: Sequence[int] | None = None) -> np.ndarray:
    """Sum the intensity inside one image region, giving a single time series.

    Parameters
    ----------
    frames : array_like
        Time-ordered image cube, frame axis first [any].
    region : sequence of int, optional
        Pixel box as ``(row_start, row_stop, column_start, column_stop)``, the
        stops exclusive; ``None`` sums the whole frame [-].

    Returns
    -------
    np.ndarray
        One value per frame [any].

    Convention
    ----------
    Bounds are half-open and in row-then-column order, the order an image array
    is indexed with -- which is the opposite of the column-then-row order
    :func:`vaft.process.camera_geometry.project_points` returns pixels in.  A
    region is stated explicitly rather than inferred, because which part of the
    image faces the low-field side depends on the camera pose.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A plain sum, not a mean: two regions of different size are not comparable in
    magnitude. Nothing checks that the region holds plasma rather than wall.

    Provenance
    ----------
    .. [1] [Jung2022]_ figure 3, the summed camera pixel intensity viewing the
       low-field side whose spectrogram is compared with the magnetic probe's.
    """
    cube = _as_cube(frames)
    if region is None:
        return cube.reshape(cube.shape[0], -1).sum(axis=1)
    if len(region) != 4:
        raise ValueError(
            "region must be (row_start, row_stop, column_start, column_stop); "
            f"got {region!r}"
        )
    r0, r1, c0, c1 = (int(value) for value in region)
    if r0 >= r1 or c0 >= c1:
        raise ValueError(f"region bounds must be increasing; got {region!r}")
    window = cube[:, r0:r1, c0:c1]
    if window.size == 0:
        raise ValueError(
            f"region {region!r} selects no pixels of a {cube.shape[1:]} frame"
        )
    return window.reshape(window.shape[0], -1).sum(axis=1)


def poisson_window(window_frames: int, *, decay_frames: float | None = None) -> np.ndarray:
    """The exponentially decreasing peak-shaped window the published analysis uses.

    Parameters
    ----------
    window_frames : int
        Window length in frames [-].
    decay_frames : float, optional
        Length over which the weight falls by ``1/e`` [-].

    Returns
    -------
    np.ndarray
        Window weights, peak in the middle [-].

    Convention
    ----------
    Periodic rather than symmetric, which is what a short-time transform wants
    and what :func:`scipy.signal.spectrogram` would build from a window name.

    Defaults
    --------
    ``decay_frames = window_frames / 8`` is an assumed value.  The publications
    name the window shape but not its rate, so this is a VAFT choice, recorded
    rather than presented as published: it leaves the window edges at about 0.02
    of the peak, tapering hard enough to suppress the transform's edge artefacts
    without narrowing the effective window far below the stated 1 ms.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Because the rate is a choice, band powers from this window are not
    numerically comparable with published values; the structure they show is.

    Provenance
    ----------
    .. [1] [Jung2022]_ section 4.1: "an exponential decreasing peak-shape time
       window of 1 ms (50 images)".
    .. [2] :func:`scipy.signal.windows.exponential`.
    """
    from scipy.signal import windows

    length = int(window_frames)
    if length < 2:
        raise ValueError(f"window_frames must be at least 2; got {window_frames!r}")
    tau = float(decay_frames) if decay_frames is not None else length / 8.0
    if tau <= 0.0:
        raise ValueError(f"decay_frames must be positive; got {decay_frames!r}")
    return windows.exponential(length, tau=tau, sym=False)


def pixelwise_spectrogram(
    frames,
    times,
    *,
    window_frames: int = SPECTRAL_WINDOW_FRAMES_50KFPS,
    overlap: float = 0.5,
    decay_frames: float | None = None,
    detrend: str | bool = "constant",
) -> PixelSpectrogram:
    """Treat every pixel as its own signal and take its short-time Fourier magnitude.

    Parameters
    ----------
    frames : array_like
        Time-ordered image cube, frame axis first [any].
    times : array_like
        Uniformly sampled frame times [s].
    window_frames : int, optional
        Short-time window length in frames [-].
    overlap : float, optional
        Fractional overlap between successive windows [-].
    decay_frames : float, optional
        Passed to :func:`poisson_window` [-].
    detrend : str or bool, optional
        Per-window detrending, passed to the transform [-].

    Returns
    -------
    PixelSpectrogram
        Frequency and time axes and the per-pixel magnitude map [any].

    Processing steps
    ----------------
    1. Build the exponential peak-shaped window of ``window_frames``.
    2. Transform every pixel's time series with that window.
    3. Return the magnitudes with the frequency and time axes first.

    Convention
    ----------
    **The time axis is returned in the caller's own base**, offset back from the
    transform's window-relative one, exactly as
    :func:`vaft.process.fluctuation.compute_spectrogram` does, so a feature can be
    read against the shot clock without correction.  Magnitudes are amplitudes,
    not densities, from the same ``mode="magnitude"`` transform that function uses.

    Defaults
    --------
    ``SPECTRAL_WINDOW_FRAMES_50KFPS = 50`` is a literature value: 1 ms at
    50 kFrames/s.  The overlap is a numerical convenience.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Cost and memory scale with the pixel count times the window count; a
    208x208 sequence is comfortable, a 1024x1280 one is not, and nothing here
    tiles the image for the caller.  A record shorter than one window raises
    rather than returning an empty map, because for imaging that is a mistake
    rather than an edge case.

    Provenance
    ----------
    .. [1] [Jung2022]_ section 4.1, which treats each pixel of a sequence of
       images as an independent signal and transforms it.
    .. [2] [CameraFFT]_, the legacy implementation of that step, which used a
       rectangular window rather than the published one.
    """
    from scipy import signal as scipy_signal

    cube = _as_cube(frames)
    time = np.asarray(times, dtype=float).reshape(-1)
    if time.size != cube.shape[0]:
        raise ValueError(
            f"times has {time.size} entries but frames holds {cube.shape[0]}; "
            "each frame needs its own time."
        )
    if time.size < 2:
        raise ValueError("a spectrogram needs at least two frames")
    steps = np.diff(time)
    if not np.all(steps > 0):
        raise ValueError("times must increase strictly")
    # The same uniformity test `vaft.process.fluctuation.compute_spectrogram`
    # applies: a frequency axis derived from a median step is meaningless if the
    # steps are not all that step, and every band selected from it would be wrong.
    spacing = float(np.median(steps))
    if np.any(np.abs(steps - spacing) > NONUNIFORM_TOLERANCE * spacing):
        raise ValueError(
            "times must be uniformly sampled for a short-time transform; the "
            f"steps span {steps.min():.6g}-{steps.max():.6g} s around a median of {spacing:.6g} s"
        )
    sample_rate = 1.0 / spacing

    segment = int(window_frames)
    if segment > cube.shape[0]:
        raise ValueError(
            f"window_frames {segment} exceeds the {cube.shape[0]} frames available"
        )
    if not 0.0 <= float(overlap) < 1.0:
        raise ValueError(f"overlap must be a fraction in [0, 1); got {overlap!r}")

    window = poisson_window(segment, decay_frames=decay_frames)
    frequency, window_time, magnitude = scipy_signal.spectrogram(
        np.moveaxis(cube, 0, -1),
        fs=sample_rate,
        window=window,
        nperseg=segment,
        noverlap=int(round(segment * float(overlap))),
        detrend=detrend,
        mode="magnitude",
        axis=-1,
    )
    # scipy appends (frequency, time) to the pixel axes; this module leads with them.
    magnitude = np.moveaxis(magnitude, (-2, -1), (0, 1))
    return PixelSpectrogram(
        time=window_time + float(time[0]),
        frequency=frequency,
        magnitude=magnitude,
    )


def track_reference_frequency(
    spectrogram,
    *,
    search_range: Sequence[float] = REFERENCE_SEARCH_RANGE_HZ,
) -> np.ndarray:
    """Follow the dominant MHD frequency a magnetic probe reports, window by window.

    Parameters
    ----------
    spectrogram : Any
        Any object carrying ``time``, ``frequency`` and a ``(frequency, time)``
        ``magnitude``, such as a magnetics spectrogram result [-].
    search_range : sequence of float, optional
        Lowest and highest frequency the peak may be found at [Hz].

    Returns
    -------
    np.ndarray
        One frequency per window, ``nan`` where no bin lies in range [Hz].

    Convention
    ----------
    The frequency is tracked, not assumed.  The published filtering is "centred
    at the dominant MHD mode frequency measured by the magnetic probe", which
    moves during a shot; a fixed 6 kHz is the value of one example, not the
    method.

    Defaults
    --------
    ``REFERENCE_SEARCH_RANGE_HZ = (3000, 15000)`` is a legacy compatibility
    value.  The legacy code searched the first fifteen bins of a magnetics
    transform whose resolution was 1 kHz and rejected anything below the fourth,
    which is this range expressed in hertz rather than in bins, so it means the
    same thing at any window length.  The lower edge matters: without it the
    peak sits on the equilibrium field's own slow variation instead of the mode.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A single argmax per window, with no continuity requirement between windows,
    so it can jump between two comparable modes from one window to the next. It
    reports the strongest component in range whether or not one is really there;
    a window with no mode still returns its loudest bin.
    :func:`vaft.process.fluctuation.track_dominant_frequency`, which this calls,
    offers a floor and a continuity limit for callers that want neither.

    Provenance
    ----------
    .. [1] [CameraFFT]_: ``[~, idx_max] = max(P1_mir(1:15, iFT))`` with windows
       whose ``idx_max`` falls below four discarded.
    .. [2] [Jung2022]_ section 4.1, which specifies the magnetic probe as the
       source of the centre frequency.
    """
    from vaft.process.fluctuation import track_dominant_frequency

    # The shared ridge tracker with no floor and no continuity requirement is
    # exactly the legacy per-window argmax; only an empty window reports NaN.
    return track_dominant_frequency(
        spectrogram, search_range=search_range, floor_ratio=0.0
    ).frequency


def mhd_band_power(
    spectrogram: PixelSpectrogram,
    *,
    centre_frequency,
    half_width: float = MHD_BAND_HALF_WIDTH_HZ,
) -> np.ndarray:
    """Integrate each pixel's spectrogram over a band around the MHD frequency.

    Parameters
    ----------
    spectrogram : PixelSpectrogram
        Per-pixel magnitude map [any].
    centre_frequency : float or array_like
        Band centre, one value or one per window [Hz].
    half_width : float, optional
        Half the band width; the band is ``centre +/- half_width`` [Hz].

    Returns
    -------
    np.ndarray
        Band magnitude per window, frame-first as ``(window, *pixel_shape)`` [any].

    Processing steps
    ----------------
    1. Select the frequency bins inside the band for each window.
    2. Sum the magnitudes over them.

    Convention
    ----------
    Band edges are **closed**, as in
    :func:`vaft.process.fluctuation.compute_band_power`, but the reduction is a
    **sum over bins, not an integral over hertz**, and the two are not
    interchangeable: that function consumes a power spectral density, where
    integrating against frequency is what produces a power, while a short-time
    transform in ``mode="magnitude"`` yields an amplitude per bin, which has no
    per-hertz meaning to integrate.  A sum also degrades correctly at the
    published settings, where a 1 ms window resolves 1 kHz and the ``+/- 0.5 kHz``
    band is exactly one bin wide -- the single-bin case the legacy
    implementation takes directly.

    A band containing no bin at all contributes zero.  A ``nan`` centre -- what
    :func:`track_reference_frequency` returns for a window it could not judge --
    yields ``nan`` for that window, not zero.

    Defaults
    --------
    ``MHD_BAND_HALF_WIDTH_HZ = 500`` is a literature value; the published images
    are filtered over ``6 +/- 0.5 kHz``.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Because this is a sum rather than a density integral, its magnitude depends
    on how many bins the band happens to hold: doubling the window length halves
    the bin width and changes the number summed. Compare images computed with the
    same window, not across windows.

    Provenance
    ----------
    .. [1] [Jung2022]_ section 4.1 and figure 3, which filter about the probe's
       dominant frequency with a 0.5 kHz bandwidth.
    .. [2] [CameraFFT]_, which takes the single Fourier bin matched to the
       magnetic probe's peak, with a three-bin mean left in the source as the
       alternative it did not use.
    """
    frequency = np.asarray(spectrogram.frequency, dtype=float).reshape(-1)
    magnitude = np.asarray(spectrogram.magnitude, dtype=float)
    n_windows = magnitude.shape[1]
    centres = np.asarray(centre_frequency, dtype=float)
    if centres.ndim == 0:
        centres = np.full(n_windows, float(centres))
    elif centres.size != n_windows:
        raise ValueError(
            f"centre_frequency has {centres.size} entries but the spectrogram has "
            f"{n_windows} windows; pass one value or one per window."
        )
    width = float(half_width)
    if width <= 0.0:
        raise ValueError(f"half_width must be positive; got {half_width!r}")

    power = np.zeros((n_windows,) + tuple(magnitude.shape[2:]), dtype=float)
    for index, centre in enumerate(centres):
        if not np.isfinite(centre):
            power[index] = np.nan
            continue
        mask = (frequency >= centre - width) & (frequency <= centre + width)
        if not np.any(mask):
            continue
        power[index] = magnitude[mask, index, ...].sum(axis=0)
    return power


def normalize_by_local_emission(
    power,
    frames,
    *,
    frame_time,
    power_time,
    window_frames: int = EMISSION_NORMALISATION_FRAMES_50KFPS,
    floor: float = 0.0,
) -> np.ndarray:
    """Divide MHD-band power by each pixel's local average emission.

    Parameters
    ----------
    power : array_like
        Band power per window, frame-first [any].
    frames : array_like
        The image cube the power was computed from, frame axis first [any].
    frame_time : array_like
        Time of each frame [s].
    power_time : array_like
        Time of each power window [s].
    window_frames : int, optional
        Width of the local emission average, in frames [-].
    floor : float, optional
        Local means at or below this are treated as carrying no emission [any].

    Returns
    -------
    np.ndarray
        Normalised power, same shape as ``power`` [-].

    Processing steps
    ----------------
    1. Take each pixel's centred rolling mean emission over ``window_frames``.
    2. Sample that mean at the frame nearest each power window's time.
    3. Divide, leaving pixels whose local mean is at or below ``floor`` at zero.

    Input semantics
    ---------------
    ``power`` and ``frames`` are on different time bases -- one value per
    transform window against one per frame -- so both bases are required rather
    than assumed alignable by index.

    Output semantics
    ----------------
    The result is dimensionless: a fluctuation power relative to the emission
    that produced it, which is what makes frames of different brightness
    comparable.  It is not a density perturbation.

    Convention
    ----------
    The local mean is centred on the analysed frame with the same shrinking-edge
    rule as :func:`subtract_temporal_background`, and is taken from the
    **unsubtracted** frames: the divisor is the emission level, not the
    fluctuation.

    Defaults
    --------
    ``EMISSION_NORMALISATION_FRAMES_50KFPS = 10`` is a literature value, the
    published ten images or 0.2 ms at 50 kFrames/s.  ``floor = 0`` is a
    hard-coded choice: only a non-positive local mean is refused, because any
    larger threshold is a judgement about what counts as dark that belongs to
    the caller and their camera.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Where the local mean is small but positive the ratio is large and noisy;
    this is inherent to the published normalisation, not an artefact of this
    implementation. A pixel refused by the floor is indistinguishable in the
    output from one with genuinely no fluctuation.

    Provenance
    ----------
    .. [1] [Jung2022]_ section 4.1: "the powers of the MHD frequency fluctuations
       divided by the averaged emission intensity (over ten images, 0.2 ms) are
       directly used as the intensities of each pixel of the final image".
    """
    band_power = np.asarray(power, dtype=float)
    cube = _as_cube(frames)
    times = np.asarray(frame_time, dtype=float).reshape(-1)
    window_times = np.asarray(power_time, dtype=float).reshape(-1)
    if times.size != cube.shape[0]:
        raise ValueError(
            f"frame_time has {times.size} entries but frames holds {cube.shape[0]}"
        )
    if window_times.size != band_power.shape[0]:
        raise ValueError(
            f"power_time has {window_times.size} entries but power holds "
            f"{band_power.shape[0]} windows"
        )
    if band_power.shape[1:] != cube.shape[1:]:
        raise ValueError(
            f"power has pixel shape {band_power.shape[1:]} but frames has "
            f"{cube.shape[1:]}"
        )

    emission = _rolling_mean(cube, int(window_frames))
    nearest = np.abs(window_times[:, None] - times[None, :]).argmin(axis=1)
    divisor = emission[nearest]
    usable = divisor > float(floor)
    return np.divide(band_power, divisor, out=np.zeros_like(band_power), where=usable)
