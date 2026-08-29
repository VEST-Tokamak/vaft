"""VEST soft X-ray fluctuation and two-filter analysis routines.

Ported from the validated ``VEST SXR Viewer`` analysis tool (``vest_sxr_viewer.py``
v5, module-level science functions; provenance: the 2026 VEST SXR thesis
presentation).  The 455xx SXR campaign acquires both digitizers -- 22577 at the
4 o'clock port (toroidal 0 deg) and 17592 at the 12 o'clock port (120 deg) -- at
125 MHz / 128 = 976562.5 Hz starting at the 285 ms trigger.

Every routine here takes plain arrays (``time`` in seconds, ``data`` shaped
``(n_channels, n_samples)``); nothing reads a CSV or an ODS except the explicit
ODS pairing helper.  Vacuum-shot PF-noise removal is a **standalone optional
step** (:func:`sxr_subtract_vacuum_reference`): no other routine applies it, so
the main processing path is always baseline -> window -> filter on exactly the
data the caller supplied.

Generic spectral analysis (PSD, spectral index, spectrogram) is *not* here --
use :mod:`vaft.process.fluctuation` on any prepared channel.
"""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import hilbert

from vaft.process.signal_processing import butterworth_bandpass, butterworth_lowpass

__all__ = [
    "SXRBandResult",
    "SXRTemperatureResult",
    "ToroidalModeCandidate",
    "hilbert_instantaneous_phase",
    "load_te_ratio_calibration",
    "rank_toroidal_mode_numbers",
    "sxr_band_signals",
    "sxr_baseline_correction",
    "sxr_cwt_spectrogram",
    "sxr_electron_temperature",
    "sxr_subtract_vacuum_reference",
    "sxr_te_pairs_from_ods",
]

#: Packaged Be/Al ratio -> Te calibration table (see ``vaft/data/README.md``).
TE_RATIO_TABLE = "legacy/sxr_te_ratio_be_al.csv"


@dataclass(frozen=True)
class SXRBandResult:
    """Baseline-corrected chord signals and their band-pass decompositions."""

    time: np.ndarray
    raw: np.ndarray
    bands: Mapping[str, np.ndarray]
    channels: tuple[int, ...]


@dataclass(frozen=True)
class SXRTemperatureResult:
    """Two-filter electron temperature per (Be, Al) chord pair."""

    time: np.ndarray
    te: np.ndarray
    rel_fluctuation: np.ndarray
    al_signal: np.ndarray
    pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ToroidalModeCandidate:
    """One candidate toroidal mode number from a two-point phase comparison."""

    n: int
    offset_deg: float
    residual_deg: float


def _as_channel_matrix(data: Any) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2:
        raise ValueError(f"data must be (n_channels, n_samples); got shape {values.shape}")
    return values


def sxr_baseline_correction(time, data, baseline_start: float) -> np.ndarray:
    """Subtract each channel's mean over the post-``baseline_start`` tail.

    The tail after the plasma is the quiet region used as the per-channel zero
    level.  When no sample lies at or beyond ``baseline_start`` the last 1000
    samples are used instead (the validated viewer's fallback).
    """
    time = np.asarray(time, dtype=float)
    values = _as_channel_matrix(data)
    if time.size != values.shape[1]:
        raise ValueError(
            f"time and data must share the sample axis; got {time.size} and {values.shape[1]}"
        )
    mask = time >= float(baseline_start)
    if np.any(mask):
        baseline = values[:, mask].mean(axis=1, keepdims=True)
    else:
        baseline = values[:, -1000:].mean(axis=1, keepdims=True)
    return values - baseline


def sxr_subtract_vacuum_reference(
    data,
    reference_data,
    *,
    cutoff: float,
    fs: float,
    order: int = 2,
) -> np.ndarray:
    """Subtract the low-passed vacuum-shot trace from each channel.

    **Optional step.** Pickup from the PF coil ramps appears identically in a
    vacuum shot; low-passing the vacuum record at ``cutoff`` and subtracting it
    channel-wise removes that common drive while leaving plasma fluctuations
    untouched.  Both inputs should already be baseline-corrected
    (:func:`sxr_baseline_correction`); records are truncated to the shorter one.

    No other routine in this module calls this -- a caller wanting PF-noise
    removal runs it explicitly and feeds the corrected matrix onward.
    """
    values = _as_channel_matrix(data)
    reference = _as_channel_matrix(reference_data)
    if values.shape[0] != reference.shape[0]:
        raise ValueError(
            f"data and reference_data must have the same channel count; got "
            f"{values.shape[0]} and {reference.shape[0]}"
        )
    n = min(values.shape[1], reference.shape[1])
    reference_lp = butterworth_lowpass(reference[:, :n], cutoff, fs, order)
    return values[:, :n] - reference_lp


def sxr_band_signals(
    time,
    data,
    *,
    baseline_start: float | None,
    bands: Mapping[str, Sequence[float]],
    fs: float,
    order: int = 2,
    time_range: Sequence[float] | None = None,
    channels: Sequence[int] | None = None,
    dead_channels: Sequence[int] = (),
) -> SXRBandResult:
    """Baseline-correct, window, and band-pass chord signals into named bands.

    Args:
        time: Sample times in seconds, shared by every channel.
        data: ``(n_channels, n_samples)`` signals (or a single 1D trace).
        baseline_start: Start of the quiet tail used as the zero level [s], or
            ``None`` when ``data`` is already baseline-corrected (for instance
            after the explicit :func:`sxr_subtract_vacuum_reference` pre-step).
        bands: ``{name: (f_low, f_high)}`` in Hz; names are the caller's.
        fs: Sampling frequency in Hz.
        order: Butterworth order for every band.
        time_range: Optional ``(t_min, t_max)`` analysis window [s].
        channels: Row indices to process; all rows when omitted.
        dead_channels: Rows whose band output is forced to zero (their raw trace
            is kept, so the dead channel stays visible in raw maps).

    Returns:
        :class:`SXRBandResult` with ``raw`` and one ``(len(channels), n_t)``
        array per band.  Vacuum-reference correction is not applied here; run
        :func:`sxr_subtract_vacuum_reference` first if wanted.
    """
    time = np.asarray(time, dtype=float)
    if baseline_start is None:
        corrected = _as_channel_matrix(data)
    else:
        corrected = sxr_baseline_correction(time, data, baseline_start)

    if time_range is not None:
        window = (time >= float(time_range[0])) & (time <= float(time_range[1]))
    else:
        window = np.ones(time.size, dtype=bool)
    if not np.any(window):
        raise ValueError(f"time_range {time_range!r} selects no samples.")
    time_win = time[window]

    selected = tuple(int(c) for c in channels) if channels is not None else tuple(
        range(corrected.shape[0])
    )
    dead = {int(c) for c in dead_channels}

    raw = np.stack([corrected[c][window] for c in selected])
    band_out: dict[str, np.ndarray] = {}
    for name, (low, high) in bands.items():
        rows = []
        for k, c in enumerate(selected):
            source = np.zeros_like(raw[k]) if c in dead else raw[k]
            rows.append(butterworth_bandpass(source, float(low), float(high), fs, order))
        band_out[str(name)] = np.stack(rows)

    return SXRBandResult(time=time_win, raw=raw, bands=band_out, channels=selected)


# ---------------------------------------------------------------------------
# Two-filter electron temperature
# ---------------------------------------------------------------------------

def load_te_ratio_calibration(path=None):
    """Load a ``te``/``ratio`` table and build the ratio -> Te interpolator.

    Defaults to the packaged VEST Be/Al calibration
    (``vaft/data/legacy/sxr_te_ratio_be_al.csv``).  Returns
    ``(interpolator, te, ratio)``; the interpolator extrapolates beyond the
    table, so out-of-range ratios yield extrapolated (not clipped) Te.
    """
    if path is None:
        from vaft.data.resources import data_path

        path = data_path(TE_RATIO_TABLE)
    table = pd.read_csv(path)
    columns = {c.strip().lower(): c for c in table.columns}
    if "te" not in columns or "ratio" not in columns:
        raise ValueError(
            "the calibration table needs 'te' and 'ratio' columns; "
            f"found {list(table.columns)}"
        )
    te = table[columns["te"]].to_numpy(dtype=float)
    ratio = table[columns["ratio"]].to_numpy(dtype=float)
    good = np.isfinite(te) & np.isfinite(ratio)
    if good.sum() < 2:
        raise ValueError("the calibration table holds fewer than two valid points.")
    interpolator = interp1d(
        ratio[good], te[good], fill_value="extrapolate", bounds_error=False
    )
    return interpolator, te[good], ratio[good]


def sxr_te_pairs_from_ods(ods: Any, array: str) -> tuple[tuple[int, int], ...]:
    """Pair Be and Al channels of one two-filter array by physical chord.

    Channels written by :func:`vaft.machine_mapping.soft_x_rays.
    soft_x_rays_from_digitizer_csv` carry identifiers
    ``{daq}:{array}:{filter}:{chord}``; matching ``chord`` numbers across the
    Be and Al blocks pairs each Be channel with the Al channel viewing the same
    line of sight, absorbing any per-block wiring reversal.
    """
    be: dict[int, int] = {}
    al: dict[int, int] = {}
    channels = ods["soft_x_rays.channel"]
    for index in range(len(channels)):
        identifier = str(channels[index]["identifier"])
        parts = identifier.split(":")
        if len(parts) != 4 or parts[1] != str(array):
            continue
        try:
            chord = int(parts[3])
        except ValueError:
            continue
        if parts[2] == "Be":
            be[chord] = index
        elif parts[2] == "Al":
            al[chord] = index
    shared = sorted(set(be) & set(al))
    if not shared:
        raise ValueError(
            f"array {array!r} has no chords with both Be and Al channels; "
            "is this a two-filter array mapped by soft_x_rays_from_digitizer_csv?"
        )
    return tuple((be[chord], al[chord]) for chord in shared)


def sxr_electron_temperature(
    time,
    data,
    pairs: Sequence[Sequence[int]],
    *,
    calibration: Callable[[np.ndarray], np.ndarray],
    baseline_start: float | None,
    fs: float,
    lowpass_cutoff: float = 50_000.0,
    al_gain: float = 1.07,
    al_threshold: float = 0.10,
    detrend_window: int = 400,
    order: int = 2,
    time_range: Sequence[float] | None = None,
) -> SXRTemperatureResult:
    """Electron temperature from the Be/Al two-filter signal ratio.

    Each ``(be_channel, al_channel)`` pair views one chord through the two
    filters; the ratio of the low-passed, baseline-corrected signals is mapped
    to Te through ``calibration`` (see :func:`load_te_ratio_calibration`).
    Samples where the conditioned Al signal is at or below ``al_threshold`` are
    invalid and returned as NaN.  ``rel_fluctuation`` is the percentage
    deviation of Te from its centered ``detrend_window``-sample rolling trend.

    Defaults are the validated VEST viewer settings: 50 kHz conditioning
    low-pass, Al relative-sensitivity gain 1.07, 0.10 V validity threshold,
    400-sample trend window.  Vacuum-reference correction is not applied here;
    run :func:`sxr_subtract_vacuum_reference` first if wanted.
    """
    time = np.asarray(time, dtype=float)
    if baseline_start is None:
        corrected = _as_channel_matrix(data)
    else:
        corrected = sxr_baseline_correction(time, data, baseline_start)

    if time_range is not None:
        window = (time >= float(time_range[0])) & (time <= float(time_range[1]))
    else:
        window = np.ones(time.size, dtype=bool)
    if not np.any(window):
        raise ValueError(f"time_range {time_range!r} selects no samples.")
    time_win = time[window]

    te_rows, rel_rows, al_rows = [], [], []
    for be_channel, al_channel in pairs:
        be = butterworth_lowpass(corrected[int(be_channel)], lowpass_cutoff, fs, order)
        al = butterworth_lowpass(corrected[int(al_channel)], lowpass_cutoff, fs, order)
        al = al * float(al_gain)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(al) > 1e-12, be / al, np.nan)
        te = np.asarray(calibration(ratio), dtype=float)

        te_win = te[window]
        al_win = al[window]
        # NaN-tolerant centered rolling mean (viewer parity): a NaN sample must
        # not poison the trend of its neighbours.
        trend = (
            pd.Series(te_win)
            .rolling(window=int(detrend_window), center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(
                np.abs(te_win) > 1e-9, (te_win - trend) / te_win * 100.0, np.nan
            )

        valid = al_win > float(al_threshold)
        te_rows.append(np.where(valid, te_win, np.nan))
        rel_rows.append(np.where(valid, rel, np.nan))
        al_rows.append(al_win)

    return SXRTemperatureResult(
        time=time_win,
        te=np.array(te_rows),
        rel_fluctuation=np.array(rel_rows),
        al_signal=np.array(al_rows),
        pairs=tuple((int(b), int(a)) for b, a in pairs),
    )


# ---------------------------------------------------------------------------
# Two-point toroidal mode number
# ---------------------------------------------------------------------------

def _wrap180(angle_deg):
    """Fold angles into (-180, +180] degrees."""
    return (np.asarray(angle_deg, dtype=float) + 180.0) % 360.0 - 180.0


def _wrapped_sawtooth(theta_deg, n: int, offset_deg: float = 0.0) -> np.ndarray:
    """Expected wrapped phase versus toroidal angle for mode number ``n``."""
    shifted = np.asarray(theta_deg, dtype=float) + offset_deg
    return 180.0 - 360.0 * ((n * shifted / 360.0) % 1.0)


def _sawtooth_offset(theta0_deg: float, phase0_deg: float, n: int) -> float:
    """Offset making the mode-``n`` curve pass exactly through one point."""
    if n == 0:
        return 0.0
    u = (180.0 - phase0_deg) / 360.0
    return (360.0 * u / n - theta0_deg) % (360.0 / abs(n))


def hilbert_instantaneous_phase(signal, time, t_eval: float):
    """Instantaneous phase and envelope of a band-limited signal at one time.

    Returns ``(phase_deg, envelope, index, phase_deg_series, envelope_series)``
    from the analytic (Hilbert) signal, evaluated at the sample nearest
    ``t_eval``.  Meaningful only for a band-passed, roughly monochromatic
    input -- filter first (:func:`sxr_band_signals`).
    """
    time = np.asarray(time, dtype=float)
    analytic = hilbert(np.asarray(signal, dtype=float))
    index = int(np.argmin(np.abs(time - float(t_eval))))
    phase_series = np.degrees(np.angle(analytic))
    envelope_series = np.abs(analytic)
    return (
        float(phase_series[index]),
        float(envelope_series[index]),
        index,
        phase_series,
        envelope_series,
    )


def rank_toroidal_mode_numbers(
    theta_a_deg: float,
    phase_a_deg: float,
    theta_b_deg: float,
    phase_b_deg: float,
    n_max: int,
) -> tuple[ToroidalModeCandidate, ...]:
    """Rank toroidal mode numbers from phases at two toroidal locations.

    For each candidate ``n`` the wrapped phase-versus-angle curve is anchored
    exactly at point A and scored by the wrapped residual at point B; candidates
    are sorted by ``(|residual|, |n|)``.

    With only two toroidal observation points, ``n`` and ``n +/- 360/|dtheta|``
    produce identical residuals -- for the VEST 0/120 deg ports that aliasing
    period is ``dn = 3``.  All degenerate candidates appear in the returned
    tuple with equal residuals; **no candidate is physically privileged**, and
    choosing among them needs independent information (frequency scaling, mode
    structure, additional ports).
    """
    theta_a, theta_b = float(theta_a_deg), float(theta_b_deg)
    if np.isclose(_wrap180(theta_a - theta_b), 0.0):
        raise ValueError(
            "the two observation points share one toroidal angle; a mode number "
            "cannot be inferred from a single location."
        )
    candidates = []
    for n in range(-int(n_max), int(n_max) + 1):
        if n == 0:
            continue
        offset = _sawtooth_offset(theta_a, float(phase_a_deg), n)
        residual = float(
            _wrap180(float(phase_b_deg) - _wrapped_sawtooth(theta_b, n, offset))
        )
        candidates.append(
            ToroidalModeCandidate(n=n, offset_deg=float(offset), residual_deg=residual)
        )
    candidates.sort(
        key=lambda c: (round(abs(c.residual_deg), 6), abs(c.n), c.n < 0)
    )
    return tuple(candidates)


def sxr_cwt_spectrogram(signal, fs: float, f0: float, f1: float, n_freq: int):
    """Continuous-wavelet magnitude scalogram via the optional ``fcwt`` package.

    Returns ``(frequency_hz, magnitude)`` with frequency ascending.  ``fcwt`` is
    not a VAFT dependency; the STFT path
    (:func:`vaft.process.fluctuation.compute_spectrogram`) covers the default
    time-frequency need without it.
    """
    try:
        import fcwt as fcwt_lib
    except ImportError as error:
        raise ImportError(
            "the optional 'fcwt' package is not installed; run 'pip install fcwt' "
            "or use vaft.process.fluctuation.compute_spectrogram (STFT) instead."
        ) from error
    frequencies, out = fcwt_lib.cwt(
        np.asarray(signal, dtype=np.float32),
        int(fs),
        float(f0),
        float(f1),
        int(n_freq),
        nthreads=1,
    )
    return np.asarray(frequencies)[::-1], np.abs(out[::-1, :])
