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
    """Zero each channel against the quiet tail after the plasma.

    Parameters
    ----------
    time : array_like
        Sample times, shared by every channel [s].
    data : array_like
        Signals as ``(n_channels, n_samples)``, or one trace [V].
    baseline_start : float
        Time from which the record is taken to be quiet [s].

    Returns
    -------
    np.ndarray
        The signals with each channel's own zero level removed [V].

    Raises
    ------
    ValueError
        The time base and the sample axis disagree in length.

    Convention
    ----------
    The zero level is per channel, not shared, because the detectors have
    different dark levels. The mean of the tail is used, not a fitted trend, so a
    drift through the shot survives.

    Defaults
    --------
    When no sample lies at or beyond *baseline_start* the last 1000 samples are
    used instead. That fallback is a validated-workflow default carried from the
    VEST SXR viewer.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Assumes the tail is genuinely quiet. A record that ends while the plasma is
    still radiating has its baseline biased by that signal, and the fallback makes
    that silent rather than an error.

    Provenance
    ----------
    .. [1] The validated VEST SXR viewer, whose baseline convention and
       last-1000-sample fallback this reproduces.
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
    zero_phase: bool = True,
) -> np.ndarray:
    """Subtract a low-passed vacuum shot to remove poloidal-field pickup.

    **An optional step.** Nothing else in this module calls it, so the main path
    is always baseline, then window, then filter, on exactly the data the caller
    supplied.

    Parameters
    ----------
    data : array_like
        Plasma-shot signals as ``(n_channels, n_samples)`` [V].
    reference_data : array_like
        Vacuum-shot signals with the same channel count [V].
    cutoff : float
        Low-pass cut-off applied to the reference [Hz].
    fs : float
        Sample rate [Hz].
    order : int, optional
        Butterworth order [-].
    zero_phase : bool, optional
        ``True`` for zero-phase forward-backward filtering via
        :func:`scipy.signal.filtfilt`, preserving temporal alignment;
        ``False`` for causal filtering introducing filter group delay [-].

    Returns
    -------
    np.ndarray
        The plasma signals with the pickup removed, truncated to the shorter
        record [V].

    Raises
    ------
    ValueError
        The two inputs have different channel counts.

    Convention
    ----------
    Pickup from the poloidal field coil ramps appears identically in a vacuum
    shot, so low-passing that record and subtracting it channel-wise removes the
    common drive while leaving plasma fluctuations untouched.

    **The low pass is zero phase by default** (``zero_phase=True``), preserving
    exact temporal alignment with the un-delayed plasma shot. Pass
    ``zero_phase=False`` to reproduce legacy causal filtering behavior.

    Both inputs should already be baseline-corrected.

    Defaults
    --------
    ``order = 2`` is a numerical convenience matching the rest of the module's
    conditioning.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Assumes the vacuum shot reproduces the plasma shot's coil programme. Records
    are truncated to the shorter of the two rather than aligned in time, so two
    records with different start triggers are subtracted misaligned.

    Provenance
    ----------
    .. [1] The validated VEST SXR viewer's optional PF-noise removal step.
    """
    values = _as_channel_matrix(data)
    reference = _as_channel_matrix(reference_data)
    if values.shape[0] != reference.shape[0]:
        raise ValueError(
            f"data and reference_data must have the same channel count; got "
            f"{values.shape[0]} and {reference.shape[0]}"
        )
    n = min(values.shape[1], reference.shape[1])
    reference_lp = butterworth_lowpass(
        reference[:, :n], cutoff, fs, order, zero_phase=zero_phase
    )
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
    zero_phase: bool = True,
) -> SXRBandResult:
    """Baseline, window and band-pass chord signals into named frequency bands.

    Parameters
    ----------
    time : array_like
        Sample times, shared by every channel [s].
    data : array_like
        Signals as ``(n_channels, n_samples)``, or one trace [V].
    baseline_start : float or None
        Start of the quiet tail, or ``None`` when the data is already corrected
        [s].
    bands : mapping of str to sequence of float
        Named ``(f_low, f_high)`` pairs; the names are the caller's [Hz].
    fs : float
        Sample rate [Hz].
    order : int, optional
        Butterworth order for every band [-].
    time_range : sequence of float, optional
        Analysis window as ``(t_min, t_max)`` [s].
    channels : sequence of int, optional
        Row indices to process; all rows when omitted [-].
    dead_channels : sequence of int, optional
        Rows whose band output is forced to zero [-].
    zero_phase : bool, optional
        ``True`` for zero-phase forward-backward filtering via
        :func:`scipy.signal.filtfilt`, preserving temporal alignment;
        ``False`` for causal filtering introducing filter group delay [-].

    Returns
    -------
    SXRBandResult
        The windowed time base, the raw windowed signals, one array per band, and
        the channel indices selected [V].

    Raises
    ------
    ValueError
        The time range selects no samples.

    Processing steps
    ----------------
    1. Baseline-correct, unless the caller says the data already is.
    2. Restrict to the analysis window.
    3. Select the requested channels.
    4. Band-pass each into every named band, substituting zeros for dead channels.

    Convention
    ----------
    **The band-pass is zero phase by default** (``zero_phase=True``), so no
    feature moves in time. That matters because the phase read off this output by
    :func:`hilbert_instantaneous_phase` feeds a toroidal mode number. Pass
    ``zero_phase=False`` if causal filtering is explicitly required.

    A dead channel's band output is zeroed but its raw trace is kept, so it stays
    visible in a raw map instead of disappearing.

    Defaults
    --------
    ``order = 2`` is a numerical convenience.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Vacuum-reference correction is not applied; run
    :func:`sxr_subtract_vacuum_reference` first when it is wanted. Band edges are
    passed straight to the filter, so a band reaching the Nyquist frequency raises
    from SciPy rather than being clipped.

    Provenance
    ----------
    .. [1] The validated VEST SXR viewer's band decomposition.
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
            rows.append(
                butterworth_bandpass(
                    source, float(low), float(high), fs, order, zero_phase=zero_phase
                )
            )
        band_out[str(name)] = np.stack(rows)

    return SXRBandResult(time=time_win, raw=raw, bands=band_out, channels=selected)


# ---------------------------------------------------------------------------
# Two-filter electron temperature
# ---------------------------------------------------------------------------

def load_te_ratio_calibration(path=None):
    """Load a ratio-to-temperature table and build its interpolator.

    Parameters
    ----------
    path : str or path-like, optional
        Calibration table with ``te`` and ``ratio`` columns. Defaults to the
        packaged VEST beryllium/aluminium table [-].

    Returns
    -------
    tuple
        The interpolator mapping ratio to temperature, the temperature column in
        electron-volts, and the ratio column, dimensionless [-].

    Raises
    ------
    ValueError
        The table lacks either column, or holds fewer than two valid points.

    Convention
    ----------
    **The interpolator extrapolates rather than clipping**, so a ratio outside the
    table returns an extrapolated temperature, not the nearest tabulated one. A
    caller that needs out-of-range samples rejected must test the ratio itself.
    Column names are matched case-insensitively after stripping.

    Defaults
    --------
    The packaged VEST beryllium/aluminium table is a diagnostic calibration,
    shipped with the package and described in the data directory's own notes.

    Applicability
    -------------
    Machine-independent.  The table is an argument; only the packaged default is
    the VEST filter pair, and a different filter combination supplies its own.

    Limitations
    -----------
    Non-finite rows are dropped silently. The two-filter method assumes a thermal
    spectrum, so the temperature it returns is meaningless for a non-thermal one.

    Provenance
    ----------
    .. [1] The packaged VEST beryllium/aluminium ratio calibration,
       ``vaft/data/legacy/sxr_te_ratio_be_al.csv``.
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
    """Pair the two filters' channels of one array by the chord they view.

    Parameters
    ----------
    ods : ODS
        Carries ``soft_x_rays.channel`` with the identifiers the VEST mapper
        writes [-].
    array : str
        Which array to pair within [-].

    Returns
    -------
    tuple of tuple of int
        One ``(beryllium_index, aluminium_index)`` pair per shared chord, ordered
        by chord number [-].

    Raises
    ------
    ValueError
        No chord in that array carries both filters.

    Convention
    ----------
    Channel identifiers are ``{daq}:{array}:{filter}:{chord}``, written by the
    VEST digitizer mapper. Pairing on the chord number rather than on position in
    the channel list is what absorbs a per-block wiring reversal: the two filter
    blocks need not be ordered the same way.

    Returns index pairs, not data, so it changes no measurement.

    Applicability
    -------------
    VEST-specific.  Depends on the identifier grammar and the filter names written
    by :func:`vaft.machine_mapping.soft_x_rays.soft_x_rays_from_digitizer_csv`;
    another machine's mapper would need its own pairing rule.

    Limitations
    -----------
    A channel whose identifier does not split into four parts, or whose chord is
    not an integer, is skipped silently, so a mis-mapped channel is absent from
    the result rather than reported.

    Provenance
    ----------
    .. [1] :func:`vaft.machine_mapping.soft_x_rays.soft_x_rays_from_digitizer_csv`,
       which writes the identifiers this parses.
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
    zero_phase: bool = True,
    time_range: Sequence[float] | None = None,
) -> SXRTemperatureResult:
    """Electron temperature from the two-filter signal ratio, chord by chord.

    Parameters
    ----------
    time : array_like
        Sample times, shared by every channel [s].
    data : array_like
        Signals as ``(n_channels, n_samples)`` [V].
    pairs : sequence of sequence of int
        ``(beryllium_index, aluminium_index)`` pairs, one per chord [-].
    calibration : callable
        Maps a signal ratio to a temperature; see
        :func:`load_te_ratio_calibration` [-].
    baseline_start : float or None
        Start of the quiet tail, or ``None`` when already corrected [s].
    fs : float
        Sample rate [Hz].
    lowpass_cutoff : float, optional
        Conditioning low-pass cut-off [Hz].
    al_gain : float, optional
        Relative sensitivity of the aluminium channel [-].
    al_threshold : float, optional
        Conditioned aluminium level below which a sample is invalid [V].
    detrend_window : int, optional
        Width of the centred rolling trend, in samples [-].
    order : int, optional
        Butterworth order [-].
    zero_phase : bool, optional
        ``True`` for zero-phase forward-backward filtering via
        :func:`scipy.signal.filtfilt`, preserving temporal alignment;
        ``False`` for causal filtering introducing filter group delay [-].
    time_range : sequence of float, optional
        Analysis window as ``(t_min, t_max)`` [s].

    Returns
    -------
    SXRTemperatureResult
        The windowed time base, the temperature per chord in electron-volts, and
        its percentage deviation from the rolling trend [-].

    Processing steps
    ----------------
    1. Baseline-correct, unless the caller says the data already is.
    2. Restrict to the analysis window.
    3. Low-pass both channels of each pair.
    4. Scale the aluminium channel by its relative sensitivity.
    5. Take the ratio and map it through the calibration.
    6. Subtract a centred rolling trend to get the relative fluctuation.
    7. Invalidate samples whose conditioned aluminium level is at or below the
       threshold.

    Convention
    ----------
    **The low pass is zero phase by default** (``zero_phase=True``), matching
    the convention of :func:`sxr_band_signals` and preserving exact temporal
    alignment with other diagnostics and fast MHD events. Pass
    ``zero_phase=False`` to reproduce legacy causal filtering behavior
    with filter group delay.

    The relative fluctuation is a percentage of the rolling trend, and the trend
    is centred, so it uses samples on both sides.

    Defaults
    --------
    The 50 kHz cut-off, the 1.07 aluminium gain, the 0.10 V threshold and the
    400-sample trend window are validated-workflow defaults, the settings of the
    validated VEST viewer. ``order = 2`` is a numerical convenience.

    Applicability
    -------------
    Machine-independent.  The defaults are VEST viewer practice and the
    calibration is supplied by the caller.

    Limitations
    -----------
    The two-filter method assumes a thermal spectrum. Invalid samples are NaN
    rather than removed, so a caller must honour them. Vacuum-reference correction
    is not applied here.

    Provenance
    ----------
    .. [1] The validated VEST SXR viewer, whose conditioning settings, validity
       threshold and detrending this reproduces.
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
        be = butterworth_lowpass(
            corrected[int(be_channel)], lowpass_cutoff, fs, order, zero_phase=zero_phase
        )
        al = butterworth_lowpass(
            corrected[int(al_channel)], lowpass_cutoff, fs, order, zero_phase=zero_phase
        )
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

    Parameters
    ----------
    signal : array_like
        A band-passed, roughly monochromatic trace [V].
    time : array_like
        Its time base [s].
    t_eval : float
        Time at which to report the phase [s].

    Returns
    -------
    tuple
        The phase in degrees and the envelope in volts at the nearest sample, that
        sample's index, and the full phase and envelope series [-].

    Convention
    ----------
    Phase comes from the analytic signal and is wrapped to the principal branch,
    so it runs within a full turn and jumps rather than accumulating. The sample
    nearest the requested time is used; no interpolation.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    **Meaningful only for a band-passed input.** The analytic signal of a
    broadband trace has a phase that does not correspond to any single mode, so
    filter first with :func:`sxr_band_signals`. The Hilbert transform is also
    poorly behaved at the record edges, so a time near either end is unreliable.

    Provenance
    ----------
    .. [1] The analytic-signal phase estimate used by the validated VEST SXR
       viewer's mode analysis.
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
    """Rank candidate toroidal mode numbers from phases measured at two locations.

    Parameters
    ----------
    theta_a_deg : float
        Toroidal angle of the first observation point [deg].
    phase_a_deg : float
        Measured phase there [deg].
    theta_b_deg : float
        Toroidal angle of the second point [deg].
    phase_b_deg : float
        Measured phase there [deg].
    n_max : int
        Largest magnitude of mode number to consider [-].

    Returns
    -------
    tuple of ToroidalModeCandidate
        Every candidate with its anchor offset and wrapped residual, best first
        [deg].

    Raises
    ------
    ValueError
        The two points share a toroidal angle, so no mode number can be inferred.

    Convention
    ----------
    For each candidate the wrapped phase-versus-angle line is anchored exactly at
    the first point and scored by the wrapped residual at the second. Zero is
    excluded, since it carries no toroidal structure. Candidates sort by absolute
    residual, then by absolute mode number, and **a positive mode number wins a
    remaining tie** over its negative counterpart, which is a tie-break, not
    physics.

    Applicability
    -------------
    Machine-independent.  The port angles are arguments; the VEST pair used in
    practice is a worked example, not a built-in.

    Limitations
    -----------
    **Two points cannot resolve aliasing.** A candidate and one differing by
    ``360/|dtheta|`` produce identical residuals; for the VEST ports at zero and
    120 degrees that period is three. Every degenerate candidate is returned with
    an equal residual and **none is physically privileged**; choosing among them
    needs independent information such as frequency scaling, mode structure, or a
    third port.

    The phases must come from the same band and the same filtering convention, or
    the residual compares two different quantities.

    Provenance
    ----------
    .. [1] The two-point toroidal mode-number estimate of the validated VEST SXR
       viewer, and its documented aliasing degeneracy.
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
    """Continuous-wavelet magnitude scalogram of one trace.

    Parameters
    ----------
    signal : array_like
        The trace [V].
    fs : float
        Sample rate [Hz].
    f0 : float
        Lowest frequency of the scalogram [Hz].
    f1 : float
        Highest frequency [Hz].
    n_freq : int
        Number of frequencies between them [-].

    Returns
    -------
    tuple of np.ndarray
        The frequency axis, ascending, and the magnitude as
        ``(n_freq, n_samples)`` [-].

    Raises
    ------
    ImportError
        The optional wavelet package is not installed.

    Convention
    ----------
    Frequencies are returned ascending, which reverses the underlying library's
    own descending order, and the magnitude rows are reversed to match. A wavelet
    transform trades frequency resolution for time resolution as frequency rises,
    unlike the fixed-window short-time transform.

    Defaults
    --------
    Single-threaded execution is a numerical convenience, keeping the call
    deterministic and avoiding a thread pool inside a per-channel loop.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Needs an optional package that is not a VAFT dependency. The short-time
    transform in :func:`vaft.process.fluctuation.compute_spectrogram` covers the
    default time-frequency need without it. The input is cast to single precision
    by that library.

    Provenance
    ----------
    .. [1] The optional ``fcwt`` continuous-wavelet implementation; the
       alternative path is :mod:`vaft.process.fluctuation`.
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
