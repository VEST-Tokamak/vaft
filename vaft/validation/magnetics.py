"""Magnetics signal-quality validation (issue #189).

One question, asked before any physics: **is this magnetic waveform itself
usable, and over which part of the discharge?**

That is deliberately narrower than what the eddy stage and EFIT already ask.
Three layers exist and must not be merged:

===========================  =============================================
diagnostics magnetics QA     is the waveform itself usable?      *(here)*
eddy magnetics QA            does it agree with the vacuum model?
EFIT QA                      does the reconstruction fit its constraints?
===========================  =============================================

A model-agreement failure is evidence about the *model*, not about the sensor,
so nothing downstream of this module may write back into the validity it
produces.  Conversely, downstream consumers should read this validity rather
than each rediscovering basic sensor health with a private heuristic.

A channel is not usable-or-not for a whole shot.  An amplifier that rails at
0.31 s leaves everything before it perfectly good, so the assessment is
per-sample and is projected into ``validity_timed``, with the scalar
``validity`` only summarizing it.

What the detectors are allowed to conclude
------------------------------------------
Thresholds that have not been justified across a representative VEST
population must not become hard gates (#189, non-goals).  So the detectors
split by how categorical their evidence is:

**Hard** -- ``-2``, invalid, excluded by a default consumer:

* non-finite samples;
* a flatlined record, or an interior run of bit-identical samples.  A noisy
  analog channel does not produce a dozen consecutive identical values; a dead
  one, a railed integrator and a dropout all do.

**Soft** -- ``1``, valid but flagged, never excluded by a default consumer:

* spikes, offset jumps, baseline drift, elevated baseline noise.

The soft set is where a threshold could misfire on real physics -- plasma
breakdown *is* a fast step in these signals -- so a misfire costs a note in the
manifest rather than a channel EFIT never sees.  ``1`` follows the convention
:mod:`vaft.machine_mapping.impa` already established for "accepted with
warnings"; it reads as "certified" in the Data Dictionary, which VAFT
deliberately narrows.

Scope is the equilibrium-relevant processed quantities,
``b_field_pol_probe[*].field`` and ``flux_loop[*].flux``.  Raw voltage validity
is a different statement about a different datum and is not assumed to be the
same: it is *read* here (a channel whose voltage never arrived cannot have a
trustworthy field) but the processed assessment is its own.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field as dataclass_field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from vaft.formula.statistics import (
    dynamic_range,
    linear_trend,
    median_absolute_deviation,
    rms,
    robust_z_scores,
)
from vaft.ods_access import path_count, path_value as lookup
from vaft.validation.imas import (
    VALIDITY_CERTIFIED,
    VALIDITY_INVALID,
    VALIDITY_VALID,
    aggregate_validity,
    read_validity,
    resolve_signal_time,
    validity_mask,
    write_validity,
)
from vaft.validation.model import ValidationStatus

__all__ = [
    "ChannelQuality",
    "MagneticsQualityConfig",
    "QualityEvent",
    "QUANTITY_BY_KIND",
    "channel_node",
    "implausible_magnitude",
    "population_peak_outliers",
    "magnetics_quality_metrics",
    "project_validity",
    "unusable_channels_at",
    "validate_magnetics_signals",
]

#: The processed quantity each magnetics channel family carries, and its unit.
QUANTITY_BY_KIND: Mapping[str, tuple[str, str]] = {
    "b_field_pol_probe": ("field", "T"),
    "flux_loop": ("flux", "Wb"),
}


@dataclass(frozen=True)
class MagneticsQualityConfig:
    """Detector thresholds, carried into the report as provenance.

    Every default is a starting point rather than a validated VEST constant,
    which is exactly why only the categorical detectors are allowed to reject.
    """

    #: Leading fraction of the record treated as the quiet baseline.  A fixed
    #: fraction rather than a detected pre-plasma window, so signal quality
    #: never depends on plasma-onset detection -- which is itself downstream of
    #: whether the signals are usable.
    baseline_fraction: float = 0.2

    #: Consecutive bit-identical samples that constitute a constant run.  A
    #: real analog channel does not produce them; a dead channel, a railed
    #: integrator and an acquisition dropout all do.
    min_constant_run: int = 16

    #: Smallest deviation any *soft* detector will report, as a fraction of the
    #: channel's own dynamic range.  A purely relative threshold has no floor,
    #: and these waveforms are low-pass filtered and integrated: their residual
    #: against a local median is numerical texture, whose robust scale is
    #: minuscule, so a scale-free z-score calls ordinary smooth wiggles
    #: outliers.  Requiring a spike or a step to be worth at least a percent of
    #: what the channel actually swung makes the detectors scale-aware without
    #: hard-coding a tesla or a weber.
    significance_floor: float = 0.01

    #: Robust-z of the residual against a local median above which a sample is
    #: a spike.
    spike_sigma: float = 8.0
    #: Longest run of consecutive outlying differences still called a spike;
    #: anything longer is sustained behaviour, not a transient.
    max_spike_samples: int = 3

    #: Robust-z of the first difference above which an uncompensated step is
    #: called an offset jump.
    offset_jump_sigma: float = 10.0

    #: How close in time two channels' offset-jump candidates must be to count
    #: as the same event.
    coherence_window: float = 5.0e-4

    #: Fraction of the assessed channels that must share a jump instant before
    #: it is read as the machine rather than as a per-channel fault.
    coherent_jump_fraction: float = 0.3

    #: Largest poloidal-field reading a VEST probe can physically report [T].
    #: VEST runs Ip <~ 250 kA and every probe sits >~ 5 cm from any conductor,
    #: so mu0*I/(2*pi*d) bounds a genuine reading at ~1 T even in the most
    #: extreme geometry; healthy channels swing 0.03-0.2 T.  A record that
    #: exceeds this anywhere is instrumentation -- a railed integrator, a
    #: miswired channel, an unconverted unit -- and not a measurement anywhere,
    #: because a probe's gain does not come and go within a shot.  Applies to
    #: ``b_field_pol_probe`` only; ``None`` disables it.
    max_plausible_field: float | None = 1.0

    #: A record whose peak exceeds this multiple of the median peak of its own
    #: family (same ``kind``, same shot) is a population outlier.  Justified on
    #: the three packaged VEST samples rather than assumed: probe H3-08 peaks
    #: at 68.5x, 21.4x and 22.8x the B-probe median on shots 39915, 41524 and
    #: 41672, several outboard probes sit at 4.5-15.8x on the latter two, and
    #: the healthiest channel in all three shots tops out at 2.6x.  The value
    #: sits inside that gap.  Within-shot and relative, so it follows the
    #: shot's own amplitude and never needs a tesla or a weber.  ``None``
    #: disables it.
    population_peak_factor: float | None = 4.0


@dataclass(frozen=True)
class QualityEvent:
    """One contiguous stretch of a waveform that a detector objected to."""

    reason: str
    start: float
    end: float
    samples: int
    validity: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelQuality:
    """The signal-quality assessment of one magnetic channel.

    Identity is by domain (``kind``, ``index``, ``quantity``), not by IDS path:
    where that projects is :func:`channel_node`'s business, so a Data Dictionary
    change lands in one function rather than in every result (#253 §9).
    """

    kind: str
    index: int
    name: str
    quantity: str
    unit: str
    status: ValidationStatus
    validity: int
    validity_timed: np.ndarray
    valid_fraction: float
    metrics: Mapping[str, float] = dataclass_field(default_factory=dict)
    events: tuple[QualityEvent, ...] = ()
    reason: str = ""

    @property
    def projectable(self) -> bool:
        """Whether there is a per-sample assessment to write into the IDS.

        A channel carrying no processed waveform has nothing to say about it.
        Writing ``-2`` there would assert the datum is invalid when the truth is
        that it does not exist -- and ``not_available`` must stay distinct from
        ``fail`` (#253).
        """
        return self.status is not ValidationStatus.NOT_AVAILABLE


def channel_node(kind: str, index: int, quantity: str) -> str:
    """The IDS node one channel's processed quantity lives at."""
    return f"magnetics.{kind}.{int(index)}.{quantity}"


# ---------------------------------------------------------------------------
# Detectors.  Pure array functions: each takes the waveform and returns either
# a per-sample mask or a list of (start, stop) index intervals, so the physics
# tests drive them directly with synthetic signals.
# ---------------------------------------------------------------------------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal ``[start, stop)`` index intervals where ``mask`` is True."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def constant_runs(data: np.ndarray, min_samples: int) -> list[tuple[int, int]]:
    """Index intervals of at least ``min_samples`` bit-identical samples.

    Exact equality, not a tolerance: the point is to catch a value that stopped
    changing at all, which is what a dead channel, a railed integrator and an
    acquisition dropout produce, and to catch nothing else.  A quiet but live
    channel still dithers in its last bits.
    """
    if data.size < min_samples:
        return []
    same = np.diff(data) == 0.0
    # A run of n equal samples has n-1 zero differences.
    return [
        (start, stop + 1)
        for start, stop in _runs(same)
        if stop + 1 - start >= min_samples
    ]


def _filled(data: np.ndarray) -> np.ndarray:
    """``data`` with non-finite samples replaced by the nearest finite one.

    A median filter propagates NaN through its whole window, which would let a
    single dead sample erase the baseline around it.  The filled values never
    reach a metric -- they exist only so the filter has something to chew on --
    and every non-finite sample is scored ``nan`` afterwards regardless.
    """
    finite = np.isfinite(data)
    if finite.all() or not finite.any():
        return np.where(finite, data, 0.0)
    index = np.arange(data.size)
    return np.interp(index, index[finite], data[finite])


def spike_intervals(
    data: np.ndarray,
    *,
    spike_sigma: float,
    max_spike_samples: int,
    significance_floor: float = 0.0,
) -> list[tuple[int, int]]:
    """Index intervals of samples that depart from their own local baseline.

    The baseline is a running median wide enough that a spike cannot outvote
    it, so what is scored is each sample's deviation from where its neighbours
    say it should be.  That is what keeps the detector off real physics: a
    median filter reproduces a ramp exactly, so plasma breakdown -- fast, but
    monotone across several samples -- leaves no residual, while a one-sample
    excursion leaves all of itself.

    Scoring the residual against its own robust scale makes ``spike_sigma``
    mean the same thing on a millitesla probe and a weber loop.  Anything
    longer than ``max_spike_samples`` is sustained behaviour rather than a
    transient and is left for the other detectors.
    """
    if data.size < 5:
        return []
    from scipy.ndimage import median_filter

    # Odd, and wide enough that a spike of the maximum accepted width is still
    # a minority of the window.
    window = 2 * int(max_spike_samples) + 3
    residual = data - median_filter(_filled(data), size=window, mode="nearest")
    scores = robust_z_scores(np.where(np.isfinite(data), residual, np.nan))
    outlying = np.isfinite(scores) & (np.abs(scores) > spike_sigma)
    outlying |= np.isinf(scores)
    # A relative score has no floor, and these waveforms are smooth enough that
    # their residual's robust scale is numerical texture.  The excursion must
    # also be worth something against what the channel actually swung.
    floor = float(significance_floor) * dynamic_range(data)
    if floor > 0.0:
        outlying &= np.abs(np.where(np.isfinite(residual), residual, 0.0)) > floor
    return [
        (start, stop)
        for start, stop in _runs(outlying)
        if stop - start <= int(max_spike_samples)
    ]


def offset_jump_samples(
    data: np.ndarray,
    *,
    offset_jump_sigma: float,
    max_spike_samples: int,
    significance_floor: float = 0.0,
) -> list[int]:
    """Sample indices at which the record steps to a new level and stays there.

    A step is a discontinuity: it happens between consecutive samples, so it
    shows as an outlying first difference that the neighbouring differences do
    not undo.  A spike's differences cancel across the event; a step's
    accumulate.

    This detector alone **cannot** tell an instrumentation step from plasma
    breakdown, which is also a step to a new level.  Nothing in one channel's
    waveform distinguishes them, which is why the caller suppresses candidates
    that appear coherently across the array (see
    :func:`validate_magnetics_signals`) and why an offset jump is only ever a
    warning.
    """
    if data.size < 5:
        return []
    difference = np.diff(np.where(np.isfinite(data), data, np.nan))
    scores = robust_z_scores(difference)
    floor = float(significance_floor) * dynamic_range(data)
    outlying = np.isfinite(scores) & (np.abs(scores) > offset_jump_sigma)
    outlying |= np.isinf(scores)

    # Wide enough to outvote an excursion of the maximum accepted spike width,
    # so the level on each side is the level the signal actually settled at.
    span = 2 * int(max_spike_samples) + 1
    jumps: list[int] = []
    for start, stop in _runs(outlying):
        if stop - start > int(max_spike_samples):
            # A long run of outlying differences is a sustained excursion, not
            # a step onto a new level.
            continue
        segment = difference[start:stop]
        net = float(np.nansum(segment))
        if abs(net) <= floor:
            continue
        # The decisive test, and the one a per-run sum cannot make: does the
        # signal stay at the new level?  A spike wider than one sample splits
        # into two outlying runs whose sums individually look like steps, and
        # only comparing the levels either side of the whole event shows that
        # they cancel.
        before = data[max(0, start - span + 1) : start + 1]
        after = data[stop : stop + span]
        if before.size == 0 or after.size == 0:
            continue
        level_change = float(np.nanmedian(after) - np.nanmedian(before))
        if abs(level_change) <= floor:
            continue
        # `difference[i]` is the step into sample i+1.
        jumps.append(int(stop))
    return jumps


# ---------------------------------------------------------------------------
# Per-channel assessment
# ---------------------------------------------------------------------------

def implausible_magnitude(data: np.ndarray, ceiling: float | None) -> bool:
    """Whether a record exceeds an absolute physical ceiling anywhere.

    A whole-record verdict, deliberately: the samples below the ceiling are not
    trustworthy either, since whatever put the channel over it (gain, wiring,
    units) was there for the whole shot.  ``False`` when no ceiling is set or
    no finite sample exists.
    """
    if ceiling is None:
        return False
    finite = data[np.isfinite(data)]
    return bool(finite.size) and bool(np.max(np.abs(finite)) > float(ceiling))


def population_peak_outliers(
    peaks: Mapping[Any, float], factor: float | None
) -> set[Any]:
    """The members of one family whose peak dwarfs the family's median peak.

    Cross-channel, like :func:`_coherent_jump_times`, and in the direction that
    is defensible: a channel *far* above every neighbour is a fault, whereas a
    quiet channel is just a quiet channel.  The median is the reference because
    it moves only once half the family is broken.  Empty when the family has
    fewer than three members -- two channels cannot vote -- or when the median
    peak is zero.
    """
    if factor is None or len(peaks) < 3:
        return set()
    finite = {key: float(peak) for key, peak in peaks.items() if np.isfinite(peak)}
    if len(finite) < 3:
        return set()
    reference = float(np.median(list(finite.values())))
    if reference <= 0.0:
        return set()
    return {key for key, peak in finite.items() if peak > float(factor) * reference}


def _leading_slice(size: int, fraction: float) -> slice:
    count = max(2, int(round(size * float(fraction))))
    return slice(0, min(size, count))


def _sample_noise(data: np.ndarray) -> float:
    """Sample-to-sample noise, robust to whatever slow signal rides on it.

    The MAD of the first difference measures the high-frequency content only,
    so it is a noise estimate even in a window where the machine is doing
    something -- which on VEST is every window, since the PF coils are already
    ramping when the magnetics record starts.  The ``sqrt(2)`` undoes the
    variance doubling that differencing introduces, so the result is
    comparable with an RMS of the signal itself.
    """
    if data.size < 2:
        return float("nan")
    return 1.4826 * median_absolute_deviation(np.diff(data)) / np.sqrt(2.0)


def _constant_run_reason(
    data: np.ndarray, finite: np.ndarray, start: int, stop: int
) -> str:
    """Name what a run of bit-identical samples most likely is.

    Every one of these is unusable data, so the distinction is diagnostic
    rather than consequential -- but the five causes need different fixes, and
    the waveform alone can separate them:

    ``flatline``      the whole record never changes: a dead channel.
    ``held_head``     the record starts held and then begins moving.
    ``held_tail``     the record stops changing and holds to the end.  On VEST
                      this is what interpolating a shorter processed window
                      onto a longer diagnostics grid produces -- `np.interp`
                      clamps, and the clamped samples are not measurements.
    ``saturation``    an interior run pinned at the record's own extreme: a
                      rail.  Indistinguishable from a genuine excursion that
                      happens to hold, which is why it is named by where it
                      sits rather than asserted as an amplifier fault.
    ``dropout``       an interior run somewhere in the middle of the range:
                      the signal simply stopped arriving.
    """
    if (start, stop) == (0, data.size) or not finite.any():
        return "flatline"
    if float(np.ptp(data[finite])) == 0.0:
        return "flatline"
    if stop >= data.size:
        return "held_tail"
    if start <= 0:
        return "held_head"
    value = data[start]
    if value in (float(np.max(data[finite])), float(np.min(data[finite]))):
        return "saturation"
    return "dropout"


@dataclass(frozen=True)
class _Detections:
    """What the detectors found in one waveform, before cross-channel review.

    Kept separate from :class:`ChannelQuality` because the offset-jump verdict
    is not a per-channel question: a step that the whole array shares is the
    machine, not thirty simultaneous acquisition faults.
    """

    time: np.ndarray
    data: np.ndarray
    seed: int
    hard: tuple[tuple[tuple[int, int], str], ...]
    spikes: tuple[tuple[int, int], ...]
    jumps: tuple[int, ...]
    metrics: dict[str, Any]


def _detect(
    time: np.ndarray,
    data: np.ndarray,
    *,
    config: MagneticsQualityConfig,
    seed: int,
    ceiling: float | None = None,
) -> _Detections:
    """Run every detector over one waveform and measure it."""
    finite = np.isfinite(data)
    hard: list[tuple[tuple[int, int], str]] = []
    for interval in _runs(~finite):
        hard.append((interval, "non_finite"))
    if implausible_magnitude(data, ceiling):
        hard.append(((0, data.size), "implausible_magnitude"))

    constant = constant_runs(data, config.min_constant_run)
    whole_record = bool(constant) and constant[0] == (0, data.size)
    for start, stop in constant:
        hard.append(((start, stop), _constant_run_reason(data, finite, start, stop)))

    spikes: tuple[tuple[int, int], ...] = ()
    jumps: tuple[int, ...] = ()
    if not whole_record and finite.any():
        spikes = tuple(
            spike_intervals(
                data,
                spike_sigma=config.spike_sigma,
                max_spike_samples=config.max_spike_samples,
                significance_floor=config.significance_floor,
            )
        )
        jumps = tuple(
            offset_jump_samples(
                data,
                offset_jump_sigma=config.offset_jump_sigma,
                max_spike_samples=config.max_spike_samples,
                significance_floor=config.significance_floor,
            )
        )

    # Reported, never raised as an event.  "Drift" is only diagnosable against a
    # window where the signal *should* be flat, and VAFT has no shot-independent
    # quiet window: VEST's magnetics record starts with the PF coils already
    # ramping, so a leading window that trends is indistinguishable from the
    # machine doing its job.  #189 asks for baseline drift as a metric, which
    # this is; turning it into a verdict needs a validated pre-excitation
    # window (see #57's vacuum-shot classification) that does not exist yet.
    leading = _leading_slice(data.size, config.baseline_fraction)
    leading_time, leading_data = time[leading], data[leading]
    noise = _sample_noise(data[finite]) if finite.any() else float("nan")
    leading_drift = linear_trend(leading_time, leading_data)
    leading_span = (
        float(leading_time[-1] - leading_time[0]) if leading_time.size > 1 else 0.0
    )
    drift_over_leading = (
        abs(leading_drift * leading_span) if np.isfinite(leading_drift) else float("nan")
    )

    metrics: dict[str, Any] = {
        "samples": int(data.size),
        "finite_fraction": float(finite.mean()) if data.size else float("nan"),
        "constant_fraction": (
            float(sum(stop - start for start, stop in constant) / data.size)
            if data.size
            else float("nan")
        ),
        "flatlined": bool(whole_record),
        "saturated_fraction": float(
            sum(stop - start for (start, stop), reason in hard if reason == "saturation")
            / data.size
        )
        if data.size
        else float("nan"),
        "spike_count": len(spikes),
        "sample_noise": noise,
        "leading_rms": rms(leading_data),
        "leading_drift_per_second": leading_drift,
        "drift_over_leading_window": drift_over_leading,
        "dynamic_range": dynamic_range(data),
        "peak_abs": float(np.max(np.abs(data[finite]))) if finite.any() else float("nan"),
    }
    return _Detections(
        time=time,
        data=data,
        seed=int(seed),
        hard=tuple(hard),
        spikes=spikes,
        jumps=jumps,
        metrics=metrics,
    )


def _coherent_jump_times(
    detections: Mapping[Any, _Detections], config: MagneticsQualityConfig
) -> list[float]:
    """Jump instants shared by enough of the array to be the machine, not a fault.

    A single waveform cannot distinguish an instrumentation step from plasma
    breakdown -- both step to a new level and stay there.  What separates them
    is that breakdown happens on every channel at once.  Coherent candidates
    are therefore dropped rather than reported, which is the "channel/family
    outlier check where defensible" #189 asks for, applied in the only
    direction that is defensible here.
    """
    candidates: list[tuple[Any, float]] = [
        (key, float(found.time[min(sample, found.time.size - 1)]))
        for key, found in detections.items()
        for sample in found.jumps
    ]
    if not candidates:
        return []
    quorum = max(2, int(np.ceil(config.coherent_jump_fraction * len(detections))))
    coherent: list[float] = []
    for _key, instant in candidates:
        sharing = {
            other
            for other, moment in candidates
            if abs(moment - instant) <= config.coherence_window
        }
        if len(sharing) >= quorum:
            coherent.append(instant)
    return coherent


def _compose(
    found: _Detections,
    *,
    config: MagneticsQualityConfig,
    coherent_jumps: Sequence[float],
) -> tuple[np.ndarray, list[QualityEvent], dict[str, Any]]:
    """Turn one channel's detections into per-sample validity codes and events."""
    time, data = found.time, found.data
    codes = np.full(data.size, found.seed, dtype=int)
    events: list[QualityEvent] = []

    def record(interval: tuple[int, int], reason: str, validity: int) -> None:
        start, stop = interval
        if validity > VALIDITY_VALID:
            # A warning cannot be combined by "worst wins": the Data Dictionary
            # orders 1 *above* 0, so `min` would silently discard it.  It marks
            # a sample that is otherwise valid, and never overrides a real
            # failure found on the same sample.
            window = codes[start:stop]
            codes[start:stop] = np.where(window == VALIDITY_VALID, validity, window)
        else:
            codes[start:stop] = np.minimum(codes[start:stop], validity)
        events.append(
            QualityEvent(
                reason=reason,
                start=float(time[start]),
                end=float(time[min(stop, time.size) - 1]),
                samples=int(stop - start),
                validity=int(validity),
            )
        )

    for interval, reason in found.hard:
        record(interval, reason, VALIDITY_INVALID)
    for interval in found.spikes:
        record(interval, "spike", VALIDITY_CERTIFIED)

    reported_jumps = 0
    for sample in found.jumps:
        instant = float(time[min(sample, time.size - 1)])
        if any(abs(instant - shared) <= config.coherence_window for shared in coherent_jumps):
            continue
        # Everything after an uncompensated step carries an unknown offset, so
        # the flag runs to the end of the record rather than marking the single
        # sample the step happened on.
        record((sample, data.size), "offset_jump", VALIDITY_CERTIFIED)
        reported_jumps += 1
    accepted = codes >= VALIDITY_VALID
    valid_times = time[accepted]
    metrics = dict(found.metrics)
    metrics.update(
        {
            "offset_jump_count": reported_jumps,
            "coherent_jump_count": len(found.jumps) - reported_jumps,
            "valid_fraction": float(accepted.mean()) if codes.size else float("nan"),
            "first_valid_time": float(valid_times[0]) if valid_times.size else float("nan"),
            "last_valid_time": float(valid_times[-1]) if valid_times.size else float("nan"),
        }
    )
    return codes, events, metrics


def _channel_name(source: Any, kind: str, index: int) -> str:
    for key in ("name", "identifier"):
        value = lookup(source, f"magnetics.{kind}.{index}.{key}")
        if isinstance(value, str) and value:
            return value
    return f"{kind}[{index}]"


def _waveform(source: Any, node: str) -> tuple[np.ndarray, np.ndarray] | None:
    """A channel's ``(time, data)`` when both are present and consistent."""
    raw = lookup(source, f"{node}.data")
    if raw is None:
        return None
    try:
        values = np.asarray(raw, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size < 2:
        return None
    time = resolve_signal_time(source, node)
    if time is None or time.size != values.size:
        return None
    return time, values


def _scalar_validity(codes: np.ndarray) -> int:
    """The scalar summary of a per-sample assessment, capped at ``valid``.

    ``aggregate_validity`` takes the worst state reached, which is right -- but
    this layer writes ``1`` to mean "flagged", and the Data Dictionary reads
    ``1`` as *certified by the diagnostic RO*.  A wholly flagged channel would
    otherwise summarize as better than a clean one, and automation has no
    business certifying anything.  So the flag lives in ``validity_timed``,
    where it is per-sample and unambiguous, and the scalar says at most
    "valid".
    """
    return min(VALIDITY_VALID, aggregate_validity(codes))


def _status(codes: np.ndarray, events: Sequence[QualityEvent]) -> ValidationStatus:
    """Summarize a per-sample assessment.

    ``FAIL`` says some sample is unusable, not that the channel must be
    discarded -- ``validity_timed`` and ``valid_fraction`` carry how much, and
    what to do about it is the consumer's policy, not this layer's (#253 §7).
    """
    if (codes < VALIDITY_VALID).any():
        return ValidationStatus.FAIL
    if events:
        return ValidationStatus.WARN
    return ValidationStatus.PASS


def _unavailable(kind: str, index: int, name: str, quantity: str, unit: str) -> ChannelQuality:
    return ChannelQuality(
        kind=kind,
        index=index,
        name=name,
        quantity=quantity,
        unit=unit,
        status=ValidationStatus.NOT_AVAILABLE,
        validity=VALIDITY_INVALID,
        validity_timed=np.empty(0, dtype=int),
        valid_fraction=float("nan"),
        reason="no processed waveform on a resolvable time base",
    )


def _whole_record_reason(found: _Detections) -> str:
    """Why a channel was condemned outright, when it was."""
    whole = {reason for (start, stop), reason in found.hard if (start, stop) == (0, found.data.size)}
    if "implausible_magnitude" in whole:
        return (
            f"peak |x| = {found.metrics['peak_abs']:.3g} exceeds the physical ceiling; "
            "not a measurement anywhere in the record"
        )
    if "population_outlier" in whole:
        return (
            f"peak |x| = {found.metrics['peak_abs']:.3g} is "
            f"{found.metrics['peak_over_family_median']:.1f}x the family's median peak"
        )
    if found.seed < VALIDITY_VALID:
        return "seeded from an invalid raw voltage"
    return ""


def validate_magnetics_signals(
    source: Any,
    *,
    config: MagneticsQualityConfig | None = None,
    kinds: Iterable[str] = tuple(QUANTITY_BY_KIND),
) -> tuple[ChannelQuality, ...]:
    """Assess every equilibrium magnetics channel's processed waveform.

    Nothing is written: the result is a report, and :func:`project_validity`
    puts the standardized part of it into the IDS.  Channels carrying no
    processed waveform come back as
    :attr:`~vaft.validation.model.ValidationStatus.NOT_AVAILABLE` with a reason
    rather than being dropped, so a missing channel stays visible and stays
    distinguishable from a failed one.

    Detection runs in two passes because one of the questions is not per
    channel: a step to a new level looks identical in a broken integrator and
    in plasma breakdown, and only the rest of the array can tell them apart.
    """
    settings = config or MagneticsQualityConfig()
    order: list[tuple[str, int, str, str, str]] = []
    unavailable: dict[tuple[str, int], ChannelQuality] = {}
    detections: dict[tuple[str, int], _Detections] = {}

    for kind in kinds:
        quantity, unit = QUANTITY_BY_KIND[kind]
        for index in range(path_count(source, f"magnetics.{kind}")):
            name = _channel_name(source, kind, index)
            order.append((kind, index, name, quantity, unit))
            waveform = _waveform(source, channel_node(kind, index, quantity))
            if waveform is None:
                unavailable[(kind, index)] = _unavailable(kind, index, name, quantity, unit)
                continue
            # The raw voltage's own verdict is an input, not a duplicate: a
            # channel whose acquisition failed cannot have a trustworthy
            # processed value either.  It can only lower the starting point,
            # and it is a statement about a different datum, so it is read
            # rather than assumed identical.
            voltage = read_validity(source, f"magnetics.{kind}.{index}.voltage")
            seed = VALIDITY_VALID if voltage is None else min(VALIDITY_VALID, int(voltage))
            detections[(kind, index)] = _detect(
                *waveform,
                config=settings,
                seed=seed,
                ceiling=settings.max_plausible_field if kind == "b_field_pol_probe" else None,
            )

    coherent = _coherent_jump_times(detections, settings)
    for kind in kinds:
        family = {
            key: found.metrics["peak_abs"]
            for key, found in detections.items()
            if key[0] == kind
        }
        reference = float(np.median([v for v in family.values() if np.isfinite(v)])) if family else float("nan")
        for key in population_peak_outliers(family, settings.population_peak_factor):
            found = detections[key]
            detections[key] = replace(
                found,
                hard=found.hard + (((0, found.data.size), "population_outlier"),),
                metrics={**found.metrics, "peak_over_family_median": found.metrics["peak_abs"] / reference},
            )

    report: list[ChannelQuality] = []
    for kind, index, name, quantity, unit in order:
        found = detections.get((kind, index))
        if found is None:
            report.append(unavailable[(kind, index)])
            continue
        codes, events, metrics = _compose(
            found, config=settings, coherent_jumps=coherent
        )
        report.append(
            ChannelQuality(
                kind=kind,
                index=index,
                name=name,
                quantity=quantity,
                unit=unit,
                status=_status(codes, events),
                validity=_scalar_validity(codes),
                validity_timed=codes,
                valid_fraction=float(metrics["valid_fraction"]),
                metrics=metrics,
                events=tuple(events),
                reason=_whole_record_reason(found),
            )
        )
    return tuple(report)


def project_validity(
    source: Any, report: Iterable[ChannelQuality]
) -> dict[str, int]:
    """Write a report's per-sample assessment into the native validity nodes.

    Only the datum's own assessment reaches the IDS -- this is signal quality
    writing signal validity, which is the one direction #253 §10 permits.
    Metrics, events and thresholds stay in the report; the IDS carries the
    standardized part downstream consumers need.

    Returns the scalar validity written per node.
    """
    written: dict[str, int] = {}
    for quality in report:
        if not quality.projectable:
            continue
        node = channel_node(quality.kind, quality.index, quality.quantity)
        written[node] = write_validity(
            source, node, quality.validity_timed, scalar=quality.validity
        )
    return written


def unusable_channels_at(
    source: Any,
    times: np.ndarray | Sequence[float],
    *,
    min_validity: int = VALIDITY_VALID,
    kinds: Iterable[str] = tuple(QUANTITY_BY_KIND),
) -> dict[tuple[str, int], np.ndarray]:
    """Which channels a consumer must not use at each instant in ``times``.

    Reads the validity already projected into the IDS -- it does not re-run the
    detectors.  That is the point of #189: basic sensor health is decided once,
    at the diagnostics stage, and every downstream module consults the answer
    instead of inventing its own.

    Returns only the channels that are unusable at *some* instant, each with a
    boolean array over ``times``, so a caller iterates over the exceptions
    rather than over the whole array.  An ODS carrying no validity yields an
    empty mapping and therefore changes nothing.
    """
    query = np.asarray(times, dtype=float).reshape(-1)
    unusable: dict[tuple[str, int], np.ndarray] = {}
    for kind in kinds:
        quantity, _unit = QUANTITY_BY_KIND[kind]
        for index in range(path_count(source, f"magnetics.{kind}")):
            node = channel_node(kind, index, quantity)
            accepted = validity_mask(
                source, node, times=query, min_validity=min_validity
            )
            if not accepted.all():
                unusable[(kind, index)] = ~accepted
    return unusable


# ---------------------------------------------------------------------------
# Manifest metrics
# ---------------------------------------------------------------------------

def _family(quality: ChannelQuality, positions: Mapping[tuple[str, int], tuple[float, float]]) -> str:
    """The EFIT-submitted family a channel belongs to.

    Classified by :func:`vaft.omas.vacuum_magnetics.probe_family`, the one place
    that owns those boundaries, so a coverage report and a forward model cannot
    disagree about what "inboard" means.  Imported lazily: it reaches the
    machine-mapping layer, which the validation core does not depend on.
    """
    from vaft.omas.vacuum_magnetics import probe_family

    position = positions.get((quality.kind, quality.index))
    if position is None:
        return "unknown"
    return probe_family(quality.kind, *position)


def _positions(source: Any, report: Iterable[ChannelQuality]) -> dict[tuple[str, int], tuple[float, float]]:
    found: dict[tuple[str, int], tuple[float, float]] = {}
    for quality in report:
        prefix = f"magnetics.{quality.kind}.{quality.index}.position"
        if quality.kind == "flux_loop":
            prefix = f"{prefix}.0"
        r, z = lookup(source, f"{prefix}.r"), lookup(source, f"{prefix}.z")
        if r is None or z is None:
            continue
        try:
            found[(quality.kind, quality.index)] = (float(r), float(z))
        except (TypeError, ValueError):
            continue
    return found


def magnetics_quality_metrics(
    source: Any,
    report: Iterable[ChannelQuality],
    *,
    config: MagneticsQualityConfig | None = None,
) -> dict[str, Any]:
    """The diagnostics-stage magnetics block for a stage manifest (#189 req 4).

    Quantitative, so a regression is visible without opening a figure, and
    report-only: nothing here is a threshold the pipeline gates on.
    """
    entries = tuple(report)
    positions = _positions(source, entries)
    channels = []
    for quality in entries:
        channels.append(
            {
                "kind": quality.kind,
                "index": quality.index,
                "name": quality.name,
                "quantity": quality.quantity,
                "unit": quality.unit,
                "family": _family(quality, positions),
                "status": str(quality.status),
                "validity": quality.validity,
                "valid_fraction": quality.valid_fraction,
                "reason": quality.reason,
                "metrics": dict(quality.metrics),
                "events": [event.as_dict() for event in quality.events],
            }
        )

    def coverage(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        """How many channels a set declares, has, and can be used from.

        ``usable`` and ``fully_usable`` are separate on purpose.  The scalar
        validity aggregates to the *worst* state a channel reached, so a
        channel good for nine tenths of the discharge reads ``invalid`` there;
        counting only those would report a healthy array as unusable.  What a
        consumer with a time window actually needs is the first count, and
        ``validity_timed`` is what tells it which samples.
        """
        return {
            "expected": len(entries),
            "present": sum(
                1
                for entry in entries
                if entry["status"] != str(ValidationStatus.NOT_AVAILABLE)
            ),
            "usable": sum(
                1
                for entry in entries
                if np.isfinite(entry["valid_fraction"]) and entry["valid_fraction"] > 0.0
            ),
            "fully_usable": sum(
                1
                for entry in entries
                if entry["status"] != str(ValidationStatus.NOT_AVAILABLE)
                and entry["validity"] >= VALIDITY_VALID
            ),
        }

    families = {
        family: coverage([entry for entry in channels if entry["family"] == family])
        for family in sorted({entry["family"] for entry in channels})
    }

    settings = config or MagneticsQualityConfig()
    summary = coverage(channels)
    summary["flagged_channels"] = sum(1 for entry in channels if entry["events"])
    summary["events"] = dict(
        Counter(event["reason"] for entry in channels for event in entry["events"])
    )
    return {
        "schema_version": 1,
        "configuration": asdict(settings),
        "summary": summary,
        "families": families,
        "channels": channels,
    }
