"""Per-loop evidence for EFIT flux-loop channel selection (issue #295, step 1).

The routine EFIT configuration carries a hand-maintained ``broken`` list that
silently zero-weights six flux loops on every shot.  Before that list can be
deleted, the automatic evidence has to be laid beside it, loop by loop and
shot by shot, so that each historical exclusion is either reproduced with a
reason or shown to be unjustified.  This module assembles that evidence; it
does not delete anything and it does not bake a verdict into a default.

Two classes of evidence, kept apart as #253 §10 requires:

``source_validity``
    What the magnetics signal-quality layer (#189) found in the waveform
    itself: the projected validity, how much of the assessment window it
    leaves usable, and the events behind it.  This is the only evidence that
    may reject a loop on its own.
``model_agreement``
    How the measured flux compares with the passive-wall forward model over a
    validated plasma-free interval (#190).  A disagreement is evidence about
    the model as much as about the loop, so by default it is *report-only*:
    it can mark a loop ``suspect`` once a threshold is set in the policy, and
    it rejects a loop only when the policy says so explicitly *and* the wall
    term is large enough at that loop for the comparison to mean anything.
    It is never written back into the source validity.

Every number that turns evidence into a state is a :class:`FluxLoopPolicy`
field and is echoed into each record, so a table produced by this module is
auditable without the code that produced it.

The manual list is positional inside the flux-loop block: combined index
``65`` is flux-loop index ``0`` after the 64 EFIT B-probes, whatever that loop
is called.  Each record therefore carries the combined one-based index, the
ODS index, the MD field code and the channel name together, because the
labels ``FL01 .. FL11`` that circulate for the list are positions, not names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from vaft.validation.model import ValidationStatus
from vaft.validation.validity import ValidityRecord, usable_fraction

__all__ = [
    "NOT_AVAILABLE",
    "REJECT_FOR_EFIT",
    "STATES",
    "SUSPECT",
    "USABLE",
    "FluxLoopAssessment",
    "FluxLoopPolicy",
    "assess_flux_loops",
    "flux_loop_evidence",
    "manual_exclusion_index",
]

USABLE = "usable"
SUSPECT = "suspect"
REJECT_FOR_EFIT = "reject_for_efit"
NOT_AVAILABLE = "not_available"
STATES = (USABLE, SUSPECT, REJECT_FOR_EFIT, NOT_AVAILABLE)


@dataclass(frozen=True)
class FluxLoopPolicy:
    """The thresholds that turn evidence into a state.  All of them are policy
    (#253 §7) and every record repeats the values it was judged under.

    ``min_valid_fraction_in_window``
        A loop must leave *more* than this fraction of the assessment window
        usable, by its own projected validity, or it is rejected.  ``0.0``
        rejects only a loop with no usable sample in the window at all.
    ``max_normalized_residual`` / ``min_correlation``
        Model-agreement thresholds.  ``None`` (the default) leaves the
        comparison report-only: the metrics are recorded, no state changes.
    ``min_wall_authority_to_score``
        Below this ratio of wall term to reading, the vacuum comparison says
        nothing about the loop and a threshold failure is not scored.  The
        model's silence is not evidence against a loop.
    ``reject_on_model_disagreement``
        Whether a scored threshold failure rejects (``True``) or merely marks
        the loop ``suspect`` (``False``, the default).
    """

    min_valid_fraction_in_window: float = 0.0
    max_normalized_residual: float | None = None
    min_correlation: float | None = None
    min_wall_authority_to_score: float = 0.1
    reject_on_model_disagreement: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FluxLoopAssessment:
    """One loop's evidence and the state the policy derives from it."""

    index: int
    name: str
    field_code: int | None
    combined_index_one_based: int
    state: str
    reasons: tuple[str, ...]
    source_validity: Mapping[str, Any]
    model_agreement: Mapping[str, Any] | None
    window: tuple[float, float] | None
    policy: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "field_code": self.field_code,
            "combined_index_one_based": self.combined_index_one_based,
            "state": self.state,
            "reasons": list(self.reasons),
            "source_validity": dict(self.source_validity),
            "model_agreement": None if self.model_agreement is None else dict(self.model_agreement),
            "window": None if self.window is None else [float(self.window[0]), float(self.window[1])],
            "policy": dict(self.policy),
        }


def manual_exclusion_index(index: int, nbprobe: int) -> int:
    """The one-based combined index the routine ``broken`` list uses for a loop.

    ``vaft.code.efit.kfile`` numbers flux loops at ``index + nbprobe`` after
    converting the list to zero-based, so the list's ``65`` is loop ``0`` when
    64 B-probes precede it.
    """
    return int(index) + int(nbprobe) + 1


def _window_selector(time: np.ndarray | None, size: int, window: tuple[float, float] | None) -> np.ndarray | None:
    if time is None or window is None:
        return None
    grid = np.asarray(time, dtype=float).reshape(-1)
    if grid.size != size:
        return None
    return (grid >= float(window[0])) & (grid <= float(window[1]))


def _source_block(
    quality: Any,
    projected: ValidityRecord | None,
    time: np.ndarray | None,
    window: tuple[float, float] | None,
) -> dict[str, Any]:
    """The intrinsic evidence for one loop.

    The validity the diagnostics stage projected into the IDS is the
    authoritative source verdict (#189: consumers read it, they do not
    rediscover it), so when ``projected`` is assessed the usable fraction is
    taken from it.  The fresh assessment in ``quality`` supplies the events and
    is the fallback for a product that carries no projection.
    """
    assessed_here = np.asarray(quality.validity_timed, dtype=int).reshape(-1)
    fresh = ValidityRecord(scalar=quality.validity, timed=assessed_here if assessed_here.size else None, time=time)
    if projected is not None and projected.assessed:
        record, source = projected, "ids"
    else:
        record, source = fresh, "assessed_here"
    size = record.timed.size if record.timed is not None else (0 if time is None else time.size)
    selector = _window_selector(time, size, window)
    if quality.status is ValidationStatus.NOT_AVAILABLE and not (projected is not None and projected.assessed):
        in_window = float("nan")
    elif selector is None:
        in_window = usable_fraction(record) if record.assessed else float(quality.valid_fraction)
    else:
        in_window = usable_fraction(record, window=selector)
    return {
        "status": quality.status.value,
        "validity": int(quality.validity),
        "valid_fraction": float(quality.valid_fraction),
        "valid_fraction_in_window": float(in_window),
        "validity_source": source,
        "window_resolved": selector is not None,
        "events": [event.reason for event in quality.events],
        "reason": str(quality.reason),
    }


def _model_verdict(
    row: Mapping[str, Any] | None, policy: FluxLoopPolicy, *, consulted: bool
) -> tuple[bool, bool, list[str]]:
    """``(scored, failed, reasons)`` for one loop's residual row."""
    if not consulted:
        return False, False, []
    if row is None:
        return False, False, ["not selected by the vacuum stage; no model comparison"]
    if row.get("status") != "evaluated":
        return False, False, [f"model comparison excluded: {row.get('reason', '')}".rstrip(": ")]
    authority = float(row.get("wall_authority", float("nan")))
    if not np.isfinite(authority) or authority < policy.min_wall_authority_to_score:
        return (
            False,
            False,
            [
                f"wall_authority {authority:.3g} below the scoring floor "
                f"{policy.min_wall_authority_to_score:.3g}; model agreement not scored"
            ],
        )
    reasons: list[str] = []
    if policy.max_normalized_residual is not None:
        value = float(row.get("normalized_residual", float("nan")))
        if not np.isfinite(value) or value > policy.max_normalized_residual:
            reasons.append(f"normalized_residual {value:.3g} > {policy.max_normalized_residual:.3g}")
    if policy.min_correlation is not None:
        value = float(row.get("correlation", float("nan")))
        if not np.isfinite(value) or value < policy.min_correlation:
            reasons.append(f"correlation {value:.3g} < {policy.min_correlation:.3g}")
    return True, bool(reasons), reasons


def assess_flux_loops(
    qualities: Iterable[Any],
    residual_rows: Sequence[Mapping[str, Any]] | None,
    *,
    window: tuple[float, float] | None,
    nbprobe: int,
    time: np.ndarray | None = None,
    projected: Mapping[int, ValidityRecord] | None = None,
    field_codes: Sequence[int | None] | None = None,
    policy: FluxLoopPolicy | None = None,
) -> tuple[FluxLoopAssessment, ...]:
    """Combine intrinsic quality and model agreement into one record per loop.

    ``qualities`` are the flux-loop entries of
    :func:`vaft.validation.magnetics.validate_magnetics_signals`;
    ``residual_rows`` the ``metrics["channels"]`` rows of
    :func:`vaft.omas.vacuum_magnetics.vacuum_residual_metrics` (any kind; only
    flux loops are matched), or ``None`` when no model comparison was run.
    ``time`` is the grid the loops' ``validity_timed`` lives on, so the usable
    fraction can be taken over ``window`` rather than the whole record.
    ``projected`` maps loop index to the validity record the IDS carries; an
    assessed record there is the authoritative source verdict and the fresh
    ``qualities`` supply the events.

    The state rule, in order:

    - ``not_available`` when the quality layer found no waveform;
    - ``reject_for_efit`` when the loop's own validity leaves no more than
      ``min_valid_fraction_in_window`` of the window usable, or -- only if
      ``reject_on_model_disagreement`` -- a scored model threshold fails;
    - ``suspect`` when a scored model threshold fails, or the loop could not
      be compared with the model although a comparison was run;
    - ``usable`` otherwise.

    Model disagreement is never written into the source validity, and an
    unscored comparison (the wall barely reaches the loop) changes nothing.
    """
    settings = policy if policy is not None else FluxLoopPolicy()
    rows: dict[int, Mapping[str, Any]] = {}
    if residual_rows is not None:
        for row in residual_rows:
            if row.get("kind") == "flux_loop":
                rows[int(row["index"])] = row
    codes = list(field_codes) if field_codes is not None else []

    assessments: list[FluxLoopAssessment] = []
    for quality in qualities:
        if quality.kind != "flux_loop":
            continue
        index = int(quality.index)
        source = _source_block(quality, None if projected is None else projected.get(index), time, window)
        row = rows.get(index)
        code = codes[index] if index < len(codes) else None
        common = dict(
            index=index,
            name=str(quality.name),
            field_code=None if code is None else int(code),
            combined_index_one_based=manual_exclusion_index(index, nbprobe),
            source_validity=source,
            model_agreement=None if row is None else dict(row),
            window=None if window is None else (float(window[0]), float(window[1])),
            policy=settings.as_dict(),
        )
        if quality.status is ValidationStatus.NOT_AVAILABLE:
            assessments.append(
                FluxLoopAssessment(state=NOT_AVAILABLE, reasons=(source["reason"] or "no waveform",), **common)
            )
            continue

        reasons: list[str] = []
        fraction = source["valid_fraction_in_window"]
        intrinsic_reject = not (np.isfinite(fraction) and fraction > settings.min_valid_fraction_in_window)
        if intrinsic_reject:
            where = "the window" if source["window_resolved"] else "the record"
            reasons.append(
                f"usable fraction in {where} {fraction:.3g} <= {settings.min_valid_fraction_in_window:.3g}"
                + (f" ({', '.join(source['events'])})" if source["events"] else "")
            )
        scored, failed, model_reasons = _model_verdict(row, settings, consulted=residual_rows is not None)
        reasons.extend(model_reasons)

        if intrinsic_reject or (failed and settings.reject_on_model_disagreement):
            state = REJECT_FOR_EFIT
        elif failed or (residual_rows is not None and (row is None or row.get("status") != "evaluated")):
            state = SUSPECT
        else:
            state = USABLE
        assessments.append(FluxLoopAssessment(state=state, reasons=tuple(reasons), **common))
    return tuple(assessments)


def flux_loop_evidence(
    ods: Any,
    *,
    policy: FluxLoopPolicy | None = None,
    window: tuple[float, float] | None = None,
    benchmark: bool = True,
    nbprobe: int | None = None,
) -> dict[str, Any]:
    """Assemble the flux-loop evidence for one processed ODS.

    Runs the magnetics quality layer on the flux loops and, with
    ``benchmark``, the plasma-free benchmark case of
    :func:`vaft.validation.vacuum_benchmark.run_benchmark_case`, whose
    validation window then becomes the assessment window unless ``window``
    is given.  A shot that cannot support a benchmark case is recorded with
    the reason and assessed on intrinsic quality alone.

    ``nbprobe`` defaults to the number of B-probes EFIT's geometry represents
    (the same clamp ``vaft.code.efit.kfile`` applies), which is what places a
    loop in the routine ``broken`` list.  The source ODS is never modified.
    """
    from vaft.validation.imas import read_validity_record, resolve_signal_time
    from vaft.validation.magnetics import channel_node, validate_magnetics_signals

    qualities = validate_magnetics_signals(ods, kinds=("flux_loop",))
    time = resolve_signal_time(ods, "magnetics.flux_loop.0.flux") if len(qualities) else None
    projected = {
        int(quality.index): read_validity_record(ods, channel_node("flux_loop", quality.index, quality.quantity))
        for quality in qualities
    }
    codes, probes = _channel_definitions()
    if nbprobe is None:
        present = len(ods["magnetics.b_field_pol_probe"]) if "magnetics.b_field_pol_probe" in ods else 0
        nbprobe = min(present, probes) if probes else present

    model: dict[str, Any] = {"consulted": bool(benchmark), "available": False, "reason": None, "case": None}
    rows: Sequence[Mapping[str, Any]] | None = None
    assessment_window = window
    if benchmark:
        from vaft.omas.vacuum_magnetics import VacuumMagneticsError
        from vaft.validation.vacuum_benchmark import BenchmarkError, run_benchmark_case

        try:
            case = run_benchmark_case(ods, window=window)
        except (BenchmarkError, VacuumMagneticsError, ValueError) as error:
            model["reason"] = str(error)
        else:
            rows = case["metrics"]["channels"]
            model["available"] = True
            model["case"] = case
            if assessment_window is None:
                assessment_window = tuple(case["validation_window"])

    assessments = assess_flux_loops(
        qualities,
        rows,
        window=assessment_window,
        nbprobe=int(nbprobe),
        time=time,
        projected=projected,
        field_codes=codes,
        policy=policy,
    )
    settings = policy if policy is not None else FluxLoopPolicy()
    return {
        "schema_version": 1,
        "window": None if assessment_window is None else [float(assessment_window[0]), float(assessment_window[1])],
        "nbprobe": int(nbprobe),
        "policy": settings.as_dict(),
        "assessments": [entry.as_dict() for entry in assessments],
        "model": model,
    }


def _channel_definitions() -> tuple[list[int | None], int]:
    """Flux-loop field codes in EFIT order, and the B-probe count EFIT represents."""
    try:
        from vaft.machine_mapping.magnetics import vest_equilibrium_magnetics_channel_definitions
    except Exception:  # pragma: no cover - the machine mapping is optional here
        return [], 0
    definitions = vest_equilibrium_magnetics_channel_definitions()
    codes = [
        None if entry.get("field_code") is None else int(entry["field_code"])
        for entry in definitions
        if entry.get("kind") == "flux_loop"
    ]
    probes = sum(1 for entry in definitions if entry.get("kind") == "b_field_pol_probe")
    return codes, probes
