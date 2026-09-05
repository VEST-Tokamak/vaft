"""Plasma-free magnetic-response benchmark for the VEST wall model (issue #190).

One question, asked of the machine model rather than of a shot:

    Can one physically consistent active-coil / passive-wall model reproduce the
    measured plasma-free magnetic response across representative VEST shots, PF
    excitations, sensors and machine eras?

That is a different question from the routine eddy-stage QA (#139), which asks
whether *this* shot's vacuum reconstruction looks right, and different again
from magnetics signal quality (#189), which asks whether the waveform is a
measurement at all.  The three must not be run together::

    #189  is the datum usable?              -> magnetics.*.validity
    #190  does it match the vacuum model?   -> residual metrics   *(here)*
    #139  may the pipeline use the result?  -> stage QA policy

The direction is strictly one way.  This module *reads* the validity #189
established and never writes it: a channel that disagrees with the forward
model is evidence about the model, and cross-diagnostic disagreement must never
invalidate source data (#253 §10).

Nothing here re-implements the forward model.  The passive-loop dynamics come
from the existing solver through
:func:`vaft.omas.process_wrapper.compute_eddy_currents`, and the synthetic
response and residual statistics from :mod:`vaft.omas.vacuum_magnetics`.  What
this module adds is what the routine stage cannot supply: case selection
without a plasma, a wall solve that measured plasma current cannot contaminate,
a check that the solver's assumed initial state has had time to be forgotten,
and cross-shot aggregation.

Two properties make the benchmark a qualification rather than a fit
------------------------------------------------------------------
**The wall is driven by measured PF currents alone.**  The routine eddy stage
lets plasma filaments drive the wall solve, which is right for processing a
plasma shot and wrong here: the plasma current would partly explain the very
response being validated.  :func:`benchmark_wall_currents` re-solves with no
filaments at all, so a case is constructible whether or not the source shot
later forms a plasma.

**Preprocessing is fixed independently of the residual.**  Nothing here re-fits
a baseline, a gain or a resistance to reduce disagreement.  The optional
resistance study varies a single global scale factor; independently fitting
hundreds of passive-loop resistances against the same magnetic data used to
validate them would turn the benchmark into an underconstrained fit.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from vaft.formula.statistics import sigma_threshold_crossing
from vaft.ods_access import path_value

if TYPE_CHECKING:  # pragma: no cover
    from vaft.machine_mapping.wall_resistance import WallResistanceCalibration

__all__ = [
    "BenchmarkError",
    "DEFAULT_HISTORY_TIME_CONSTANTS",
    "BENCHMARK_CASE_SCHEMA",
    "MIN_COIL_DRIVE_FRACTION",
    "PLASMA_FREE_EVIDENCE_SCHEMA",
    "coil_drive_check",
    "PlasmaFreeInterval",
    "aggregate_benchmark",
    "benchmark_wall_currents",
    "plasma_free_interval",
    "run_benchmark_case",
    "solver_history_check",
    "wall_time_constants",
]

#: How many of the wall's slowest time constants must elapse between the start
#: of the solver's input and the start of the validation window.  The solver
#: begins from ``I_wall = 0``; after ``n`` time constants that assumption has
#: decayed by ``exp(-n)``, so three leaves under 5% of it.
DEFAULT_HISTORY_TIME_CONSTANTS = 3.0

#: Smallest fraction of the shot's peak PF current that must appear inside the
#: validation window for a plasma-free eddy score to mean anything (see
#: :func:`coil_drive_check`).  The gate is independent of how the interval was
#: found, and it earned its place on 41524: the legacy Ip discharge detector
#: fired on PF pickup ~0.8 ms *before* the solenoid, so the nominal plasma-free
#: window carried 0.003 of the shot's drive and the eddy score there was a
#: ratio of two noise numbers.  With the interval ending at the plasma onset
#: of the shared timing policy (#409) the packaged shots 39915, 41524 and
#: 41672 all reach 1.00 of their shot peak inside the window; the gate stays,
#: for the shot whose coils genuinely fire late.
MIN_COIL_DRIVE_FRACTION = 0.10

#: Sigma above the early-record noise band at which the plasma current is
#: considered to have emerged.  Matches the residual-onset convention in
#: :mod:`vaft.omas.vacuum_magnetics` so the two detectors speak the same way.
ONSET_SIGMA = 5.0

#: Fraction of the record used as the reference noise band for that detector.
ONSET_REFERENCE_FRACTION = 0.2

#: Schema of ``plasma_free_evidence`` (#409): the boundary comes from the shared
#: plasma-timing policy and its provenance is recorded; the two retired
#: detectors are reported under ``legacy`` until this reaches 3.
PLASMA_FREE_EVIDENCE_SCHEMA = 2

#: Schema of a :func:`run_benchmark_case` record; follows the evidence schema.
BENCHMARK_CASE_SCHEMA = 2


class BenchmarkError(ValueError):
    """Raised when a shot cannot supply a usable plasma-free benchmark case."""


class PlasmaFreeInterval(dict):
    """The plasma-free stretch of one shot, with the evidence for it.

    A ``dict`` so it serializes into a manifest unchanged, with attribute
    access for the three fields callers actually branch on.
    """

    @property
    def start(self) -> float:
        return float(self["start"])

    @property
    def end(self) -> float:
        return float(self["end"])

    @property
    def case_type(self) -> str:
        return str(self["case_type"])


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------

def _signal(ods: Any, path: str) -> np.ndarray | None:
    """A waveform at ``path``, or ``None`` when the ODS does not carry one.

    Through the shared non-mutating accessor (issue #118): an ODS creates paths
    on access, so probing for an absent ``magnetics.ip`` -- which a vacuum shot
    genuinely has -- would leave a malformed branch behind that fails the next
    consistency check, and ``flat()`` does not show it, so the damage stays
    invisible until the ODS is saved.
    """
    values = path_value(ods, path)
    if values is None:
        return None
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return array if array.size >= 2 else None




def plasma_free_interval(
    ods: Any,
    *,
    sigma: float = ONSET_SIGMA,
    reference_fraction: float = ONSET_REFERENCE_FRACTION,
) -> PlasmaFreeInterval:
    """The interval of a shot over which no plasma contributes, with evidence.

    A shot that never forms a plasma is a benchmark case over its whole record
    -- the situation the routine eddy stage treats as "nothing to validate"
    and #190 treats as the cleanest case there is.  A shot that does form one
    contributes the stretch before it.

    The boundary is the plasma-free boundary of the shared timing policy
    (:func:`vaft.omas.plasma_timing.plasma_timing`, issue #409): the plasma
    window's onset -- the slow H-alpha line when it is usable, optical, so the
    coil-firing pickup every magnetic diagnostic carries cannot trigger it --
    or the plasma-current principal pulse when that starts earlier, since a
    plasma-*free* stretch wants the earliest evidence of plasma from any
    source.  The interval is half-open, ``[start, boundary)``, on every grid.

    A shot is a vacuum case only on evidence: no plasma current recorded at
    all, or a current whose principal pulse the detector examined and found
    absent.  A current the product marks unusable, or one the policy cannot
    read, cannot certify a plasma-free interval and raises
    :class:`BenchmarkError`.  The two detectors this replaced -- the legacy
    Ip discharge detector and a sigma crossing of the current, both of which
    fired on PF pickup -- are reported under ``legacy`` until the evidence
    schema reaches 3.  ``sigma`` feeds only that block.
    """
    from vaft.omas.plasma_timing import PlasmaTimingError, plasma_timing
    from vaft.omas.vacuum_magnetics import plasma_free_boundary
    from vaft.validation.imas import resolve_signal_time

    time = _signal(ods, "pf_active.time")
    if time is None:
        raise BenchmarkError(
            "the ODS carries no pf_active time grid, so no interval can be defined"
        )
    record = (float(time[0]), float(time[-1]))

    ip_data = _signal(ods, "magnetics.ip.0.data")
    ip_time = resolve_signal_time(ods, "magnetics.ip.0") if ip_data is not None else None
    has_ip = ip_time is not None and ip_data is not None and ip_time.size == ip_data.size
    evidence: dict[str, Any] = {
        "schema_version": PLASMA_FREE_EVIDENCE_SCHEMA,
        "method": "plasma-free boundary of the shared timing policy: the plasma "
        "window's onset (H-alpha by label, validated fast line, plasma-current "
        "principal pulse) or the current's principal pulse when it starts earlier",
        "record": list(record),
    }

    boundary = float("nan")
    if not has_ip:
        evidence["reason"] = "no usable magnetics.ip waveform"
    else:
        try:
            timing = plasma_timing(ods)
        except (PlasmaTimingError, ValueError, TypeError) as exc:
            raise BenchmarkError(
                f"the plasma current cannot certify a plasma-free interval: {exc}"
            ) from exc
        evidence["plasma_timing"] = timing.summary()
        if timing.found:
            boundary, source = plasma_free_boundary(timing)
            evidence.update({"boundary": boundary, "boundary_source": source})
        elif "ip_unusable" in timing.flags or timing.ip is None:
            raise BenchmarkError(
                "the plasma current is marked unusable, so no plasma-free interval can be "
                f"certified ({timing.fallback_reason})"
            )
        else:
            evidence["reason"] = f"no source shows a plasma ({timing.fallback_reason})"

        from vaft.machine_mapping.magnetics import vfit_plasma_mgods_startend

        start, end = vfit_plasma_mgods_startend(ods)
        span = float(ip_time[-1] - ip_time[0])
        reference = ip_time < ip_time[0] + reference_fraction * span
        fine = sigma_threshold_crossing(ip_time, ip_data, reference, sigma=sigma)
        evidence["legacy"] = {
            "discharge_detector_onset": float(start) if start >= 0 and end > start else float("nan"),
            "sigma_crossing_onset": float(fine),
        }

    if not np.isfinite(boundary) or boundary <= record[0]:
        interval = PlasmaFreeInterval(
            start=record[0],
            end=record[1],
            case_type="vacuum",
            plasma_free_evidence=evidence,
        )
    else:
        # The boundary itself: the half-open evaluation window then excludes
        # every sample at or after it on whichever grid a consumer reads.
        evidence["boundary_on_pf_grid"] = bool(np.any(np.isclose(time, boundary, rtol=0.0, atol=1e-9)))
        interval = PlasmaFreeInterval(
            start=record[0],
            end=min(boundary, record[1]),
            case_type="pre_plasma",
            plasma_free_evidence=evidence,
        )

    if has_ip:
        from vaft.formula.statistics import noise_band

        inside = (ip_time >= interval.start) & (ip_time < interval.end)
        span = float(ip_time[-1] - ip_time[0])
        reference = ip_time < ip_time[0] + reference_fraction * span
        centre, width = noise_band(ip_data[reference])
        # Reported together on purpose: the residual current in a nominally
        # plasma-free interval is only interpretable against the noise the
        # detector called it consistent with, and a reader should be able to
        # see both rather than take the verdict on trust.
        evidence.update(
            {
                "max_abs_ip_in_interval": (
                    float(np.max(np.abs(ip_data[inside]))) if inside.any() else float("nan")
                ),
                "ip_reference_mean": float(centre),
                "ip_reference_std": float(width),
            }
        )
    return interval


# ---------------------------------------------------------------------------
# The PF-only wall solve
# ---------------------------------------------------------------------------

def benchmark_wall_currents(
    ods: Any,
    *,
    resistance_scale: float = 1.0,
    calibration: "WallResistanceCalibration | None" = None,
    dt_sub: float = 5.0e-5,
) -> Any:
    """Re-solve the passive-wall currents driven by measured PF currents alone.

    Returns a **copy** with ``pf_passive.loop.*.current`` replaced.  The source
    ODS is never touched, so a benchmark can be run against a routinely
    processed shot without disturbing the product the pipeline wrote.

    The routine eddy stage passes plasma filaments into the same solver, which
    is correct for a plasma shot and disqualifying here: the measured plasma
    current would partly explain the response the benchmark is checking.
    Passing no filaments is the whole difference, and it means the wall
    solution is unchanged by any plasma-current waveform the ODS happens to
    carry.

    ``resistance_scale`` multiplies every ``pf_passive.loop.*.resistance`` by
    one global factor, for the low-dimensional sensitivity study #190 permits.
    It is deliberately a single scalar: fitting hundreds of loop resistances
    against the magnetic data used to validate them would make the benchmark an
    underconstrained fit rather than an independent qualification.

    ``calibration`` replaces the shipped resistances with the nominal hoop
    resistance times that vintage's 31 band factors
    (:mod:`vaft.machine_mapping.wall_resistance`). ``None`` leaves the ODS's
    own values in place -- byte-identical to before this seam existed -- and
    the vintage the shipped asset was built from reproduces them exactly, so
    passing it is a no-op that makes the wall's provenance explicit. This is
    the seam a #308 calibration fit varies; ``resistance_scale`` still
    multiplies on top.
    """
    from vaft.omas.process_wrapper import compute_eddy_currents

    working = copy.deepcopy(ods)
    scale = float(resistance_scale)
    if scale <= 0.0:
        raise BenchmarkError(f"resistance_scale must be positive, got {scale}")
    if calibration is not None:
        from vaft.machine_mapping.wall_resistance import calibrated_resistance

        for index, value in enumerate(calibrated_resistance(working, calibration)):
            working[f"pf_passive.loop.{index}.resistance"] = float(value)
    if scale != 1.0:
        for index in range(len(working["pf_passive.loop"])):
            path = f"pf_passive.loop.{index}.resistance"
            working[path] = float(working[path]) * scale
    compute_eddy_currents(working, plasma=[], ip=[], dt_sub=dt_sub)
    if "pf_passive.time" in working:
        working["pf_passive.ids_properties.homogeneous_time"] = 1
    return working


def wall_time_constants(ods: Any) -> np.ndarray:
    """The passive structure's L/R decay times, slowest first.

    The eddy solve integrates ``M dI/dt = -R I - L dI_active/dt``, so the
    homogeneous decay is governed by the generalized eigenvalues of the
    ``(R, L)`` pencil, and the slowest of them sets how long the solver's
    assumed initial state persists.  They are read from the same
    segment-wise wall eigenbasis the reduced-wall work builds (vaft #473):
    at full rank the reduced inductance ``L_r = V^T L V`` with ``R_r = I``
    has exactly the global spectrum, so the QA and the basis cannot report
    two different walls.
    """
    from vaft.omas.process_wrapper import compute_wall_mode_basis_ods, compute_impedance_matrices_ods
    from vaft.process.wall_modes import WallModeError, global_time_constants

    try:
        basis = compute_wall_mode_basis_ods(ods)
        _resistance, _coupling, inductance = compute_impedance_matrices_ods(ods, [])
        taus = global_time_constants(basis, inductance)
    except WallModeError as exc:
        raise BenchmarkError(
            "the passive structure has no well-posed decaying modes: " + str(exc)
        ) from exc
    taus = taus[np.isfinite(taus) & (taus > 0.0)]
    if taus.size == 0:
        raise BenchmarkError(
            "the passive structure has no decaying mode; its resistance or "
            "inductance matrix is degenerate"
        )
    return np.sort(taus)[::-1]


def solver_history_check(
    ods: Any,
    validation_start: float,
    *,
    n_tau: float = DEFAULT_HISTORY_TIME_CONSTANTS,
    time_constants: np.ndarray | None = None,
) -> dict[str, Any]:
    """Whether the solver has forgotten its assumed initial state by ``validation_start``.

    The solver starts from ``I_wall = 0`` at the beginning of its input.  Inside
    an already-established PF transient that is simply wrong, and the error
    decays with the wall's own slowest time constant -- so a benchmark window
    that opens too early measures the initial condition, not the wall model.

    Reports rather than raises: an under-supported case is still informative,
    and #190 asks for it to be rejected *or explicitly warned about*.
    """
    constants = wall_time_constants(ods) if time_constants is None else time_constants
    slowest = float(constants[0])
    time = _signal(ods, "pf_active.time")
    if time is None:
        raise BenchmarkError("the ODS carries no pf_active time grid")
    solver_start = float(time[0])
    available = float(validation_start) - solver_start
    required = float(n_tau) * slowest
    return {
        "solver_start": solver_start,
        "validation_start": float(validation_start),
        "available_history": available,
        "required_history": required,
        "slowest_wall_time_constant": slowest,
        "n_tau": float(n_tau),
        # exp(-n) of the assumed zero-current state survives at the window's start.
        "residual_initial_condition": float(np.exp(-available / slowest))
        if slowest > 0
        else float("nan"),
        "sufficient": bool(available >= required),
    }


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def _pf_excitation(ods: Any) -> dict[str, Any]:
    """Which coils drove this case, and how hard.

    The grouping key for "many sensors go bad whenever a particular PF response
    dominates": a case is characterized by which coils actually carried current,
    not by a configuration label that may not match what the shot did.
    """
    coils: list[dict[str, Any]] = []
    for index in range(len(ods["pf_active.coil"])):
        current = _signal(ods, f"pf_active.coil.{index}.current.data")
        if current is None:
            continue
        peak = float(np.max(np.abs(current)))
        coils.append(
            {
                "index": index,
                "name": str(ods.get(f"pf_active.coil.{index}.name", f"PF{index}") or f"PF{index}"),
                "peak_abs_current": peak,
            }
        )
    driven = [entry for entry in coils if entry["peak_abs_current"] > 0.0]
    return {
        "current_source": "measured pf_active.coil.*.current.data",
        "active_coils": [entry["name"] for entry in driven],
        "coils": coils,
    }


def coil_drive_check(ods: Any, window: tuple[float, float]) -> dict[str, Any]:
    """Whether the coils drove the vessel inside ``window`` at all.

    The wall term is a response to the coils, so a window in which the coils
    barely moved is instrument baseline and the residual improvement measured
    there is a ratio of two noise numbers.  ``coil_drive_fraction`` is the
    peak coil current inside the half-open window as a fraction of the shot's
    peak (the shot peak, which VEST reaches during the plasma phase, is the
    only current scale the ODS itself supplies); ``sufficiently_driven``
    compares it with :data:`MIN_COIL_DRIVE_FRACTION`.  A precondition,
    reported rather than raised, like :func:`solver_history_check`.  It
    answers "did the coils move here", not "is the wall excited here": a coil
    that ramped before the window still drives a decaying wall current.

    Every key is always present.  A coil whose current is stored on a grid of
    a different length than ``pf_active.time`` cannot be windowed and is
    listed in ``skipped_coils`` rather than silently counted as zero drive.
    """
    time = _signal(ods, "pf_active.time")
    start, end = float(window[0]), float(window[1])
    report: dict[str, Any] = {
        "window": [start, end],
        "min_coil_drive_fraction": MIN_COIL_DRIVE_FRACTION,
        "shot_peak_abs_current": None,
        "window_peak_abs_current": None,
        "coil_drive_fraction": None,
        "sufficiently_driven": False,
        "skipped_coils": [],
        "reason": "",
    }
    if time is None:
        report["reason"] = "the ODS carries no usable pf_active time grid"
        return report
    inside = (time >= start) & (time < end)
    shot_peak = 0.0
    window_peak = 0.0
    for index in range(len(ods["pf_active.coil"])):
        current = _signal(ods, f"pf_active.coil.{index}.current.data")
        if current is None:
            continue
        shot_peak = max(shot_peak, float(np.max(np.abs(current))))
        if current.size != inside.size:
            report["skipped_coils"].append(
                str(ods.get(f"pf_active.coil.{index}.name", f"PF{index}") or f"PF{index}")
            )
            continue
        if inside.any():
            window_peak = max(window_peak, float(np.max(np.abs(current[inside]))))
    report["shot_peak_abs_current"] = shot_peak
    report["window_peak_abs_current"] = window_peak
    if shot_peak <= 0.0:
        report["reason"] = "no coil carried current anywhere in the record"
        return report
    fraction = window_peak / shot_peak
    report["coil_drive_fraction"] = float(fraction)
    report["sufficiently_driven"] = bool(fraction >= MIN_COIL_DRIVE_FRACTION)
    if not report["sufficiently_driven"]:
        report["reason"] = (
            f"peak coil current inside the window is {fraction:.3g} of the shot peak, "
            f"below {MIN_COIL_DRIVE_FRACTION:g}: the eddy score here is a ratio of noise"
        )
    if report["skipped_coils"]:
        report["reason"] = (report["reason"] + "; " if report["reason"] else "") + (
            f"{len(report['skipped_coils'])} coil(s) off the pf_active grid were not windowed"
        )
    return report


def _static_model(ods: Any) -> dict[str, Any]:
    """The machine-model revision a case was evaluated against.

    Without this a residual is not reproducible: the same shot against a
    different coupling matrix or resistance set is a different measurement of
    the model.
    """
    resistances = np.array(
        [
            float(ods[f"pf_passive.loop.{index}.resistance"])
            for index in range(len(ods["pf_passive.loop"]))
        ]
    )
    outline = _signal(ods, "wall.description_2d.0.limiter.unit.0.outline.r")
    from vaft.machine_mapping.wall_resistance import identify_calibration

    try:
        calibration = identify_calibration(ods)
    except (KeyError, ValueError, TypeError) as reason:  # unbanded or foreign passive model
        calibration = {"key": None, "error": str(reason)}
    return {
        "wall_calibration": calibration,
        "passive_loop_count": int(resistances.size),
        "pf_coil_count": int(len(ods["pf_active.coil"])),
        "passive_resistance_sum": float(resistances.sum()),
        "passive_resistance_median": float(np.median(resistances)),
        "machine_version": ods.get("dataset_description.data_entry.machine", None),
        "wall_outline_points": int(
            0 if outline is None else outline.size
        ),
    }


def run_benchmark_case(
    ods: Any,
    *,
    shot: int | None = None,
    machine_era: str | None = None,
    window: tuple[float, float] | None = None,
    per_family: int | None = None,
    resistance_scale: float = 1.0,
    calibration: "WallResistanceCalibration | None" = None,
    dt_sub: float = 5.0e-5,
    n_tau: float = DEFAULT_HISTORY_TIME_CONSTANTS,
    min_samples: int = 2,
) -> dict[str, Any]:
    """Evaluate one plasma-free case: PF-only wall solve, then measured vs model.

    The solver-input window and the validation window are deliberately
    different.  The solver is driven over the whole plasma-free stretch so the
    wall currents have a history; the comparison opens only once the assumed
    zero-current initial state has decayed, which
    :func:`solver_history_check` computes from the wall's own slowest time
    constant rather than from a guessed margin.

    ``per_family=None`` evaluates every usable channel, which is what
    qualifying a machine model needs; #139's compact subset stays available for
    routine per-shot QA and for plotting.

    Returns a manifest carrying the provenance #190 asks for alongside the
    metrics, so a residual can be traced back to the model revision that
    produced it.
    """
    from vaft.omas.vacuum_magnetics import (
        synthetic_vacuum_magnetics,
        vacuum_residual_metrics,
    )

    interval = plasma_free_interval(ods)
    solver_window = (interval.start, interval.end)

    constants = wall_time_constants(ods)
    if window is None:
        # Snapped to the first solver sample at or after the requirement, so the
        # window opens on a real sample and the history it reports is genuinely
        # satisfied rather than equal to the bound up to float error.
        grid = _signal(ods, "pf_active.time")
        opens = interval.start + float(n_tau) * float(constants[0])
        later = grid[grid >= opens] if grid is not None else np.empty(0)
        validation_window = (
            (float(later[0]), interval.end) if later.size else (opens, interval.end)
        )
    else:
        validation_window = (float(window[0]), float(window[1]))

    history = solver_history_check(
        ods, validation_window[0], n_tau=n_tau, time_constants=constants
    )
    if validation_window[1] <= validation_window[0]:
        raise BenchmarkError(
            f"the plasma-free stretch {solver_window} is shorter than the "
            f"{history['required_history']:.4g} s of solver history the wall's "
            f"{history['slowest_wall_time_constant']:.4g} s time constant requires; "
            "this shot cannot support a benchmark case"
        )

    working = benchmark_wall_currents(
        ods, resistance_scale=resistance_scale, calibration=calibration, dt_sub=dt_sub
    )
    channels = synthetic_vacuum_magnetics(
        working,
        per_family=per_family,
        window=validation_window,
        validity_window=validation_window,
    )
    metrics = vacuum_residual_metrics(
        channels, window=validation_window, min_samples=min_samples
    )

    excluded = [
        {"channel": row["name"], "kind": row["kind"], "reason": row["reason"]}
        for row in metrics["channels"]
        if row["status"] != "evaluated"
    ]
    return {
        "schema_version": BENCHMARK_CASE_SCHEMA,
        "shot": None if shot is None else int(shot),
        "machine_era": machine_era,
        "case_type": interval.case_type,
        "solver_input_window": list(solver_window),
        "validation_window": list(validation_window),
        "plasma_free_evidence": interval["plasma_free_evidence"],
        "pf_excitation": _pf_excitation(ods),
        "coil_drive": coil_drive_check(ods, validation_window),
        "static_model": {
            **_static_model(ods),
            "resistance_scale": float(resistance_scale),
            "applied_calibration": None if calibration is None else {
                "key": calibration.key,
                "digest": calibration.digest(),
                "source": calibration.source,
            },
            "wall_time_constants": {
                "slowest": float(constants[0]),
                "median": float(np.median(constants)),
                "fastest": float(constants[-1]),
            },
        },
        "solver": {"dt_sub": float(dt_sub), **history},
        "channels": {
            "selected": [
                row["name"] for row in metrics["channels"] if row["status"] == "evaluated"
            ],
            "excluded": excluded,
        },
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Across cases
# ---------------------------------------------------------------------------

def _median(values: Sequence[float]) -> float:
    finite = np.array([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.median(finite)) if finite.size else float("nan")


def _group(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    for name in sorted({str(row[key]) for row in rows}):
        members = [row for row in rows if str(row[key]) == name]
        grouped[name] = {
            "cases": len({row["case"] for row in members}),
            "channels": len({row["channel"] for row in members}),
            "median_improvement": _median([row["improvement"] for row in members]),
            "median_normalized_residual": _median(
                [row["normalized_residual"] for row in members]
            ),
            "median_correlation": _median([row["correlation"] for row in members]),
            "worst_channel": min(
                members, key=lambda row: row["improvement"]
                if np.isfinite(row["improvement"]) else np.inf
            )["channel"],
        }
    return grouped


def aggregate_benchmark(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Residual structure across cases, channels, excitations and machine eras.

    The cross-tabs exist so a discrepancy can be attributed rather than merely
    noted.  Which axis a poor result concentrates on is the diagnosis:

    ==========================================  ==================================
    one channel, across many excitations        probe calibration / geometry
    many channels, whenever one PF dominates    coil geometry / current calibration
    a similar timescale mismatch everywhere     passive-wall resistance / coupling
    a change across geometry revisions          static-model provenance
    one shot among consistent neighbours        acquisition / baseline / timing
    ==========================================  ==================================

    No thresholds and no verdict: #190 is explicit that broad scientific
    acceptance bounds must wait until the VEST benchmark distribution has been
    inspected.
    """
    rows: list[dict[str, Any]] = []
    undriven: list[str] = []
    listed = list(cases)
    for position, case in enumerate(listed):
        label = str(case.get("shot") if case.get("shot") is not None else position)
        excitation = ",".join(case.get("pf_excitation", {}).get("active_coils", [])) or "none"
        # A case whose coils never moved inside its window (coil_drive_check)
        # contributes rows for inspection but not to the cross-case spreads:
        # its improvements are ratios of noise, not evidence about the wall.
        driven = bool(case.get("coil_drive", {}).get("sufficiently_driven", True))
        if not driven:
            undriven.append(label)
        for row in case.get("metrics", {}).get("channels", []):
            if row["status"] != "evaluated":
                continue
            rows.append(
                {
                    "case": label,
                    "channel": row["name"],
                    "kind": row["kind"],
                    "family": row["family"],
                    "excitation": excitation,
                    "machine_era": str(case.get("machine_era") or "unknown"),
                    "driven": driven,
                    "improvement": row["improvement"],
                    "normalized_residual": row["normalized_residual"],
                    "correlation": row["correlation"],
                    "wall_authority": row.get("wall_authority", float("nan")),
                }
            )

    if not rows:
        return {
            "schema_version": 1,
            "case_count": len(listed),
            "status": "empty",
            "reason": "no case produced an evaluated channel",
        }

    scored = [row for row in rows if row["driven"]]
    return {
        "schema_version": 1,
        "case_count": len(listed),
        "channel_rows": len(rows),
        "by_case": _group(rows, "case"),
        "by_channel": _group(rows, "channel"),
        "by_family": _group(rows, "family"),
        "by_excitation": _group(rows, "excitation"),
        "by_machine_era": _group(rows, "machine_era"),
        "undriven_cases": undriven,
        "summary": {
            "median_improvement": _median([row["improvement"] for row in scored]),
            "improved_fraction": float(
                np.mean([row["improvement"] > 0.0 for row in scored])
            ) if scored else float("nan"),
            "median_normalized_residual": _median(
                [row["normalized_residual"] for row in scored]
            ),
            "median_wall_authority": _median([row["wall_authority"] for row in scored]),
            "driven_channel_rows": len(scored),
        },
    }
