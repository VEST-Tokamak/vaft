#!/usr/bin/env python3
"""Generate EFIT constraints OMAS ODS from the diagnostics and eddy products.

The constraint builder needs `magnetics`, `pf_active` and `tf` from the
diagnostics stage and `pf_passive` from the eddy stage.  Each stage stores only
what it owns, so the two are unioned here through
`vaft.database.composition.compose_stage_products`, which also refuses a pairing
whose time grids say the products came from different runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np

import vaft.omas
from vaft.code.efit import correct_flux_loop, generate_constraints_ods as build_constraints
from vaft.code.efit.applicability import not_applicable_constraints
from vaft.database.composition import EddyNoOutputError, compose_stage_products
from vaft.machine_mapping.utils import PlasmaTimingPolicy, resolve_plasma_timing_policy
from vaft.omas.plasma_timing import plasma_timing
from vaft.omas.vest_upstream import efit_constraint_inputs_absent_from_data, machine_era_for_shot
from vaft.validation.imas import is_condemned_channel, resolve_signal_time, resolve_signal_waveform
from vaft.validation.validity import VALIDITY_SUSPECT


LOGGER = logging.getLogger("vaft.generate_constraints_ods")
DEFAULT_UNCERTAINTY = [1e-4, 1e-4, 5e-2, 3e-2, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2]
DEFAULT_WEIGHTING = [1, 1, 1, 0.1, 0.1, 0.1, 0.01, 0.01]


def _csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "on"}


def _table_dir(text: str) -> str:
    if not text:
        return ""
    return text if text.endswith("/") else text + "/"


def _assessment_present(ods) -> bool:
    """Whether the diagnostics stage projected a validity verdict into the magnetics."""
    from vaft.validation.efit_channels import EFIT_KINDS, EFIT_QUANTITY
    from vaft.validation.imas import read_validity_record

    for kind in EFIT_KINDS:
        count = len(ods[f"magnetics.{kind}"]) if f"magnetics.{kind}" in ods else 0
        for index in range(count):
            if read_validity_record(ods, f"magnetics.{kind}.{index}.{EFIT_QUANTITY[kind]}").assessed:
                return True
    return False


def _decisions_for(ods, times, *, require_assessment: bool):
    """The per-channel, per-slice decisions the constraint builder consumes (#296).

    Formed by the acceptance policy from the validity the diagnostics stage
    projected (#189) and nothing else: the routine manual exclusion list is
    gone (#295), so a channel is a constraint unless the assessment says it
    is not.  No detector runs here: a
    product that carries no assessment is refused when ``require_assessment``
    (the routine setting), otherwise every channel is usable by default
    (#424) and the log says so.  Sensor health is decided once, at the
    diagnostics stage; a product that predates it is re-run through that
    stage, not patched here.
    """
    from vaft.validation.efit_channels import decide_efit_channels, efit_probe_count

    if not _assessment_present(ods):
        message = (
            "the magnetics carry no diagnostics-stage assessment (validity/validity_timed); "
            "re-run the diagnostics stage on this product"
        )
        if require_assessment:
            raise ValueError(message + " -- or pass --detect-broken false to proceed with every channel usable")
        LOGGER.warning("%s; proceeding with every channel usable by default (#424)", message)
    nbprobe = efit_probe_count(ods)
    return decide_efit_channels(ods, times, nbprobe=nbprobe)


def _recovery_for(option: int, ods, count: int, shot: int | None = None):
    """The compatibility recovery backend selected by ``--gaussian-fit-option``.

    With *shot*, the probe families are the registry's for that shot (#2164).
    """
    if int(option) <= 0:
        return None
    from functools import partial

    from vaft.code.efit.recovery import gaussian_probe_recovery, probe_families

    return partial(
        gaussian_probe_recovery,
        mode=int(option),
        families=probe_families(ods["magnetics"], count=count, shot=shot),
        uncertainty="legacy",
    )


def _log_decisions(decisions) -> None:
    from vaft.validation.channel_decision import STATE_NAMES

    LOGGER.info("channel decisions (channel-slices per state): %s", json.dumps(decisions.summary()))
    for (kind, index), decision in sorted(decisions.entries.items()):
        if decision.all_usable:
            continue
        states = {STATE_NAMES[code] for code in decision.state.tolist()}
        LOGGER.info(
            "  %s[%d] %s: %s at %d/%d slices (%s)",
            kind, index, decision.name, "/".join(sorted(states)),
            int((decision.state != 0).sum()), decision.n_slices, "; ".join(decision.reasons),
        )


ANALYSIS_RANGE_FALLBACK = "analysis_range_fallback"


class NoPlasmaCurrentError(ValueError):
    """Every selected instant is below ``CUTIP``: EFIT would answer each with a vacuum solution."""

    def __init__(self, message: str, *, count: int, threshold: float):
        super().__init__(message)
        self.count = count
        self.threshold = threshold


class ConstraintWindow(NamedTuple):
    """The EFIT constraint window and where it came from.

    A named tuple rather than a dataclass: the tests load this script by path
    without registering it in ``sys.modules``, which a dataclass with
    postponed annotations cannot survive.
    """

    start: float
    end: float
    source: str
    flags: tuple[str, ...]
    agreement: str | None
    fallback_reason: str | None
    record: dict

    @property
    def fallback(self) -> bool:
        return ANALYSIS_RANGE_FALLBACK in self.flags


def _constraint_window(ods, *, policy: PlasmaTimingPolicy | None = None) -> ConstraintWindow:
    """The shared ``plasma_analysis`` range intersected with the detected plasma window.

    The range comes from the timing policy in ``vest.yaml`` (issue #409); the
    window from ``vaft.omas.plasma_timing`` -- the slow H-alpha line, then the
    fast one, then the plasma current.  When no source shows a plasma the
    whole range is used and the choice is flagged ``analysis_range_fallback``
    with the reason, so a vacuum shot's slices are visibly not plasma slices;
    :func:`_no_plasma_window_reason` then decides whether that fallback is
    the shot's "not applicable" verdict (#2195).
    """
    policy = policy or resolve_plasma_timing_policy()
    ip_time = resolve_signal_time(ods, "magnetics.ip.0")  # the node's own time, else magnetics.time
    if ip_time is None or ip_time.size == 0:
        raise ValueError("magnetics.ip.0.time is empty")
    base_start = max(float(policy.window.tstart), float(ip_time[0]))
    base_end = min(float(policy.window.tend), float(ip_time[-1]))
    if base_end <= base_start:
        raise ValueError(
            f"the plasma current record {ip_time[0]:.4f}-{ip_time[-1]:.4f} s does not overlap the "
            f"{policy.window.name} range {policy.window.tstart}-{policy.window.tend} s"
        )

    # A product without a usable plasma current cannot be constrained at all;
    # plasma_timing's PlasmaTimingError is the right message and propagates.
    timing = plasma_timing(ods, policy=policy)
    if timing.found:
        return ConstraintWindow(
            max(base_start, float(timing.onset)),
            min(base_end, float(timing.offset)),
            str(timing.source),
            tuple(timing.flags),
            timing.agreement,
            timing.fallback_reason,
            timing.record(),
        )
    return ConstraintWindow(
        base_start, base_end, "analysis_range",
        tuple(timing.flags) + (ANALYSIS_RANGE_FALLBACK,),
        timing.agreement, timing.fallback_reason, timing.record(),
    )


def _select_times(
    ods,
    timeset: str,
    tstep: float,
    tstart: float | None,
    tend: float | None,
    *,
    policy: PlasmaTimingPolicy | None = None,
) -> tuple[np.ndarray, ConstraintWindow | None]:
    """The EFIT constraint instants, and the window they were cut from (``None`` in manual mode).

    Auto mode takes :func:`_constraint_window`, clamps it to ``--tstart``/``--tend``
    when given, snaps both ends to the ``tstep`` grid and includes the end;
    manual mode is exactly ``np.arange(tstart, tend, tstep)``.
    """
    if timeset == "manual":
        if tstart is None or tend is None:
            raise ValueError("manual timeset requires --tstart and --tend")
        return np.arange(tstart, tend, tstep, dtype=float), None

    window = _constraint_window(ods, policy=policy)
    start = window.start if tstart is None else max(window.start, tstart)
    end = window.end if tend is None else min(window.end, tend)
    if timeset == "auto":
        start = round(start / tstep) * tstep
        end = round(end / tstep) * tstep
    if end <= start:
        return np.array([start], dtype=float), window
    return np.arange(start, end + 0.5 * tstep, tstep, dtype=float), window


def _drop_vacuum_times(
    ods, times: np.ndarray, *, average_window: float
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Remove the instants EFIT would answer with a vacuum solution (#76).

    The constraint window comes from the plasma timing -- H-alpha and the Ip
    pulse -- which says when there *was* a discharge, not when there was
    enough current to reconstruct. Slices below EFIT's `CUTIP` were therefore
    still being submitted, and EFIT answered them the way it is supposed to:
    `ivacum = 1`, and an a-file whose `limloc` is `VAC`.

    That is not a failure, but it is not an equilibrium either, and counting
    it as one inflates every yield figure. Measured over the reference set,
    11 of 89 selected slices are below the cut -- and three of 39915's
    seventeen produced g-files were vacuum, so its plasma yield is 14 of 23
    rather than 17 of 26.

    The current is box-averaged with `box_average`, the same function and the
    same half-window the constraint builder uses, so a slice is judged on the
    number EFIT will actually be given as `PLASMA` rather than on an
    instantaneous value that can differ at the boundary.
    """
    from vaft.code.efit.legacy import box_average
    from vaft.machine_mapping.equilibrium_regime import vest_equilibrium_regime_policy

    threshold = float(vest_equilibrium_regime_policy().vacuum_current_amperes)
    if threshold <= 0.0:
        return times, []

    # The same resolver the window selector uses: a homogeneous-time product
    # stores only `magnetics.time`, and reading `ip.0.time` by name skipped the
    # cut exactly there (cold review 0.7.0 data F10, #1888).
    waveform = resolve_signal_waveform(ods, "magnetics.ip.0")
    if waveform is None:
        LOGGER.warning(
            "no magnetics.ip.0 to judge the vacuum cut by; every selected time is kept"
        )
        return times, []
    clock, data = waveform

    keep, dropped = [], []
    for value in times:
        try:
            current = box_average(clock, data, float(value), float(average_window), what="ip")
        except Exception:
            keep.append(float(value))
            continue
        if abs(float(current)) < threshold:
            dropped.append((float(value), float(current)))
        else:
            keep.append(float(value))

    if not keep:
        raise NoPlasmaCurrentError(
            f"every one of the {times.size} selected instants carries less than "
            f"{threshold} A of plasma current, so EFIT would return a vacuum solution "
            "for all of them. The plasma timing found a discharge the current does not "
            "support; check the window before reconstructing.",
            count=int(times.size), threshold=threshold,
        )
    return np.asarray(keep, dtype=float), dropped


def _efit_not_applicable(
    ods, error: NoPlasmaCurrentError, *, average_window: float = 0.0005
) -> str | None:
    """Why EFIT does not apply to this shot, or ``None`` when the cut is a fault (#205).

    The plasma current decides: a shot whose current stays below ``CUTIP``
    over the whole *diagnostics window* never carried a plasma EFIT could
    reconstruct, and that is the result.  The window is what the product
    holds: the mapper stores ``magnetics.ip.0`` on the plasma-analysis grid
    (0.26-0.36 s on the routine configuration) and crops the Rogowski record
    to it, so nothing here sees the rest of the shot.  When the current does
    reach ``CUTIP`` somewhere in the window but no selected instant does, the
    constraint window missed the discharge, and that stays an error.

    "Reaches ``CUTIP``" is judged on the current a constraint would see: box
    averaged over ``average_window`` like :func:`_drop_vacuum_times` does,
    and in the discharge direction -- the sign of the pulse the plasma timing
    accepted (``|Ip|`` when it accepted none).  Raw samples are not the
    measure: on the class shots 48886-48937 (#1731) single-sample ringing
    reads 41 kA where every millisecond averages below 9 kA, and a PF pickup
    swing of -12 to -22 kA ahead of a +5 kA pulse is current in the wrong
    direction, not a discharge the window missed -- when H-alpha is dark.
    Under a lit window an opposite-sign current at ``CUTIP`` stays a fault.

    A current that is low because the sensor is dead is not a vacuum shot
    either: the diagnostics stage condemns such a plasma-current Rogowski in
    ``magnetics.rogowski_coil.0.current.validity`` (its whole-record verdict,
    #1373; ``magnetics.ip.0`` itself carries no validity node), and that
    stays a visible failure too.

    The class from :func:`vaft.omas.shot_class.shot_class` -- what the
    new-shot worker records -- is quoted in the reason but its label does not
    decide: on vestserver 48927 (peak 3.3 kA, H-alpha dark) its pulse detector
    reads the pickup as a ``Plasma`` pulse.  The light does veto, when the
    current shows no pulse: H-alpha saw a window that a current never
    reaching ``CUTIP`` does not show, which may be a dead Ip channel under a
    real discharge, and that must stay a visible failure.  When the timing
    found a current pulse under the light, the channel is alive and the
    discharge is simply too small (48913/48924: 2.4-2.8 kA, lit).
    """
    waveform = resolve_signal_waveform(ods, "magnetics.ip.0")  # node time, else magnetics.time
    if waveform is None:
        return None
    clock, current = waveform
    if is_condemned_channel(ods, "magnetics.rogowski_coil.0.current", min_validity=VALIDITY_SUSPECT):
        return None
    finite = np.isfinite(current) & np.isfinite(clock)
    if not finite.any():
        return None
    current, clock = current[finite], clock[finite]
    from vaft.code.efit.legacy import box_average_series

    # The current each constraint instant would see there (#2162 B3.10).
    averaged = box_average_series(clock, current, float(average_window))

    verdict = None
    try:
        from vaft.omas.shot_class import shot_class

        verdict = shot_class(ods)
    except Exception as exc:  # the class is a note here, never the reason to fail
        label = f"shot_class unavailable ({type(exc).__name__}: {exc})"
    else:
        label = f"shot_class {verdict.label} ({verdict.reason}" + (
            f"; flags {', '.join(verdict.flags)})" if verdict.flags else ")"
        )
    pulse = verdict.timing.ip if verdict is not None else None
    pulse_found = pulse is not None and pulse.found
    direction = None
    if pulse_found and pulse.start is not None and pulse.end is not None:
        inside = (clock >= pulse.start) & (clock <= pulse.end)
        if inside.any():
            strongest = averaged[inside][np.argmax(np.abs(averaged[inside]))]
            direction = float(np.sign(strongest)) or None
    peak = float(np.max(direction * averaged)) if direction else float(np.max(np.abs(averaged)))
    if peak >= error.threshold:
        return None
    # The timing's pulse detector accepts only positive pulses, so the
    # direction above can be a small pickup pulse ahead of a large current of
    # the other sign.  That current is dismissed as pickup only in the dark
    # (48932-48937); under a lit window it may be the discharge, so fail.
    if direction and verdict is not None and verdict.optical_window:
        if float(np.max(-direction * averaged)) >= error.threshold:
            return None
    if verdict is not None and verdict.optical_window and not pulse_found:
        return None
    measure = "in the discharge direction" if direction else "either sign"
    return (
        f"peak |Ip| {max(peak, 0.0) / 1e3:.1f} kA over the diagnostics window "
        f"({measure}, box-averaged over \u00b1{average_window * 1e3:g} ms), below CUTIP {error.threshold:g} A "
        f"(all {error.count} constraint instants below it); {label}"
    )


#: Detector flags that say the current could not be judged, not that it showed
#: no discharge: a verdict of "no plasma" must not rest on them (#2195 review).
_UNJUDGED_CURRENT_FLAGS = frozenset({
    "ip_unusable", "baseline_inside_search", "record_too_short",
    "reference_too_short", "reference_not_finite",
})


def _current_flags(window: ConstraintWindow) -> set[str]:
    """Every timing flag plus the plasma-current detector's own window and onset flags.

    The timing's flags include the light's crop flags too, so a degenerate
    H-alpha baseline also withholds the verdict -- the safe side.
    """
    flags = set(window.flags)
    ip = (window.record or {}).get("ip") or {}
    flags.update(((ip.get("onset") or {}).get("flags")) or ())
    flags.update(ip.get("flags") or ())
    return flags


def _reverse_current_pulse(ods, window: ConstraintWindow) -> bool:
    """Whether the sign-flipped current shows a pulse the Ip detector accepts.

    The detector accepts positive pulses only (VEST stores Ip positive-going),
    so a discharge recorded with the wrong sign -- a sign fault, #275 -- shows
    no pulse at all.  Running the same detector, with the same rule and span,
    on ``-Ip`` tells that apart from pickup.  Any failure to judge counts as a
    pulse: the verdict is then withheld.
    """
    from omas import ODS

    from vaft.omas.plasma_timing import AnalysisSpan, ip_window

    try:
        clock, current = resolve_signal_waveform(ods, "magnetics.ip.0")
        span = AnalysisSpan(**window.record["span"])
        mirror = ODS(consistency_check=False)
        mirror["magnetics.ip.0.time"] = np.asarray(clock, dtype=float)
        mirror["magnetics.ip.0.data"] = -np.asarray(current, dtype=float)
        return bool(ip_window(mirror, span=span).found)
    except Exception as exc:  # cannot judge: withhold the verdict, but say why
        LOGGER.warning("sign-flipped current could not be judged (%s: %s); no no-plasma verdict",
                       type(exc).__name__, exc)
        return True


def _kept_currents(ods, kept_times, *, average_window: float) -> np.ndarray:
    """The box-averaged current each kept constraint instant would be given as ``PLASMA``."""
    from vaft.code.efit.legacy import box_average

    clock, data = resolve_signal_waveform(ods, "magnetics.ip.0")
    values = []
    for value in kept_times:
        try:
            values.append(float(box_average(clock, data, float(value), float(average_window), what="ip")))
        except Exception:
            values.append(float("nan"))
    return np.asarray(values, dtype=float)


def _no_plasma_window_reason(
    ods,
    window: ConstraintWindow | None,
    kept_times: np.ndarray,
    *,
    cut: int,
    average_window: float = 0.0005,
) -> str | None:
    """Why the equilibrium branch does not apply when the timing found no plasma (#2195).

    :func:`vaft.omas.plasma_timing.plasma_timing` returns ``found = False``
    when neither source *shows* a plasma.  The constraint window then falls
    back to the whole analysis range, and the vacuum cut used to keep
    whatever instants PF pickup pushed past ``CUTIP`` -- 39417, 45919 and
    46406 each reached EFIT with one to three such instants (#2166 D2).  Not
    showing one is not yet evidence of its absence, so the verdict is drawn
    only where the evidence is positive:

    * a usable H-alpha line answered for the light and its detector accepted
      no window (45919, 46406: no light above the noise beyond the excursions
      the onset rules reject), while the judged current shows no pulse either:
      two independent detectors agree that no discharge happened.  Their
      refusals are trusted as they stand -- a sub-millisecond flash under an
      impulsive current is refused by both, and a wrong-signed current of any
      size under that light is pickup, the dark-window reading
      :func:`_efit_not_applicable` already applies (#205); or
    * no light could be judged (39417 has no H-alpha line) and the current
      alone shows it: every kept instant is wrong-signed (VEST stores Ip
      positive-going) and the sign-flipped current shows no pulse either, so
      the instants are pickup excursions, not a discharge recorded with the
      wrong sign.  A positive kept instant, or a reverse pulse, keeps the old
      fallback -- that may be a real discharge the detector missed.

    Withheld, too, when the current could not be judged at all: the timing
    refused it, its detector could not form a reference
    (:data:`_UNJUDGED_CURRENT_FLAGS`), or the diagnostics stage condemned the
    plasma-current Rogowski (#1373) -- a dead sensor under a real discharge
    is a fault to see, the same guard :func:`_efit_not_applicable` applies.

    It is asked after the vacuum cut, about the ``kept_times`` the cut left
    (``cut`` were dropped): a fallback shot whose every instant is below
    ``CUTIP`` already got the #205 verdict and keeps its wording.  The verdict
    travels the #205 way (not-applicable constraints; the k-file, EFIT and
    EFIT-ODS steps report ``skipped: not applicable``); the diagnostics and
    eddy products, which the vacuum benchmark reads, are untouched.  Manual
    mode has no window and is never judged here.
    """
    if window is None or not window.fallback:
        return None
    if _current_flags(window) & _UNJUDGED_CURRENT_FLAGS:
        return None
    if is_condemned_channel(ods, "magnetics.rogowski_coil.0.current", min_validity=VALIDITY_SUSPECT):
        return None
    kept = int(np.asarray(kept_times).size)
    record = window.record or {}
    if record.get("optical") is not None:
        evidence = "a usable H-alpha line shows no discharge window and the plasma current shows no discharge pulse"
    else:
        currents = _kept_currents(ods, kept_times, average_window=average_window)
        if not currents.size or not np.all(currents < 0.0) or _reverse_current_pulse(ods, window):
            return None
        evidence = (
            "no H-alpha line could be judged; the plasma current shows no pulse in either "
            f"direction and every kept instant is wrong-signed (peak {currents.min() / 1e3:.1f} kA)"
        )
    return (
        f"no plasma window in the plasma-analysis range {window.start:.4f}-{window.end:.4f} s: "
        f"{evidence} ({window.fallback_reason}); the vacuum cut kept {kept} of {kept + cut} "
        "constraint instants"
    )


def _window_comment(
    window: ConstraintWindow | None,
    times: np.ndarray,
    vacuum: Sequence[tuple[float, float]] = (),
) -> str:
    """The one-line provenance written to ``equilibrium.ids_properties.comment``.

    The vacuum cut is named here because it changes the denominator of every
    yield figure computed from the product: without it a reader cannot tell
    whether a missing instant was dropped before EFIT or failed inside it.
    """
    span = f"EFIT constraint times {float(times[0]):.4f}-{float(times[-1]):.4f} s"
    if window is None:
        text = f"{span}: manual"
    else:
        text = f"{span}: plasma window from {window.source}"
        if window.agreement:
            text += f", agreement {window.agreement}"
        if window.fallback:
            text += f"; analysis-range fallback: {window.fallback_reason}"
    if vacuum:
        text += (
            f"; {len(vacuum)} instant(s) below CUTIP dropped before EFIT "
            f"({', '.join(f'{t:.4f}s' for t, _ in vacuum)})"
        )
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--eddy-ods", "--input", required=True, type=Path, help="Input eddy ODS (pf_passive).")
    parser.add_argument(
        "--diagnostics-ods",
        required=True,
        type=Path,
        help="The diagnostics ODS the eddy product was computed from.",
    )
    parser.add_argument(
        "--eddy-manifest",
        default=None,
        type=Path,
        help="The eddy stage manifest, for the input-provenance check.",
    )
    parser.add_argument(
        "--diagnostics-manifest",
        default=None,
        type=Path,
        help="The diagnostics stage manifest: a required input it records as absent "
        "from the raw data makes the result 'not applicable' instead of an error.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output constraints ODS JSON.")
    parser.add_argument("--efit-table-dir", default="", help="EFIT table/input directory written into kfiles.")
    parser.add_argument("--timeset", default="auto", choices=["auto", "manual"], help="EFIT constraint time selection mode.")
    parser.add_argument("--tstep", default=0.001, type=float, help="EFIT time step in seconds.")
    parser.add_argument(
        "--average-window",
        default=0.0005,
        type=float,
        help="Half-width in seconds of the box average each constraint is taken over (issue #433/#468).",
    )
    parser.add_argument("--tstart", default=None, type=float, help="Manual lower time bound.")
    parser.add_argument("--tend", default=None, type=float, help="Manual upper time bound.")
    parser.add_argument("--uncertainty", default=",".join(str(v) for v in DEFAULT_UNCERTAINTY))
    parser.add_argument("--weighting", default=",".join(str(v) for v in DEFAULT_WEIGHTING))
    parser.add_argument(
        "--detect-broken",
        default="false",
        help=(
            "Require the diagnostics-stage assessment: refuse a product whose magnetics carry no "
            "projected validity. With false, such a product runs with every channel usable (#424). "
            "The assessment itself is never run here (issue #296)."
        ),
    )
    parser.add_argument("--fl-correct-option", default=0, type=int, help="Reserved for future flux-loop correction.")
    parser.add_argument(
        "--gaussian-fit-option",
        default=1,
        type=int,
        help="Recovery backend: 0 none, 1 Gaussian family fit for rejected probes, 2 for every probe.",
    )
    # Accepted and ignored: the constraints product no longer records a
    # profile basis it does not choose; the k-file stage's configuration does (#869).
    parser.add_argument("--npprime", default=None, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--nffprime", default=None, type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # force=True: vaft.database.raw installs a root handler at import time, which makes
    # basicConfig() a no-op without this, silently dropping our INFO logs.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    if args.npprime is not None or args.nffprime is not None:
        LOGGER.warning(
            "--npprime/--nffprime are ignored: the constraints product no longer records "
            "a profile basis; the k-file stage's EFIT configuration chooses it (#869)"
        )
    try:
        ods, composition = compose_stage_products(
            diagnostics=args.diagnostics_ods,
            eddy=args.eddy_ods,
            eddy_manifest=args.eddy_manifest,
        )
    except EddyNoOutputError as error:
        # The eddy stage recorded that this shot has no passive currents to
        # compute (a PF circuit that was never recorded, #1568). That is a
        # result about the shot, carried on the way a vacuum shot's is (#205):
        # without it an unscoped run failed here and the worker retried the
        # shot to `gave_up` for a product that can never exist.
        reason = f"eddy produced no output: {error.reason}"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        vaft.omas.save(not_applicable_constraints(reason), args.output)
        LOGGER.info("EFIT not applicable to shot %s: %s", args.shot, reason)
        return 0
    LOGGER.info("composed inputs: %s", json.dumps(composition, default=str))
    # A shot whose raw dump has no TF waveform (42675, #1799) cannot be
    # reconstructed: BTOR has no other source. Recognised only from the
    # diagnostics stage's own record of the absence; a TF missing for any other
    # reason reaches build_constraints and fails there, by name.
    absent = (
        efit_constraint_inputs_absent_from_data(args.diagnostics_ods, args.diagnostics_manifest)
        if args.diagnostics_manifest is not None and "tf.b_field_tor_vacuum_r.data" not in ods
        else None
    )
    if absent is not None:
        reason = f"required input absent from the raw data: {absent}"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        vaft.omas.save(not_applicable_constraints(reason), args.output)
        LOGGER.info("EFIT not applicable to shot %s: %s", args.shot, reason)
        return 0
    times, window = _select_times(ods, args.timeset, args.tstep, args.tstart, args.tend)
    if times.size == 0:
        raise ValueError("No EFIT constraint times selected")
    try:
        times, vacuum = _drop_vacuum_times(ods, times, average_window=args.average_window)
    except NoPlasmaCurrentError as error:
        reason = _efit_not_applicable(ods, error, average_window=args.average_window)
        if reason is None:
            raise
        # A result, not a failure: the k-file and EFIT steps pass it on as
        # `skipped: not applicable`, the way they pass on efit.run=false.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        vaft.omas.save(not_applicable_constraints(reason), args.output)
        LOGGER.info("EFIT not applicable to shot %s: %s", args.shot, reason)
        return 0
    if vacuum:
        LOGGER.info(
            "vacuum cut: dropped %d of %d instants below CUTIP (%s)",
            len(vacuum),
            len(vacuum) + times.size,
            ", ".join(f"{t:.3f}s at {c / 1e3:.1f} kA" for t, c in vacuum),
        )
    no_plasma = _no_plasma_window_reason(
        ods, window, times, cut=len(vacuum), average_window=args.average_window
    )
    if no_plasma is not None:
        # A result, not a failure (#2195): the timing found no plasma, so the
        # instants pickup lifted past CUTIP are not equilibria to reconstruct.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        vaft.omas.save(not_applicable_constraints(no_plasma), args.output)
        LOGGER.info("plasma timing: %s", json.dumps(window.record, default=str))
        LOGGER.info("EFIT not applicable to shot %s: %s", args.shot, no_plasma)
        return 0
    ods["equilibrium.time"] = times
    ods["equilibrium.ids_properties.comment"] = _window_comment(window, times, vacuum)
    if window is not None:
        LOGGER.info("plasma timing: %s", json.dumps(window.record, default=str))
        if window.fallback:
            LOGGER.warning(
                "No plasma window found; EFIT constraint times cover the whole analysis range "
                "%.4f-%.4f s (%s)", window.start, window.end, window.fallback_reason,
            )
    fl_correct_coeff = None
    if args.fl_correct_option:
        if window is None or window.fallback:
            # The correction fits the pre-plasma stretch; without a detected
            # plasma there is no such stretch to fit, so it is skipped -- said
            # here rather than raised from inside the fit.
            LOGGER.warning("flux-loop correction skipped: no plasma window to fit before")
        else:
            fl_correct_coeff = correct_flux_loop(ods, window=(window.start, window.end))

    decisions = _decisions_for(ods, times, require_assessment=_bool(args.detect_broken))
    _log_decisions(decisions)
    from vaft.validation.efit_channels import efit_probe_count

    recovery = _recovery_for(args.gaussian_fit_option, ods, efit_probe_count(ods), shot=args.shot)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Generating constraints for shot %s at %d time slices", args.shot, len(times))
    build_constraints(
        ods,
        args.shot,
        str(args.output.parent),
        _table_dir(args.efit_table_dir),
        times,
        _csv_floats(args.uncertainty),
        _csv_floats(args.weighting),
        fl_correct_coeff=fl_correct_coeff,
        decisions=decisions,
        recovery=recovery,
        average_window=args.average_window,
        # Which era this shot belongs to is a VEST fact, resolved here and
        # handed in; the writer only compares it with what the table claims
        # for itself (#805).
        expected_table_era=machine_era_for_shot(args.shot).name,
    )

    produced = args.output.parent / f"{args.shot}_constraints.json"
    if produced != args.output and produced.exists():
        shutil.move(str(produced), str(args.output))
    if not args.output.exists():
        raise FileNotFoundError(f"Expected constraints output was not created: {args.output}")
    LOGGER.info("Constraints ODS saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
