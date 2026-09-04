#!/usr/bin/env python3
"""Generate EFIT constraints OMAS ODS from eddy-current diagnostics ODS."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import NamedTuple

import numpy as np
from omas import load_omas_json

from vaft.code.efit import correct_flux_loop, generate_constraints_ods as build_constraints
from vaft.machine_mapping.utils import PlasmaTimingPolicy, resolve_plasma_timing_policy
from vaft.omas.plasma_timing import PlasmaTimingError, plasma_timing
from vaft.validation.imas import resolve_signal_time


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


def _detect_broken_bpol_probes(ods) -> list[int]:
    """One-based indexes of Bpol probes the shared quality layer condemns outright.

    The fallback for products that carry no projected assessment (see
    :func:`_condemned_by_diagnostics_stage`).  It used to be a detector of its
    own -- a 99th-percentile amplitude against a 12-MAD band inside three
    hard-coded index banks -- which flagged the same probe as
    :mod:`vaft.validation.magnetics` by a different rule, so a product
    assessed before and after regeneration could disagree about one channel.
    It now runs the same detectors the diagnostics stage projects (physical
    ceiling, geometric-family population vote), on the fly, and returns the
    probes they reject for the whole record.
    """
    from vaft.validation.magnetics import validate_magnetics_signals
    from vaft.validation.model import ValidationStatus

    report = validate_magnetics_signals(ods, kinds=("b_field_pol_probe",))
    return sorted(
        quality.index + 1
        for quality in report
        if quality.status is not ValidationStatus.NOT_AVAILABLE
        and quality.valid_fraction == 0.0
    )


def _condemned_by_diagnostics_stage(ods) -> list[int] | None:
    """One-based legacy indexes the diagnostics stage condemned, or ``None``
    when the magnetics carry no assessment at all.

    Since #189/#343 the diagnostics stage assesses every channel and projects
    the verdict into the native validity nodes; ``vaft.code.efit.kfile`` folds
    those channels into the legacy ``broken`` list on its own, flux loops at
    ``index + nbprobe``. The list here comes from kfile's own rule so the log
    names exactly what the writer will exclude. When an assessment is present
    the amplitude detector below is redundant and must not vote -- two
    detectors disagreeing on one channel is worse than one. It stays only as a
    fallback for products that predate the assessment.
    """
    from vaft.code.efit.kfile import _condemned_channels
    from vaft.validation.imas import read_validity

    assessed = False
    for kind, quantity in (("b_field_pol_probe", "field"), ("flux_loop", "flux")):
        count = len(ods[f"magnetics.{kind}"]) if f"magnetics.{kind}" in ods else 0
        if any(
            read_validity(ods, f"magnetics.{kind}.{index}.{quantity}") is not None
            for index in range(count)
        ):
            assessed = True
            break
    if not assessed:
        return None
    nbprobe = len(ods["magnetics.b_field_pol_probe"]) if "magnetics.b_field_pol_probe" in ods else 0
    # kfile works zero-based and converts the script's one-based list itself.
    return sorted(index + 1 for index in _condemned_channels(ods, nbprobe))


def _resolve_broken(ods, explicit: list[int], *, detect: bool) -> list[int]:
    """The ``broken`` list handed to the constraint writer.

    Explicit indexes always apply. With ``detect``, projected validity wins
    when present (kfile folds it in regardless; it is listed here so the log
    says what will be excluded); otherwise the amplitude detector runs.
    """
    broken = set(explicit)
    if detect:
        condemned = _condemned_by_diagnostics_stage(ods)
        if condemned is not None:
            LOGGER.info(
                "magnetics carry a diagnostics-stage assessment; kfile folds its "
                "condemned channels %s (one-based, flux loops offset by the probe "
                "count) into broken -- amplitude detector not run",
                condemned,
            )
        else:
            detected = _detect_broken_bpol_probes(ods)
            LOGGER.warning(
                "magnetics carry no diagnostics-stage assessment (product predates "
                "#189); falling back to the 12-MAD amplitude detector: %s",
                detected,
            )
            broken |= set(detected)
    return sorted(broken)


ANALYSIS_RANGE_FALLBACK = "analysis_range_fallback"


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
    with the reason, so a vacuum shot's slices are visibly not plasma slices.
    """
    policy = policy or resolve_plasma_timing_policy()
    ip_time = resolve_signal_time(ods, "magnetics.ip.0")  # the node's own time, else magnetics.time
    if ip_time is None or ip_time.size == 0:
        raise ValueError("magnetics.ip.0.time is empty")
    base_start = max(float(policy.window.tstart), float(ip_time[0]))
    base_end = min(float(policy.window.tend), float(ip_time[-1]))

    try:
        timing = plasma_timing(ods, policy=policy)
    except PlasmaTimingError as exc:
        return ConstraintWindow(
            base_start, base_end, "analysis_range", (ANALYSIS_RANGE_FALLBACK,), None,
            str(exc), {"error": str(exc)},
        )
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


def _window_comment(window: ConstraintWindow | None, times: np.ndarray) -> str:
    """The one-line provenance written to ``equilibrium.ids_properties.comment``."""
    span = f"EFIT constraint times {float(times[0]):.4f}-{float(times[-1]):.4f} s"
    if window is None:
        return f"{span}: manual"
    text = f"{span}: plasma window from {window.source}"
    if window.agreement:
        text += f", agreement {window.agreement}"
    if window.fallback:
        text += f"; analysis-range fallback: {window.fallback_reason}"
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--eddy-ods", "--input", required=True, type=Path, help="Input eddy ODS JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Output constraints ODS JSON.")
    parser.add_argument("--efit-table-dir", default="", help="EFIT table/input directory written into kfiles.")
    parser.add_argument("--timeset", default="auto", choices=["auto", "manual"], help="EFIT constraint time selection mode.")
    parser.add_argument("--tstep", default=0.001, type=float, help="EFIT time step in seconds.")
    parser.add_argument("--tstart", default=None, type=float, help="Manual lower time bound.")
    parser.add_argument("--tend", default=None, type=float, help="Manual upper time bound.")
    parser.add_argument("--uncertainty", default=",".join(str(v) for v in DEFAULT_UNCERTAINTY))
    parser.add_argument("--weighting", default=",".join(str(v) for v in DEFAULT_WEIGHTING))
    parser.add_argument("--broken", default="", help="Comma-separated one-based broken diagnostic indices.")
    parser.add_argument(
        "--detect-broken",
        default="false",
        help=(
            "Exclude probes the diagnostics stage condemned (projected validity); "
            "for products without an assessment, fall back to the 12-MAD amplitude detector."
        ),
    )
    parser.add_argument("--fl-correct-option", default=0, type=int, help="Reserved for future flux-loop correction.")
    parser.add_argument("--gaussian-fit-option", default=1, type=int, help="Gaussian fit option forwarded to EFIT constraints.")
    parser.add_argument("--npprime", default=2, type=int, help="EFIT KPPCUR value.")
    parser.add_argument("--nffprime", default=2, type=int, help="EFIT KFFCUR value.")
    args = parser.parse_args()

    # force=True: vaft.database.raw installs a root handler at import time, which makes
    # basicConfig() a no-op without this, silently dropping our INFO logs.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    ods = load_omas_json(str(args.eddy_ods), consistency_check=False)
    broken = _resolve_broken(ods, _csv_ints(args.broken), detect=_bool(args.detect_broken))
    times, window = _select_times(ods, args.timeset, args.tstep, args.tstart, args.tend)
    if times.size == 0:
        raise ValueError("No EFIT constraint times selected")
    ods["equilibrium.time"] = times
    ods["equilibrium.ids_properties.comment"] = _window_comment(window, times)
    if window is not None:
        LOGGER.info("plasma timing: %s", json.dumps(window.record, default=str))
        if window.fallback:
            LOGGER.warning(
                "No plasma window found; EFIT constraint times cover the whole analysis range "
                "%.4f-%.4f s (%s)", window.start, window.end, window.fallback_reason,
            )
    fl_correct_coeff = correct_flux_loop(ods) if args.fl_correct_option else None

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
        broken=broken,
        fit=args.gaussian_fit_option,
        fl_correct_coeff=fl_correct_coeff,
        FFCUR=args.nffprime,
        PPCUR=args.npprime,
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
