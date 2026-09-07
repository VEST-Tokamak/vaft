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
from vaft.omas.plasma_timing import plasma_timing
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


def _recovery_for(option: int, ods, count: int):
    """The compatibility recovery backend selected by ``--gaussian-fit-option``."""
    if int(option) <= 0:
        return None
    from functools import partial

    from vaft.code.efit.recovery import gaussian_probe_recovery, probe_families

    return partial(
        gaussian_probe_recovery,
        mode=int(option),
        families=probe_families(ods["magnetics"], count=count),
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
    parser.add_argument("--npprime", default=2, type=int, help="EFIT KPPCUR value.")
    parser.add_argument("--nffprime", default=2, type=int, help="EFIT KFFCUR value.")
    args = parser.parse_args()

    # force=True: vaft.database.raw installs a root handler at import time, which makes
    # basicConfig() a no-op without this, silently dropping our INFO logs.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    ods = load_omas_json(str(args.eddy_ods), consistency_check=False)
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

    recovery = _recovery_for(args.gaussian_fit_option, ods, efit_probe_count(ods))

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
        FFCUR=args.nffprime,
        PPCUR=args.npprime,
        decisions=decisions,
        recovery=recovery,
        average_window=args.average_window,
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
