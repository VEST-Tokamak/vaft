"""Is this shot's magnetics fit to reconstruct from? (issues #189, #190, #295, #296)

Every EFIT study — cadence (#468), termination (#171), initialization (#196),
domain and grid (#459), profile-model uncertainty (#579) — reconstructs from
the same magnetics, and none of them says whether that input was sound. The
pieces to answer it all exist and none of them answers it alone: the quality
layer reports per-channel metrics without judging a shot, the flux-loop
back-test judges flux loops only, and the vacuum benchmark asks a
machine-model question. This script composes them into one per-shot verdict
over the window the routine pipeline would actually reconstruct.

    PYTHONPATH=$PWD python workflow/magnetics_quality/scan_magnetics_quality.py \\
        --packaged-samples --table test/data/magnetics_quality.json --markdown /tmp/mq.md
    PYTHONPATH=$PWD python workflow/magnetics_quality/scan_magnetics_quality.py \\
        --shots 39915,41524 --source main --table /tmp/mq.json

The layering of ``vaft.validation`` is respected and not crossed. Model
disagreement never changes the verdict: #190 is explicit that scientific
acceptance bounds wait for the VEST distribution, so the benchmark's residual
columns are reported beside the verdict as corroboration, never folded into
it. What decides the verdict is only what the channel decisions say EFIT will
see, which is the #296 contract read at the routine slice times.

The table is what the tests pin. Shots the source does not carry are recorded
as ``absent`` and shots whose assessment raises are recorded as ``error``:
which shots could not be judged is part of the answer, not a gap in it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from vaft.validation.channel_decision import MISSING, RECOVERED, REJECTED, STATE_NAMES, SUSPECT, USABLE
from vaft.validation.efit_channels import decide_efit_channels, efit_probe_count
from vaft.omas.vacuum_magnetics import quality_gate
from vaft.validation.flux_loop_assessment import flux_loop_evidence
from vaft.validation.imas import read_validity_record
from vaft.validation.magnetics import (
    channel_node,
    magnetics_quality_metrics,
    validate_magnetics_signals,
)

LOGGER = logging.getLogger("scan_magnetics_quality")

SCHEMA = 1
REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = REPOSITORY / "test" / "data" / "magnetics_quality.json"
WRAPPER = REPOSITORY / "workflow" / "automatic_pipeline_1_routine_data_processing" / "generate_constraints_ods.py"
PACKAGED_SHOTS = (39915, 41524, 41672)
#: Rows are checkpointed this often, so an aborted scan keeps what it did.
CHECKPOINT_EVERY = 10

VERDICT_FIT = "fit"
VERDICT_DEGRADED = "degraded"
VERDICT_UNFIT = "unfit"


@dataclass(frozen=True)
class FitnessPolicy:
    """What "fit to reconstruct from" means, in one place.

    These are *coverage* floors, not signal thresholds: the detectors and
    their thresholds live in ``MagneticsQualityConfig`` and are echoed into
    the report beside these. A family below its witness floor cannot
    constrain the boundary on its side of the machine however clean the
    remaining channels are, which is why the count and not the fraction is
    the test.

    ``min_family_witnesses`` is deliberately the *minimum over slices*: a
    family that loses its members part-way through the window is as unusable
    for those slices as one that never had them.
    """

    #: Usable-or-suspect channels each family must keep at every slice.
    min_family_witnesses: int = 3
    #: Families exempt from the floor because VEST has too few members to spare.
    small_families: tuple[str, ...] = ("inboard_flux_loop", "outboard_flux_loop", "flux_loop", "other")
    #: Usable-or-suspect fraction of the whole EFIT-facing set, at every slice.
    min_usable_fraction: float = 0.75
    #: A shot is degraded, not unfit, while the measured record still covers
    #: this fraction of the window; below it the slices are held values.
    min_window_coverage: float = 1.0
    #: Reject the shot outright when the record covers less than this.
    min_window_coverage_unfit: float = 0.9

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _wrapper():
    """The routine constraint wrapper, loaded by path (it is not a package)."""
    spec = importlib.util.spec_from_file_location("generate_constraints_ods_wrapper", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_shots(text: str) -> list[int]:
    """``"39915,41000-41010"`` -> the shots it names, in order, without repeats."""
    shots: list[int] = []
    for piece in str(text).replace(" ", "").split(","):
        if not piece:
            continue
        if "-" in piece[1:]:
            first, _, last = piece.partition("-")
            start, end = int(first), int(last)
            if end < start:
                raise ValueError(f"reversed shot range {piece!r}")
            shots.extend(range(start, end + 1))
        else:
            shots.append(int(piece))
    return list(dict.fromkeys(shots))


def load_shot(shot: int, *, source: str | None, packaged: bool):
    """The pre-EFIT product for ``shot``, or ``None`` when it is not carried.

    Packaged shots load the full ``pipeline-until-efit`` product rather than
    the compact sample: the compact one omits ``em_coupling``, and without it
    the vacuum-model layer cannot be consulted at all.
    """
    from omas import load_omas_json

    if packaged:
        from vaft.data.resources import data_path

        path = Path(data_path(f"samples/{shot}/source/pipeline-until-efit.json.gz"))
        if not path.is_file():
            return None
        import gzip
        import tempfile

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = handle.read()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            tmp.write(payload)
            staged = tmp.name
        try:
            return load_omas_json(staged, consistency_check=False)
        finally:
            Path(staged).unlink(missing_ok=True)

    import vaft.database as db

    # `open`, not `load`: the eager path downloads the whole domain through
    # hsget and is refused by the domain ACLs for ordinary readers, while the
    # lazy path serves the same shot.  A sweep only ever reads.
    return db.open(shot, source=source)


def refresh_coupling(ods, shot: int) -> str | None:
    """Re-map ``em_coupling`` from the canonical asset; returns what it did.

    The packaged pre-EFIT products predate the repair of #347/#373 and carry a
    ``mutual_passive_passive`` that is asymmetric by 1.27e-3, which the wall-
    mode basis refuses as ill-posed -- so the vacuum model cannot be consulted
    on them as they stand.  Re-mapping installs the repaired, exactly
    reciprocal asset, which is what the pipeline would carry today.  The
    magnetics are untouched: this decides only whether the *model* layer can
    speak, never whether a channel is usable.
    """
    from vaft.machine_mapping.em_coupling import em_coupling

    before = "em_coupling.mutual_passive_passive" in ods
    try:
        em_coupling(ods, shot=int(shot))
    except Exception as error:
        return f"could not re-map: {type(error).__name__}: {error}"[:160]
    return "re-mapped from the canonical asset" + ("" if before else " (product carried none)")


def _window_and_times(ods, *, tstep: float) -> tuple[np.ndarray, dict[str, Any]]:
    """The routine EFIT slice times and the window they were cut from."""
    wrapper = _wrapper()
    times, window = wrapper._select_times(ods, "auto", tstep, None, None)
    record = {
        "start": float(window.start) if window else float(times[0]),
        "end": float(window.end) if window else float(times[-1]),
        "source": str(window.source) if window else "manual",
        "flags": list(window.flags) if window else [],
        "agreement": (window.agreement if window else None),
        "tstep": float(tstep),
        "slices": int(times.size),
    }
    return np.asarray(times, dtype=float), record


def _measured_span(quality_channels: Iterable[dict[str, Any]], window: dict[str, Any]) -> dict[str, Any]:
    """When the magnetics actually stop measuring, against the EFIT window.

    Shot 39915 is the case this exists for: its magnetics are processed over a
    range that ends before the diagnostics grid does and are then interpolated
    onto it, so every channel holds its last value from 0.34 s. A held value
    is not a measurement, and a slice taken there is not constrained by one.
    """
    last = [
        float(entry["metrics"]["last_valid_time"])
        for entry in quality_channels
        if entry["metrics"].get("last_valid_time") is not None
        and np.isfinite(float(entry["metrics"].get("last_valid_time", np.nan)))
    ]
    first = [
        float(entry["metrics"]["first_valid_time"])
        for entry in quality_channels
        if entry["metrics"].get("first_valid_time") is not None
        and np.isfinite(float(entry["metrics"].get("first_valid_time", np.nan)))
    ]
    if not last or not first:
        return {"first": None, "last": None, "covers_window": None, "coverage_fraction": None}
    start, end = window["start"], window["end"]
    span = max(end - start, 0.0)
    measured_end = float(np.median(last))
    measured_start = float(np.median(first))
    covered = max(min(measured_end, end) - max(measured_start, start), 0.0)
    # The median says what the record as a whole does; the count says whether
    # any single channel stops early, which a median of 87 cannot show.
    whole = sum(1 for a, b in zip(first, last) if a <= start and b >= end)
    return {
        "first": measured_start,
        "last": measured_end,
        "earliest_last": float(min(last)),
        "covers_window": bool(measured_start <= start and measured_end >= end),
        "coverage_fraction": float(covered / span) if span > 0 else 1.0,
        "channels_covering_window": whole,
        "channels_with_a_span": len(last),
    }


def _condemned(quality_channels: Iterable[dict[str, Any]]) -> list[str]:
    """Channels with no usable sample anywhere in the record.

    Never ``validity <= -2``: the scalar aggregates to the worst state a
    channel ever reached, so a channel that measures cleanly through the
    window and holds its last value afterwards reads condemned there.  What
    condemns a channel is having nothing usable at all -- the ``is_condemned``
    rule of :mod:`vaft.validation.validity`, applied to the assessment rather
    than to a projection the product may not carry.
    """
    out = []
    for entry in quality_channels:
        if entry["status"] == "not_available":
            continue
        fraction = entry.get("valid_fraction")
        if fraction is None or not np.isfinite(float(fraction)):
            continue
        if float(fraction) <= 0.0:
            out.append(f"{entry['kind']}[{entry['index']}] {entry['name']}")
    return out


def _projection_coverage(ods, quality_channels: Iterable[dict[str, Any]], nbprobe: int) -> dict[str, int]:
    """How many channels the product already carried a projected validity for.

    A count and not a flag: "some channels carry a record" is the case that
    misleads, because the decision layer reads the projection per channel and
    treats an absent one as unassessed-and-therefore-usable.
    """
    total = 0
    with_record = 0
    facing = 0
    facing_with_record = 0
    for entry in quality_channels:
        total += 1
        is_facing = entry["kind"] == "flux_loop" or int(entry["index"]) < int(nbprobe)
        facing += 1 if is_facing else 0
        record = read_validity_record(ods, channel_node(entry["kind"], int(entry["index"]), entry["quantity"]))
        if record.scalar is not None or record.timed is not None:
            with_record += 1
            facing_with_record += 1 if is_facing else 0
    return {
        "channels": total,
        "with_projected_validity": with_record,
        "efit_facing": facing,
        "efit_facing_with_projected_validity": facing_with_record,
    }


def _family_of(kind: str, index: int, quality_channels: list[dict[str, Any]]) -> str:
    for entry in quality_channels:
        if entry["kind"] == kind and int(entry["index"]) == int(index):
            return str(entry["family"])
    return "other"


def _decision_summary(decisions, quality_channels: list[dict[str, Any]], policy: FitnessPolicy) -> dict[str, Any]:
    """What EFIT sees at every slice, per family and in total.

    The counts are over the EFIT-facing set only, and ``usable`` here means
    usable *or* suspect: a suspect channel is still fitted, at a reduced
    weight, which is a different statement from a rejected one.
    """
    n_slices = int(np.asarray(decisions.times).size)
    per_family: dict[str, np.ndarray] = {}
    totals = np.zeros(n_slices, dtype=int)
    expected = 0
    states = {name: 0 for name in STATE_NAMES}
    for (kind, index), decision in decisions.entries.items():
        family = _family_of(kind, index, quality_channels)
        counted = (decision.state == USABLE) | (decision.state == SUSPECT) | (decision.state == RECOVERED)
        per_family.setdefault(family, np.zeros(n_slices, dtype=int))
        per_family[family] += counted.astype(int)
        totals += counted.astype(int)
        expected += 1
        for code in decision.state.tolist():
            states[STATE_NAMES[int(code)]] += 1
    families = {
        family: {
            "min_witnesses": int(counts.min()),
            "max_witnesses": int(counts.max()),
            "exempt": family in policy.small_families,
        }
        for family, counts in sorted(per_family.items())
    }
    fraction = totals / expected if expected else np.zeros(n_slices)
    return {
        "channels": expected,
        "slices": n_slices,
        "state_slice_counts": states,
        "families": families,
        "min_usable_fraction": float(fraction.min()) if expected else 0.0,
        "usable_at_every_slice": int(
            sum(
                1
                for decision in decisions.entries.values()
                if np.all((decision.state == USABLE) | (decision.state == RECOVERED))
            )
        ),
        "rejected_channels": sorted(
            f"{kind}[{index}]"
            for (kind, index), decision in decisions.entries.items()
            if np.any(decision.state == REJECTED)
        ),
        "missing_channels": sorted(
            f"{kind}[{index}]"
            for (kind, index), decision in decisions.entries.items()
            if np.any(decision.state == MISSING)
        ),
    }


def _verdict(decisions_summary: dict[str, Any], span: dict[str, Any], policy: FitnessPolicy) -> dict[str, Any]:
    """`fit` / `degraded` / `unfit`, and the reasons, from coverage alone."""
    reasons: list[str] = []
    verdict = VERDICT_FIT

    for family, record in decisions_summary["families"].items():
        if record["exempt"]:
            continue
        if record["min_witnesses"] < policy.min_family_witnesses:
            verdict = VERDICT_UNFIT
            reasons.append(
                f"family {family} keeps only {record['min_witnesses']} witnesses at some slice, "
                f"below the floor of {policy.min_family_witnesses}"
            )

    if decisions_summary["min_usable_fraction"] < policy.min_usable_fraction:
        verdict = VERDICT_UNFIT
        reasons.append(
            f"only {decisions_summary['min_usable_fraction']:.2f} of the EFIT-facing set is usable at "
            f"some slice, below {policy.min_usable_fraction}"
        )

    coverage = span.get("coverage_fraction")
    if coverage is None:
        if verdict == VERDICT_FIT:
            verdict = VERDICT_DEGRADED
        reasons.append("the measured span could not be determined")
    elif coverage < policy.min_window_coverage_unfit:
        verdict = VERDICT_UNFIT
        reasons.append(
            f"the measured record covers {coverage:.2f} of the window, below {policy.min_window_coverage_unfit}"
        )
    elif coverage < policy.min_window_coverage:
        if verdict == VERDICT_FIT:
            verdict = VERDICT_DEGRADED
        reasons.append(
            f"the measured record ends inside the window (covers {coverage:.2f}); "
            f"slices past {span['last']:.4f} s are held values, not measurements"
        )

    if decisions_summary["rejected_channels"]:
        if verdict == VERDICT_FIT:
            verdict = VERDICT_DEGRADED
        reasons.append(
            f"{len(decisions_summary['rejected_channels'])} channel(s) rejected at some slice: "
            + ", ".join(decisions_summary["rejected_channels"])
        )
    if decisions_summary["missing_channels"]:
        if verdict == VERDICT_FIT:
            verdict = VERDICT_DEGRADED
        reasons.append("missing: " + ", ".join(decisions_summary["missing_channels"]))

    return {"verdict": verdict, "reasons": reasons}


def _model_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    """The vacuum-model layer, reported beside the verdict and never inside it."""
    model = dict(evidence.get("model") or {})
    assessments = evidence.get("assessments") or []
    states: dict[str, int] = {}
    residuals: list[float] = []
    authority: list[float] = []
    for entry in assessments:
        states[str(entry.get("state"))] = states.get(str(entry.get("state")), 0) + 1
        agreement = entry.get("model_agreement") or {}
        for key, sink in (("normalized_residual", residuals), ("wall_authority", authority)):
            value = agreement.get(key)
            if value is not None and np.isfinite(float(value)):
                sink.append(float(value))
    return {
        "consulted": bool(model.get("consulted")),
        "available": bool(model.get("available")),
        "reason": model.get("reason"),
        "window_source": model.get("window_source"),
        "loop_states": states,
        "normalized_residual": {
            "median": float(np.median(residuals)) if residuals else None,
            "max": float(max(residuals)) if residuals else None,
            "n": len(residuals),
        },
        "wall_authority": {
            "median": float(np.median(authority)) if authority else None,
            "min": float(min(authority)) if authority else None,
            "n": len(authority),
        },
    }


def scan_shot(shot: int, *, source: str | None, packaged: bool, policy: FitnessPolicy, tstep: float) -> dict[str, Any]:
    """One row: the verdict for one shot, with the evidence behind it.

    Deliberately carries no timing: the table is committed, and a duration
    would make every regeneration a diff without telling a reader anything
    about the shot.
    """
    row: dict[str, Any] = {"shot": int(shot)}
    try:
        ods = load_shot(shot, source=source, packaged=packaged)
    except Exception as error:  # a source that cannot answer is a row, not a crash
        row.update(status="absent", reason=f"{type(error).__name__}: {error}"[:200])
        return row
    if ods is None or "magnetics" not in ods:
        row.update(status="absent", reason="the source carries no magnetics for this shot")
        return row

    try:
        times, window = _window_and_times(ods, tstep=tstep)
        report = validate_magnetics_signals(ods)
        metrics = magnetics_quality_metrics(ods, report)
        nbprobe = efit_probe_count(ods)

        # A product carries no marker saying whether it was ever assessed:
        # "all valid" and "never looked at" are the same bytes.  Say which
        # this one is, because the decision layer reads *projected* validity
        # and an unassessed product therefore yields "everything usable" --
        # including channels the assessment condemns outright.
        projected = _projection_coverage(ods, metrics["channels"], nbprobe)

        # Decide on a gated copy, so the verdict describes the shot's signals
        # rather than whether someone remembered to project them.  The source
        # is never written; this is the one direction #253 permits.
        gated, gate = quality_gate(ods, window=(window["start"], window["end"]), report=report)
        decisions = decide_efit_channels(gated, times, nbprobe=nbprobe)
        summary = _decision_summary(decisions, metrics["channels"], policy)
        summary["gate"] = gate.record()
        summary["projection_in_product"] = projected
        span = _measured_span(metrics["channels"], window)
        coupling = refresh_coupling(ods, shot)
        try:
            evidence = flux_loop_evidence(ods, nbprobe=nbprobe)
            model = _model_summary(evidence)
        except Exception as error:
            model = {"consulted": False, "available": False, "reason": f"{type(error).__name__}: {error}"[:160]}
        model["coupling"] = coupling
        verdict = _verdict(summary, span, policy)
    except Exception as error:
        row.update(status="error", reason=f"{type(error).__name__}: {error}"[:300])
        return row

    condemned = sorted(_condemned(metrics["channels"]))
    row.update(
        status="assessed",
        window=window,
        nbprobe=int(nbprobe),
        coverage=metrics["summary"],
        families=metrics["families"],
        condemned=condemned,
        events=metrics["summary"].get("events", {}),
        measured_span=span,
        decisions=summary,
        model=model,
        **verdict,
    )
    return row


def summarize(rows: list[dict[str, Any]], policy: FitnessPolicy) -> dict[str, Any]:
    assessed = [row for row in rows if row.get("status") == "assessed"]
    verdicts: dict[str, int] = {}
    for row in assessed:
        verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1
    condemned: dict[str, int] = {}
    for row in assessed:
        for name in row.get("condemned", []):
            condemned[name] = condemned.get(name, 0) + 1
    return {
        "shots": len(rows),
        "assessed": len(assessed),
        "absent": sum(1 for row in rows if row.get("status") == "absent"),
        "errors": [row["shot"] for row in rows if row.get("status") == "error"],
        "verdicts": dict(sorted(verdicts.items())),
        "condemned_by_shot_count": dict(sorted(condemned.items(), key=lambda item: (-item[1], item[0]))),
        "policy": policy.as_dict(),
    }


def _finite(value: Any) -> Any:
    """Replace non-finite numbers with ``None``, recursively.

    The table is written with ``allow_nan=False`` so it stays valid JSON for
    any reader.  Without this, a single NaN metric on one shot would raise at
    write time and take the whole scan's rows with it.
    """
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_table(path: Path, *, shots_requested: str, policy: FitnessPolicy, rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA,
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shots_requested": shots_requested,
        "summary": _finite(summarize(rows, policy)),
        "rows": _finite(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def markdown(payload: dict[str, Any]) -> str:
    lines = ["# Magnetics quality: is this shot fit to reconstruct from?", ""]
    summary = payload["summary"]
    lines.append(
        f"{summary['assessed']} of {summary['shots']} shots assessed"
        + (f", {summary['absent']} absent" if summary["absent"] else "")
        + (f", {len(summary['errors'])} errors" if summary["errors"] else "")
        + ". Verdicts: "
        + ", ".join(f"{count} {name}" for name, count in summary["verdicts"].items())
        + "."
    )
    lines.append("")
    lines.append("| shot | verdict | window [s] | slices | EFIT channels | usable frac (min) | measured to [s] | covers window | condemned | model |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in payload["rows"]:
        if row.get("status") != "assessed":
            lines.append(f"| {row['shot']} | {row.get('status')} | – | – | – | – | – | – | – | {row.get('reason', '')[:40]} |")
            continue
        window, decisions, span = row["window"], row["decisions"], row["measured_span"]
        model = row["model"]
        residual = model.get("normalized_residual", {}).get("median")
        lines.append(
            f"| {row['shot']} | **{row['verdict']}** | {window['start']:.3f}–{window['end']:.3f} "
            f"| {window['slices']} | {decisions['channels']} | {decisions['min_usable_fraction']:.2f} "
            f"| {'–' if span['last'] is None else f'{span["last"]:.4f}'} "
            f"| {span['covers_window']} | {len(row['condemned'])} "
            f"| {'–' if residual is None else f'{residual:.3f}'} |"
        )
    lines.append("")
    for row in payload["rows"]:
        if row.get("status") != "assessed":
            continue
        lines.append(f"## {row['shot']} — {row['verdict']}")
        lines.append("")
        for reason in row["reasons"] or ["no finding"]:
            lines.append(f"- {reason}")
        if row["condemned"]:
            lines.append(f"- condemned (no usable sample anywhere): {', '.join(row['condemned'])}")
        projection = row["decisions"].get("projection_in_product") or {}
        if projection:
            lines.append(
                f"- the product carried a projected validity for "
                f"{projection['efit_facing_with_projected_validity']} of {projection['efit_facing']} "
                f"EFIT-facing channels ({projection['with_projected_validity']} of "
                f"{projection['channels']} overall); this assessment was run and gated into a copy"
            )
        families = ", ".join(
            f"{name} {record['min_witnesses']}" + ("*" if record["exempt"] else "")
            for name, record in row["decisions"]["families"].items()
        )
        lines.append(f"- witnesses per family, minimum over slices (* exempt): {families}")
        events = row.get("events") or {}
        if events:
            lines.append("- events: " + ", ".join(f"{name} {count}" for name, count in sorted(events.items())))
        model = row["model"]
        if model.get("consulted"):
            residual = model["normalized_residual"]
            authority = model.get("wall_authority", {})
            lines.append(
                "- vacuum model: "
                + ", ".join(f"{state} {count}" for state, count in sorted(model["loop_states"].items()))
                + f"; normalized residual median {residual['median']:.3f}, max {residual['max']:.3f}"
                + (f"; wall authority median {authority['median']:.3f}" if authority.get("median") is not None else "")
            )
        else:
            lines.append(f"- vacuum model: not consulted ({model.get('reason')})")
        lines.append("")
    lines.append("Policy: " + json.dumps(payload["summary"]["policy"], sort_keys=True))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shots", help="range 'A-B' and/or comma-separated shots")
    group.add_argument("--packaged-samples", action="store_true", help=f"the packaged shots {PACKAGED_SHOTS}")
    parser.add_argument("--source", default=None, help="database source for --shots (default: the configured one)")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--tstep", type=float, default=0.001, help="EFIT slice spacing used to read the decisions")
    parser.add_argument("--min-family-witnesses", type=int, default=FitnessPolicy.min_family_witnesses)
    parser.add_argument("--min-usable-fraction", type=float, default=FitnessPolicy.min_usable_fraction)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    policy = FitnessPolicy(
        min_family_witnesses=args.min_family_witnesses,
        min_usable_fraction=args.min_usable_fraction,
    )
    packaged = bool(args.packaged_samples)
    shots = list(PACKAGED_SHOTS) if packaged else parse_shots(args.shots)
    requested = "packaged" if packaged else args.shots

    rows: list[dict[str, Any]] = []
    for position, shot in enumerate(shots, start=1):
        LOGGER.info("[%d/%d] %s", position, len(shots), shot)
        rows.append(scan_shot(shot, source=args.source, packaged=packaged, policy=policy, tstep=args.tstep))
        print(f"{shot}: {rows[-1].get('verdict', rows[-1].get('status'))}", flush=True)
        if position % CHECKPOINT_EVERY == 0:
            write_table(args.table, shots_requested=requested, policy=policy, rows=rows)

    payload = write_table(args.table, shots_requested=requested, policy=policy, rows=rows)
    text = markdown(payload)
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    else:
        print(text)
    print(f"table: {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
