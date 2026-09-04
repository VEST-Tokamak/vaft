"""Flux-loop evidence for EFIT channel selection (issue #295, step 1).

Source validity and model agreement are separate evidence, only the first
rejects on its own, every threshold is policy and is echoed into the record,
and the manual list's positions are printed beside the loops they name.  No
test here encodes the historical ``broken`` list as an expected outcome.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from vaft.validation.flux_loop_assessment import (
    NOT_AVAILABLE,
    REJECT_FOR_EFIT,
    SUSPECT,
    USABLE,
    FluxLoopPolicy,
    assess_flux_loops,
    flux_loop_evidence,
    manual_exclusion_index,
)
from vaft.validation.imas import VALIDITY_INVALID, VALIDITY_VALID, read_validity_record, write_validity
from vaft.validation.magnetics import ChannelQuality, QualityEvent
from vaft.validation.model import ValidationStatus

SCRIPT = Path(__file__).parents[1] / "workflow/efit_channel_selection/backtest_flux_loops.py"
SPEC = importlib.util.spec_from_file_location("backtest_flux_loops", SCRIPT)
BACKTEST = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BACKTEST)

TIME = np.linspace(0.26, 0.36, 101)
WINDOW = (0.28, 0.30)


def _quality(index: int, timed, *, name: str | None = None, status=None, events=()) -> ChannelQuality:
    codes = np.asarray(timed, dtype=int).reshape(-1)
    fraction = float((codes >= VALIDITY_VALID).mean()) if codes.size else float("nan")
    return ChannelQuality(
        kind="flux_loop",
        index=index,
        name=name or f"loop{index}",
        quantity="flux",
        unit="Wb",
        status=status or (ValidationStatus.FAIL if fraction < 1.0 else ValidationStatus.PASS),
        validity=int(codes.min()) if codes.size else VALIDITY_INVALID,
        validity_timed=codes,
        valid_fraction=fraction,
        events=tuple(QualityEvent(reason=r, start=0.0, end=0.0, samples=1, validity=VALIDITY_INVALID) for r in events),
    )


def _unavailable(index: int) -> ChannelQuality:
    return ChannelQuality(
        kind="flux_loop", index=index, name=f"loop{index}", quantity="flux", unit="Wb",
        status=ValidationStatus.NOT_AVAILABLE, validity=VALIDITY_INVALID,
        validity_timed=np.empty(0, dtype=int), valid_fraction=float("nan"),
        reason="no processed waveform on a resolvable time base",
    )


def _row(index: int, *, status="evaluated", improvement=0.9, normalized_residual=0.05, correlation=0.99, wall_authority=1.0, reason=""):
    return {
        "name": f"loop{index}", "kind": "flux_loop", "index": index, "status": status, "reason": reason,
        "improvement": improvement, "normalized_residual": normalized_residual,
        "correlation": correlation, "wall_authority": wall_authority,
    }


def _assess(qualities, rows, **policy):
    return assess_flux_loops(
        qualities, rows, window=WINDOW, nbprobe=64, time=TIME, field_codes=[18, 19, 20],
        policy=FluxLoopPolicy(**policy),
    )


# ---------------------------------------------------------------------------
# The state rule
# ---------------------------------------------------------------------------

def test_not_available_channels_are_not_rejected():
    (entry,) = _assess([_unavailable(0)], [_row(0)])
    assert entry.state == NOT_AVAILABLE
    assert entry.reasons == ("no processed waveform on a resolvable time base",)
    assert entry.source_validity["status"] == "not_available"


def test_a_loop_with_no_usable_sample_in_the_window_is_rejected_for_efit():
    """Good after the window, dead inside it: the window decides, and the
    record-wide fraction is reported beside it rather than instead of it."""
    timed = np.where((TIME >= WINDOW[0]) & (TIME <= WINDOW[1]), VALIDITY_INVALID, VALIDITY_VALID)
    (entry,) = _assess([_quality(0, timed, events=("saturated",))], [_row(0)])
    assert entry.state == REJECT_FOR_EFIT
    assert entry.source_validity["valid_fraction_in_window"] == 0.0
    assert entry.source_validity["valid_fraction"] > 0.5
    assert entry.reasons[0].startswith("usable fraction in the window 0 <= 0")
    assert "saturated" in entry.reasons[0]


def test_a_held_tail_outside_the_window_keeps_a_loop_usable():
    timed = np.where(TIME > 0.34, VALIDITY_INVALID, VALIDITY_VALID)
    (entry,) = _assess([_quality(0, timed, events=("held_tail",))], [_row(0)])
    assert entry.state == USABLE
    assert entry.source_validity["valid_fraction_in_window"] == 1.0
    assert entry.source_validity["events"] == ["held_tail"]


def test_model_disagreement_alone_is_report_only_by_default():
    (entry,) = _assess([_quality(0, np.zeros(TIME.size))], [_row(0, improvement=-0.2, normalized_residual=0.9)])
    assert entry.state == USABLE
    assert entry.reasons == ()
    assert entry.model_agreement["normalized_residual"] == 0.9


def test_a_set_model_threshold_marks_a_loop_suspect_not_rejected():
    (entry,) = _assess([_quality(0, np.zeros(TIME.size))], [_row(0, normalized_residual=0.9)], max_normalized_residual=0.5)
    assert entry.state == SUSPECT
    assert entry.reasons == ("normalized_residual 0.9 > 0.5",)


def test_model_disagreement_rejects_only_when_policy_says_so_and_the_wall_reaches_the_loop():
    strong = _row(0, normalized_residual=0.9, wall_authority=1.0)
    weak = _row(1, normalized_residual=0.9, wall_authority=0.02)
    qualities = [_quality(0, np.zeros(TIME.size)), _quality(1, np.zeros(TIME.size))]
    reached, unreached = _assess(qualities, [strong, weak], max_normalized_residual=0.5, reject_on_model_disagreement=True)
    assert reached.state == REJECT_FOR_EFIT
    assert unreached.state == USABLE
    assert unreached.reasons == ("wall_authority 0.02 below the scoring floor 0.1; model agreement not scored",)


def test_a_loop_the_vacuum_stage_did_not_compare_is_suspect_when_a_comparison_was_run():
    qualities = [_quality(0, np.zeros(TIME.size)), _quality(1, np.zeros(TIME.size))]
    with_rows = _assess(qualities, [_row(0), _row(1, status="excluded", reason="only 1 usable sample(s)")])
    assert [entry.state for entry in with_rows] == [USABLE, SUSPECT]
    assert with_rows[1].reasons == ("model comparison excluded: only 1 usable sample(s)",)
    missing = _assess(qualities, [_row(0)])
    assert missing[1].state == SUSPECT and "not selected by the vacuum stage" in missing[1].reasons[0]
    # No comparison at all is not a mark against anyone.
    assert [entry.state for entry in _assess(qualities, None)] == [USABLE, USABLE]


def test_reasons_policy_and_both_evidence_classes_are_recorded_on_every_row():
    policy = dict(max_normalized_residual=0.5, min_correlation=0.8)
    entries = _assess([_quality(0, np.zeros(TIME.size)), _unavailable(1)], [_row(0, correlation=0.3)], **policy)
    for entry in entries:
        assert entry.policy == FluxLoopPolicy(**policy).as_dict()
        assert entry.window == WINDOW
        assert set(entry.source_validity) >= {"status", "validity", "valid_fraction", "valid_fraction_in_window", "events"}
    assert entries[0].reasons == ("correlation 0.3 < 0.8",)
    assert entries[1].model_agreement is None
    json.dumps([entry.as_dict() for entry in entries], default=float)


def test_combined_index_and_field_code_are_reported_together():
    """The manual list is positional; a row must say which loop a position is."""
    entries = _assess([_quality(2, np.zeros(TIME.size), name="Flux Loop - #5")], None)
    (entry,) = entries
    assert entry.combined_index_one_based == manual_exclusion_index(2, 64) == 67
    assert entry.field_code == 20 and entry.name == "Flux Loop - #5" and entry.index == 2


# ---------------------------------------------------------------------------
# End to end on a synthetic machine
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_shot():
    from test_vacuum_benchmark import _machine, _synthesize

    return _synthesize(_machine())


def test_flux_loop_evidence_runs_on_a_synthetic_machine(synthetic_shot):
    ods = copy.deepcopy(synthetic_shot)
    evidence = flux_loop_evidence(ods, nbprobe=2)
    assert evidence["model"]["available"], evidence["model"]["reason"]
    assert evidence["window"] is not None
    states = {entry["index"]: entry["state"] for entry in evidence["assessments"]}
    assert states == {0: USABLE, 1: USABLE}
    assert all(entry["combined_index_one_based"] == entry["index"] + 3 for entry in evidence["assessments"])
    assert all(entry["model_agreement"]["status"] == "evaluated" for entry in evidence["assessments"])


def test_a_model_disagreement_never_changes_source_validity(synthetic_shot):
    """A loop whose measurement is three times what the wall model predicts
    is suspect under a threshold, and its validity nodes are byte-identical
    before and after (#253 §10)."""
    ods = copy.deepcopy(synthetic_shot)
    base = "magnetics.flux_loop.1.flux"
    ods[f"{base}.data"] = 3.0 * np.asarray(ods[f"{base}.data"])
    before = read_validity_record(ods, base)
    # The synthetic wall reaches its loops at ~8 % of the reading, so the
    # scoring floor is lowered to let the comparison count at all.
    policy = FluxLoopPolicy(max_normalized_residual=0.2, min_wall_authority_to_score=0.0)
    evidence = flux_loop_evidence(ods, nbprobe=2, policy=policy)
    after = read_validity_record(ods, base)
    assert before == after
    entry = {row["index"]: row for row in evidence["assessments"]}[1]
    assert entry["state"] == SUSPECT
    assert all(reason.startswith("normalized_residual") for reason in entry["reasons"])
    # The product carried no projection, so the intrinsic evidence is the fresh
    # assessment; whatever it says (a tripled loop is a population outlier to
    # it), the IDS nodes were not written and the state came from the model.
    assert entry["source_validity"]["validity_source"] == "assessed_here"
    assert after.assessed is False


def test_the_source_validity_the_diagnostics_stage_wrote_decides_rejection(synthetic_shot):
    ods = copy.deepcopy(synthetic_shot)
    time = np.asarray(ods["magnetics.time"], dtype=float)
    write_validity(ods, "magnetics.flux_loop.0.flux", np.full(time.size, VALIDITY_INVALID))
    evidence = flux_loop_evidence(ods, nbprobe=2, benchmark=False)
    entry = {row["index"]: row for row in evidence["assessments"]}[0]
    assert entry["state"] == REJECT_FOR_EFIT
    assert entry["source_validity"]["validity_source"] == "ids"
    assert entry["source_validity"]["valid_fraction_in_window"] == 0.0
    other = {row["index"]: row for row in evidence["assessments"]}[1]
    assert other["state"] == USABLE and other["source_validity"]["validity_source"] == "assessed_here"
    assert evidence["model"] == {"consulted": False, "available": False, "reason": None, "case": None}


# ---------------------------------------------------------------------------
# The packaged shot and the back-test script
# ---------------------------------------------------------------------------

def test_the_packaged_shot_yields_a_row_per_flux_loop():
    from vaft.omas.sample import sample_ods

    evidence = flux_loop_evidence(sample_ods())
    assert evidence["nbprobe"] == 64, "the routine list is positioned after EFIT's 64 B-probes"
    assert len(evidence["assessments"]) == 11
    assert evidence["model"]["available"], evidence["model"]["reason"]
    for entry in evidence["assessments"]:
        assert entry["state"] != NOT_AVAILABLE
        assert entry["model_agreement"] is not None
        assert entry["field_code"] is not None
        assert entry["combined_index_one_based"] == entry["index"] + 65
    names = {entry["combined_index_one_based"]: entry["name"] for entry in evidence["assessments"]}
    # The shipped list's positions 65..68 are the outboard loops #3..#6; its
    # "FL10" (74) is loop #14, and physical Flux Loop #10 sits at 70.
    assert names[65] == "Flux Loop - #3" and names[70] == "Flux Loop - #10" and names[74] == "Flux Loop - #14"


def test_agreement_vocabulary_covers_every_state():
    assert BACKTEST.agreement(REJECT_FOR_EFIT, True) == BACKTEST.AGREE_REJECT
    assert BACKTEST.agreement(REJECT_FOR_EFIT, False) == BACKTEST.FALSE_REJECTION
    assert BACKTEST.agreement(USABLE, True) == BACKTEST.UNREPRODUCED_EXCLUSION
    assert BACKTEST.agreement(SUSPECT, True) == BACKTEST.UNREPRODUCED_EXCLUSION
    assert BACKTEST.agreement(USABLE, False) == BACKTEST.AGREE_KEEP
    assert BACKTEST.agreement(NOT_AVAILABLE, True) == NOT_AVAILABLE


def test_the_manual_list_is_read_from_the_routine_config_not_hard_coded():
    manual = BACKTEST.routine_manual_broken()
    assert manual == sorted(manual) and all(65 <= item <= 75 for item in manual)
    assert "constraints" in BACKTEST.ROUTINE_CONFIG.read_text()


def test_the_backtest_script_writes_the_schema(tmp_path):
    output, markdown = tmp_path / "flux_loops.json", tmp_path / "flux_loops.md"
    assert BACKTEST.main(["--packaged-sample", "--output", str(output), "--markdown", str(markdown), "--manual-broken", "65,74"]) == 0
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 1
    assert payload["configuration"]["manual_broken"] == [65, 74]
    assert payload["configuration"]["manual_source"] == "argument"
    assert set(payload["summary"]) == set(BACKTEST.AGREEMENTS)
    mapping = payload["configuration"]["mapping"]
    assert [row["combined_index_one_based"] for row in mapping] == list(range(65, 76))
    assert all({"index", "field_code", "name", "manual_excluded"} <= set(row) for row in mapping)
    (case,) = payload["cases"]
    assert case["shot"] == 39915 and len(case["loops"]) == 11
    assert case["model"]["coil_drive"] is not None and case["model"]["plasma_free_evidence"] is not None
    assert sum(case["summary"].values()) == 11
    assert all(row["agreement"] in BACKTEST.AGREEMENTS for row in case["loops"])
    table = markdown.read_text()
    assert "| 39915 | 0 | 65 | Flux Loop - #3 | 18 | excluded |" in table
    assert "| 39915 | 5 | 70 | Flux Loop - #10 | 25 | kept |" in table
