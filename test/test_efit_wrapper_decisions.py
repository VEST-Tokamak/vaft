"""The routine constraint wrapper forms decisions; it detects nothing (issue #296)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from omas import ODS

from vaft.validation.channel_decision import REASON_MANUAL_LIST, REJECTED
from vaft.validation.imas import write_validity

SCRIPT = (
    Path(__file__).parents[1]
    / "workflow/automatic_pipeline_1_routine_data_processing/generate_constraints_ods.py"
)
SPEC = importlib.util.spec_from_file_location("generate_constraints_ods", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

TIME = np.linspace(0.26, 0.36, 2500)


def _array(n: int = 68, loops: int = 2) -> ODS:
    ods = ODS(consistency_check=False)
    ods["magnetics.time"] = TIME
    for index in range(n):
        ods[f"magnetics.b_field_pol_probe.{index}.field.data"] = 0.05 * np.sin(2 * np.pi * 20 * TIME)
        ods[f"magnetics.b_field_pol_probe.{index}.position.r"] = 0.05
        ods[f"magnetics.b_field_pol_probe.{index}.position.z"] = 0.1
    for index in range(loops):
        ods[f"magnetics.flux_loop.{index}.flux.data"] = 0.01 * np.cos(2 * np.pi * 20 * TIME)
        ods[f"magnetics.flux_loop.{index}.position.0.r"] = 0.6
        ods[f"magnetics.flux_loop.{index}.position.0.z"] = 0.3
    return ods


def _assess(ods, *, bad_probe=None, bad_loop=None):
    for index in range(len(ods["magnetics.b_field_pol_probe"])):
        code = -2 if index == bad_probe else 0
        write_validity(ods, f"magnetics.b_field_pol_probe.{index}.field", [code] * TIME.size, scalar=code)
    for index in range(len(ods["magnetics.flux_loop"])):
        code = -2 if index == bad_loop else 0
        write_validity(ods, f"magnetics.flux_loop.{index}.flux", [code] * TIME.size, scalar=code)


def test_the_wrapper_never_runs_a_detector(monkeypatch):
    import vaft.validation.magnetics as quality

    def boom(*args, **kwargs):
        raise AssertionError("the wrapper must not assess signals")

    monkeypatch.setattr(quality, "validate_magnetics_signals", boom)
    ods = _array()
    _assess(ods, bad_probe=9)
    decisions = MODULE._decisions_for(ods, [0.30, 0.31], manual=[3], require_assessment=True)
    assert decisions.get("b_field_pol_probe", 9).state.tolist() == [REJECTED, REJECTED]
    assert decisions.get("b_field_pol_probe", 2).reasons == (f"{REASON_MANUAL_LIST}:3",)


def test_an_unassessed_product_is_refused_when_required_and_warned_otherwise(caplog):
    ods = _array()
    with pytest.raises(ValueError, match="no diagnostics-stage assessment"):
        MODULE._decisions_for(ods, [0.30], manual=[], require_assessment=True)
    with caplog.at_level("WARNING", logger="vaft.generate_constraints_ods"):
        decisions = MODULE._decisions_for(ods, [0.30], manual=[], require_assessment=False)
    assert "every channel usable by default" in caplog.text
    assert all(decision.all_usable for decision in decisions.entries.values())


def test_flux_loop_combined_indexes_use_the_efit_probe_count():
    """68 probes in the IDS, 64 in EFIT's geometry: the manual list's 66 is
    flux loop 1, not probe 65 -- and the trailing probes get no decision."""
    ods = _array(n=68)
    _assess(ods, bad_loop=1)
    decisions = MODULE._decisions_for(ods, [0.30], manual=[66], require_assessment=True)
    assert decisions.indices("b_field_pol_probe") == tuple(range(64))
    loop = decisions.get("flux_loop", 1)
    assert loop.state.tolist() == [REJECTED]
    assert loop.reasons == ("condemned_whole_record", f"{REASON_MANUAL_LIST}:66")


def test_the_recovery_backend_follows_the_fit_option():
    ods = _array()
    assert MODULE._recovery_for(0, ods, 64) is None
    backend = MODULE._recovery_for(1, ods, 64)
    assert backend.keywords["mode"] == 1 and backend.keywords["uncertainty"] == "legacy"
    assert len(backend.keywords["families"].inboard) == 64
