"""The constraint builder consumes decisions and only translates them (issue #296)."""

from __future__ import annotations

import ast
import json
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pytest

import vaft
import vaft.code.efit.kfile as kfile
from vaft.code.efit.recovery import gaussian_probe_recovery, probe_families
from vaft.validation.channel_decision import (
    REASON_CONDEMNED,
    RECOVERED,
    REJECTED,
    ChannelDecisions,
    rejected_decision,
    usable_decision,
)
from vaft.validation.efit_channels import decide_efit_channels, efit_probe_count

UNCERTAINTY = [1e-4, 1e-4, 5e-2, 3e-2, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2]
WEIGHTING = [1, 1, 1, 0.1, 0.1, 0.1, 0.01, 0.01]
MANUAL = [65, 66, 67, 68, 72, 74]


def _build(monkeypatch, ods, times, **kwargs):
    """Run the builder in memory; the compact sample stops it in the namelist
    block (no pf_passive.time), after every constraint leaf is written."""
    monkeypatch.setattr(kfile, "save_omas_json", lambda *a, **k: None)
    ods["equilibrium.time"] = times
    try:
        kfile.generate_constraints_ods(ods, 39915, "/nonexistent", "", times, UNCERTAINTY, WEIGHTING, **kwargs)
    except ValueError as error:
        if "sample points is empty" not in str(error):
            raise
    return ods["equilibrium"]


def _leaves(EQ, n_slices):
    out = {}
    for i in range(n_slices):
        for family, n in (("bpol_probe", 64), ("flux_loop", 11)):
            for j in range(n):
                for leaf in ("measured", "weight", "measured_error_upper"):
                    path = f"time_slice.{i}.constraints.{family}.{j}.{leaf}"
                    if path in EQ:
                        out[path] = float(EQ[path])
    return out


@pytest.fixture(scope="module")
def times():
    return np.asarray(vaft.omas.sample_ods()["equilibrium.time"], dtype=float)


def test_legacy_args_and_explicit_decisions_agree_on_the_packaged_shot(monkeypatch, times):
    with pytest.warns(DeprecationWarning, match="broken=/fit="):
        legacy = _leaves(_build(monkeypatch, vaft.omas.sample_ods(), times, broken=MANUAL, fit=1), times.size)

    ods = vaft.omas.sample_ods()
    nbprobe = efit_probe_count(ods)
    decisions = decide_efit_channels(ods, times, nbprobe=nbprobe, manual_rejections=MANUAL)
    families = probe_families(ods["magnetics"], count=nbprobe)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        explicit = _leaves(_build(
            monkeypatch, ods, times,
            decisions=decisions,
            recovery=partial(gaussian_probe_recovery, mode=1, families=families, uncertainty="legacy"),
        ), times.size)
    assert legacy.keys() == explicit.keys() and len(legacy) == 2025
    assert all(legacy[key] == explicit[key] for key in legacy)

    # Against the stored product every leaf agrees except the condemned
    # probe's weight at the fitted slices: the sample predates the per-slice
    # validity exclusion and still carries the nominal weight there.
    stored = _leaves(vaft.omas.sample_ods()["equilibrium"], times.size)
    differing = sorted(key for key in stored if stored[key] != explicit.get(key))
    assert {key.split(".", 2)[2] for key in differing} == {"constraints.bpol_probe.25.weight"}
    assert all(stored[key] == 0.1 and explicit[key] == 0.0 for key in differing)


def test_the_product_records_the_decisions_it_was_built_from(monkeypatch, times):
    ods = vaft.omas.sample_ods()
    decisions = decide_efit_channels(ods, times, manual_rejections=MANUAL)
    EQ = _build(monkeypatch, ods, times, decisions=decisions)
    record = json.loads(EQ["code.parameters.channel_decisions"])
    assert record["schema_version"] == 1 and record["times"] == times.tolist()
    notable = {(entry["kind"], entry["index"]): entry for entry in record["entries"]}
    assert notable[("b_field_pol_probe", 25)]["state"] == ["rejected"] * times.size
    assert notable[("b_field_pol_probe", 25)]["reasons"] == [REASON_CONDEMNED]
    assert notable[("flux_loop", 9)]["reasons"] == ["manual_exclusion_list:74"]
    assert ("flux_loop", 4) not in notable  # all usable: not listed


def test_the_145_placeholder_is_untouched_and_listed_as_missing(monkeypatch, times):
    ods = vaft.omas.sample_ods()
    del ods["magnetics.flux_loop.4.flux.data"]
    decisions = decide_efit_channels(ods, times, manual_rejections=MANUAL)
    assert decisions.get("flux_loop", 4).state.tolist() == [3] * times.size
    EQ = _build(monkeypatch, ods, times, decisions=decisions)
    for i in range(times.size):
        base = f"time_slice.{i}.constraints.flux_loop.4"
        assert float(EQ[f"{base}.measured"]) == 0.0 and float(EQ[f"{base}.weight"]) == 0.0
        assert EQ[f"{base}.source"] == ods["magnetics.flux_loop.4.identifier"]
    record = json.loads(EQ["code.parameters.channel_decisions"])
    assert next(e for e in record["entries"] if e["index"] == 4 and e["kind"] == "flux_loop")["state"][0] == "missing"


def test_rejected_at_one_slice_and_usable_at_another_in_the_product(monkeypatch, times):
    ods = vaft.omas.sample_ods()
    decisions = decide_efit_channels(ods, times)
    state = np.zeros(times.size, dtype=np.int8)
    state[3] = REJECTED
    factor = np.ones(times.size)
    factor[3] = 0.0
    partial_reject = decisions.get("flux_loop", 5)
    from vaft.validation.channel_decision import ChannelDecision

    decisions = decisions.replaced(ChannelDecision("flux_loop", 5, partial_reject.name, state, factor, reasons=("projected_validity",)))
    EQ = _build(monkeypatch, ods, times, decisions=decisions)
    weights = [float(EQ[f"time_slice.{i}.constraints.flux_loop.5.weight"]) for i in range(times.size)]
    assert weights[3] == 0.0 and all(w == 0.01 for i, w in enumerate(weights) if i != 3)


def test_a_recovered_value_lands_in_measured_with_its_own_error(monkeypatch, times):
    ods = vaft.omas.sample_ods()
    nbprobe = efit_probe_count(ods)
    decisions = decide_efit_channels(ods, times, nbprobe=nbprobe, manual_rejections=[3])  # probe 2 by the list
    families = probe_families(ods["magnetics"], count=nbprobe)
    EQ = _build(
        monkeypatch, ods, times, decisions=decisions,
        recovery=partial(gaussian_probe_recovery, mode=1, families=families, uncertainty="residual_rms"),
    )
    record = json.loads(EQ["code.parameters.channel_decisions"])
    entry = next(e for e in record["entries"] if e["kind"] == "b_field_pol_probe" and e["index"] == 2)
    assert entry["provenance"] == "gaussian_fit4"
    fitted = [i for i, s in enumerate(entry["state"]) if s == "recovered"]
    assert fitted, "the packaged shot has slices above the Ip floor"
    i = fitted[0]
    base = f"time_slice.{i}.constraints.bpol_probe.2"
    assert float(EQ[f"{base}.measured"]) == pytest.approx(entry["value"][i])
    assert float(EQ[f"{base}.measured_error_upper"]) == pytest.approx(entry["uncertainty"][i])
    assert float(EQ[f"{base}.weight"]) == 0.1  # a manual-list rejection is undone by recovery


def test_the_builder_reads_no_validity_and_runs_no_detector():
    tree = ast.parse(Path(kfile.__file__).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_constraints_ods")
    names = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(function) if isinstance(n, ast.Attribute)
    }
    forbidden = {
        "validity", "validity_timed", "unusable_channels_at", "is_condemned_channel", "_condemned_channels",
        "validate_magnetics_signals", "apply_validity_exclusions", "optimize", "gauss_fit4", "min_gauss_fit4",
    }
    assert not (names & forbidden), names & forbidden


def test_the_applier_invents_no_constraint(monkeypatch, times):
    ods = vaft.omas.sample_ods()
    decisions = decide_efit_channels(ods, times)
    extra = usable_decision("flux_loop", 40, "ghost", times.size)
    decisions = decisions.replaced(extra)
    EQ = _build(monkeypatch, ods, times, decisions=decisions)
    assert "time_slice.0.constraints.flux_loop.40" not in EQ
    assert len(EQ["time_slice.0.constraints.flux_loop"]) == 11
