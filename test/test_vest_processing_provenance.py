"""Effective processing provenance per shot (issue #195)."""

import pytest

from vaft.machine_mapping.provenance import vest_processing_provenance
from vaft.machine_mapping.utils import (
    resolve_shot_revisions,
    resolve_shot_revisions_with_provenance,
    resolve_vest_diagnostic,
)


def test_provenance_needs_no_raw_data():
    """Pure config resolution, so it is safe for manifests and reporting."""
    record = vest_processing_provenance(46403)
    assert record["shot"] == 46403


@pytest.mark.parametrize(
    ("shot", "expected_mode", "expected_window"),
    [
        (43760, "subtract", (7250, 8750)),
        (43761, "subtract", (6000, 7251)),
        (46403, "subtract_fl10_windowed", (6000, 7251)),
        (47116, "subtract_fl10_windowed", (6000, 7251)),
        (47117, "disabled", (6000, 7251)),
    ],
)
def test_plasma_current_era_is_recoverable(shot, expected_mode, expected_window):
    record = vest_processing_provenance(shot)["plasma_current"]
    assert record["reference"]["mode"] == expected_mode
    assert record["baseline"]["analysis_window"] == expected_window
    assert record["source_field"] == 109  # raw-first; field 102 never reintroduced


@pytest.mark.parametrize(
    ("shot", "enabled"),
    [(46403, True), (47116, True), (47117, False)],
)
def test_fl10_compensation_enabled_flag_tracks_the_mode(shot, enabled):
    record = vest_processing_provenance(shot)["plasma_current"]
    assert record["reference"]["compensation_enabled"] is enabled


@pytest.mark.parametrize(
    ("shot", "pf1", "pf5"),
    [
        (45964, -5.0e4, -1.0e4),
        (45965, -1.0e4, -1.0e4),
        (48371, -1.0e4, -1.0e4),
        (48372, -1.0e4, -5.0e3),
    ],
)
def test_pf_gain_era_is_recoverable(shot, pf1, pf5):
    gains = vest_processing_provenance(shot)["pf_active"]["coil_gains"]
    assert gains[0] == pytest.approx(pf1)
    assert gains[4] == pytest.approx(pf5)


def test_pf6_saturation_repair_policy_is_discoverable():
    repair = vest_processing_provenance(45965)["pf_active"]["saturation_repair"]
    assert repair[5]["value"] == pytest.approx(-5000.0)
    assert repair[5]["tolerance"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("shot", "daq_mode", "probe_baseline"),
    [
        (43684, "legacy", 5000),
        (43685, "legacy", 1750),
        (46403, "legacy", 1750),
        (46404, "native_daq", 1750),
    ],
)
def test_equilibrium_magnetics_era_is_recoverable(shot, daq_mode, probe_baseline):
    record = vest_processing_provenance(shot)["equilibrium_magnetics"]
    assert record["daq_mode"] == daq_mode
    assert record["probe_baseline_end"] == probe_baseline


def test_shot_39204_geometry_gap_is_visible_in_provenance():
    record = vest_processing_provenance(39204)["equilibrium_magnetics"]
    assert record["geometry_supported"] is False
    assert record["required_geometry_version"] == "2310"

    supported = vest_processing_provenance(39205)["equilibrium_magnetics"]
    assert supported["geometry_supported"] is True
    assert supported["required_geometry_version"] is None


@pytest.mark.parametrize("shot", [43760, 43761, 46403, 46404, 47117, 48372])
def test_revision_bounds_actually_contain_the_shot(shot):
    """A reported era must be one the shot really falls inside."""
    record = vest_processing_provenance(shot)
    for revision in (
        record["plasma_current"]["reference"]["revision"],
        record["plasma_current"]["baseline"]["revision"],
        record["pf_active"]["coil_gains_revision"],
        record["equilibrium_magnetics"]["revision"],
    ):
        bounds = revision["revision_bounds"]
        if bounds is None:
            continue
        if bounds["from_shot"] is not None:
            assert shot >= bounds["from_shot"]
        if bounds["to_shot"] is not None:
            assert shot <= bounds["to_shot"]


# --------------------------------------------------------------------------
# Resolver backward compatibility
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shot", [0, 20259, 43761, 45965, 47117, 48372])
@pytest.mark.parametrize("diagnostic", ["plasma_current", "pf_active", "equilibrium_magnetics"])
def test_resolve_vest_diagnostic_default_return_is_unchanged(shot, diagnostic):
    plain = resolve_vest_diagnostic(shot, diagnostic)
    with_prov, provenance = resolve_vest_diagnostic(shot, diagnostic, with_provenance=True)
    assert plain == with_prov
    assert set(provenance) == {"context", "revision_index", "revision_bounds"}


def test_resolve_shot_revisions_matches_its_provenance_counterpart():
    base = {"value": 1}
    revisions = [{"from_shot": 10, "to_shot": 20, "value": 2}, {"from_shot": 21, "value": 3}]
    for shot in (5, 10, 20, 21, 99):
        plain = resolve_shot_revisions(base, revisions, shot, context="t")
        resolved, _ = resolve_shot_revisions_with_provenance(base, revisions, shot, context="t")
        assert plain == resolved


def test_shot_outside_every_era_reports_no_revision():
    base = {"value": 1}
    revisions = [{"from_shot": 10, "to_shot": 20, "value": 2}]
    resolved, provenance = resolve_shot_revisions_with_provenance(
        base, revisions, 5, context="t"
    )
    assert resolved == {"value": 1}
    assert provenance["revision_index"] is None
    assert provenance["revision_bounds"] is None
