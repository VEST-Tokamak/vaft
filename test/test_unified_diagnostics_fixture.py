"""Offline physical and provenance checks for the cross-shot fixture."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("omas")

from vaft.data import unified_diagnostics_fixture, unified_diagnostics_manifest


@pytest.fixture(scope="module")
def fixture_data():
    return unified_diagnostics_manifest(), unified_diagnostics_fixture()


def test_fixture_contract_and_source_identity(fixture_data):
    manifest, ods = fixture_data
    assert manifest["physical_discharge"] is False
    assert manifest["geometry_reference"]["source_shot"] == 39915
    assert manifest["geometry_reference"]["pf_geometry_version"] == "1906"
    assert "shot" not in manifest
    assert "dataset_description.data_entry.pulse" not in ods
    assert "Cross-shot composite" in ods["dataset_description.ids_properties.comment"]
    kinetic = {name for name, record in manifest["sources"].items() if record["source_shot"] == 48224}
    assert kinetic == {
        "thomson_scattering", "charge_exchange", "core_profiles", "equilibrium"
    }
    assert manifest["sources"]["core_profiles"]["value_kind"].startswith("fitted")
    assert manifest["sources"]["langmuir_probes"]["source_shot"] == 42699
    assert manifest["sources"]["soft_x_rays"]["source_shot"] == 45531


def test_time_axes_are_kept_separate(fixture_data):
    _, ods = fixture_data
    assert len(ods["equilibrium.time_slice"]) == 1
    assert float(ods["core_profiles.time"][0]) == pytest.approx(0.3, abs=0.003)
    assert 0.35 <= float(ods["langmuir_probes.embedded.0.time"][0]) < 0.46
    assert len(ods["langmuir_probes.embedded.0.t_e.validity_timed"]) == len(
        ods["langmuir_probes.embedded.0.time"]
    )
    horizontal = np.asarray(ods["interferometer.channel.0.n_e_line.time"])
    vertical = np.asarray(ods["interferometer.channel.1.n_e_line.time"])
    assert len(horizontal) <= 192 and len(vertical) <= 192
    assert not np.array_equal(horizontal, vertical)
    assert "interferometer.time" not in ods


def test_chords_remain_chords_and_probes_remain_local(fixture_data):
    _, ods = fixture_data
    for family, prefix in (("soft_x_rays", "channel.0"), ("interferometer", "channel.0")):
        assert np.isfinite(ods[f"{family}.{prefix}.line_of_sight.first_point.r"])
        assert np.isfinite(ods[f"{family}.{prefix}.line_of_sight.second_point.r"])
    assert "n_e_line" in ods["interferometer.channel.0"]
    assert "n_e" not in ods["interferometer.channel.0"]
    assert "position.r" in ods["langmuir_probes.embedded.0"]
    assert len(ods["thomson_scattering.channel"]) > 0
    assert len(ods["charge_exchange.channel"]) > 0
