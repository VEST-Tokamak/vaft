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


def test_machine_view_keeps_diagnostic_geometry_distinct(fixture_data):
    from vaft.plot.backend.recipes import _build_machine_poloidal

    _, ods = fixture_data
    model = _build_machine_poloidal(
        ods, overlay=("interferometer", "langmuir_probes", "soft_x_rays")
    )
    interferometer = [layer for layer in model.layers if layer.label == "Interferometer LOS"]
    langmuir = [layer for layer in model.layers if layer.label == "Triple Langmuir probes"]
    sxr = [layer for layer in model.layers if layer.label == "Soft X-ray LOS"]
    assert len(interferometer) == 1
    assert interferometer[0].kind == "polyline"
    assert len(interferometer[0].r) == 3  # reflected 94 GHz chord
    assert len(langmuir) == 1 and langmuir[0].kind == "points"
    assert len(langmuir[0].r) == len(ods["langmuir_probes.embedded"])
    assert len(sxr) == 1 and sxr[0].kind == "polyline"
    assert not any(layer.label == "B-field Probes" for layer in model.layers)


def test_kinetic_overview_keeps_local_measurement_meanings(fixture_data):
    from vaft.plot.backend.recipes import build_model

    _, ods = fixture_data
    panels = build_model("kinetic_overview_profiles", [("fixture", ods)])
    assert [panel.title for panel in panels.models] == ["n_e", "T_e", "T_i", "V_phi"]
    assert "Cross-shot composite" in panels.suptitle
    assert "shot 48224 equilibrium" in panels.suptitle
    assert "shot 39915" in panels.suptitle
    assert all(panel.series for panel in panels.models)
    assert all(
        "interferometer" not in series.label.lower() and "soft x-ray" not in series.label.lower()
        for panel in panels.models for series in panel.series
    )
    assert {series.role for series in panels.models[0].series} == {"measurement", "reconstruction"}
    assert all("shot 48224" in series.label for panel in panels.models for series in panel.series)

    radius_panels = build_model("kinetic_overview_profiles", [("fixture", ods)], coordinate="R")
    assert {series.role for series in radius_panels.models[0].series} == {"measurement", "derived"}
    assert not any(series.role == "reconstruction" for panel in radius_panels.models for series in panel.series)
    assert "shot 42699" in radius_panels.models[0].series[-1].label


def test_probe_only_overview_uses_major_radius_without_invention(fixture_data):
    from omas import ODS
    from vaft.plot.backend.recipes import build_model, missing_required_path

    _, ods = fixture_data
    probe_only = ODS(consistency_check=False)
    probe_only["langmuir_probes"] = ods["langmuir_probes"]
    assert missing_required_path(probe_only, "kinetic_overview_profiles") is None
    model = build_model("kinetic_overview_profiles", [("probe", probe_only)])
    assert [len(panel.series) for panel in model.models] == [1, 1, 0, 0]
    assert all(panel.coordinate_label == "Major radius R [m]" for panel in model.models)
    with pytest.raises(ValueError, match="no compatible local kinetic profiles"):
        build_model("kinetic_overview_profiles", [("probe", probe_only)], coordinate="rho_tor_norm")


def test_kinetic_overview_renders_through_both_adapters(fixture_data):
    import matplotlib.pyplot as plt
    import vaft.omas
    import vaft.imas
    from pathlib import Path

    _, ods = fixture_data
    figure, axes = vaft.omas.plot_kinetic_overview_profiles(ods)
    assert len(figure.axes) == 4
    assert "Cross-shot composite" in figure._suptitle.get_text()
    plt.close(figure)
    figure, axes = vaft.omas.plot_machine_geometry_poloidal(ods)
    assert "projected onto geometry reference shot 39915" in axes.get_title()
    labels = [text for axis in figure.axes for text in axis.get_legend_handles_labels()[1]]
    assert "Interferometer LOS" in labels
    plt.close(figure)
    figure, axes = vaft.omas.plot_equilibrium_field_psi(ods)
    assert "source shot 48224, PF era 2507" in axes.get_title()
    assert "projected onto geometry reference shot 39915" not in axes.get_title()
    plt.close(figure)

    fixture_path = Path(__file__).parents[1] / "vaft/data/unified/vest_diagnostics/omas.json.gz"
    with vaft.imas.load(fixture_path) as source:
        figure, axes = vaft.imas.plot_kinetic_overview_profiles(source)
        assert len(figure.axes) == 4
        assert "Cross-shot composite" in figure._suptitle.get_text()
        plt.close(figure)
