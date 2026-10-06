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
    langmuir = [layer for layer in model.layers if layer.label == "Langmuir sites"]
    sxr = [layer for layer in model.layers if layer.label == "SXR LOS"]
    assert len(interferometer) == 1
    assert interferometer[0].kind == "polyline"
    # The three stored corners remain on the sampled Cartesian LOS, including
    # its reflection. Sampling also preserves the true cylindrical R-Z curve.
    assert len(interferometer[0].r) == 64
    assert interferometer[0].r[31] == pytest.approx(
        ods["interferometer.channel.0.line_of_sight.second_point.r"]
    )
    assert len(langmuir) == 1 and langmuir[0].kind == "points"
    assert sum(layer.kind == "points" for layer in model.layers) == len(ods["langmuir_probes.embedded"])
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


def test_kinetic_overview_maps_measurements_through_the_time_matched_slice(fixture_data):
    """The Thomson sample is picked by time, so the equilibrium slice must be too.

    Two slices with different axes: the plotted psi_N must follow the slice
    nearest ``core_profiles.time``, not slice 0 whatever the target time is.
    """
    from omas import ODS

    from vaft.omas.sample import sample_ods
    from vaft.plot.backend.recipes import build_model
    from vaft.process.profile import equilibrium_mapping_points

    _, fixture = fixture_data
    sample = sample_ods()
    times = np.asarray(sample["equilibrium.time"], dtype=float)
    assert times.size >= 2
    two = ODS(consistency_check=False)
    two["thomson_scattering"] = fixture["thomson_scattering"]
    for new, old in ((0, 0), (1, times.size - 1)):
        two[f"equilibrium.time_slice.{new}"] = sample[f"equilibrium.time_slice.{old}"]
    two["equilibrium.time"] = np.array([times[0], times[-1]])
    for leaf in ("equilibrium.vacuum_toroidal_field.r0", "equilibrium.ids_properties.cocos"):
        if leaf in sample:
            two[leaf] = sample[leaf]
    channels = len(fixture["thomson_scattering.channel"])
    r = np.array([float(fixture[f"thomson_scattering.channel.{i}.position.r"]) for i in range(channels)])
    z = np.array([float(fixture[f"thomson_scattering.channel.{i}.position.z"]) for i in range(channels)])
    targets = (float(times[0]), float(times[-1]))
    expected = {when: equilibrium_mapping_points(two, r, z, time=when).psi_norm for when in targets}
    assert not np.allclose(np.nan_to_num(expected[targets[0]]), np.nan_to_num(expected[targets[1]]))

    plotted = {}
    for when in targets:
        two["core_profiles.time"] = np.array([when])
        panels = build_model("kinetic_overview_profiles", [("two", two)], coordinate="psi_norm")
        measured = panels.models[0].series[0]
        assert measured.role == "measurement"
        assert all(np.isclose(value, expected[when]).any() for value in measured.x), when
        plotted[when] = measured.x
    assert not np.array_equal(plotted[targets[0]], plotted[targets[1]])


def test_kinetic_overview_omits_an_unavailable_coordinate_instead_of_raising(fixture_data):
    """An equilibrium without q cannot give rho_tor_norm; the overview still builds.

    ``available_plots`` promises the plot on IDS presence, so ``auto`` must
    resolve to a coordinate the equilibrium supplies, and an explicit
    unavailable coordinate omits the mapped series as the contract says.
    """
    from omas import ODS

    from vaft.omas.sample import sample_ods
    from vaft.plot.backend.kinetic_overview import _LABELS
    from vaft.plot.backend.recipes import build_model, missing_required_path

    _, fixture = fixture_data
    sample = sample_ods()
    without_q = ODS(consistency_check=False)
    without_q["thomson_scattering"] = fixture["thomson_scattering"]
    without_q["equilibrium"] = sample["equilibrium"]
    for index in range(len(without_q["equilibrium.time_slice"])):
        prefix = f"equilibrium.time_slice.{index}.profiles_1d"
        for leaf in ("q", "rho_tor_norm", "phi"):
            if f"{prefix}.{leaf}" in without_q:
                del without_q[f"{prefix}.{leaf}"]
    assert missing_required_path(without_q, "kinetic_overview_profiles") is None

    panels = build_model("kinetic_overview_profiles", [("h", without_q)])
    assert [len(panel.series) for panel in panels.models] == [1, 1, 0, 0]
    assert panels.models[0].series[0].role == "measurement"
    assert all(panel.coordinate_label == _LABELS["rho_pol_norm"] for panel in panels.models)
    with pytest.raises(ValueError, match="no compatible local kinetic profiles"):
        build_model("kinetic_overview_profiles", [("h", without_q)], coordinate="rho_tor_norm")


def test_probe_only_overview_uses_major_radius_without_invention(fixture_data):
    from omas import ODS
    import vaft.omas
    from vaft.plot.backend.recipes import build_model, missing_required_path

    _, ods = fixture_data
    probe_only = ODS(consistency_check=False)
    probe_only["langmuir_probes"] = ods["langmuir_probes"]
    assert missing_required_path(probe_only, "kinetic_overview_profiles") is None
    discovered = vaft.omas.available_plots(probe_only, query="kinetic", available_only=True)
    assert any(record.name == "kinetic_overview_profiles" for record in discovered)
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


def _calibration_assets():
    import yaml

    from vaft.data.resources import data_path

    with (data_path("unified/vest_diagnostics") / "manifest.yaml").open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    return manifest["camera_calibration_reference"]["assets"]


def test_sha256_pinned_calibration_assets_are_checked_out_with_lf():
    """The manifest hashes the LF bytes, so Git must not rewrite them to CRLF on a
    Windows checkout (that broke every fixture test on Windows CI only)."""
    import subprocess

    from vaft.data.resources import data_path

    paths = [str(data_path(asset["path"])) for asset in _calibration_assets()]
    assert paths
    out = subprocess.run(
        ["git", "check-attr", "text", "eol", "--", *paths],
        cwd=data_path(), capture_output=True, text=True,
    )
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    for path in paths:
        attrs = dict(
            line.rsplit(": ", 2)[1:] for line in out.stdout.splitlines() if line.startswith(path)
        )
        assert attrs.get("text") == "unset" or attrs.get("eol") == "lf", (path, out.stdout)


def test_calibration_checksum_still_rejects_crlf_content(tmp_path, monkeypatch):
    """The check stays byte-exact: a CRLF copy of a calibration asset must fail."""
    import shutil

    from vaft.data import resources

    real = resources.data_path
    asset = _calibration_assets()[0]["path"]
    shutil.copytree(real("unified/vest_diagnostics"), tmp_path / "unified/vest_diagnostics")
    target = tmp_path / asset
    target.parent.mkdir(parents=True)
    for other in _calibration_assets():
        shutil.copyfile(real(other["path"]), tmp_path / other["path"])
    monkeypatch.setattr(resources, "data_path", lambda name="": tmp_path / name if name else tmp_path)
    assert unified_diagnostics_manifest()["camera_calibration_reference"]["assets"]
    target.write_bytes(target.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="Camera calibration reference checksum mismatch"):
        unified_diagnostics_manifest()
