"""Registered gyrokinetic / turbulent-transport plots (#1591 PR-C)."""

from __future__ import annotations

import copy

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft.omas
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import build_model, missing_required_path

from _synthetic_inputs import make_gyrokinetics_local, make_turbulent_transport


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def gk():
    return make_gyrokinetics_local(None)


@pytest.fixture(scope="module")
def transport():
    return make_turbulent_transport(None)


def test_growth_rate_and_frequency_read_the_imas_normalisation(gk):
    model = build_model("gyrokinetics_spectrum_growth_rate", normalize_entries(gk))
    assert model.metadata["x"] == "binormal_wavevector_norm"
    assert "GKDB" in model.metadata["normalisation"]
    assert model.series[0].y[0] == pytest.approx(
        gk["gyrokinetics_local.linear.wavevector.0.eigenmode.0.growth_rate_norm"])
    freq = build_model("gyrokinetics_spectrum_frequency", normalize_entries(gk))
    assert "ion dia. < 0" in freq.y_label


def test_flux_spectrum_is_per_species_and_says_quasilinear(gk):
    model = build_model("gyrokinetics_spectrum_energy_flux", normalize_entries(gk))
    assert len(model.series) == 3
    assert "quasilinear" in model.title


def test_eigenfunction_keeps_the_complex_amplitude(gk):
    model = build_model("gyrokinetics_profile_eigenfunction", normalize_entries(gk))
    magnitude, real, imag = (s.y for s in model.series)
    assert np.allclose(magnitude, np.hypot(real, imag))
    assert np.any(np.abs(imag) > 0)


def test_overview_composes_what_the_run_carries_and_drops_the_rest(gk):
    model = build_model("gyrokinetics_overview", normalize_entries(gk))
    assert len(model.models) == 5   # growth, frequency, flux, eigenfunction, state
    linear_only = copy.deepcopy(gk)
    del linear_only["gyrokinetics_local.non_linear"]
    smaller = build_model("gyrokinetics_overview", normalize_entries(linear_only))
    assert len(smaller.models) == 4
    assert any("R0/Ln" in line for line in smaller.models[-1].lines)


def test_unmatched_entries_are_named_on_the_figure(gk):
    other = copy.deepcopy(gk)
    other["gyrokinetics_local.flux_surface.r_minor_norm"] = 0.9 * gk["gyrokinetics_local.flux_surface.r_minor_norm"]
    model = build_model("gyrokinetics_spectrum_growth_rate", [("a", gk), ("b", other)])
    assert "unmatched: surface (r_minor_norm)" in model.title
    matched = build_model("gyrokinetics_spectrum_growth_rate", [("a", gk), ("b", copy.deepcopy(gk))])
    assert "unmatched" not in matched.title


def test_turbulent_transport_profiles_one_pair_per_model(transport):
    model = build_model("turbulent_transport_profile_energy_flux", normalize_entries(transport))
    labels = [s.label for s in model.series]
    assert labels == ["TGLF e", "TGLF i", "CGYRO e", "CGYRO i"]


def test_the_omas_facades_draw_each_registered_plot(gk, transport):
    for name, ods in (
        ("gyrokinetics_spectrum_growth_rate", gk), ("gyrokinetics_spectrum_frequency", gk),
        ("gyrokinetics_spectrum_energy_flux", gk), ("gyrokinetics_spectrum_particle_flux", gk),
        ("gyrokinetics_profile_eigenfunction", gk), ("gyrokinetics_overview", gk),
        ("turbulent_transport_profile_energy_flux", transport),
        ("turbulent_transport_profile_particle_flux", transport),
        ("turbulent_transport_overview", transport),
    ):
        figure, _ = getattr(vaft.omas, f"plot_{name}")(ods)
        assert figure is not None, name


def test_an_input_without_the_ids_is_not_offered(transport, gk):
    assert missing_required_path(transport, "gyrokinetics_overview") is not None
    assert missing_required_path(gk, "turbulent_transport_overview") is not None


def test_single_ky_runs_merge_into_one_scan_and_foreign_runs_are_refused():
    import copy

    from vaft.machine_mapping.gyrokinetics import merge_linear_scan

    one = make_gyrokinetics_local(None)
    two = copy.deepcopy(one)
    wave = "gyrokinetics_local.linear.wavevector.0"
    two[f"{wave}.binormal_wavevector_norm"] = one[f"{wave}.binormal_wavevector_norm"] / 2
    merged = merge_linear_scan([one, two])
    ky = [merged[f"gyrokinetics_local.linear.wavevector.{k}.binormal_wavevector_norm"]
          for k in range(len(merged["gyrokinetics_local.linear.wavevector"]))]
    assert ky == sorted(ky) and len(ky) == 2 * len(one["gyrokinetics_local.linear.wavevector"])
    foreign = copy.deepcopy(two)
    foreign["gyrokinetics_local.species.0.temperature_norm"] *= 1.1
    with pytest.raises(ValueError, match="species.0.temperature_norm"):
        merge_linear_scan([one, foreign])


def test_an_unconverged_initial_value_mode_is_not_drawn_unless_asked():
    """growth_rate_tolerance is written only for a converged run; a run stopped at
    MAX_TIME holds a last-step value, not an eigenvalue (the ES low-ky points of #1484)."""
    from vaft.machine_mapping.gyrokinetics import merge_linear_scan

    one = make_gyrokinetics_local(None)
    junk = copy.deepcopy(one)
    wave = "gyrokinetics_local.linear.wavevector.0"
    junk[f"{wave}.binormal_wavevector_norm"] = one[f"{wave}.binormal_wavevector_norm"] / 3
    junk[f"{wave}.eigenmode.0.growth_rate_norm"] = 50.0
    del junk[f"{wave}.eigenmode.0.growth_rate_tolerance"]
    scan = merge_linear_scan([one, junk])
    entries = normalize_entries(scan)
    default = build_model("gyrokinetics_spectrum_growth_rate", entries)
    assert np.nanmax(np.concatenate([s.y for s in default.series])) < 50.0
    asked = build_model("gyrokinetics_spectrum_growth_rate", entries, include_unconverged=True)
    assert np.nanmax(np.concatenate([s.y for s in asked.series])) == pytest.approx(50.0)


def _unconverged_scan():
    from vaft.machine_mapping.gyrokinetics import merge_linear_scan

    one = make_gyrokinetics_local(None)
    junk = copy.deepcopy(one)
    wave = "gyrokinetics_local.linear.wavevector.0"
    junk[f"{wave}.binormal_wavevector_norm"] = one[f"{wave}.binormal_wavevector_norm"] / 3
    junk[f"{wave}.eigenmode.0.growth_rate_norm"] = 50.0
    del junk[f"{wave}.eigenmode.0.growth_rate_tolerance"]
    return merge_linear_scan([one, junk])


def _largest_drawn(model):
    return float(np.nanmax(np.concatenate([s.y for s in model.series])))


def test_include_unconverged_is_an_option_of_the_public_entry_points():
    """The guide says to pass include_unconverged=True; the facade, extract and the
    overview must take it, and the plots that never read it must refuse it by name
    (cold review 0.8.0 delta-absorb-17-transport F2)."""
    import vaft.plot
    from vaft.plot.backend.options import OPTION_SCHEMA

    assert OPTION_SCHEMA["include_unconverged"].kind == "bool"
    scan = _unconverged_scan()
    assert _largest_drawn(vaft.plot.extract("gyrokinetics_spectrum_growth_rate", scan)) < 50.0
    asked = vaft.plot.extract("gyrokinetics_spectrum_growth_rate", scan, include_unconverged=True)
    assert _largest_drawn(asked) == pytest.approx(50.0)
    figure, ax = vaft.omas.plot_gyrokinetics_spectrum_growth_rate(scan, include_unconverged=True)
    assert max(float(np.nanmax(line.get_ydata())) for line in ax.get_lines()) == pytest.approx(50.0)
    figure, ax = vaft.omas.plot_gyrokinetics_spectrum_growth_rate(scan)
    assert max(float(np.nanmax(line.get_ydata())) for line in ax.get_lines()) < 50.0
    def finite_points(model):
        return int(sum(np.isfinite(s.y).sum() for s in model.series))

    frequency = vaft.plot.extract("gyrokinetics_spectrum_frequency", scan)
    assert finite_points(vaft.plot.extract("gyrokinetics_spectrum_frequency", scan,
                                           include_unconverged=True)) == finite_points(frequency) + 1
    overview = vaft.plot.extract("gyrokinetics_overview", scan, include_unconverged=True)
    assert _largest_drawn(overview.models[0]) == pytest.approx(50.0)
    assert _largest_drawn(vaft.plot.extract("gyrokinetics_overview", scan).models[0]) < 50.0
    for name in ("gyrokinetics_spectrum_energy_flux", "gyrokinetics_profile_eigenfunction"):
        with pytest.raises(ValueError, match="does not take an option named 'include_unconverged'"):
            vaft.plot.extract(name, scan, include_unconverged=True)


def test_labels_name_the_run_once(gk):
    """An entry label that already names the run is not repeated, and inside the
    overview (whose suptitle names the run) a lone mode carries no label at all."""
    model = build_model("gyrokinetics_spectrum_growth_rate", [("CGYRO", gk), ("run B", copy.deepcopy(gk))])
    assert [s.label for s in model.series] == ["CGYRO EM", "run B CGYRO EM"]
    overview = build_model("gyrokinetics_overview", normalize_entries(gk))
    assert overview.models[0].series[0].label == ""
    assert [s.label for s in overview.models[2].series] == ["Z=1", "Z=6", "e"]


def test_the_flux_axis_stops_where_the_flux_has_fallen_off(gk):
    wide = copy.deepcopy(gk)
    nl = "gyrokinetics_local.non_linear"
    ky = np.array([0.2, 0.4, 0.8, 1.6, 10.0, 40.0])
    wide[f"{nl}.binormal_wavevector_norm"] = ky
    wide[f"{nl}.fluxes_2d_k_x_sum.energy_phi_potential"] = np.outer([1.0, 0.6, 0.3], [1, 2, 1, 0.5, 0, 0])
    del wide[f"{nl}.fluxes_2d_k_x_sum.particles_phi_potential"]
    model = build_model("gyrokinetics_spectrum_energy_flux", normalize_entries(wide))
    assert model.x_limits == pytest.approx((0.0, 1.76))
    assert model.series[0].x[-1] == 40.0          # display range only, nothing dropped


def test_the_plotting_guide_names_only_registered_recipes():
    """Every standardized recipe the gyrokinetic guide names exists (cold review 0.8.0 F4)."""
    import re
    from pathlib import Path

    from vaft.plot.backend.recipes import RECIPES

    guide = Path(__file__).resolve().parents[1] / "docs" / "_guide" / "Gyrokinetic_plotting.md"
    names = set(re.findall(r"`((?:gyrokinetics|turbulent_transport)_(?:spectrum|profile|overview)[a-z_]*)`",
                           guide.read_text(encoding="utf-8")))
    assert names, "the guide no longer names any standardized recipe"
    assert names <= set(RECIPES), sorted(names - set(RECIPES))
