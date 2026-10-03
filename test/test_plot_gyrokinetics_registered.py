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
