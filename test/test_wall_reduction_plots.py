"""Plot contracts for the reduced-wall study views (vaft #494)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft.omas as vomas
from vaft.plot.backend.recipes import RECIPES
from vaft.plot.models import Field2D, Panels


@pytest.fixture(scope="module")
def packaged():
    from vaft.omas.sample import sample_ods
    from vaft.validation.wall_reduction import observation_set, order_convergence, wall_system

    ods = sample_ods()
    system = wall_system(ods)
    observation = observation_set(ods, n_coils=system["n_coils"])
    rows = order_convergence(
        system, observation=observation, rules=("tau", "output_weight", "moments"),
        orders=(19, 76), drives=("shot",),
    )
    return ods, rows


def test_the_convergence_view_has_one_panel_per_metric_and_one_series_per_rule(packaged):
    ods, rows = packaged
    model = RECIPES["passive_structure_overview_wall_reduction"].builder(
        ods, rows=rows, metrics=("probe", "flux_loop", "grid_psi")
    )
    assert isinstance(model, Panels) and len(model.models) == 3
    for panel in model.models:
        assert [series.label for series in panel.series] == ["tau", "output_weight", "moments"]
        assert panel.log_y
        for series in panel.series:
            assert np.all(np.diff(series.x) > 0) and np.all(series.y > 0)


def test_the_convergence_view_refuses_a_drive_without_rows(packaged):
    ods, rows = packaged
    with pytest.raises(ValueError, match="no reduced-wall rows"):
        RECIPES["passive_structure_overview_wall_reduction"].builder(ods, rows=rows, drive="step")


def test_the_flux_map_is_masked_to_the_limiter_and_names_its_error(packaged):
    ods, rows = packaged
    model = RECIPES["passive_structure_field_wall_reduction"].builder(
        ods, which="difference", rule="moments", M=10, grid_shape=(17, 25)
    )
    assert isinstance(model, Field2D) and model.values.shape == (25, 17)
    assert np.isnan(model.values).any() and np.isfinite(model.values).any()
    assert "region error" in model.title and "10 moment patterns" in model.title
    assert model.overlays and model.overlays[0].label == "limiter"
    with pytest.raises(ValueError, match="which"):
        RECIPES["passive_structure_field_wall_reduction"].builder(ods, which="other")


def test_the_difference_map_is_smaller_than_the_full_map(packaged):
    ods, rows = packaged
    build = RECIPES["passive_structure_field_wall_reduction"].builder
    full = build(ods, which="full", rule="moments", M=30, grid_shape=(17, 25))
    diff = build(ods, which="difference", rule="moments", M=30, grid_shape=(17, 25))
    assert np.nanmax(np.abs(diff.values)) < 0.05 * np.nanmax(np.abs(full.values))


def test_the_adapters_render_figures(packaged):
    ods, rows = packaged
    figure, axes = vomas.plot_passive_structure_overview_wall_reduction(ods, rows=rows, metrics=("probe",))
    assert np.asarray(axes).ravel()[0].get_yscale() == "log"
    plt.close(figure)
    figure, axes = vomas.plot_passive_structure_field_wall_reduction(ods, rule="moments", M=10, grid_shape=(9, 13))
    assert "region error" in axes.get_title()
    plt.close(figure)
