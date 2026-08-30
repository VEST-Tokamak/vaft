"""Contract for the composed machine and equilibrium overview plots.

These are the views a first tutorial reaches for: one picture that orients a
reader in the machine, and one that shows what the plasma is doing. They are
public API, so their composition is pinned here rather than left to whichever
notebook happens to call them.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import vaft
from vaft.omas._plot_recipes import _build_machine_poloidal, _build_machine_topview


@pytest.fixture(scope="module")
def sample():
    return vaft.omas.sample_ods()


@pytest.fixture(scope="module")
def sample_with_soft_x_rays():
    """The packaged sample carries no soft_x_rays IDS, so add one to test with."""
    from vaft.machine_mapping.soft_x_rays import soft_x_rays

    ods = vaft.omas.sample_ods()
    csv = vaft.data.data_path("legacy") / "digitizer_17592_45531.csv"
    if not csv.is_file():
        pytest.skip("the soft X-ray reference CSV needs a repository checkout")
    soft_x_rays(ods, 45531, 17592, data_root=vaft.data.data_path("legacy"))
    return ods


# ---------------------------------------------------------------------------
# Composed poloidal view
# ---------------------------------------------------------------------------


def test_machine_poloidal_composes_the_structural_geometry(sample):
    model = _build_machine_poloidal(sample)
    assert model.layers, "the composed view must draw something"
    assert model.title == "Machine Cross-section"


def test_machine_poloidal_includes_soft_x_ray_sight_lines(sample_with_soft_x_rays):
    """The sight lines belong in the orientation view, not only in their own plot."""
    without = _build_machine_poloidal(vaft.omas.sample_ods())
    with_sxr = _build_machine_poloidal(sample_with_soft_x_rays)
    assert len(with_sxr.layers) > len(without.layers)
    labels = [layer.label for layer in with_sxr.layers if layer.label]
    assert "Soft X-ray LOS" in labels


def test_soft_x_ray_sight_lines_collapse_to_one_legend_entry(sample_with_soft_x_rays):
    """Forty labelled sight lines orient nobody; the composite gets one entry."""
    model = _build_machine_poloidal(sample_with_soft_x_rays)
    sxr_labels = [
        layer.label for layer in model.layers if layer.label and "Soft X-ray" in layer.label
    ]
    assert len(sxr_labels) == 1

    # The dedicated plot still labels every channel, which is its job.
    standalone = vaft.omas.plot_soft_x_rays_geometry_lines_of_sight
    figure, axes = standalone(sample_with_soft_x_rays, show=False)
    named = [line.get_label() for line in axes.get_lines()]
    assert len([name for name in named if not name.startswith("_")]) > 1


def test_composed_view_does_not_draw_the_wall_twice(sample_with_soft_x_rays):
    from vaft.omas._plot_recipes import _build_lines_of_sight, _wall_layers

    wall = len(_wall_layers(sample_with_soft_x_rays))
    assert wall  # the sample has a limiter outline
    composed = _build_lines_of_sight(
        sample_with_soft_x_rays, label_channels=False, include_wall=False
    )
    standalone = _build_lines_of_sight(sample_with_soft_x_rays)
    assert len(standalone.layers) - len(composed.layers) == wall


def test_machine_poloidal_renders(sample):
    figure, axes = vaft.omas.plot_machine_geometry_poloidal(sample, show=False)
    assert figure is not None and axes is not None


# ---------------------------------------------------------------------------
# Composed top view
# ---------------------------------------------------------------------------


def test_top_view_draws_the_vessel(sample):
    """Without the vessel the top view has nothing to orient against."""
    model = _build_machine_topview(sample)
    labels = [layer.label for layer in model.layers if layer.label]
    assert "Vessel outboard" in labels
    assert "Vessel inboard" in labels


def test_top_view_vessel_encloses_the_plasma(sample):
    model = _build_machine_topview(sample)
    radii = {
        layer.label: float(np.nanmax(np.hypot(np.asarray(layer.r), np.asarray(layer.z))))
        for layer in model.layers
        if layer.label
    }
    assert radii["Vessel outboard"] >= radii["Plasma outboard"]


def test_top_view_renders(sample):
    figure, axes = vaft.omas.plot_machine_geometry_topview(sample, show=False)
    assert figure is not None and axes is not None


# ---------------------------------------------------------------------------
# Equilibrium profile overview
# ---------------------------------------------------------------------------


def test_equilibrium_profile_overview_is_public_api():
    assert hasattr(vaft.omas, "plot_equilibrium_overview_profiles")
    assert hasattr(vaft.plot, "equilibrium_overview_profiles")
    assert "equilibrium_overview_profiles" in {
        row["name"] for row in vaft.plot.available_plots()
    }


def test_equilibrium_profile_overview_composes_p_j_and_q():
    from vaft.omas._plot_recipes import RECIPES

    assert RECIPES["equilibrium_overview_profiles"].members == (
        "equilibrium_profile_pressure",
        "equilibrium_profile_j_tor",
        "equilibrium_profile_q",
    )


def test_equilibrium_profile_overview_hides_panels_with_no_data(sample):
    """The packaged sample stores p and q but no j_tor; no blank panel may show."""
    figure, axes = vaft.omas.plot_equilibrium_overview_profiles(sample, show=False)
    visible = [panel for panel in np.asarray(axes).ravel() if panel.get_visible()]
    titles = [panel.get_title() for panel in visible]
    assert titles == ["Pressure", "Safety Factor q"]
    assert all(panel.get_lines() for panel in visible)


def test_equilibrium_profile_overview_draws_j_when_the_equilibrium_provides_it():
    """A reconstruction that stores j_tor gets all three panels."""
    kinetic = vaft.data.data_path("kineticEfit/ods_48224_300ms.json")
    if not kinetic.is_file():
        pytest.skip("the kinetic reference ODS needs a repository checkout")
    ods = vaft.omas.load(kinetic)
    figure, axes = vaft.omas.plot_equilibrium_overview_profiles(ods, show=False)
    visible = [panel for panel in np.asarray(axes).ravel() if panel.get_visible()]
    assert "Toroidal Current Density" in [panel.get_title() for panel in visible]
    assert len(visible) == 3
