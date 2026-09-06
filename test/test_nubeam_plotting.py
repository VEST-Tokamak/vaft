"""Views of a NUBEAM run.

Figure introspection rather than image comparison, matching the rest of the
plotting suite. No NUBEAM installation is needed: every fixture is synthetic.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vaft.code.nubeam.outputs import (
    NUBEAMBirthMarkers,
    NUBEAMLostParticles,
    NUBEAMOutputs,
    NUBEAMPowerBalance,
)
from vaft.plot import nubeam as nbp
from vaft.plot.models import GeometryLayers, Panels, Profile1D

ZONES = 8


@pytest.fixture
def outputs(tmp_path):
    """A NUBEAM result with every optional product present."""
    return NUBEAMOutputs(
        workdir=tmp_path,
        runid="TESTRUN",
        profiles={
            "pbe": np.linspace(10.0, 1.0, ZONES),
            "pbi": np.linspace(2.0, 0.2, ZONES),
            "curbeam": np.linspace(5.0, 0.5, ZONES),
            # Species-resolved, as NUBEAM writes it.
            "nbeami": np.linspace(1e17, 1e16, ZONES).reshape(1, ZONES),
        },
        birth=NUBEAMBirthMarkers(
            path=tmp_path / "birth.cdf",
            species="H",
            count=4,
            # Centimetres and degrees, as the birth file stores them.
            columns={
                "r": np.array([20.0, 40.0, 60.0, 50.0]),
                "z": np.array([-10.0, 0.0, 10.0, 5.0]),
                "zeta": np.array([0.0, 90.0, 180.0, 270.0]),
                "wght": np.array([1e14, 2e14, 3e14, 4e14]),
            },
        ),
        lost=NUBEAMLostParticles(
            path=tmp_path / "xplasma_out.cdf",
            count=3,
            # Metres, as the lost-particle record stores them.
            columns={
                "rlost": np.array([0.51, 0.60, 0.69]),
                "zlost": np.array([-0.01, 0.20, 0.44]),
                "lstype": np.array([1.0, 1.0, 2.0]),
                "energy": np.array([9.6, 9.8, 10.0]),
                "ptcl": np.array([5e14, 4e14, 3e14]),
            },
        ),
        power_balance=(
            NUBEAMPowerBalance(
                species="H beam ion",
                entries={
                    "injected power (W)": 2.0e5,
                    "electron heating": -1.138e5,
                    "shine-through": -3.852e4,
                },
                residual=-1468.0,
            ),
        ),
    )


def _plots(outputs):
    return (
        ("profile", lambda **kw: nbp.nubeam_profile(outputs, "pbe", **kw)),
        ("deposition_poloidal", lambda **kw: nbp.nubeam_deposition_poloidal(outputs, **kw)),
        ("deposition_topview", lambda **kw: nbp.nubeam_deposition_topview(outputs, **kw)),
        ("lost_fast_ions", lambda **kw: nbp.nubeam_lost_fast_ions(outputs, **kw)),
    )


# --------------------------------------------------------------------------
# The renderer contract, inherited from the shared bodies
# --------------------------------------------------------------------------


def test_each_plot_returns_a_figure_and_axes(outputs):
    for name, call in _plots(outputs):
        figure, axes = call()
        assert isinstance(figure, plt.Figure), name
        assert isinstance(axes, plt.Axes), name
        plt.close(figure)


def test_supplied_axes_are_used_without_creating_a_figure(outputs):
    for name, call in _plots(outputs):
        figure, axes = plt.subplots()
        before = set(plt.get_fignums())
        _, returned = call(ax=axes)
        assert returned is axes, name
        assert set(plt.get_fignums()) == before, name
        plt.close(figure)


def test_nothing_is_shown_by_default(outputs, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    for _, call in _plots(outputs):
        figure, _ = call()
        plt.close(figure)
    assert shown == []


def test_show_is_honoured_when_asked(outputs, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    figure, _ = nbp.nubeam_profile(outputs, "pbe", show=True)
    plt.close(figure)
    assert shown == [True]


def test_builders_produce_the_shared_view_models(outputs):
    assert isinstance(nbp.build_nubeam_profile(outputs, "pbe"), Profile1D)
    assert isinstance(nbp.build_nubeam_deposition_topview(outputs), GeometryLayers)
    assert isinstance(nbp.build_nubeam_lost_fast_ions(outputs), GeometryLayers)
    assert isinstance(nbp.build_nubeam_power_accounting(outputs), Panels)


# --------------------------------------------------------------------------
# Units and coordinates -- the part most likely to go silently wrong
# --------------------------------------------------------------------------


def test_profile_labels_carry_nubeams_own_unit(outputs):
    """pbe is watts per zone, not a power density. The label must not imply one."""
    figure, axes = nbp.nubeam_profile(outputs, "pbe")
    label = axes.get_ylabel()
    assert "Electron heating" in label
    assert "W" in label
    assert "m$^{-3}$" not in label and "/m" not in label
    plt.close(figure)


def test_a_boundary_rho_grid_is_reduced_to_zone_centres(outputs):
    """NUBEAM's rho grid holds boundaries; profiles are zone averages."""
    boundaries = np.linspace(0.0, 1.0, ZONES + 1)
    model = nbp.build_nubeam_profile(outputs, "pbe", rho=boundaries)
    x = model.series[0].x
    assert x.size == ZONES
    np.testing.assert_allclose(x, 0.5 * (boundaries[:-1] + boundaries[1:]))
    assert "rho" in model.coordinate_label


def test_without_rho_the_abscissa_is_not_called_rho(outputs):
    """An index must not be passed off as a flux coordinate."""
    model = nbp.build_nubeam_profile(outputs, "pbe")
    assert "rho" not in model.coordinate_label
    assert "index" in model.coordinate_label


def test_a_mismatched_rho_is_refused(outputs):
    with pytest.raises(nbp.NUBEAMPlotError):
        nbp.build_nubeam_profile(outputs, "pbe", rho=np.linspace(0, 1, ZONES + 5))


def test_deposition_markers_are_converted_from_centimetres(outputs):
    model = nbp.build_nubeam_deposition_poloidal(outputs)
    layer = model.layers[0]
    np.testing.assert_allclose(layer.r, [0.20, 0.40, 0.60, 0.50])
    np.testing.assert_allclose(layer.z, [-0.10, 0.0, 0.10, 0.05])
    assert model.x_label == "R [m]"


def test_topview_projects_from_degrees(outputs):
    """x = R cos(zeta), y = R sin(zeta), with zeta in degrees."""
    model = nbp.build_nubeam_deposition_topview(outputs)
    markers = model.layers[0]
    # 0 deg, 90 deg, 180 deg, 270 deg at 0.2, 0.4, 0.6, 0.5 m
    np.testing.assert_allclose(markers.r, [0.20, 0.0, -0.60, 0.0], atol=1e-12)
    np.testing.assert_allclose(markers.z, [0.0, 0.40, 0.0, -0.50], atol=1e-12)
    assert model.x_label == "x [m]" and model.y_label == "y [m]"


def test_lost_particles_are_not_rescaled(outputs):
    """Already metres, unlike the birth file. A second /100 would be silent."""
    model = nbp.build_nubeam_lost_fast_ions(outputs)
    everything = np.concatenate([layer.r for layer in model.layers])
    assert everything.min() >= 0.5 and everything.max() <= 0.7


# --------------------------------------------------------------------------
# Loss channels
# --------------------------------------------------------------------------


def test_loss_channels_are_split_and_counted(outputs):
    model = nbp.build_nubeam_lost_fast_ions(outputs)
    labels = [layer.label for layer in model.layers]
    assert any("prompt loss (2)" in label for label in labels)
    assert any("orbit loss (1)" in label for label in labels)


def test_a_prompt_only_run_offers_no_orbit_layer(outputs):
    """The VEST case is entirely prompt; an empty 'orbit loss' entry would lie."""
    outputs.lost.columns["lstype"] = np.array([1.0, 1.0, 1.0])
    model = nbp.build_nubeam_lost_fast_ions(outputs)
    assert len(model.layers) == 1
    assert "prompt loss (3)" in model.layers[0].label


def test_the_title_does_not_name_a_single_loss_channel(outputs):
    """NUBEAM's log says 'bad orbit loss'; the data here says prompt."""
    model = nbp.build_nubeam_lost_fast_ions(outputs)
    assert "orbit" not in model.title.lower()


# --------------------------------------------------------------------------
# Power accounting
# --------------------------------------------------------------------------


def test_power_accounting_reports_nubeams_own_numbers(outputs):
    model = nbp.build_nubeam_power_accounting(outputs)
    text = "\n".join(model.models[0].lines)
    assert "injected 200.0 kW" in text
    # 1.138e5 / 2.0e5
    assert "56.9%" in text
    # 3.852e4 / 2.0e5
    assert "19.3%" in text
    assert "residual" in text


# --------------------------------------------------------------------------
# Degrading when a product is absent
# --------------------------------------------------------------------------


def test_an_absent_profile_names_what_is_available(outputs):
    with pytest.raises(nbp.NUBEAMPlotError) as error:
        nbp.build_nubeam_profile(outputs, "pfuse")
    message = str(error.value)
    assert "pfuse" in message
    assert "hydrogen" in message
    assert "pbe" in message


@pytest.mark.parametrize(
    ("attribute", "builder", "expected"),
    [
        ("birth", nbp.build_nubeam_deposition_poloidal, "nltrk_dep0"),
        ("birth", nbp.build_nubeam_deposition_topview, "nltrk_dep0"),
        ("lost", nbp.build_nubeam_lost_fast_ions, "xplasma_out"),
    ],
)
def test_a_missing_product_is_reported_actionably(outputs, attribute, builder, expected):
    setattr(outputs, attribute, None)
    with pytest.raises(nbp.NUBEAMPlotError) as error:
        builder(outputs)
    assert expected in str(error.value)


def test_a_run_with_no_power_balance_is_reported(outputs):
    outputs.power_balance = ()
    with pytest.raises(nbp.NUBEAMPlotError) as error:
        nbp.build_nubeam_power_accounting(outputs)
    assert "step log" in str(error.value)


def test_a_run_that_lost_nothing_is_reported_rather_than_drawn_empty(outputs):
    outputs.lost.count = 0
    with pytest.raises(nbp.NUBEAMPlotError):
        nbp.build_nubeam_lost_fast_ions(outputs)


# --------------------------------------------------------------------------
# The result is not modified
# --------------------------------------------------------------------------


def test_plotting_does_not_mutate_the_result(outputs):
    import copy

    before = copy.deepcopy(outputs.profiles["pbe"])
    birth_before = copy.deepcopy(outputs.birth.columns["r"])
    lost_before = copy.deepcopy(outputs.lost.columns["rlost"])

    for _, call in _plots(outputs):
        figure, _ = call()
        plt.close(figure)

    np.testing.assert_array_equal(outputs.profiles["pbe"], before)
    np.testing.assert_array_equal(outputs.birth.columns["r"], birth_before)
    np.testing.assert_array_equal(outputs.lost.columns["rlost"], lost_before)


def test_the_result_bundle_and_the_native_container_are_interchangeable(outputs):
    """Callers hold a NUBEAMResult; builders should not make them unwrap it."""
    from vaft.code.nubeam.outputs import NUBEAMResult

    bundle = NUBEAMResult(returncode=0, workdir=outputs.workdir, outputs_native=outputs)
    from_bundle = nbp.build_nubeam_profile(bundle, "pbe")
    from_native = nbp.build_nubeam_profile(outputs, "pbe")

    # Compare the contents: a dataclass __eq__ over NumPy arrays is ambiguous.
    assert from_bundle.y_label == from_native.y_label
    assert from_bundle.y_unit == from_native.y_unit
    assert from_bundle.title == from_native.title
    assert len(from_bundle.series) == len(from_native.series)
    for left, right in zip(from_bundle.series, from_native.series):
        np.testing.assert_array_equal(left.x, right.x)
        np.testing.assert_array_equal(left.y, right.y)


# --------------------------------------------------------------------------
# Titles
# --------------------------------------------------------------------------


def test_a_profile_title_does_not_repeat_the_y_label(outputs):
    """The y-label already says 'Electron heating'; the title carries only
    provenance, so a grid of these does not read as eight copies of one
    sentence."""
    model = nbp.build_nubeam_profile(outputs, "pbe")
    assert "Electron heating" in model.y_label
    assert "Electron heating" not in model.title
    assert "TESTRUN" in model.title


def test_a_title_can_be_suppressed_for_a_panel_grid(outputs):
    assert nbp.build_nubeam_profile(outputs, "pbe", title="").title == ""
    assert nbp.build_nubeam_deposition_topview(outputs, title="").title == ""
    assert nbp.build_nubeam_lost_fast_ions(outputs, title="").title == ""


def test_a_title_can_be_replaced(outputs):
    model = nbp.build_nubeam_profile(outputs, "pbe", title="Shot 12345")
    assert model.title == "Shot 12345"


def test_geometry_titles_say_what_the_view_is(outputs):
    """Unlike a profile, 'Z [m]' says nothing about the content."""
    assert "top view" in nbp.build_nubeam_deposition_topview(outputs).title
    assert "poloidal" in nbp.build_nubeam_deposition_poloidal(outputs).title


def test_a_run_without_an_id_gets_no_dangling_separator(outputs):
    outputs.runid = ""
    assert not nbp.build_nubeam_profile(outputs, "pbe").title.endswith("--")
    assert "--" not in nbp.build_nubeam_deposition_topview(outputs).title
