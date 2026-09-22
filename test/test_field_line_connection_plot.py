"""The connection-length map read back out of the IDS stand-in (issue #1099).

Everything here goes through the writer, because what is under test is
whether ``plasma_initiation.b_field_lines`` carries enough to draw the figure
-- the structure was chosen for field-line tracing without being designed for
it, and the interesting failures are at that seam.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from omas import ODS

matplotlib.use("Agg")

import vaft.plot
from vaft.machine_mapping.field_line_topology import write_b_field_lines
from vaft.plot import registry
from vaft.plot.backend.recipes import RECIPES

NAME = "field_line_topology_field_connection_length"

GRID_R = np.array([1.0, 1.5, 2.0, 2.5])
GRID_Z = np.array([-1.0, 0.0])

#: The value assertions below are about the numbers, so they ask for the
#: linear view; the log default has its own tests at the end.
LINEAR = {"scale": "linear"}


def traced(ods=None, *, order=None, time=0.1, phi_deg=30.0, open_fraction=0.5):
    """One plane whose length is ``100 R + z``, so every node is distinct."""
    ods = ODS() if ods is None else ods
    r, z = np.meshgrid(GRID_R, GRID_Z)
    r, z = r.ravel(), z.ravel()
    lengths = 100.0 * r + z
    if order is not None:
        r, z, lengths = r[order], z[order], lengths[order]
    write_b_field_lines(
        ods, grid_r=GRID_R, grid_z=GRID_Z, starting_r=r, starting_z=z,
        lengths=lengths, open_fraction=open_fraction, time=time, phi_deg=phi_deg,
    )
    return ods


def expected():
    r, z = np.meshgrid(GRID_R, GRID_Z)
    return 100.0 * r + z


def test_the_map_is_the_lengths_on_their_own_grid():
    model = RECIPES[NAME].builder(traced(), **LINEAR)
    np.testing.assert_allclose(model.r, GRID_R)
    np.testing.assert_allclose(model.z, GRID_Z)
    np.testing.assert_allclose(model.values, expected())
    assert model.values.shape == (GRID_Z.size, GRID_R.size)


def test_the_image_is_built_from_the_positions_not_from_the_row_order():
    """A FLARE mesh runs its first axis fastest while its header prints the
    axes the other way round, and the maps are usually square, so a builder
    that reshaped the flat list would transpose the picture and still fit.
    Scrambling the lines changes nothing here."""
    scrambled = np.random.default_rng(0).permutation(GRID_R.size * GRID_Z.size)
    np.testing.assert_allclose(
        RECIPES[NAME].builder(traced(order=scrambled), **LINEAR).values, expected()
    )


def test_a_transposed_grid_would_not_quietly_fit():
    """The grid is 4 x 2, so an order swap cannot be absorbed by a reshape --
    which is exactly why a square grid is the dangerous case and this one is
    deliberately not square."""
    model = RECIPES[NAME].builder(traced(), **LINEAR)
    assert model.values.shape != model.values.T.shape


def test_the_title_names_the_angle_the_time_and_the_open_fraction():
    """b_field_lines has no toroidal coordinate, so a map drawn without its
    angle says nothing about where it is."""
    title = RECIPES[NAME].builder(traced(phi_deg=-12.0, time=0.25), **LINEAR).title
    assert "φ = -12°" in title
    assert "t = 0.25 s" in title
    assert "50% of lines reach the wall" in title


def test_an_angle_the_writer_did_not_record_is_left_out_rather_than_invented():
    ods = traced()
    ods["plasma_initiation.code.parameters"] = "<parameters/>"
    title = RECIPES[NAME].builder(ods, **LINEAR).title
    assert "φ" not in title and "Connection length" in title


def test_time_names_a_plane_and_anything_else_raises():
    """A plane is a traced object, not a sample of a signal: there is nothing
    between two of them to interpolate, and the nearest one is not the one
    the caller asked for.  The first version snapped to it."""
    ods = traced(time=0.1, phi_deg=0.0)
    traced(ods, time=0.4, phi_deg=90.0)
    assert "φ = 90°" in RECIPES[NAME].builder(ods, time=0.4, **LINEAR).title
    assert "φ = 0°" in RECIPES[NAME].builder(ods, time=0.1, **LINEAR).title
    for requested in (0.25, 9.0):
        with pytest.raises(ValueError, match="not one of the traced planes"):
            RECIPES[NAME].builder(ods, time=requested, **LINEAR)


# ---------------------------------------------------------------------------
# The value axis
# ---------------------------------------------------------------------------


def test_the_map_is_logarithmic_by_default_and_says_so():
    """A connection length runs from about a metre to the tracing limit, so
    on a linear ramp every value below the top decade shares one colour and
    the lobe structure the map exists to show is invisible."""
    model = RECIPES[NAME].builder(traced())
    assert model.value_scale == "log"
    assert "log scale" in model.value_label
    linear = RECIPES[NAME].builder(traced(), **LINEAR)
    assert linear.value_scale == "linear" and "log" not in linear.value_label
    np.testing.assert_allclose(model.values, linear.values)


def test_a_zero_length_cell_is_blanked_and_counted_rather_than_clipped():
    """Zero cannot sit on a log axis.  Dropping it to the smallest colour
    would read as a short connection; blanking it reads as no value, and the
    title says how many."""
    ods = traced()
    entry = "plasma_initiation.b_field_lines.0"
    lengths = np.asarray(ods[f"{entry}.lengths"]).copy()
    lengths[0] = 0.0
    ods[f"{entry}.lengths"] = lengths
    model = RECIPES[NAME].builder(ods)
    assert np.isnan(model.values[0, 0])
    assert "1 cell(s) of zero length left blank" in model.title
    assert RECIPES[NAME].builder(ods, **LINEAR).values[0, 0] == 0.0


def test_an_unknown_scale_is_refused_by_the_option_schema():
    from vaft.plot.backend.options import validate_options

    with pytest.raises(ValueError, match="must be one of log, linear"):
        validate_options(NAME, {"scale": "cubic"})


def test_the_log_map_renders_with_a_logarithmic_norm():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    figure, axes = plt.subplots()
    getattr(vaft.plot, NAME)(RECIPES[NAME].builder(traced()), ax=axes)
    assert any(isinstance(c.norm, LogNorm) for c in axes.collections)
    plt.close(figure)


def test_an_ods_with_nothing_traced_says_so():
    with pytest.raises(ValueError, match="nothing has been traced"):
        RECIPES[NAME].builder(ODS(), **LINEAR)


def test_a_line_that_does_not_sit_on_the_declared_grid_is_refused():
    """The writer guards this too; the builder guards it again because an ODS
    can reach a figure from anywhere, not only from this writer."""
    ods = traced()
    positions = np.asarray(ods["plasma_initiation.b_field_lines.0.starting_positions.r"])
    positions = positions.copy()
    positions[0] = 1.25
    ods["plasma_initiation.b_field_lines.0.starting_positions.r"] = positions
    with pytest.raises(ValueError, match="do not lie on grid.dim1"):
        RECIPES[NAME].builder(ods, **LINEAR)


def test_two_lines_on_one_node_are_refused_rather_than_overwritten():
    ods = traced()
    entry = "plasma_initiation.b_field_lines.0"
    positions = np.asarray(ods[f"{entry}.starting_positions.r"]).copy()
    positions[1] = positions[0]
    ods[f"{entry}.starting_positions.r"] = positions
    with pytest.raises(ValueError, match="share a grid node"):
        RECIPES[NAME].builder(ods, **LINEAR)


def test_a_line_the_tracer_could_not_follow_keeps_its_node():
    """A NaN length is a real result -- the VEST startup tracer writes one for
    every seed outside the wall -- so the map must carry it rather than read
    the empty cell as two lines colliding."""
    ods = traced()
    entry = "plasma_initiation.b_field_lines.0"
    lengths = np.asarray(ods[f"{entry}.lengths"]).copy()
    lengths[2] = np.nan
    ods[f"{entry}.lengths"] = lengths
    drawn = RECIPES[NAME].builder(ods, **LINEAR).values
    assert np.isnan(drawn[0, 2])
    assert np.isfinite(drawn).sum() == lengths.size - 1


def test_a_length_missing_for_a_line_is_refused():
    ods = traced()
    entry = "plasma_initiation.b_field_lines.0"
    ods[f"{entry}.lengths"] = np.asarray(ods[f"{entry}.lengths"])[:-1]
    with pytest.raises(ValueError, match="each line needs all three"):
        RECIPES[NAME].builder(ods, **LINEAR)


# ---------------------------------------------------------------------------
# The registry contract
# ---------------------------------------------------------------------------


def test_the_plot_is_registered_under_its_own_subject():
    spec = registry.get_spec(NAME)
    assert spec.subject == "field_line_topology"
    assert spec.view == "field" and spec.quantity == "connection_length"
    assert spec.ids == ("plasma_initiation",)
    assert all(path.startswith("plasma_initiation.") for path in spec.required_paths)
    assert hasattr(vaft.plot, NAME)


def test_the_renderer_draws_onto_the_axes_it_is_given():
    import matplotlib.pyplot as plt

    model = RECIPES[NAME].builder(traced(), **LINEAR)
    figure, axes = plt.subplots()
    drawn_figure, drawn_axes = getattr(vaft.plot, NAME)(model, ax=axes)
    assert drawn_figure is figure and drawn_axes is axes
    plt.close(figure)


# ---------------------------------------------------------------------------
# The Field2D option the log map is built on
# ---------------------------------------------------------------------------


def test_field_2d_refuses_a_log_scale_over_non_positive_values():
    """Clipping them to the smallest colour would read as a small value;
    the caller masks them, so the map shows where there is none."""
    from vaft.plot.models import Field2D

    grid = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="strictly positive"):
        Field2D(r=grid, z=grid, values=np.array([[0.0, 1.0], [2.0, 3.0]]),
                value_scale="log")
    # NaN is not a value, so a masked cell is fine.
    Field2D(r=grid, z=grid, values=np.array([[np.nan, 1.0], [2.0, 3.0]]),
            value_scale="log")


def test_field_2d_refuses_a_scale_it_does_not_know():
    from vaft.plot.models import Field2D

    grid = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match='"linear" or "log"'):
        Field2D(r=grid, z=grid, values=np.ones((2, 2)), value_scale="symlog")


def test_the_plotly_backend_takes_the_decades_in_the_data():
    """Plotly contours have no logarithmic colour axis, so the decades are
    taken in the data and given back on the colorbar's ticks -- a reader
    still reads metres."""
    plotly = pytest.importorskip("plotly")  # noqa: F841
    from vaft.plot.plotly.fields import render_field_2d

    figure = render_field_2d(RECIPES[NAME].builder(traced()))
    trace = [t for t in figure.data if t.meta and t.meta.get("vaft") == "field"][0]
    drawn = np.asarray(trace.z, dtype=float)
    finite = drawn[np.isfinite(drawn)]
    assert finite.max() < 3.0, "the decades should be in the data, not the raw metres"
    assert trace.colorbar.ticktext, "the ticks must give the metres back"
