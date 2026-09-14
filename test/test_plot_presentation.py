"""Issue #689: plot geometry policy, ``format=`` and ``theme=``, opt-in.

Three separate things: how a kind of plot occupies a canvas, how large the
rendering is, and which visual grammar it uses.  What has to hold first is
that with neither preset every renderer draws exactly what it drew before
-- nothing in the suite asserted a figure size or an rcParam until now --
and that global Matplotlib state is untouched afterwards.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
from vaft.plot import renderers
from vaft.plot.backend.options import STYLE_OPTIONS, validate_options
from vaft.plot.models import (
    Field2D,
    GeometryLayer,
    GeometryLayers,
    Image2D,
    LineSeries,
    Panels,
    Profile1D,
    Series,
)
from vaft.plot.presentation import (
    EQUILIBRIUM_ROLE,
    FORMATS,
    GEOMETRY,
    THEMES,
    Presentation,
    resolve_presentation,
    rz_extent,
)

BASE_RENDERERS = (
    "render_line_series", "render_profile_1d", "render_field_2d", "render_geometry_layers",
    "render_geometry_3d_layers", "render_image_2d", "render_power_spectrum",
    "render_spectrogram", "render_panels",
)


@pytest.fixture(scope="module")
def sample():
    return vaft.omas.load(vaft.data.sample(39915, representation="omas"))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _minimal(name):
    from test_plot_contract import _minimal_model

    model_type = {
        "render_line_series": "LineSeries", "render_profile_1d": "Profile1D", "render_field_2d": "Field2D",
        "render_geometry_layers": "GeometryLayers", "render_geometry_3d_layers": "Geometry3DLayers",
        "render_image_2d": "Image2D", "render_power_spectrum": "PowerSpectrum",
        "render_spectrogram": "Spectrogram", "render_panels": "Panels",
    }[name]
    import vaft.plot.models as models

    return _minimal_model(getattr(models, model_type))


def _fingerprint(figure, axes):
    """What a reader would notice: canvas, fonts, the first line's look."""
    first = None
    for axis in np.asarray(axes, dtype=object).ravel():
        if getattr(axis, "lines", None):
            line = axis.lines[0]
            first = (line.get_linewidth(), line.get_color(), line.get_linestyle(), line.get_marker())
            break
    axis0 = np.asarray(axes, dtype=object).ravel()[0]
    return (
        tuple(figure.get_size_inches()),
        axis0.xaxis.label.get_fontsize(),
        axis0.title.get_fontsize(),
        first,
    )


def _wall(z_half: float, r_min: float = 0.1, r_max: float = 0.8) -> GeometryLayer:
    return GeometryLayer(
        r=np.array([r_min, r_max, r_max, r_min]), z=np.array([-z_half, -z_half, z_half, z_half]),
        kind="polygon", label="wall",
    )


def _boundary(z_half: float) -> GeometryLayer:
    theta = np.linspace(0, 2 * np.pi, 40)
    return GeometryLayer(
        r=0.45 + 0.2 * np.cos(theta), z=z_half * np.sin(theta), kind="polygon", label="Boundary",
        role=EQUILIBRIUM_ROLE,
    )


# ---------------------------------------------------------------------------
# the opt-in promise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", BASE_RENDERERS)
def test_no_preset_draws_exactly_what_it_drew_before(name):
    from vaft.plot.renderers import geometry

    render = getattr(renderers, name, None) or getattr(geometry, name)
    plain = render(_minimal(name), show=False)
    explicit = render(_minimal(name), show=False, format=None, theme=None)
    assert _fingerprint(plain[0], plain[1]) == _fingerprint(explicit[0], explicit[1])


def test_the_canonical_plots_are_untouched_without_a_preset(sample):
    before = dict(matplotlib.rcParams)
    for call in (
        lambda **k: vaft.omas.plot_plasma_current_time(sample, **k),
        lambda **k: vaft.omas.plot_equilibrium_field_psi(sample, **k),
        lambda **k: vaft.omas.plot_flux_loop_time_flux(sample, layout="subplots", **k),
    ):
        plain = call()
        explicit = call(format=None, theme=None)
        assert _fingerprint(*plain) == _fingerprint(*explicit)
    assert dict(matplotlib.rcParams) == before


def test_global_rcparams_are_what_they_were_afterwards(sample):
    before = dict(matplotlib.rcParams)
    vaft.omas.plot_plasma_current_time(sample, format="single_column", theme="monochrome")
    assert dict(matplotlib.rcParams) == before
    with pytest.raises(TypeError):
        renderers.render_line_series(object(), theme="technical")  # raised inside the context
    assert dict(matplotlib.rcParams) == before


# ---------------------------------------------------------------------------
# format: how large
# ---------------------------------------------------------------------------

def test_formats_resolve_deterministic_widths():
    assert FORMATS["single_column"].width_in == 3.375
    assert FORMATS["double_column"].width_in == 7.0
    assert FORMATS["screen"].width_in == 6.5
    model = _minimal("render_line_series")
    for name, width in (("single_column", 3.375), ("double_column", 7.0), ("screen", 6.5)):
        sizes = {tuple(renderers.render_line_series(model, format=name)[0].get_size_inches()) for _ in range(2)}
        assert len(sizes) == 1 and next(iter(sizes))[0] == width
    # screen is a canonical width, not an alias of the legacy default.
    assert tuple(renderers.render_line_series(model)[0].get_size_inches()) != tuple(
        renderers.render_line_series(model, format="screen")[0].get_size_inches()
    )


def test_geometries_take_different_heights_under_one_format():
    heights = {}
    for name in ("render_line_series", "render_profile_1d", "render_field_2d"):
        figure, _ = getattr(renderers, name)(_minimal(name), format="double_column")
        heights[name] = figure.get_size_inches()[1]
    assert heights["render_line_series"] < heights["render_profile_1d"] < heights["render_field_2d"]


def test_panel_count_does_not_multiply_the_width():
    x = np.linspace(0, 1, 5)
    widths = set()
    for ncols in (1, 2, 3):
        model = Panels(
            models=tuple(LineSeries(series=(Series(x=x, y=x * i, label=str(i)),), y_label="y") for i in range(3)),
            ncols=ncols,
        )
        widths.add(renderers.render_panels(model, format="double_column")[0].get_size_inches()[0])
    assert widths == {7.0}


def test_a_time_trace_is_the_landscape_strip_its_policy_says():
    figure, _ = renderers.render_line_series(_minimal("render_line_series"), format="screen")
    assert tuple(figure.get_size_inches()) == pytest.approx((6.5, 6.5 * GEOMETRY["LineSeries"].aspect))
    figure, _ = renderers.render_spectrogram(_minimal("render_spectrogram"), format="screen")
    assert figure.get_size_inches()[1] == pytest.approx(6.5 * GEOMETRY["Spectrogram"].aspect)


def test_minimal_really_switches_the_grid_off():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        figure, axes = renderers.render_line_series(_minimal("render_line_series"), theme="minimal")
        _, caller_axes = plt.subplots()
        renderers.render_line_series(_minimal("render_line_series"), ax=caller_axes, theme="minimal")
    for axis in (axes, caller_axes):
        assert not any(line.get_visible() for line in axis.get_xgridlines())


def test_format_and_type_sizes_compose():
    model = _minimal("render_line_series")
    _, small = renderers.render_line_series(model, format="single_column")
    _, screen = renderers.render_line_series(model, format="screen")
    assert small.xaxis.label.get_fontsize() == 8.0 and screen.xaxis.label.get_fontsize() == 10.0
    assert small.lines[0].get_linewidth() < screen.lines[0].get_linewidth()


# ---------------------------------------------------------------------------
# R-Z geometry: equal coordinates, height from the machine, never the LCFS
# ---------------------------------------------------------------------------

def test_an_rz_view_takes_its_height_from_the_machine():
    tall = GeometryLayers(layers=(_wall(1.2), _boundary(0.3)))
    short = GeometryLayers(layers=(_wall(0.5), _boundary(0.3)))
    size_tall = renderers.render_geometry_layers(tall, format="single_column")[0].get_size_inches()
    size_short = renderers.render_geometry_layers(short, format="single_column")[0].get_size_inches()
    assert size_tall[1] > size_short[1]
    # The format's width is a maximum for an R-Z view: when the height ceiling
    # binds, the canvas narrows with it rather than framing the axes in margin.
    assert size_short[0] == 3.375 and size_tall[0] <= 3.375
    assert size_tall[1] != size_tall[0], "equal coordinate scaling is not a square canvas"
    figure, axes = renderers.render_geometry_layers(tall, format="single_column")
    assert axes.get_aspect() == 1.0


def test_the_boundary_alone_never_sizes_the_canvas():
    with_small = GeometryLayers(layers=(_wall(1.0), _boundary(0.2)))
    with_large = GeometryLayers(layers=(_wall(1.0), _boundary(0.9)))
    a = renderers.render_geometry_layers(with_small, format="screen")[0].get_size_inches()
    b = renderers.render_geometry_layers(with_large, format="screen")[0].get_size_inches()
    np.testing.assert_array_equal(a, b)
    assert rz_extent(GeometryLayers(layers=(_boundary(0.9),))) is None
    from vaft.plot import presentation as pres

    only_boundary = renderers.render_geometry_layers(GeometryLayers(layers=(_boundary(0.9),)), format="screen")[0]
    # The portrait ratio the R-Z renderers always used, fitted snugly.
    expected = pres._snug(6.5, pres._RZ_FALLBACK_ASPECT, pres._RZ_AXES_FRACTION, FORMATS["screen"].max_height_in)
    assert tuple(only_boundary.get_size_inches()) == pytest.approx(expected)


def test_a_field_takes_its_extent_from_its_overlays_then_its_grid():
    r, z = np.linspace(0.1, 0.8, 5), np.linspace(-0.5, 0.5, 6)
    bare = Field2D(r=r, z=z, values=np.outer(z, r), value_label="psi")
    assert rz_extent(bare) == pytest.approx((0.7, 1.0))
    with_wall = Field2D(r=r, z=z, values=np.outer(z, r), value_label="psi", overlays=(_wall(1.2),))
    assert rz_extent(with_wall) == pytest.approx((0.7, 2.4))
    with_boundary = Field2D(r=r, z=z, values=np.outer(z, r), value_label="psi", overlays=(_boundary(0.9),))
    assert rz_extent(with_boundary) == pytest.approx((0.7, 1.0))


def test_the_packaged_psi_map_is_sized_by_the_wall(sample):
    figure, axes = vaft.omas.plot_equilibrium_field_psi(sample, format="double_column")
    width, height = figure.get_size_inches()
    assert width <= 7.0 and height > width
    assert axes.get_aspect() == 1.0
    # The canvas follows the axes: most of the width is drawn on, not margin.
    figure.canvas.draw()
    assert axes.get_position().width > 0.45


def test_an_image_keeps_its_pixel_ratio():
    wide = Image2D(values=np.zeros((10, 40)), value_label="counts")
    tall = Image2D(values=np.zeros((40, 10)), value_label="counts")
    w = renderers.render_image_2d(wide, format="screen")[0].get_size_inches()
    t = renderers.render_image_2d(tall, format="screen")[0].get_size_inches()
    assert w[1] < t[1]
    assert GEOMETRY["Image2D"].kind == "native"


# ---------------------------------------------------------------------------
# theme: which visual grammar
# ---------------------------------------------------------------------------

def test_themes_resolve_deterministic_tokens():
    tokens = {name: Presentation(None, theme).rc() for name, theme in THEMES.items()}
    assert set(tokens) == {"technical", "minimal", "monochrome"}
    assert tokens["technical"] == Presentation(None, THEMES["technical"]).rc()
    assert tokens["technical"]["xtick.direction"] == "in" and tokens["minimal"]["xtick.direction"] == "out"
    assert tokens["minimal"]["axes.spines.top"] is False and tokens["technical"]["axes.spines.top"] is True
    assert "accessible" not in THEMES


def test_monochrome_does_not_rely_on_colour():
    x = np.linspace(0, 1, 50)
    model = LineSeries(series=tuple(Series(x=x, y=x * i, label=f"s{i}") for i in range(6)), y_label="y")
    figure, axes = renderers.render_line_series(model, theme="monochrome")
    looks = [(line.get_linestyle(), line.get_marker()) for line in axes.lines[:6]]
    assert len(set(looks)) == 6
    for line in axes.lines[:6]:
        r, g, b, _ = matplotlib.colors.to_rgba(line.get_color())
        assert r == g == b, "monochrome draws in greys"


def test_technical_draws_from_its_own_cycle():
    figure, axes = renderers.render_line_series(_minimal("render_line_series"), theme="technical")
    assert matplotlib.colors.to_hex(axes.lines[0].get_color()) == "#000000"
    assert axes.xaxis.majorTicks[0]._tickdir == "in"
    _, minimal_axes = renderers.render_line_series(_minimal("render_line_series"), theme="minimal")
    assert minimal_axes.xaxis.majorTicks[0]._tickdir == "out"
    assert not minimal_axes.spines["top"].get_visible()
    # A geometry stack with several entries colours them from the theme cycle.
    stack = GeometryLayers(layers=(
        GeometryLayer(r=np.array([0.1, 0.8]), z=np.array([0.0, 0.0]), label="a", entry="one"),
        GeometryLayer(r=np.array([0.1, 0.8]), z=np.array([0.2, 0.2]), label="b", entry="two"),
    ))
    _, axes = renderers.render_geometry_layers(stack, theme="technical")
    colours = {matplotlib.colors.to_hex(line.get_color()) for line in axes.lines}
    assert colours <= {c.lower() for c in THEMES["technical"].colors}


def test_a_theme_leaves_the_scientific_semantics_alone(sample):
    plain = vaft.omas.plot_flux_loop_time_flux(sample, selection="all")
    themed = vaft.omas.plot_flux_loop_time_flux(sample, selection="all", theme="monochrome")
    for axes in (plain[1], themed[1]):
        assert axes.get_ylabel() == plain[1].get_ylabel()
    assert themed[1].get_ylim() == plain[1].get_ylim()
    invalid = [line for line in themed[1].lines if "(invalid)" in line.get_label()]
    plain_invalid = [line for line in plain[1].lines if "(invalid)" in line.get_label()]
    assert len(invalid) == len(plain_invalid)
    for line in invalid:
        assert line.get_color() == "0.65" and line.get_linestyle() == "--"


# ---------------------------------------------------------------------------
# ownership and the paths that cannot apply a preset
# ---------------------------------------------------------------------------

def test_a_caller_owned_canvas_is_not_taken(sample):
    figure, axis = plt.subplots(figsize=(4.0, 3.0))
    before = set(plt.get_fignums())
    with pytest.raises(TypeError, match="ax="):
        vaft.omas.plot_plasma_current_time(sample, ax=axis, format="single_column")
    vaft.omas.plot_plasma_current_time(sample, ax=axis, theme="technical")
    assert tuple(figure.get_size_inches()) == (4.0, 3.0)
    assert set(plt.get_fignums()) == before
    assert matplotlib.colors.to_hex(axis.lines[0].get_color()) == "#000000"


def test_figsize_and_format_together_are_refused():
    with pytest.raises(ValueError, match="figsize= and format="):
        renderers.render_line_series(_minimal("render_line_series"), figsize=(3, 3), format="screen")
    with pytest.raises(ValueError, match="format must be one of"):
        resolve_presentation("a4", None)
    with pytest.raises(ValueError, match="theme must be one of"):
        resolve_presentation(None, "dark")
    assert resolve_presentation(None, None) is None


def test_paths_that_cannot_apply_a_preset_say_so(sample):
    pytest.importorskip("plotly")
    with pytest.raises(NotImplementedError, match="backend='plotly'"):
        vaft.omas.plot_plasma_current_time(sample, backend="plotly", format="single_column")
    with pytest.raises(NotImplementedError, match="interactive=True"):
        vaft.omas.plot_plasma_current_time(sample, interactive=True, interaction_backend="none", theme="minimal")


def test_the_option_schema_knows_the_presets_and_offers_no_control(sample):
    assert {"format", "theme"} <= STYLE_OPTIONS
    validate_options("plasma_current_time", {"format": "single_column", "theme": "technical"})
    from vaft.plot.controls import controls_for

    record = next(r for r in vaft.omas.available_plots(sample) if r.name == "plasma_current_time")
    assert not {"format", "theme"} & {c.name for c in controls_for(record)}
