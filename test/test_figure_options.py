"""Reproducible figure options, the typography contract and the slide/poster formats (issue #1421)."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft.omas as vo
from vaft.cli.plot import _parser
from vaft.plot import FigureOptions, renderers
from vaft.plot.figure_options import as_figure_options, reproduce_cli, reproduce_python
from vaft.plot.models import Field2D, LineSeries, Series
from vaft.plot import presentation as pres


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _lines(n=3):
    t = np.linspace(0.0, 1.0, 50)
    return LineSeries(
        series=tuple(Series(x=t, y=np.sin(t * (k + 1)), label=f"trace {k}") for k in range(n)),
        x_label="time", x_unit="s", y_label=r"$I_p$", y_unit="kA", title="Plasma current",
    )


def _field():
    r = np.linspace(0.1, 0.9, 30)
    z = np.linspace(-0.8, 0.8, 40)
    values = np.sin(3 * r)[None, :] * np.cos(2 * z)[:, None]
    return Field2D(r=r, z=z, values=values, value_label="signed field")


# ---------------------------------------------------------------------------
# intent: only what is set, and the same thing in every spelling
# ---------------------------------------------------------------------------

def test_an_empty_options_object_is_no_options():
    assert not FigureOptions()
    assert as_figure_options(FigureOptions()) is None
    assert as_figure_options({}) is None
    assert FigureOptions().cli_args() == []


def test_only_explicit_fields_are_serialised():
    options = FigureOptions(xlim=[0.2, 0.4], legend_ncols=2)
    assert options.explicit() == {"xlim": (0.2, 0.4), "legend_ncols": 2}
    assert options.python() == "FigureOptions(xlim=(0.2, 0.4), legend_ncols=2)"


ROUND_TRIP = FigureOptions(
    font_family=("Arial", "DejaVu Sans"), math_fontset="stixsans", font_size=11,
    xlim=(0.25, 0.35), yscale="log", xlabel="t [s]", title="",
    tick_direction="out", minor_ticks=True, mirror_ticks=False, grid=True,
    legend=True, legend_loc="upper left", legend_ncols=2, legend_frame=False,
    clim=(-1, 1), cmap="RdBu_r", norm="centered", panel_labels=True, dpi=600, transparent=True,
)


def test_cli_and_python_reproduce_the_same_options():
    assert FigureOptions.from_cli(ROUND_TRIP.cli_args()[1::2]) == ROUND_TRIP
    assert eval(ROUND_TRIP.python(), {"FigureOptions": FigureOptions}) == ROUND_TRIP


def test_the_reproduced_command_parses_back_through_vaft_plot():
    args = reproduce_cli(
        "plasma_current_time", 39915, format="slide", theme="minimal",
        figure_options=ROUND_TRIP, out="figure.pdf",
    )
    assert args[:3] == ["vaft", "plot", "plasma_current_time"]
    parsed = _parser().parse_args(args[2:])
    assert (parsed.shot, parsed.format, parsed.theme, parsed.out) == ([39915], "slide", "minimal", "figure.pdf")
    assert FigureOptions.from_cli(parsed.figure_option) == ROUND_TRIP


def test_inherited_presets_are_not_written_out():
    args = reproduce_cli("plasma_current_time", 39915)
    assert args == ["vaft", "plot", "plasma_current_time", "--shot", "39915"]
    code = reproduce_python("plasma_current_time", 39915, theme="technical")
    assert "format=" not in code and "figure_options" not in code
    assert code.endswith("vaft.database.plot_plasma_current_time(39915, theme='technical')")


@pytest.mark.parametrize("bad", [
    {"xlim": (1, 1)}, {"xscale": "logit"}, {"norm": "power"}, {"legend_loc": (0.1, 0.2)},
    {"cmap": "no_such_map"}, {"dpi": 0}, {"legend_ncols": 0}, {"math_fontset": "comic"},
    {"colour": "red"},
])
def test_a_bad_value_is_refused_at_construction(bad):
    with pytest.raises(ValueError):
        FigureOptions.from_dict(bad)


# ---------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------

def test_axes_ticks_and_legend_options_reach_the_figure():
    options = FigureOptions(
        xlim=(0.2, 0.8), yscale="symlog", ylabel="custom", minor_ticks=True,
        mirror_ticks=True, legend_loc="lower left", legend_ncols=2, legend_frame=False,
    )
    figure, axes = renderers.render_line_series(_lines(), figure_options=options)
    assert axes.get_xlim() == (0.2, 0.8)
    assert axes.get_yscale() == "symlog" and axes.get_ylabel() == "custom"
    assert axes.xaxis.get_minor_locator().__class__.__name__ != "NullLocator"
    legend = axes.get_legend()
    assert legend._ncols == 2 and not legend.get_frame_on()
    assert [t.get_text() for t in legend.get_texts()] == ["trace 0", "trace 1", "trace 2"]


def test_legend_false_removes_it_and_title_empty_hides_it():
    figure, axes = renderers.render_line_series(_lines(), figure_options={"legend": False, "title": ""})
    assert axes.get_legend() is None
    assert axes.get_title() == ""


def test_a_lone_trace_gains_no_legend_from_a_placement_alone():
    figure, axes = renderers.render_line_series(_lines(1), figure_options={"legend_loc": "upper left"})
    assert axes.get_legend() is None


def test_colour_scale_options_reach_the_mappable_and_its_colorbar():
    options = FigureOptions(clim=(-0.5, 0.5), cmap="RdBu_r", norm="centered")
    figure, axes = renderers.render_field_2d(_field(), figure_options=options)
    mappables = [a for ax in figure.axes for a in (*ax.images, *ax.collections) if a.get_array() is not None]
    assert mappables
    for mappable in mappables:
        assert mappable.get_cmap().name == "RdBu_r"
        assert isinstance(mappable.norm, matplotlib.colors.CenteredNorm)
        assert mappable.norm.vcenter == 0.0


def test_panel_labels_go_on_each_data_axes_of_a_composite():
    from vaft.plot.models import Panels

    figure, axes = renderers.render_panels(
        Panels(models=(_lines(), _lines(), _field()), ncols=2), figure_options={"panel_labels": True},
    )
    labels = sorted(
        t.get_text() for ax in figure.axes for t in ax.texts if t.get_gid() == "vaft-panel-label"
    )
    assert labels == ["(a)", "(b)", "(c)"]


def test_typography_overrides_win_over_the_format_and_keep_its_ratios():
    figure, axes = renderers.render_line_series(
        _lines(), format="single_column", figure_options={"font_size": 12, "math_fontset": "cm"},
    )
    fmt = pres.FORMATS["single_column"]
    assert axes.xaxis.label.get_size() == pytest.approx(12 * fmt.label_scale)
    assert axes.get_xticklabels()[0].get_size() == pytest.approx(12 * fmt.tick_scale)
    assert matplotlib.rcParams["mathtext.fontset"] != "cm", "the override leaked out of the render"


def test_options_apply_on_a_caller_owned_axes_too():
    figure, ax = plt.subplots()
    renderers.render_line_series(_lines(), ax=ax, figure_options={"xlim": (0.1, 0.9)})
    assert ax.get_xlim() == (0.1, 0.9)


def test_the_adapters_and_options_validation_accept_them():
    ods = vo.sample_ods()
    figure, axes = vo.plot_plasma_current_time(ods, figure_options={"xlim": (0.25, 0.35)})
    axis = axes if hasattr(axes, "get_xlim") else np.ravel(axes)[0]
    assert axis.get_xlim() == (0.25, 0.35)
    with pytest.raises(TypeError, match="figure_options"):
        vo.extract_plasma_current_time(ods, figure_options={"xlim": (0.25, 0.35)})
    with pytest.raises(NotImplementedError, match="figure_options"):
        vo.plot_plasma_current_time(ods, backend="plotly", figure_options={"xlim": (0.2, 0.3)})


def test_save_applies_the_export_fields_and_embeds_truetype(tmp_path):
    figure, _ = renderers.render_line_series(_lines())
    path = FigureOptions(dpi=72, transparent=True).save(figure, tmp_path / "f.pdf")
    data = path.read_bytes()
    assert b"/FontFile2" in data, "PDF text is not embedded as TrueType (Type 42)"
    figure, _ = renderers.render_line_series(_lines())
    png = FigureOptions(dpi=50, transparent=True).save(figure, tmp_path / "f.png", bbox_inches=None)
    image = plt.imread(png)
    assert image.shape[2] == 4 and image[0, 0, 3] == 0.0, "transparent= did not reach savefig"
    width_px = image.shape[1]
    assert width_px == pytest.approx(pres.FORMATS["screen"].width_in * 50, abs=2)


# ---------------------------------------------------------------------------
# typography contract and fonts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(pres.THEMES))
def test_every_theme_names_a_math_font_and_a_stack_ending_in_the_bundled_font(name):
    theme = pres.THEMES[name]
    assert theme.math_fontset in ("dejavusans", "stixsans", "stix", "cm", "dejavuserif")
    rc = pres.Presentation(format=None, theme=theme).rc()
    assert rc["mathtext.fontset"] == theme.math_fontset
    assert rc["font.family"][-1] == "DejaVu Sans"


def test_a_missing_font_is_dropped_and_warned_about_once(monkeypatch):
    monkeypatch.setattr(pres, "_WARNED_FALLBACKS", set())
    stack = ("No Such Font 1421", "Another Missing 1421", "DejaVu Sans")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert pres.resolve_font_family(stack) == ["DejaVu Sans"]
        assert pres.resolve_font_family(stack) == ["DejaVu Sans"]
    fallbacks = [w for w in caught if issubclass(w.category, pres.FontFallbackWarning)]
    assert len(fallbacks) == 1
    assert "No Such Font 1421" in str(fallbacks[0].message) and "DejaVu Sans" in str(fallbacks[0].message)


def test_without_helvetica_the_minimal_theme_falls_back_in_order(monkeypatch):
    """What a plain Linux host (vestserver) sees: no Helvetica, no Arial."""
    from matplotlib import font_manager

    present = [e for e in font_manager.fontManager.ttflist if e.name not in ("Helvetica", "Arial")]
    monkeypatch.setattr(font_manager.fontManager, "ttflist", present)
    monkeypatch.setattr(pres, "_WARNED_FALLBACKS", set())
    with pytest.warns(pres.FontFallbackWarning, match="Helvetica"):
        family = pres.resolve_font_family(pres.THEMES["minimal"].font_family)
    assert "Helvetica" not in family and "Arial" not in family
    assert family[-1] == "DejaVu Sans"


def test_no_font_file_ships_with_vaft():
    from pathlib import Path

    import vaft

    root = Path(vaft.__file__).parent
    shipped = [p for p in root.rglob("*") if p.suffix.lower() in (".ttf", ".otf", ".woff", ".woff2")]
    assert shipped == []


# ---------------------------------------------------------------------------
# slide and poster
# ---------------------------------------------------------------------------

def test_the_existing_formats_are_unchanged():
    assert pres.DEFAULT_FORMAT == "screen"
    assert (pres.FORMATS["screen"].width_in, pres.FORMATS["screen"].base_font_pt) == (6.5, 10.0)
    assert (pres.FORMATS["single_column"].width_in, pres.FORMATS["single_column"].base_font_pt) == (3.375, 8.0)
    assert (pres.FORMATS["double_column"].width_in, pres.FORMATS["double_column"].base_font_pt) == (7.0, 8.0)


@pytest.mark.parametrize("name", ["slide", "poster"])
def test_presentation_formats_are_large_type_and_heavy_lines(name):
    fmt = pres.FORMATS[name]
    assert fmt.base_font_pt >= 18 and fmt.line_scale >= 2.0
    figure, axes = renderers.render_line_series(_lines(), format=name, theme="technical")
    width, height = figure.get_size_inches()
    assert width <= fmt.width_in + 1e-9 and height <= fmt.max_height_in + 1e-9
    assert axes.xaxis.label.get_size() == pytest.approx(fmt.base_font_pt * fmt.label_scale)
    assert axes.lines[0].get_linewidth() == pytest.approx(pres.THEMES["technical"].line_pt * fmt.line_scale)


def test_a_slide_figure_fits_a_16_by_9_slide():
    fmt = pres.FORMATS["slide"]
    assert fmt.width_in <= 13.333 and fmt.max_height_in <= 7.5
