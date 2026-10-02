"""Slide/poster formats, font fallback and the norm/panel/export figure options (issue #1421).

The follow-up to #1483: what the conference figures need on top of the
library part -- presentation formats for a projector and a poster, a font
stack that degrades quietly on a host without Helvetica, and the remaining
reproducible options from the issue (colour normalisation, panel marks,
export resolution and background).
"""

from __future__ import annotations

import json
import shlex
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
from vaft.plot import DataSource, FigureComposition, FigureOptions, PlotRequest, renderers, save_figure
from vaft.plot import presentation as pres
from vaft.plot._panel_grid import PANEL_LABEL_GID
from vaft.plot.models import Field2D, Image2D, LineSeries, Series


@pytest.fixture(scope="module")
def ods():
    return vaft.omas.sample_ods(39915)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _marks(figure):
    return sorted(t.get_text() for ax in figure.axes for t in ax.texts if t.get_gid() == PANEL_LABEL_GID)


def _signed_field():
    r = np.linspace(0.1, 0.9, 30)
    z = np.linspace(-0.8, 0.8, 40)
    return Field2D(r=r, z=z, values=0.3 + np.sin(3 * r)[None, :] * np.cos(2 * z)[:, None], value_label="b")


# --- formats ----------------------------------------------------------------------


def test_the_existing_formats_are_unchanged():
    assert pres.DEFAULT_FORMAT == "screen"
    assert (pres.FORMATS["screen"].width_in, pres.FORMATS["screen"].base_font_pt) == (6.5, 10.0)
    assert (pres.FORMATS["single_column"].width_in, pres.FORMATS["single_column"].base_font_pt) == (3.375, 8.0)
    assert (pres.FORMATS["double_column"].width_in, pres.FORMATS["double_column"].base_font_pt) == (7.0, 8.0)


@pytest.mark.parametrize("name", ["slide", "poster"])
def test_presentation_formats_are_large_type_and_heavy_lines(ods, name):
    fmt = pres.FORMATS[name]
    assert fmt.base_font_pt >= 18 and fmt.line_scale >= 2.0
    figure, axes = vaft.omas.plot_plasma_current_time(ods, format=name, theme="technical")
    width, height = figure.get_size_inches()
    assert width <= fmt.width_in + 1e-9 and height <= fmt.max_height_in + 1e-9
    assert axes.xaxis.label.get_size() == pytest.approx(fmt.base_font_pt * fmt.label_scale)
    assert axes.get_lines()[0].get_linewidth() == pytest.approx(pres.THEMES["technical"].line_pt * fmt.line_scale)


def test_a_slide_figure_fits_a_16_by_9_slide_and_the_cli_names_it():
    from vaft.cli.plot import _parser

    fmt = pres.FORMATS["slide"]
    assert fmt.width_in <= 13.333 and fmt.max_height_in <= 7.5
    assert "slide" in _parser().format_help() and "poster" in _parser().format_help()


# --- fonts ------------------------------------------------------------------------


def test_every_theme_stack_ends_in_the_bundled_font():
    for theme in pres.THEMES.values():
        assert pres.Presentation(format=None, theme=theme).rc()["font.family"][-1] == "DejaVu Sans"
    assert "Liberation Sans" in pres.THEMES["minimal"].font_family


def test_a_missing_font_is_dropped_and_warned_about_once(monkeypatch):
    monkeypatch.setattr(pres, "_WARNED_FALLBACKS", set())
    stack = ("No Such Font 1421", "Another Missing 1421", "DejaVu Sans")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert pres.resolve_font_family(stack) == ["DejaVu Sans"]
        assert pres.resolve_font_family(stack) == ["DejaVu Sans"]
    fallbacks = [w for w in caught if issubclass(w.category, pres.FontFallbackWarning)]
    assert len(fallbacks) == 1 and "No Such Font 1421" in str(fallbacks[0].message)


def test_without_helvetica_the_minimal_theme_falls_back_in_order(monkeypatch):
    """What a plain Linux host (vestserver) sees: no Helvetica, no Arial."""
    from matplotlib import font_manager

    present = [e for e in font_manager.fontManager.ttflist if e.name not in ("Helvetica", "Arial")]
    monkeypatch.setattr(font_manager.fontManager, "ttflist", present)
    monkeypatch.setattr(pres, "_WARNED_FALLBACKS", set())
    with pytest.warns(pres.FontFallbackWarning, match="Helvetica"):
        family = pres.resolve_font_family(pres.THEMES["minimal"].font_family)
    assert "Helvetica" not in family and "Arial" not in family and family[-1] == "DejaVu Sans"


def test_a_font_option_goes_through_the_same_fallback(monkeypatch):
    monkeypatch.setattr(pres, "_WARNED_FALLBACKS", set())
    with pytest.warns(pres.FontFallbackWarning):
        rc = FigureOptions(font_family=("No Such Font 1421",)).rc()
    assert rc["font.family"] == ["DejaVu Sans"]


def test_no_font_file_ships_with_vaft():
    root = Path(vaft.__file__).parent
    assert [p for p in root.rglob("*") if p.suffix.lower() in (".ttf", ".otf", ".woff", ".woff2")] == []


# --- norm -------------------------------------------------------------------------


def test_a_contour_map_is_drawn_under_the_norm_with_levels_to_match():
    from vaft.plot.figure_options import figure_options_scope

    with figure_options_scope(FigureOptions(norm="centered")):
        figure, axes = renderers.render_field_2d(_signed_field())
    contour = next(m for m in axes.collections if getattr(m, "colorbar", None) is not None)
    assert isinstance(contour.norm, matplotlib.colors.CenteredNorm) and contour.norm.vcenter == 0.0
    assert contour.norm.halfrange == pytest.approx(np.nanmax(np.abs(_signed_field().values)))


def test_norm_through_an_adapter_and_a_centred_clim(ods):
    figure, _ = vaft.omas.plot_equilibrium_field_2d(ods, figure_options={"norm": "centered", "clim": (-2, 4)})
    contour = next(m for ax in figure.axes for m in ax.collections if getattr(m, "colorbar", None) is not None)
    assert contour.norm.vcenter == pytest.approx(1.0) and contour.norm.halfrange == pytest.approx(3.0)


def test_an_image_takes_the_norm_after_drawing():
    model = Image2D(values=np.arange(1, 13.0).reshape(3, 4))
    figure, axes = renderers.render_image_2d(model)
    FigureOptions(norm="log").apply(figure)
    images = [im for im in axes.images if getattr(im, "colorbar", None) is not None]
    assert images
    for image in images:
        assert isinstance(image.norm, matplotlib.colors.LogNorm)
        assert image.norm.vmin == 1.0 and image.norm.vmax == 12.0


def test_a_log_norm_refuses_what_it_cannot_draw():
    with pytest.raises(ValueError, match="positive clim"):
        FigureOptions(norm="log", clim=(-1, 1))
    with pytest.raises(ValueError, match="norm must be one of"):
        FigureOptions(norm="power")
    r = np.linspace(0, 1, 5)
    negative = Field2D(r=r, z=r, values=-np.ones((5, 5)) - r[None, :])
    from vaft.plot.figure_options import figure_options_scope

    with figure_options_scope(FigureOptions(norm="log")), pytest.warns(UserWarning, match="positive values"):
        figure, axes = renderers.render_field_2d(negative)
    contour = next(m for m in axes.collections if getattr(m, "colorbar", None) is not None)
    assert not isinstance(contour.norm, matplotlib.colors.LogNorm)


def test_no_norm_keeps_the_canonical_colours(ods):
    plain, _ = vaft.omas.plot_equilibrium_field_2d(ods)
    other, _ = vaft.omas.plot_equilibrium_field_2d(ods, figure_options={"title": "x"})
    first = next(m for ax in plain.axes for m in ax.collections if getattr(m, "colorbar", None) is not None)
    second = next(m for ax in other.axes for m in ax.collections if getattr(m, "colorbar", None) is not None)
    assert type(first.norm) is type(second.norm) and list(first.levels) == list(second.levels)


# --- panel labels -----------------------------------------------------------------


def test_a_single_plot_gets_one_mark(ods):
    figure, _ = vaft.omas.plot_plasma_current_time(ods, figure_options={"panel_labels": True})
    assert _marks(figure) == ["(a)"]


def test_an_overview_is_marked_once_per_panel(ods):
    figure, axes = vaft.omas.plot_equilibrium_overview(ods, figure_options={"panel_labels": True})
    marks = _marks(figure)
    assert len(marks) == len(set(marks)) and marks[0] == "(a)"


def test_a_composition_mark_follows_the_option_both_ways(ods):
    stacked = FigureComposition.stack(["plasma_current_time", "equilibrium_time_q95"])
    figure, _ = vaft.omas.compose(stacked, ods, figure_options={"panel_labels": True})
    assert _marks(figure) == ["(a)", "(b)"]
    labelled = FigureComposition.from_dict({**stacked.to_dict(), "panel_labels": True})
    figure, _ = vaft.omas.compose(labelled, ods, figure_options={"panel_labels": False})
    assert _marks(figure) == []
    figure, _ = vaft.omas.compose(labelled, ods, figure_options={"panel_labels": True})
    assert _marks(figure) == ["(a)", "(b)"]


# --- export -----------------------------------------------------------------------


def test_save_figure_reads_the_export_fields_and_a_keyword_still_wins(tmp_path):
    t = np.linspace(0, 1, 20)
    model = LineSeries(series=(Series(x=t, y=t),), x_label="t", y_label="y")
    figure, _ = renderers.render_line_series(model, format="screen")
    path = save_figure(figure, tmp_path / "a.png", figure_options={"dpi": 40, "transparent": True}, bbox_inches=None)
    image = plt.imread(path)
    assert image.shape[1] == pytest.approx(pres.FORMATS["screen"].width_in * 40, abs=2)
    assert image[0, 0, 3] == 0.0
    figure, _ = renderers.render_line_series(model, format="screen")
    path = save_figure(figure, tmp_path / "b.png", figure_options={"dpi": 40}, dpi=20, bbox_inches=None)
    assert plt.imread(path).shape[1] == pytest.approx(pres.FORMATS["screen"].width_in * 20, abs=2)


def test_drawing_ignores_the_export_fields(ods):
    plain, _ = vaft.omas.plot_plasma_current_time(ods)
    exported, _ = vaft.omas.plot_plasma_current_time(ods, figure_options={"dpi": 600, "transparent": True})
    assert plain.get_size_inches().tolist() == exported.get_size_inches().tolist()
    assert plain.dpi == exported.dpi


def test_the_cli_writes_with_the_requested_dpi(tmp_path):
    from vaft.cli.plot import main

    request = PlotRequest(DataSource("sample", 39915), plot="plasma_current_time", format="screen",
                          figure_options=FigureOptions(dpi=30, panel_labels=True, norm="linear"))
    out = tmp_path / "ip.png"
    command = request.to_cli(out=str(out))
    assert '"dpi":30' in command.replace(" ", "")
    assert main(shlex.split(command)[2:]) == 0
    assert plt.imread(out).shape[1] < pres.FORMATS["screen"].width_in * 40


def test_the_new_fields_are_intent_like_the_others():
    options = FigureOptions(norm="centered", panel_labels=True, dpi=600, transparent=False)
    data = options.to_dict()
    assert data == {"norm": "centered", "panel_labels": True, "dpi": 600.0, "transparent": False}
    assert FigureOptions.from_dict(json.loads(json.dumps(data))) == options
    request = PlotRequest(DataSource("sample", 39915), plot="plasma_current_time", figure_options=options)
    assert PlotRequest.from_dict(request.to_dict()) == request
    assert "'norm': 'centered'" in request.to_python()


def test_plotly_names_what_it_does_not_apply_and_takes_a_transparent_page(ods):
    with pytest.warns(UserWarning, match="norm.*panel_labels.*dpi|panel_labels|norm"):
        figure = vaft.omas.plot_plasma_current_time(
            ods, backend="plotly", figure_options={"norm": "linear", "panel_labels": True, "dpi": 300, "transparent": True},
        )
    assert figure.layout.paper_bgcolor == "rgba(0,0,0,0)"


def test_a_narrow_clim_under_a_norm_saturates_instead_of_leaving_holes():
    from vaft.plot.figure_options import figure_options_scope

    r = np.linspace(0.1, 0.9, 30)
    z = np.linspace(-0.8, 0.8, 40)
    wide = Field2D(r=r, z=z, values=np.linspace(1, 1000, 1200).reshape(40, 30), value_label="x")
    with figure_options_scope(FigureOptions(norm="log", clim=(10, 100))):
        figure, axes = renderers.render_field_2d(wide)
    contour = next(m for m in axes.collections if getattr(m, "colorbar", None) is not None)
    assert contour.extend == "both"
    with figure_options_scope(FigureOptions(norm="linear", clim=(-2, 2))):
        figure, axes = renderers.render_field_2d(_signed_field())
    contour = next(m for m in axes.collections if getattr(m, "colorbar", None) is not None)
    # Matplotlib's own levels over the data; the norm only spaces the colours.
    assert min(contour.levels) <= np.nanmin(_signed_field().values) + 0.2


def test_a_field_without_a_colorbar_keeps_its_colours():
    from vaft.plot.figure_options import figure_options_scope

    with figure_options_scope(FigureOptions(norm="centered")):
        figure, axes = renderers.render_field_2d(_signed_field(), colorbar=False)
    assert all(not isinstance(m.norm, matplotlib.colors.CenteredNorm) for m in axes.collections)


def test_a_centred_norm_takes_a_single_bound_as_its_half_range():
    from vaft.plot.figure_options import _make_norm

    norm = _make_norm("centered", np.array([-1.0, 3.0]), (None, 5.0))
    assert norm.vcenter == 0.0 and norm.halfrange == 5.0
