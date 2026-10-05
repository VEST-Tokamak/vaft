"""Reproducible figure options and requests over the #689 presentation (issue #1421)."""

from __future__ import annotations

import json
import shlex

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import vaft
from vaft.plot import DataSource, FigureComposition, FigureOptions, PlotRequest
from vaft.plot.presentation import THEMES, Presentation


@pytest.fixture(scope="module")
def ods():
    return vaft.omas.sample_ods(39915)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# --- typography -----------------------------------------------------------------


def test_every_theme_pairs_its_text_face_with_a_math_font():
    assert {name: theme.math_fontset for name, theme in THEMES.items()} == {
        "technical": "dejavusans", "minimal": "stixsans", "monochrome": "dejavusans",
    }
    assert Presentation(None, THEMES["minimal"]).rc()["mathtext.fontset"] == "stixsans"


def test_a_themed_label_is_set_in_the_theme_s_math_font(ods):
    figure, axes = vaft.omas.plot_equilibrium_profile_q(ods, theme="minimal")
    label = axes.xaxis.label
    assert "$" in label.get_text(), "the profile axis is MathText"
    assert label.get_fontproperties().get_math_fontfamily() == "stixsans"


def test_vector_files_keep_their_text(tmp_path, ods):
    figure, _ = vaft.omas.plot_plasma_current_time(ods)
    path = vaft.plot.save_figure(figure, tmp_path / "ip.pdf")
    assert b"/FontFile2" in path.read_bytes(), "TrueType embedded, not Type 3 outlines"
    figure, _ = vaft.omas.plot_plasma_current_time(ods)
    svg = vaft.plot.save_figure(figure, tmp_path / "ip.svg").read_text()
    assert "<text" in svg and "Plasma Current" in svg
    assert matplotlib.rcParams["pdf.fonttype"] == 3, "set for the save only"


# --- the options ------------------------------------------------------------------


def test_options_hold_intent_and_validate():
    options = FigureOptions(xlim=(0.3, None), legend=False, font_family="Arial")
    assert options.to_dict() == {"xlim": [0.3, None], "legend": False, "font_family": ["Arial"]}
    assert FigureOptions.from_dict(options.to_dict()) == options
    assert not FigureOptions() and options
    with pytest.raises(ValueError, match="low must be below high"):
        FigureOptions(xlim=(1, 0))
    with pytest.raises(ValueError, match="xscale must be one of"):
        FigureOptions(xscale="loglog")
    with pytest.raises(ValueError, match="positive ylim"):
        FigureOptions(yscale="log", ylim=(0, 10))
    with pytest.raises(ValueError, match="takes no colour"):
        FigureOptions.from_dict({"colour": "red"})


def test_inherited_presentation_is_unchanged_without_options(ods):
    plain, axis = vaft.omas.plot_plasma_current_time(ods, format="single_column", theme="technical")
    same, again = vaft.omas.plot_plasma_current_time(ods, format="single_column", theme="technical", figure_options={})
    assert axis.get_xlim() == again.get_xlim() and axis.title.get_fontsize() == again.title.get_fontsize()
    assert tuple(plain.get_size_inches()) == tuple(same.get_size_inches())


def test_options_layer_over_the_format_and_the_theme(ods):
    before = dict(plt.rcParams)
    _, axis = vaft.omas.plot_plasma_current_time(
        ods, format="single_column", theme="technical",
        figure_options={"xlim": (0.30, 0.33), "title": "Ip", "font_size": 10, "minor_ticks": True,
                        "legend": False, "ylabel": "I_p [kA]"},
    )
    assert axis.get_xlim() == pytest.approx((0.30, 0.33))
    assert axis.get_title() == "Ip" and axis.get_ylabel() == "I_p [kA]"
    # 10 pt over the format's 8 pt: every size scaled by the same 1.25
    assert axis.title.get_fontsize() == pytest.approx(10.0)
    assert axis.xaxis.label.get_fontsize() == pytest.approx(10.0)
    assert axis.xaxis.get_ticklabels()[0].get_fontsize() == pytest.approx(9.0)
    assert axis.get_legend() is None and axis.xaxis.get_minor_ticks()
    assert dict(plt.rcParams) == before, "no global rcParams change"


def test_scalar_field_and_legend_options(ods):
    figure, axis = vaft.omas.plot_equilibrium_field_psi(
        ods, figure_options={"cmap": "magma", "clim": (-10, 0), "colorbar_label": "psi"},
    )
    fields = [m for m in axis.collections if getattr(m, "colorbar", None) is not None]
    overlays = [m for m in axis.collections if getattr(m, "colorbar", None) is None and hasattr(m, "get_cmap")
                and m.get_array() is not None]
    assert fields and all(m.get_cmap().name == "magma" and m.get_clim() == (-10, 0) for m in fields)
    assert all(m.get_cmap().name != "magma" for m in overlays), "a fixed-colour overlay keeps its colours"
    assert [a.get_ylabel() for a in figure.axes if a.get_label() == "<colorbar>"] == ["psi"]
    _, axis = vaft.omas.plot_flux_loop_time_voltage(
        ods, figure_options={"legend": True, "legend_loc": "upper left", "legend_ncols": 2, "legend_frame": False},
    )
    legend = axis.get_legend()
    assert legend is not None and legend._ncols == 2 and not legend.get_frame_on()


def test_plotly_and_compositions_take_the_same_options(ods):
    figure = vaft.omas.plot_plasma_current_time(
        ods, backend="plotly", figure_options={"xlim": (0.30, 0.33), "title": "Ip", "yscale": "log", "ylim": (1, 100)},
    )
    assert tuple(figure.layout.xaxis.range) == (0.30, 0.33) and figure.layout.title.text == "Ip"
    assert figure.layout.yaxis.type == "log" and tuple(figure.layout.yaxis.range) == pytest.approx((0.0, 2.0))
    _, axes = vaft.omas.compose(FigureComposition.stack(["plasma_current_time", "equilibrium_time_q95"]), ods,
                                figure_options={"xlim": (0.30, 0.33), "title": "stack"})
    assert all(a.get_xlim() == pytest.approx((0.30, 0.33)) for a in axes)
    assert axes[0].figure._suptitle.get_text() == "stack"


def test_interactive_and_animation_refuse_figure_options_for_now(ods):
    with pytest.raises(TypeError, match="do not take it yet"):
        vaft.omas.plot_equilibrium_field_psi(ods, interactive=True, interaction_backend="none", figure_options={"title": "x"})


# --- the request ------------------------------------------------------------------


def _request(**kwargs):
    return PlotRequest(
        DataSource("sample", (39915,)), plot="plasma_current_time", options={"yunit": "kA"},
        format="single_column", theme="technical", figure_options=FigureOptions(xlim=(0.30, 0.33)), **kwargs,
    )


def test_a_request_writes_only_intent():
    request = PlotRequest(DataSource("sample", 39915), plot="plasma_current_time")
    assert request.to_dict() == {"source": {"kind": "sample", "values": [39915]}, "plot": "plasma_current_time"}
    code = request.to_python()
    assert "format" not in code and "theme" not in code and "figure_options" not in code
    assert request.to_cli() == "vaft plot plasma_current_time --sample 39915"
    data = _request().to_dict()
    assert json.loads(json.dumps(data)) == data and PlotRequest.from_dict(data) == _request()


def test_the_python_code_draws_the_same_figure():
    request = _request()
    drawn = request.render()[1]
    namespace: dict = {}
    exec(request.to_python(), namespace)
    replayed = namespace["axes"]
    assert replayed.get_xlim() == drawn.get_xlim() == pytest.approx((0.30, 0.33))
    assert replayed.get_ylabel() == drawn.get_ylabel()
    assert replayed.figure.get_size_inches().tolist() == drawn.figure.get_size_inches().tolist()


def test_the_cli_command_draws_the_same_figure(tmp_path):
    from vaft.cli.plot import main

    out = tmp_path / "ip.png"
    command = _request().to_cli(out=str(out))
    assert "--option yunit=kA" in command and "--format single_column" in command
    assert main(shlex.split(command)[2:]) == 0 and out.stat().st_size > 0
    composed = PlotRequest(DataSource("sample", (39915,)),
                           composition=FigureComposition.stack(["plasma_current_time", "equilibrium_time_q95"]))
    out2 = tmp_path / "stack.png"
    assert main(shlex.split(composed.to_cli(out=str(out2)))[2:]) == 0 and out2.stat().st_size > 0
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps(composed.to_dict()))
    out3 = tmp_path / "again.png"
    assert main(["--request", str(request_file), "--out", str(out3)]) == 0 and out3.stat().st_size > 0


def test_database_requests_use_the_database_api():
    request = PlotRequest(DataSource("shot", (39915, 41524), "main"), plot="plasma_current_time", backend="plotly")
    assert "figure = vaft.database.plot_plasma_current_time([39915, 41524], source='main', backend='plotly')" \
        in request.to_python()
    assert request.to_cli() == "vaft plot plasma_current_time --shot 39915 --shot 41524 --source main --backend plotly"
    composed = PlotRequest(DataSource("shot", (39915,)),
                           composition=FigureComposition.stack(["plasma_current_time"]))
    assert "vaft.database.plotting.compose(composition, 39915)" in composed.to_python()


def test_pylustrator_finishing_starts_first_and_only_in_the_text():
    code = _request().to_python(pylustrator=True)
    assert code.index("pylustrator.start()") < code.index("import vaft")
    assert "pylustrator" not in _request().to_python()
    import pathlib

    sources = pathlib.Path(vaft.plot.__file__).parent
    assert not any("import pylustrator\n" in path.read_text(encoding="utf-8") for path in sources.rglob("*.py")
                   if path.name != "request.py"), "never imported by the library"


def test_requests_refuse_what_cannot_be_drawn():
    with pytest.raises(ValueError, match="either one plot or one composition"):
        PlotRequest(DataSource("sample", (39915,)))
    with pytest.raises(ValueError, match="does not apply them"):
        PlotRequest(DataSource("sample", (39915,)), plot="plasma_current_time", backend="plotly", theme="minimal")
    with pytest.raises(ValueError, match="field of the request"):
        PlotRequest(DataSource("sample", (39915,)), plot="plasma_current_time", options={"theme": "minimal"})
    with pytest.raises(ValueError, match="namespace applies to database shots"):
        DataSource("sample", (39915,), "main")


# --- regressions from the cold review -------------------------------------------


def test_line_scale_applies_once_in_a_composition(ods):
    _, single = vaft.omas.plot_plasma_current_time(ods, theme="technical")
    base = single.get_lines()[0].get_linewidth()
    _, axes = vaft.omas.compose(FigureComposition.stack(["plasma_current_time", "equilibrium_time_q95"]), ods,
                                theme="technical", figure_options={"line_scale": 2})
    assert axes[0].get_lines()[0].get_linewidth() == pytest.approx(2 * base)
    _, scaled = vaft.omas.plot_plasma_current_time(ods, theme="technical", figure_options={"line_scale": 2})
    assert scaled.get_lines()[0].get_linewidth() == pytest.approx(2 * base)


def test_a_caller_s_other_axes_are_left_alone(ods):
    figure, (mine, theirs) = plt.subplots(1, 2)
    theirs.plot([0, 1], [0, 1])
    theirs.set_ylabel("theirs")
    vaft.omas.plot_plasma_current_time(ods, ax=mine, figure_options={"xlim": (0.30, 0.33), "ylabel": "X", "title": "T"})
    assert mine.get_xlim() == pytest.approx((0.30, 0.33)) and mine.get_title() == "T"
    assert theirs.get_xlim() != pytest.approx((0.30, 0.33)) and theirs.get_ylabel() == "theirs"
    assert figure._suptitle is None


def test_plotly_scalar_fields_and_log_axes(ods):
    figure = vaft.omas.plot_equilibrium_field_psi(
        ods, backend="plotly", figure_options={"cmap": "magma", "clim": (-10, 0), "colorbar_label": "psi"},
    )
    shown = [t for t in figure.data if getattr(t, "showscale", None) is not False and getattr(t, "colorscale", None)]
    assert shown and all(t.zmin == -10 and t.zmax == 0 for t in shown if hasattr(t, "zmin"))
    assert any(t.colorbar.title.text == "psi" for t in shown)
    already_log = vaft.omas.plot_plasma_current_time(
        ods, backend="plotly", figure_options={"yscale": "log"},
    )
    already_log.update_yaxes(type="log")
    FigureOptions(ylim=(1, 100)).apply_plotly(already_log)
    assert tuple(already_log.layout.yaxis.range) == pytest.approx((0.0, 2.0)), "decades on a log axis"
    with pytest.warns(UserWarning, match="does not apply legend_ncols"):
        vaft.omas.plot_plasma_current_time(ods, backend="plotly", figure_options={"legend_ncols": 2})


def test_a_rebuilt_legend_keeps_its_title(ods):
    figure, axis = plt.subplots()
    axis.plot([0, 1], label="a")
    axis.legend(title="T")
    FigureOptions(legend_loc="lower right").apply(figure)
    assert axis.get_legend().get_title().get_text() == "T"


def test_empty_options_are_no_options(ods):
    result = vaft.omas.plot_equilibrium_field_psi(ods, interactive=True, interaction_backend="none", figure_options={})
    plt.close(result.figure)


def test_inputs_and_flags_that_cannot_go_together(tmp_path):
    import pathlib

    from vaft.cli.plot import main

    source = DataSource("file", pathlib.Path("a b/eq.json"))
    assert source.values == (str(pathlib.Path("a b/eq.json")),), "the native path, separators and all"
    import numpy as np

    assert DataSource("sample", np.int64(39915)).values == (39915,)
    with pytest.raises(SystemExit):
        main(["plasma_current_time", "--sample", "39915", "--source", "main"])
    with pytest.raises(SystemExit):
        main(["--request", '{"source": {"kind": "sample", "values": [39915]}, "plot": "plasma_current_time"}',
              "--theme", "minimal"])
