"""Table and text-summary views are first-class canonical views (issue #1180).

A ``table`` or ``text`` view is built like any plot -- the adapter selects and
reduces the data into a typed model -- and presented by a renderer that returns
text: fixed-width in a terminal, HTML in a notebook, Markdown on export.  The
models keep numbers and stored units; the renderer formats them through the
display policy.  The Matplotlib presentation keywords are refused.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pytest

import vaft
import vaft.omas
import vaft.plot
from vaft.plot import registry
from vaft.plot.models import (
    LineSeries,
    Series,
    Table,
    TableCell,
    TableColumn,
    TextItem,
    TextPanel,
    TextSection,
    TextSummary,
)
from vaft.plot.renderers.tables import (
    RenderedTable,
    RenderedTextSummary,
    TextView,
    format_quantity,
    render_table,
    render_text_summary,
)

NAMES = ("equilibrium_table_summary", "equilibrium_table_fit_quality", "equilibrium_text_summary")


def _quiet(function, *args, **kwargs):
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


@pytest.fixture(scope="module")
def sample():
    return _quiet(vaft.omas.load, str(vaft.data.data_path("samples/39915/omas.json.gz")))


def _table() -> Table:
    return Table(
        columns=(
            TableColumn("Quantity"),
            TableColumn("Value", kind="value", units="column"),
            TableColumn("Check", kind="status"),
        ),
        rows=(
            ("Ip", TableCell(172_400.0, unit="A", subject="equilibrium"), TableCell(status="pass")),
            ("beta_N", TableCell(0.05089, quantity="beta_n", subject="equilibrium"), TableCell(status="warn")),
            ("q_95", TableCell(3.21), TableCell()),
            ("W_mhd", TableCell(None, unit="J"), TableCell(status="fail", note="energy not stored")),
        ),
        title="Example",
        caption="A caption.",
        missing="not stored",
    )


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def test_models_keep_structured_values_not_strings():
    table = _table()
    assert table.rows[0][1].value == 172_400.0 and table.rows[0][1].unit == "A"
    assert isinstance(table.rows[0][0], TableCell) and table.rows[0][0].value == "Ip"
    assert table.column("Check")[1].status == "warn"
    assert table.rows[3][1].missing
    assert TableCell(np.float64(2.5)).value == 2.5 and type(TableCell(np.int64(3)).value) is int
    assert TableCell(float("nan")).missing


def test_models_validate_their_shape_and_vocabulary():
    with pytest.raises(ValueError, match="2 cells for 1 columns"):
        Table(columns=(TableColumn("a"),), rows=((1, 2),))
    with pytest.raises(ValueError, match="status"):
        TableCell(1.0, status="red")
    with pytest.raises(ValueError, match="kind"):
        TableColumn("a", kind="colour")
    with pytest.raises(ValueError, match="one value"):
        TableCell(np.arange(3))
    with pytest.raises(TypeError, match="TextSection"):
        TextSummary(sections=("not a section",))
    import omas

    with pytest.raises(TypeError, match="view models only"):
        TableCell(omas.ODS())


def test_text_summary_is_not_a_text_panel():
    summary = TextSummary(sections=(TextSection("A", (TextItem("x", 1.0), "a statement")),))
    assert not isinstance(summary, TextPanel) and not issubclass(TextPanel, TextSummary)
    assert summary.section("A").items[1].is_statement
    assert vaft.plot.TextSummary is TextSummary and vaft.plot.Table is Table


def test_models_round_trip_to_xarray():
    ds = _table().to_xarray(plot_name="x")
    assert ds.attrs["model"] == "Table" and ds.attrs["plot_name"] == "x"
    assert list(ds["column_01"].values) == [172_400.0, 0.05089, 3.21, None]
    summary = TextSummary(sections=(TextSection("A", (TextItem("x", 1.0, unit="m"),)),))
    assert list(summary.to_xarray()["unit"].values) == ["m"]


# ---------------------------------------------------------------------------
# rendering: deterministic text, Markdown and HTML, through the display policy
# ---------------------------------------------------------------------------


def test_units_follow_the_display_policy():
    assert format_quantity(TableCell(172_400.0, unit="A", subject="equilibrium")) == ("172.4", "kA")
    assert format_quantity(TableCell(0.05089, quantity="beta_n", subject="equilibrium")) == ("0.05089", "%·m·T/MA")
    assert format_quantity(TableCell(0.0123, quantity="beta_t", subject="equilibrium")) == ("1.23", "%")
    assert format_quantity(TableCell(0.32, unit="s", display_unit="ms", format=".2f")) == ("320.00", "ms")
    assert format_quantity(TableCell(3, unit="")) == ("3", "")
    # A unit the policy cannot convert is shown as stored.
    assert format_quantity(TableCell(1.5, unit="Pa/Wb")) == ("1.5", "Pa/Wb")
    assert format_quantity(TableCell(None, unit="A")) == (None, "")


def test_a_table_renders_deterministic_text():
    expected = "\n".join([
        "Example",
        "",
        "Quantity       Value  Unit      Check",
        "--------  ----------  --------  --------",
        "Ip             172.4  kA        PASS",
        "beta_N       0.05089  %·m·T/MA  WARN",
        "q_95            3.21",
        "W_mhd     not stored            FAIL [1]",
        "",
        "A caption.",
        "",
        "[1] energy not stored",
    ])
    rendered = render_table(_table())
    assert isinstance(rendered, RenderedTable) and isinstance(rendered, TextView)
    assert rendered.text() == expected
    assert str(rendered) == repr(rendered) == expected
    assert render_table(_table()).text() == expected


def test_a_table_renders_markdown_and_html_without_colour():
    rendered = render_table(_table())
    markdown = rendered.markdown()
    assert markdown.splitlines()[:4] == [
        "**Example**",
        "",
        "| Quantity | Value | Unit | Check |",
        "|---|---:|---|---|",
    ]
    assert "| Ip | 172.4 | kA | PASS |" in markdown
    assert rendered._repr_markdown_() == markdown
    html = rendered._repr_html_()
    assert html == rendered.html()
    assert "<caption>Example</caption>" in html and "<td style=\"text-align: right\">172.4</td>" in html
    assert 'data-status="warn"' in html
    assert "color" not in html.lower() and "#" not in html


def test_header_units_and_inline_units():
    cells = (TableCell(1000.0, unit="A"), TableCell(2000.0, unit="A"))
    header = Table(columns=(TableColumn("I", kind="value", units="header"),), rows=((cells[0],), (cells[1],)))
    assert render_table(header).text().splitlines()[0].strip() == "I [kA]"
    mixed = Table(
        columns=(TableColumn("v", kind="value", units="header"),),
        rows=((TableCell(1000.0, unit="A"),), (TableCell(2.0, unit="m"),)),
    )
    assert render_table(mixed).text().splitlines()[2:] == ["1 kA", " 2 m"]


def test_a_text_summary_renders_sections():
    summary = TextSummary(
        sections=(
            TextSection("Slice", (TextItem("shot", 39915), TextItem("time", 0.32, unit="s", display_unit="ms", format=".2f"))),
            TextSection("Globals", (
                TextItem("Ip", 78_580.0, unit="A", subject="equilibrium"),
                TextItem("W_mhd", None, unit="J"),
                TextItem("li_3", 0.5, note="derived"),
                "a statement",
            )),
        ),
        title="Summary",
    )
    rendered = render_text_summary(summary)
    assert isinstance(rendered, RenderedTextSummary)
    assert rendered.text() == "\n".join([
        "Summary",
        "=======",
        "",
        "Slice",
        "-----",
        "  shot  39915",
        "  time  320.00 ms",
        "",
        "Globals",
        "-------",
        "  Ip     78.58 kA",
        "  W_mhd  not stored",
        "  li_3   0.5 [1]",
        "  - a statement",
        "",
        "[1] derived",
    ])
    assert "- **Ip:** 78.58 kA" in rendered.markdown()
    assert "<h4>Globals</h4>" in rendered.html()


def test_renderers_refuse_the_wrong_model_and_print_on_show(capsys):
    with pytest.raises(TypeError, match="vaft.plot.models.Table"):
        render_table(LineSeries(series=(Series(x=[0, 1], y=[0, 1]),)))
    with pytest.raises(TypeError, match="vaft.plot.models.TextSummary"):
        render_text_summary(_table())
    for spec in registry.specs():
        if spec.view in registry.NON_GRAPHICAL_VIEWS:
            with pytest.raises(TypeError, match="vaft.plot.models"):
                spec.renderer(Series(x=[0.0, 1.0], y=[0.0, 1.0]))
    rendered = render_table(_table(), show=True)
    assert capsys.readouterr().out == rendered.text() + "\n"
    render_table(_table())
    assert capsys.readouterr().out == ""


def test_save_writes_the_export_its_extension_names(tmp_path):
    rendered = render_table(_table())
    assert rendered.save(tmp_path / "t.md") == str(tmp_path / "t.md")
    assert (tmp_path / "t.md").read_text(encoding="utf-8") == rendered.markdown() + "\n"
    rendered.save(tmp_path / "t.txt")
    assert (tmp_path / "t.txt").read_text(encoding="utf-8") == rendered.text() + "\n"
    rendered.save(tmp_path / "t.html")
    assert (tmp_path / "t.html").read_text(encoding="utf-8").startswith('<div class="vaft-table-view">')
    with pytest.raises(ValueError, match=".txt, .md or .html"):
        rendered.save(tmp_path / "t.png")


def test_the_renderers_import_no_matplotlib():
    import subprocess
    import sys

    code = (
        "import sys; import vaft.plot.renderers.tables as t; "
        "print('matplotlib.pyplot' in sys.modules)"
    )
    # vaft.plot itself imports the Matplotlib renderers; the module's own imports do not.
    source = open(t_path := __import__("vaft.plot.renderers.tables", fromlist=["x"]).__file__).read()
    assert "matplotlib" not in "".join(line for line in source.splitlines() if line.startswith(("import", "from")))
    del code, subprocess, sys, t_path


# ---------------------------------------------------------------------------
# registry and discovery
# ---------------------------------------------------------------------------


def test_table_and_text_are_views_and_their_plots_are_registered():
    assert {"table", "text"} <= set(registry.VIEWS)
    assert set(registry.NON_GRAPHICAL_VIEWS) == {"table", "text"}
    for name in NAMES:
        spec = registry.get_spec(name)
        assert spec.subject == "equilibrium"
        assert name == f"{spec.subject}_{spec.view}_{spec.quantity}"
        assert spec.model is (TextSummary if spec.view == "text" else Table)
        assert name in vaft.plot.__all__ and getattr(vaft.plot, name) is spec.renderer
    assert vaft.plot.render_table is render_table and vaft.plot.render_text_summary is render_text_summary


def test_discovery_lists_them_by_view(sample):
    tables = {record.name for record in vaft.plot.available_plots(view="table")}
    assert {"equilibrium_table_summary", "equilibrium_table_fit_quality"} <= tables
    texts = [record for record in vaft.plot.available_plots(view="text")]
    assert "equilibrium_text_summary" in {record.name for record in texts}
    assert all(record.view == "text" for record in texts)
    found = {record.name: record for record in vaft.omas.available_plots(sample, view="table")}
    assert set(found) >= {"equilibrium_table_summary", "equilibrium_table_fit_quality"}
    record = found["equilibrium_table_summary"]
    # Nothing draws it, and there is no figure for a control to redraw.
    assert record.backends == () and "controls" not in record.interaction
    assert "presented as text" in str(vaft.plot.available_plots(query="equilibrium", view="table", detail=True))


# ---------------------------------------------------------------------------
# adapters: plot_ == render(extract_), time= snapping, refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_plot_is_the_rendered_extract(name, sample):
    model = _quiet(getattr(vaft.omas, f"extract_{name}"), sample)
    rendered = _quiet(getattr(vaft.omas, f"plot_{name}"), sample)
    assert isinstance(model, registry.get_spec(name).model)
    assert isinstance(rendered, TextView) and rendered.model == model
    assert rendered.text() == registry.get_spec(name).renderer(model).text()
    assert _quiet(vaft.plot.extract, name, sample) == model


def test_the_summary_is_the_slice_the_overview_draws(sample):
    from vaft.omas._plot_recipes import build_model, normalize_entries

    table = _quiet(vaft.omas.extract_equilibrium_table_summary, sample)
    overview = _quiet(build_model, "equilibrium_overview", normalize_entries(sample))
    assert table.title == overview.suptitle
    panel = next(member for member in overview.models if isinstance(member, TextPanel))
    for line, row in zip(panel.lines, table.rows):
        label, cell = row
        text, unit = format_quantity(cell)
        assert line.split()[0] == label.value.split()[0]
        assert (text or "not stored") in line and unit in line
    ip = dict((row[0].value, row[1]) for row in table.rows)["Ip"]
    assert ip.unit == "A" and format_quantity(ip)[1] == "kA"


def test_time_snaps_to_a_stored_slice_as_the_overview_does(sample):
    model = _quiet(vaft.omas.extract_equilibrium_table_summary, sample, time=0.3195)
    assert model.title.startswith("Equilibrium slice #39915 — t = 319.00 ms (slice 4 of 9")
    assert "nearest stored slice to t = 319.50 ms" in model.title
    by_slice = _quiet(vaft.omas.extract_equilibrium_table_summary, sample, time_slice=3)
    assert by_slice.rows == model.rows
    summary = _quiet(vaft.omas.extract_equilibrium_text_summary, sample, time=0.3195)
    assert summary.section("Slice").items[2].value == "4 of 9"
    fit = _quiet(vaft.omas.extract_equilibrium_table_fit_quality, sample, time=0.3195)
    assert "slice 3 at t=319.00 ms" in fit.title
    with pytest.raises(ValueError, match="either time= or time_slice="):
        _quiet(vaft.omas.extract_equilibrium_table_summary, sample, time=0.3, time_slice=1)


def test_the_text_summary_has_identity_globals_and_shape(sample):
    summary = _quiet(vaft.omas.extract_equilibrium_text_summary, sample)
    assert [section.title for section in summary.sections] == ["Slice", "Global quantities", "Shape"]
    identity = {item.label: item.value for item in summary.section("Slice").items}
    assert identity["shot"] == 39915 and identity["slice"] == "5 of 9"
    shape = {item.label for item in summary.section("Shape").items}
    assert {"minor radius", "elongation"} <= shape


def test_the_fit_quality_table_reports_efit_metrics(sample):
    from vaft.omas.efit_quality import fit_quality_metrics

    table = _quiet(vaft.omas.extract_equilibrium_table_fit_quality, sample, time_slice=2)
    fit = fit_quality_metrics(sample, time_slice=2)
    families = [cell.value for cell in table.column("Family")]
    assert families[-1] == "Total"
    assert table.column("χ²")[-1].value == pytest.approx(fit["chi_squared_total"])
    probes = families.index("Poloidal probes")
    assert table.column("z RMS")[probes].value == pytest.approx(fit["families"]["bpol_probe"]["z_rms"])


@pytest.mark.parametrize(
    "keywords, refused",
    [
        ({"format": "screen"}, "format="),
        ({"theme": "minimal"}, "theme="),
        ({"figure_options": {"xlim": [0, 1]}}, "figure_options="),
        ({"backend": "plotly"}, "backend='plotly'"),
        ({"ax": object()}, "ax="),
        ({"interactive": True}, "interactive=True"),
        ({"animation": True}, "animation=True"),
        ({"cmap": "viridis"}, "cmap="),
    ],
)
def test_matplotlib_presentation_keywords_are_refused(sample, keywords, refused):
    with pytest.raises(TypeError, match="returns text, not a figure") as raised:
        _quiet(vaft.omas.plot_equilibrium_table_summary, sample, **keywords)
    assert refused in str(raised.value)


def test_extract_refuses_rendering_keywords(sample):
    with pytest.raises(TypeError, match="draws nothing"):
        vaft.omas.extract_equilibrium_table_summary(sample, format="screen")


def test_missing_data_is_refused_before_anything_is_built():
    import omas

    empty = omas.ODS(consistency_check=False)
    empty["magnetics.ip.0.time"] = np.array([0.0, 1.0])
    empty["magnetics.ip.0.data"] = np.array([0.0, 1.0])
    for name in NAMES:
        # A computed view diagnoses the absence itself, more precisely than a path.
        with pytest.raises(ValueError, match="no usable equilibrium slice|no time slices"):
            getattr(vaft.omas, f"plot_{name}")(empty)
        assert name not in {row.name for row in vaft.omas.available_plots(empty)}


def test_plot_show_prints(sample, capsys):
    rendered = _quiet(vaft.omas.plot_equilibrium_table_summary, sample)
    vaft.omas.plot_equilibrium_table_summary(sample, show=True)
    assert rendered.text() in capsys.readouterr().out


def test_the_imas_adapter_builds_the_same_models(sample):
    imas = pytest.importorskip("imas")
    import vaft.imas

    entry = imas.DBEntry(str(vaft.data.data_path("samples/39915/imas.nc")), "r", dd_version="3.41.0")
    try:
        for name in NAMES:
            native = _quiet(getattr(vaft.imas, f"plot_{name}"), entry)
            assert isinstance(native, TextView)
            assert isinstance(_quiet(getattr(vaft.imas, f"extract_{name}"), entry), registry.get_spec(name).model)
    finally:
        entry.close()


def test_the_database_adapter_opens_what_the_view_declares(sample, monkeypatch):
    from unittest.mock import Mock, patch
    from types import ModuleType

    import vaft.database as database
    from vaft.database import plotting

    open_ods = Mock(return_value=sample)
    module = ModuleType("vaft.database.lazy_ods")
    module.open_ods, module.h5pyd = open_ods, None
    monkeypatch.setattr(plotting, "stored_ids", lambda shot, source=None: tuple(sample.keys()))
    with patch.dict("sys.modules", {"vaft.database.lazy_ods": module}):
        rendered = _quiet(database.plot_equilibrium_table_summary, 39915)
        model = _quiet(database.extract_equilibrium_table_summary, 39915)
    assert isinstance(rendered, RenderedTable) and rendered.model == model
    assert "equilibrium" in open_ods.call_args.kwargs["ids"]


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------


def test_the_cli_prints_a_table_view(capsys):
    from vaft.cli import plot as plot_cli

    code = _quiet_cli(plot_cli.main, ["equilibrium_table_summary", "--sample", "39915", "--option", "time=0.3195"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("Equilibrium slice #39915 — t = 319.00 ms")
    assert "Ip " in out and " kA" in out


@pytest.mark.parametrize("suffix, start", [(".md", "**Equilibrium slice"), (".txt", "Equilibrium slice"),
                                           (".html", '<div class="vaft-table-view">')])
def test_the_cli_writes_the_export_out_names(tmp_path, capsys, suffix, start):
    from vaft.cli import plot as plot_cli

    target = tmp_path / f"summary{suffix}"
    code = _quiet_cli(plot_cli.main, ["equilibrium_table_summary", "--sample", "39915", "--out", str(target)])
    assert code == 0 and capsys.readouterr().out.strip() == str(target)
    assert target.read_text(encoding="utf-8").startswith(start)


def test_the_cli_refuses_presentation_flags_for_a_table_view(capsys):
    from vaft.cli import plot as plot_cli

    code = _quiet_cli(plot_cli.main, ["equilibrium_table_summary", "--sample", "39915", "--theme", "minimal"])
    assert code == 1 and "theme=" in capsys.readouterr().err


def test_the_cli_shot_path_writes_the_export(tmp_path, sample, monkeypatch, capsys):
    from unittest.mock import Mock, patch
    from types import ModuleType

    from vaft.cli import plot as plot_cli
    from vaft.database import plotting

    module = ModuleType("vaft.database.lazy_ods")
    module.open_ods, module.h5pyd = Mock(return_value=sample), None
    monkeypatch.setattr(plotting, "stored_ids", lambda shot, source=None: tuple(sample.keys()))
    target = tmp_path / "fit.md"
    with patch.dict("sys.modules", {"vaft.database.lazy_ods": module}):
        code = _quiet_cli(plot_cli.main, ["equilibrium_table_fit_quality", "--shot", "39915", "--out", str(target)])
    assert code == 0 and capsys.readouterr().out.strip() == str(target)
    assert "| Family |" in target.read_text(encoding="utf-8")


def _quiet_cli(main, argv):
    with contextlib.redirect_stderr(io.StringIO()) as _err, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = main(argv)
    # Re-emit what the command reported, so capsys sees it.
    import sys

    sys.stderr.write(_err.getvalue())
    return code


def test_a_composed_figure_refuses_a_table_cell():
    from vaft.plot import FigureCell

    with pytest.raises(ValueError, match="presented as text"):
        FigureCell("equilibrium_table_summary")
