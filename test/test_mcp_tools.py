"""The VAFT MCP tools answer exactly what the VAFT APIs they wrap answer (#1423).

These call the tool functions directly, so they need no ``mcp`` SDK: the
protocol round trip is ``test_mcp_server.py``.  Parity is semantic -- the
same formula, check, plot declaration or extracted numbers -- and every result
must survive strict JSON (no NaN, no NumPy, no tuples-as-keys).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os

import numpy as np
import pytest

from vaft.mcp import _tools as tools
from vaft.mcp._jsonable import Bounded, preview_strides


def _json(result):
    """Strict JSON round trip: what an MCP client actually receives."""
    return json.loads(json.dumps(result, allow_nan=False))


# -- inventory ----------------------------------------------------------------


def test_the_tool_set_is_the_curated_read_only_list():
    assert [tool.__name__ for tool in tools.TOOLS] == [
        "get_capabilities",
        "get_capability",
        "search_formulas",
        "describe_formula",
        "search_processes",
        "describe_process",
        "list_validation_checks",
        "describe_validation_check",
        "list_plots",
        "describe_plot",
        "get_plot_requirements",
        "extract_plot_data",
        "list_samples",
        "describe_sample",
        "list_boundaries",
        "describe_boundary",
        "get_atlas_summary",
    ]
    forbidden = ("write", "save", "publish", "run", "exec", "shell", "delete", "upload", "solve")
    assert not [t.__name__ for t in tools.TOOLS if any(word in t.__name__ for word in forbidden)]


def test_no_tool_parameter_names_a_data_representation():
    """The schema is semantic: no ODS/IDS/representation knob leaks into it (#1423)."""
    import inspect

    for tool in tools.TOOLS:
        for name in inspect.signature(tool).parameters:
            assert not any(word in name.lower() for word in ("ods", "ids", "imas", "omas", "representation")), (
                tool.__name__, name)
        assert tool.__doc__, tool.__name__


# -- capabilities ---------------------------------------------------------------


def test_capabilities_are_the_help_topics():
    from vaft._help import help as vaft_help
    from vaft._help import topics

    result = _json(tools.get_capabilities())
    assert [row["name"] for row in result["topics"]] == list(topics())
    assert result["overview"] == _json(vaft_help().as_dict())
    assert _json(tools.get_capability("validation")) == _json(vaft_help("validation").as_dict())


def test_capability_items_route_to_the_dedicated_tools():
    result = tools.get_capability("validation", "verification.structure")
    assert result["result"] == tools.describe_validation_check("verification.structure")
    assert "vaft plot" in tools.get_capability("cli", "plot")["result"]
    with pytest.raises(tools.ToolInputError, match="choose from"):
        tools.get_capability("nonsense")


# -- formulas, processes, validation -----------------------------------------------


def test_describe_formula_is_the_catalog_entry_without_source_code():
    from vaft.formula import catalog

    name = catalog.list_formulas()[0].qualname
    expected = catalog.describe(name).as_dict()
    result = _json(tools.describe_formula(name))
    assert "code" not in result["source"]
    expected["source"].pop("code")
    assert result == _json(expected)


def test_search_formulas_matches_the_catalog_search_and_is_bounded():
    from vaft.formula import catalog

    direct = [spec.qualname for spec in catalog.search("safety factor")]
    result = _json(tools.search_formulas("safety factor", limit=3))
    assert result["total"] == len(direct)
    assert [row["id"] for row in result["items"]] == direct[:3]
    assert result["truncated"] is (len(direct) > 3)
    with pytest.raises(tools.ToolInputError):
        tools.describe_formula("no_such_formula_anywhere")


def test_describe_process_is_the_catalog_entry_without_source_code():
    from vaft.process import catalog

    name = catalog.search("spectrogram")[0].qualname
    expected = catalog.describe(name).as_dict()
    expected["source"].pop("code")
    assert _json(tools.describe_process(name)) == _json(expected)
    assert [row["id"] for row in tools.search_processes("spectrogram", limit=500)["items"]] == [
        spec.qualname for spec in catalog.search("spectrogram")
    ]


def test_validation_metadata_is_the_registry():
    from vaft.validation.registry import CHECKS, MEASURES

    listed = _json(tools.list_validation_checks())
    assert [row["key"] for row in listed["items"]] == list(CHECKS)
    for key, spec in CHECKS.items():
        row = _json(tools.describe_validation_check(key))
        expected = _json(dataclasses.asdict(spec))
        assert {k: row[k] for k in expected} == expected
        assert row["measure_description"] == MEASURES[spec.measure]
    category = next(iter(CHECKS.values())).category
    assert {row["category"] for row in tools.list_validation_checks(category)["items"]} == {category}
    with pytest.raises(tools.ToolInputError, match="known"):
        tools.describe_validation_check("nope.nope")


# -- plots ------------------------------------------------------------------------


def test_plot_listing_is_the_registry_catalog():
    import vaft.plot

    direct = [record.name for record in vaft.plot.available_plots()]
    result = _json(tools.list_plots(limit=500))
    assert [row["name"] for row in result["items"]] == direct
    queried = _json(tools.list_plots(query="q"))
    assert [row["name"] for row in queried["items"]] == [r.name for r in vaft.plot.available_plots(query="q")]
    assert tools.list_plots(limit=2)["truncated"] is True


def test_plot_description_and_requirements_are_the_declarations():
    import vaft.plot

    name = "equilibrium_profile_q"
    record = next(r for r in vaft.plot.available_plots() if r.name == name)
    described = _json(tools.describe_plot(name))
    assert described["required_paths"] == list(record.required_paths)
    assert described["model"] == record.model
    requirements = _json(tools.get_plot_requirements(name))["paths"]
    direct = vaft.plot.dd(name)
    assert len(requirements) == len(direct)
    for row, path in zip(requirements, direct):
        assert row["canonical"] == path.canonical
        assert row["role"] == path.role
        assert row["units"] == path.units
        assert row["coordinate"] == path.coordinate
        assert row["fallback_coordinate"] == list(path.fallback_coordinate)
        assert row["attrs"] == dict(path.attrs)
    with pytest.raises(tools.ToolInputError):
        tools.describe_plot("no_such_plot")


def test_extracted_numbers_match_direct_extraction():
    pytest.importorskip("omas")
    import vaft.plot
    from vaft.omas.sample import sample_ods

    name, points = "equilibrium_profile_q", 40
    result = _json(tools.extract_plot_data(name, shot=39915, max_points=points))
    model = vaft.plot.extract(name, sample_ods(39915))
    assert result["model"] == type(model).__name__
    for got, series in zip(result["data"]["series"], model.series):
        for axis in ("x", "y"):
            direct = np.asarray(getattr(series, axis))
            summary = got[axis]
            assert summary["shape"] == list(direct.shape)
            finite = direct[np.isfinite(direct)]
            assert summary["min"] == pytest.approx(finite.min())
            assert summary["max"] == pytest.approx(finite.max())
            (stride,) = summary["stride"]
            assert len(summary["values"]) <= points
            expected = direct[::stride]
            assert [np.nan if v is None else v for v in summary["values"]] == pytest.approx(
                expected.tolist(), nan_ok=True)
    assert result["data"]["y_label"] == model.y_label


def test_extraction_of_a_2d_map_is_bounded():
    pytest.importorskip("omas")
    result = _json(tools.extract_plot_data("equilibrium_field_psi", max_points=25))
    values = result["data"]["values"]
    assert values["shape"] == [129, 129]
    assert np.prod(values["preview_shape"]) <= 25
    assert len(json.dumps(result)) < 200_000
    assert any("overlays" in entry for entry in result["truncated"])


def test_extraction_refuses_unknown_input():
    with pytest.raises(tools.ToolInputError, match="no plot"):
        tools.extract_plot_data("no_such_plot")
    with pytest.raises(ValueError, match="no packaged reference shot"):
        tools.extract_plot_data("equilibrium_profile_q", shot=1)
    with pytest.raises(tools.ToolInputError, match="max_points"):
        tools.extract_plot_data("equilibrium_profile_q", max_points=0)


# -- samples and boundaries ---------------------------------------------------------


def test_samples_are_the_packaged_samples():
    from vaft.data import available_samples, sample_manifest

    listed = _json(tools.list_samples())
    assert [row["shot"] for row in listed["items"]] == [int(s) for s in available_samples()]
    assert next(r for r in listed["items"] if r["shot"] == 39915)["installed"] is True
    assert _json(tools.describe_sample(39915)) == _json(Bounded()(sample_manifest(39915)))
    with pytest.raises(tools.ToolInputError):
        tools.describe_sample(1)


def test_boundaries_are_the_registry_without_callables():
    from vaft.formula import boundaries

    listed = _json(tools.list_boundaries())
    assert [row["key"] for row in listed["items"]] == list(boundaries.list_boundaries())
    family = listed["items"][0]["family"]
    assert [r["key"] for r in tools.list_boundaries(family)["items"]] == list(boundaries.list_boundaries(family))
    entry = boundaries.get_boundary("greenwald")
    described = _json(tools.describe_boundary("greenwald"))
    assert described["target"]["unit"] == entry.target.unit
    assert [q["name"] for q in described["inputs"]] == [q.name for q in entry.inputs]
    assert "function" not in described or described["function"] is None
    with pytest.raises(tools.ToolInputError):
        tools.describe_boundary("nope")
    with pytest.raises(tools.ToolInputError):
        tools.list_boundaries("no_family")


# -- atlas tables ---------------------------------------------------------------------


@pytest.fixture
def atlas(tmp_path, monkeypatch):
    root = tmp_path / "atlas"
    root.mkdir()
    (root / "atlas.csv").write_text(
        "shot,efit_quality,efit_lineage,ip\n"
        "1,good,v1,1.0\n2,good,v2,\n3,poor,v1,3.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(tools.ATLAS_ENV, str(root))
    return root


def test_atlas_summary_counts_groups_and_bounds_rows(atlas):
    result = _json(tools.get_atlas_summary(limit=2))
    assert result["path"] == "atlas.csv"
    assert result["rows"] == 3
    assert [c["name"] for c in result["columns"]] == ["shot", "efit_quality", "efit_lineage", "ip"]
    assert {r["value"]: r["rows"] for r in result["counts"]["efit_quality"]} == {"good": 2, "poor": 1}
    assert len(result["head"]) == 2 and result["head_truncated"] is True
    assert result["head"][1]["ip"] is None  # NaN is not JSON
    custom = tools.get_atlas_summary("atlas.csv", group_by=["shot", "absent"])
    assert custom["missing_group_columns"] == ["absent"]


def test_atlas_refuses_paths_outside_its_directory(atlas, tmp_path):
    outside = tmp_path / "secret.csv"
    outside.write_text("a\n1\n", encoding="utf-8")
    for path in ("../secret.csv", str(outside)):
        with pytest.raises(tools.ToolInputError, match="outside"):
            tools.get_atlas_summary(path)
    if hasattr(os, "symlink"):
        try:
            (atlas / "link.csv").symlink_to(outside)
        except OSError:  # Windows without symlink privilege
            pass
        else:
            with pytest.raises(tools.ToolInputError, match="outside"):
                tools.get_atlas_summary("link.csv")
            # an escaping link is not offered either, so the single table is still chosen
            assert tools.get_atlas_summary()["path"] == "atlas.csv"
    (atlas / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(tools.ToolInputError, match="csv"):
        tools.get_atlas_summary("notes.txt")


def test_atlas_needs_its_directory(monkeypatch):
    monkeypatch.delenv(tools.ATLAS_ENV, raising=False)
    with pytest.raises(tools.ToolInputError, match=tools.ATLAS_ENV):
        tools.get_atlas_summary()


# -- the bounded converter ------------------------------------------------------------


def test_bounded_conversion_is_strict_json_and_records_what_it_cut():
    bounded = Bounded(max_items=3, max_points=10)
    array = np.arange(30.0)
    array[0], array[3] = np.nan, np.inf
    value = {"a": array, "b": tuple(range(5)), "f": len, "s": np.float32(2.5)}
    result = _json(bounded(value))
    assert "f" not in result
    assert result["b"] == [0, 1, 2]
    assert result["s"] == 2.5
    assert result["a"]["finite_count"] == 28 and result["a"]["min"] == 1.0 and result["a"]["max"] == 29.0
    assert result["a"]["values"][:2] == [None, None] and len(result["a"]["values"]) <= 10
    assert any(entry.startswith("$.b") for entry in bounded.truncated)
    assert preview_strides((129, 129), 25) == (26, 26)
    assert math.prod(math.ceil(n / s) for n, s in zip((129, 129), preview_strides((129, 129), 25))) <= 25
