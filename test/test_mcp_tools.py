"""The VAFT MCP tools answer exactly what the VAFT APIs they wrap answer (#1423).

These call the tool functions directly, so they need no ``mcp`` SDK: the
protocol round trip is ``test_mcp_server.py``.  Parity is semantic -- the
same formula, check, plot declaration or extracted numbers -- and every result
must survive strict JSON (no NaN, no NumPy, no tuples-as-keys), carry its
``truncated`` record, and stay bounded in size.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os

import numpy as np
import pytest

from vaft.mcp import _tools as tools
from vaft.mcp._jsonable import REPORTED_PATHS, Bounded, bounded_json, preview_strides


def _json(result):
    """Strict JSON round trip: what an MCP client actually receives."""
    return json.loads(json.dumps(result, allow_nan=False))


def _body(result):
    """A tool result without its ``truncated`` record, as strict JSON."""
    data = _json(result)
    record = data.pop("truncated")
    assert set(record) == {"count", "paths"} and len(record["paths"]) <= REPORTED_PATHS
    return data


def _preview_values(node) -> int:
    """How many array values a converted result carries in its previews."""
    if isinstance(node, dict):
        if "shape" in node and "dtype" in node:
            return int(np.prod(node.get("preview_shape") or [0])) if node.get("values") is not None else 0
        return sum(_preview_values(v) for v in node.values())
    if isinstance(node, list):
        return sum(_preview_values(v) for v in node)
    return 0


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


def test_every_result_reports_what_it_truncated():
    calls = [
        tools.get_capabilities(),
        tools.get_capability("plot"),
        tools.get_capability("formula", "equilibrium.kink_safety_factor"),
        tools.get_capability("cli", "plot"),
        tools.search_formulas("safety", limit=1),
        tools.describe_formula("equilibrium.kink_safety_factor"),
        tools.list_validation_checks(),
        tools.describe_validation_check("verification.structure"),
        tools.list_plots(limit=1),
        tools.describe_plot("equilibrium_profile_q"),
        tools.get_plot_requirements("equilibrium_profile_q"),
        tools.list_samples(),
        tools.describe_sample(39915),
        tools.list_boundaries(),
        tools.describe_boundary("greenwald"),
    ]
    for result in calls:
        _body(result)
    assert tools.list_plots(limit=1)["truncated"]["count"] == 1


# -- capabilities ---------------------------------------------------------------


def test_capabilities_are_the_help_topics():
    from vaft._help import help as vaft_help
    from vaft._help import topics

    result = _body(tools.get_capabilities())
    assert [row["name"] for row in result["topics"]] == list(topics())
    assert result["overview"] == _json(vaft_help().as_dict())
    assert _body(tools.get_capability("validation")) == _json(vaft_help("validation").as_dict())


def test_capability_items_route_to_the_dedicated_tools():
    result = _json(tools.get_capability("validation", "verification.structure"))
    assert result["result"] == _body(tools.describe_validation_check("verification.structure"))
    assert "vaft plot" in tools.get_capability("cli", "plot")["result"]
    with pytest.raises(tools.ToolInputError, match="choose from"):
        tools.get_capability("nonsense")
    with pytest.raises(tools.ToolInputError, match="shot number"):
        tools.get_capability("data", "not-a-shot")


def test_capability_summaries_come_from_the_overview():
    result = _body(tools.get_capabilities())
    summaries = {row["name"]: row["summary"] for row in result["topics"]}
    assert summaries["formula"] and summaries["database"]


def test_no_output_carries_a_secret_or_the_home_directory(monkeypatch, tmp_path):
    """Help reports *that* HSDS is configured; values and the account's paths stay local."""
    home = tmp_path / "home-of-someone"
    home.mkdir()
    (home / ".hscfg").write_text(
        "hs_endpoint = http://hsds.invalid\nhs_username = someone\n"
        "hs_password = pw-in-file-1234\nhs_api_key = key-in-file-5678\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HS_PASSWORD", "pw-in-env-9876")
    monkeypatch.setenv("HS_API_KEY", "key-in-env-5432")
    monkeypatch.chdir(tmp_path)
    text = json.dumps([tools.get_capability("database"), tools.get_capability("code"), tools.get_capabilities()])
    for secret in ("pw-in-file-1234", "key-in-file-5678", "pw-in-env-9876", "key-in-env-5432", str(home)):
        assert secret not in text
    # The redaction also covers strings a tool did not expect to carry them.
    scrubbed = tools._finish({"note": f"{home}/x and pw-in-env-9876"})
    assert scrubbed["note"] == "~/x and <redacted>"


def test_redaction_respects_path_components_and_keeps_colliding_keys(monkeypatch, tmp_path):
    home = tmp_path / "yun"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    result = tools._redacted({"a": f"set ({home}/.hscfg)", "b": f"{home}ho/x", f"k{home}": 1, "k~": 2})
    assert result == {"a": "set (~/.hscfg)", "b": f"{home}ho/x", "k~": 1, "k~#2": 2}


# -- formulas, processes, validation -----------------------------------------------


def test_describe_formula_is_the_catalog_entry_without_source_code():
    from vaft.formula import catalog

    name = catalog.list_formulas()[0].qualname
    expected = catalog.describe(name).as_dict()
    result = _body(tools.describe_formula(name))
    assert "code" not in result["source"]
    expected["source"].pop("code")
    assert result == _json(expected)


def test_search_formulas_matches_the_catalog_search_and_is_bounded():
    from vaft.formula import catalog

    direct = [spec.qualname for spec in catalog.search("safety factor")]
    result = _json(tools.search_formulas("safety factor", limit=3))
    assert result["total"] == len(direct)
    assert [row["id"] for row in result["items"]] == direct[:3]
    assert result["truncated"]["count"] == (1 if len(direct) > 3 else 0)
    with pytest.raises(tools.ToolInputError):
        tools.describe_formula("no_such_formula_anywhere")


def test_describe_process_is_the_catalog_entry_without_source_code():
    from vaft.process import catalog

    name = catalog.search("spectrogram")[0].qualname
    expected = catalog.describe(name).as_dict()
    expected["source"].pop("code")
    assert _body(tools.describe_process(name)) == _json(expected)
    assert [row["id"] for row in tools.search_processes("spectrogram", limit=500)["items"]] == [
        spec.qualname for spec in catalog.search("spectrogram")
    ]


def test_validation_metadata_is_the_registry():
    from vaft.validation.registry import CHECKS, MEASURES

    listed = _json(tools.list_validation_checks())
    assert [row["key"] for row in listed["items"]] == list(CHECKS)
    for key, spec in CHECKS.items():
        row = _body(tools.describe_validation_check(key))
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


def test_plot_description_and_requirements_are_the_declarations():
    import vaft.plot

    name = "equilibrium_profile_q"
    record = next(r for r in vaft.plot.available_plots() if r.name == name)
    described = _body(tools.describe_plot(name))
    assert described["required_paths"] == list(record.required_paths)
    assert described["model"] == record.model
    requirements = _body(tools.get_plot_requirements(name))["paths"]
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
    assert _preview_values(result["data"]) <= points  # the budget covers the whole result
    for got, series in zip(result["data"]["series"], model.series):
        for axis in ("x", "y"):
            direct = np.asarray(getattr(series, axis))
            summary = got[axis]
            assert summary["shape"] == list(direct.shape)
            finite = direct[np.isfinite(direct)]
            assert summary["min"] == pytest.approx(finite.min())
            assert summary["max"] == pytest.approx(finite.max())
            (stride,) = summary["stride"]
            expected = direct[::stride]
            assert [np.nan if v is None else v for v in summary["values"]] == pytest.approx(
                expected.tolist(), nan_ok=True)
    assert result["data"]["y_label"] == model.y_label


@pytest.mark.parametrize(
    "name", ["diagnostics_overview", "magnetics_overview", "b_field_probe_time_field", "equilibrium_field_psi"]
)
def test_the_largest_views_stay_bounded_at_the_defaults(name):
    """Panel overviews held ~300 arrays: ~880 kB at the defaults before the whole-result budget."""
    pytest.importorskip("omas")
    result = _json(tools.extract_plot_data(name))
    size = len(json.dumps(result))
    assert size <= tools.MAX_EXTRACT_BYTES + 5_000, size
    assert _preview_values(result["data"]) <= 200
    assert len(result["truncated"]["paths"]) <= REPORTED_PATHS
    biggest = _json(tools.extract_plot_data(name, max_points=tools.MAX_POINTS))
    assert len(json.dumps(biggest)) <= tools.MAX_EXTRACT_BYTES + 5_000


def test_extraction_refuses_file_options_and_oversized_computations():
    with pytest.raises(tools.ToolInputError, match="local files"):
        tools.extract_plot_data("camera_visible_image", options={"pose_path": "~/.hscfg"})
    with pytest.raises(tools.ToolInputError, match="grid_shape"):
        tools.extract_plot_data("vacuum_field", options={"grid_shape": [20000, 20000]})
    with pytest.raises(tools.ToolInputError, match="ceiling"):
        tools.extract_plot_data("vacuum_field", options={"resolution": 10**6})
    with pytest.raises(tools.ToolInputError, match="limit is"):
        tools.extract_plot_data("vacuum_field", options={"seeds": [[0.3, 0.0]] * 1000})


def test_extraction_refuses_unknown_input():
    with pytest.raises(tools.ToolInputError, match="no plot"):
        tools.extract_plot_data("no_such_plot")
    with pytest.raises(ValueError, match="no packaged reference shot"):
        tools.extract_plot_data("equilibrium_profile_q", shot=1)
    with pytest.raises(tools.ToolInputError, match="max_points"):
        tools.extract_plot_data("equilibrium_profile_q", max_points=0)
    for options in ({"_panel_member": 1}, {"bogus": 1}, {"figsize": [3, 3]}):
        with pytest.raises(tools.ToolInputError, match="not extraction options"):
            tools.extract_plot_data("equilibrium_profile_q", options=options)
    assert "coordinate" in tools.extraction_option_names()


# -- samples and boundaries ---------------------------------------------------------


def test_samples_are_the_packaged_samples():
    from vaft.data import available_samples, sample_manifest

    listed = _json(tools.list_samples())
    assert [row["shot"] for row in listed["items"]] == [int(s) for s in available_samples()]
    assert next(r for r in listed["items"] if r["shot"] == 39915)["installed"] is True
    assert _body(tools.describe_sample(39915)) == _json(Bounded()(sample_manifest(39915)))
    with pytest.raises(tools.ToolInputError):
        tools.describe_sample(1)


def test_boundaries_are_the_registry_without_callables():
    from vaft.formula import boundaries

    listed = _json(tools.list_boundaries())
    assert [row["key"] for row in listed["items"]] == list(boundaries.list_boundaries())
    family = listed["items"][0]["family"]
    assert [r["key"] for r in tools.list_boundaries(family)["items"]] == list(boundaries.list_boundaries(family))
    entry = boundaries.get_boundary("greenwald")
    described = _body(tools.describe_boundary("greenwald"))
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
        "shot,efit_quality,efit_lineage,ip,note\n"
        f"1,good,v1,1.0,{'x' * 500}\n2,good,v2,,a\n3,poor,v1,3.0,b\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(tools.ATLAS_ENV, str(root))
    return root


def test_atlas_summary_counts_groups_and_bounds_rows(atlas):
    result = _json(tools.get_atlas_summary(limit=2))
    assert result["path"] == "atlas.csv"
    assert result["rows"] == 3
    assert [c["name"] for c in result["columns"]] == ["shot", "efit_quality", "efit_lineage", "ip", "note"]
    assert {r["value"]: r["rows"] for r in result["counts"]["efit_quality"]} == {"good": 2, "poor": 1}
    assert len(result["head"]) == 2 and result["head_truncated"] is True
    assert result["head"][1]["ip"] is None  # NaN is not JSON
    assert len(result["head"][0]["note"]) == tools.MAX_ATLAS_CELL
    assert result["truncated"]["count"] >= 1
    custom = tools.get_atlas_summary("atlas.csv", group_by=["shot", "absent"])
    assert custom["missing_group_columns"] == ["absent"]
    with pytest.raises(tools.ToolInputError, match="at most"):
        tools.get_atlas_summary(group_by=["a", "b", "c", "d"])


def test_atlas_parquet_tables_read_the_same(atlas):
    pytest.importorskip("pyarrow")
    import pandas as pd

    pd.read_csv(atlas / "atlas.csv").to_parquet(atlas / "atlas.parquet")
    result = _json(tools.get_atlas_summary("atlas.parquet", limit=1))
    assert result["rows"] == 3 and len(result["head"]) == 1
    assert {r["value"]: r["rows"] for r in result["counts"]["efit_lineage"]} == {"v1": 2, "v2": 1}


def test_atlas_caps_columns_and_file_size(atlas, monkeypatch):
    monkeypatch.setattr(tools, "MAX_ATLAS_COLUMNS", 2)
    result = _json(tools.get_atlas_summary())
    assert result["column_count"] == 5 and len(result["columns"]) == 2
    assert all(len(row) == 2 for row in result["head"])
    monkeypatch.setattr(tools, "MAX_ATLAS_BYTES", 10)
    with pytest.raises(tools.ToolInputError, match="limit"):
        tools.get_atlas_summary()


def test_atlas_refuses_paths_outside_its_directory_with_one_message(atlas, tmp_path):
    outside = tmp_path / "secret.csv"
    outside.write_text("a\n1\n", encoding="utf-8")
    messages = set()
    for path in ("../secret.csv", str(outside), "C:\\secret.csv", "\\\\server\\share\\x.csv",
                 "//server/share/x.csv", "sub/../../secret.csv", "missing.csv", "atlas.txt", "~/x.csv"):
        with pytest.raises(tools.ToolInputError) as caught:
            tools.get_atlas_summary(path)
        messages.add(str(caught.value).replace(repr(path[:200]), "<path>"))
    assert len(messages) == 1, messages
    if hasattr(os, "symlink"):
        try:
            (atlas / "link.csv").symlink_to(outside)
        except OSError:  # Windows without symlink privilege
            pass
        else:
            with pytest.raises(tools.ToolInputError, match="no readable atlas table"):
                tools.get_atlas_summary("link.csv")
            # an escaping link is not offered either, so the single table is still chosen
            assert tools.get_atlas_summary()["path"] == "atlas.csv"


def test_atlas_needs_its_directory(monkeypatch):
    monkeypatch.delenv(tools.ATLAS_ENV, raising=False)
    with pytest.raises(tools.ToolInputError, match=tools.ATLAS_ENV):
        tools.get_atlas_summary()


# -- the bounded converter ------------------------------------------------------------


def test_bounded_conversion_is_strict_json_and_records_what_it_cut():
    array = np.arange(30.0)
    array[0], array[3] = np.nan, np.inf
    value = {"a": array, "b": tuple(range(5)), "f": len, "s": np.float32(2.5), "t": "abcdefgh"}
    converter = Bounded(max_items=3, max_points=10, max_string=5)
    inner = _json(converter(value))
    assert "f" not in inner
    assert inner["b"] == [0, 1, 2]
    assert inner["s"] == 2.5
    assert inner["t"] == "abcde"
    assert inner["a"]["finite_count"] == 28 and inner["a"]["min"] == 1.0 and inner["a"]["max"] == 29.0
    assert inner["a"]["values"][:2] == [None, None] and len(inner["a"]["values"]) <= 10
    assert any(entry.startswith("$.b") for entry in converter.truncated)
    narrow = Bounded(max_keys=2)
    assert list(_json(narrow({"x": 1, "y": 2, "z": 3}))) == ["x", "y"]
    assert narrow.report()["count"] == 1
    assert _json(Bounded()({1: "one", "1": "uno"})) == {"1": "one", "1#2": "uno"}
    assert preview_strides((129, 129), 25) == (26, 26)
    assert math.prod(math.ceil(n / s) for n, s in zip((129, 129), preview_strides((129, 129), 25))) <= 25


def test_the_byte_cap_holds_for_long_strings_and_says_when_nothing_fits():
    data, converter = bounded_json({f"k{i}": "x" * 19_000 for i in range(10)}, budget=10, max_bytes=50_000)
    assert len(json.dumps(data)) <= 50_000 and converter.truncated
    data, converter = bounded_json({"k" * 5000 + str(i): 1 for i in range(400)}, budget=10, max_bytes=1_000)
    assert len(json.dumps(data)) <= 1_000 and converter.truncated


def test_the_point_budget_is_shared_and_the_byte_cap_holds():
    many = {f"k{i}": np.arange(1000.0) for i in range(100)}
    data, converter = bounded_json(many, budget=200, max_bytes=1_000_000)
    assert _preview_values(data) <= 200
    data, converter = bounded_json(many, budget=200, max_bytes=5_000)
    assert len(json.dumps(data)) <= 5_000 or converter.max_items == 1
    assert converter.report()["count"] >= 1 and len(converter.report()["paths"]) <= REPORTED_PATHS
