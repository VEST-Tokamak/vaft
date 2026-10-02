"""MCP Phase 2: datasets by shot or artifact, equilibrium summaries, campaign atlas tables (#188).

The atlas tests build a miniature atlas in each lane's own schema format --
JSON Schema (K, Z, D), the transport ``columns`` map (T), the stability
``column_dictionary`` with ``{t}/{r}/{s}`` patterns (N, version 1 spellings
and version 2) and MANIFEST units (V) -- because the real one lives on
vestserver.  Dataset tests read the packaged shot 39915 and compare with a
direct call to the API the tool wraps.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaft.mcp import _atlas
from vaft.mcp import _tools as tools


def _json(result):
    return json.loads(json.dumps(result, allow_nan=False))


# -- a miniature campaign atlas ------------------------------------------------------


def _write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _json_schema(columns):
    return json.dumps({"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
                       "properties": {c: {"type": ["string", "null"], "description": d} for c, d in columns.items()}})


@pytest.fixture
def atlas(tmp_path, monkeypatch):
    root = tmp_path / "atlas"
    _write(root, "v1/state.csv",
           "contract_version,shot,time_efit_s,efit_lineage,efit_quality,r_w,q95\n"
           "1,39915,0.319,magnetics,good,1.1,6.6\n"
           "1,39915,0.32,magnetics,admissible,1.4,6.3\n"
           "1,39915,0.32,electron_kinetic,admissible,1.2,6.2\n"
           "1,42929,0.3,magnetics,good,,5.0\n")
    _write(root, "v1/schema/state.schema.json",
           _json_schema({"shot": "VEST shot number", "r_w": "stored-energy ratio", "q95": "edge safety factor"}))
    _write(root, "v1/MANIFEST.json", json.dumps({
        "contract_version": "1", "generated_at": "2026-10-01T00:00:00Z", "vaft_git": "abc123",
        "command": ["build_state.py", "--out", "/home/someone/secret/place"]}))
    _write(root, "transport/atlas.csv",
           "shot,time_efit_s,efit_lineage,r_over_a,efit_quality,tglf_config,qe_gb\n"
           "39915,0.319,magnetics,0.3,good,tglf-sat3-em-bper,1.5\n"
           "39915,0.319,magnetics,0.5,good,tglf-sat3-em-bper,2.5\n")
    _write(root, "transport/schema.json", json.dumps({
        "description": "TGLF atlas", "columns": {
            "qe_gb": {"unit": "gB", "category": "tglf_predicted", "definition": "electron heat flux"}}}))
    _write(root, "transport_sensitivity/sensitivity.csv",
           "shot,time_efit_s,efit_lineage,r_over_a,tglf_config,q_tot_gb\n"
           "39915,0.319,magnetics,0.5,tglf-sat0-es,1.0\n"
           "39915,0.319,magnetics,0.5,tglf-sat3-es,5.0\n")
    _write(root, "transport_sensitivity/sensitivity_pairs.csv",
           "shot,time_efit_s,efit_lineage,r_over_a,q_tot_gb_sat_spread_es\n"
           "39915,0.319,magnetics,0.3,0.4\n39915,0.319,magnetics,0.5,1.3\n39915,0.32,magnetics,0.5,\n")
    _write(root, "transport_sensitivity/schema.json", json.dumps({
        "columns": {}, "pair_columns": {"q_tot_gb_sat_spread_es": {"unit": "-", "definition": "SAT spread"}}}))
    # Lane N, atlas version 1: the pre-contract spellings.
    _write(root, "stability/atlas_n.csv",
           "shot,time_efit_s,efit_lineage,efit_label,n_tor,dcon_full_256_W_t,ideal_stable_full_edge,atlas_version\n"
           "39915,0.319,magnetics-only,good,1,0.5,True,1\n"
           "39915,0.319,magnetics-only,good,2,-0.2,False,1\n"
           "39915,0.32,electron-kinetic,admissible,1,-0.1,False,1\n")
    _write(root, "stability/atlas_surfaces.csv",
           "shot,time_efit_s,efit_lineage,efit_label,n_tor,solver,m,delta_prime\n"
           "39915,0.319,magnetics-only,good,1,rdcon,2,3.0\n39915,0.319,magnetics-only,good,1,stride,2,2.0\n")
    _write(root, "stability/schema.json", json.dumps({
        "rules": ["no cross-n combination"],
        "column_patterns": {"{t}": "full | trunc", "{r}": "256 | 512"},
        "column_dictionary": {
            "efit_label": {"type": "str", "unit": "", "meaning": "#1331 label"},
            "dcon_{t}_{r}_W_t": {"type": "float", "unit": "", "meaning": "DCON total energy"}}}))
    _write(root, "stability/README.md", "# Stability atlas\nNever combine values across n.\n")
    _write(root, "zeff/zeff.csv",
           "contract_version,shot,t_start_s,t_end_s,window_kind,status,zeff,conductivity_model\n"
           "zeff-1,39916,0.30,0.32,flattop,ok,1.74,redl\n")
    _write(root, "confinement/table.csv",
           "record_id,shot,time_efit_s,efit_lineage,efit_quality,tau_e_th_s,accepted\nVEST:39915:319,39915,0.319,magnetics,good,0.001,True\n")
    _write(root, "lane_v/efit_base.csv",
           "shot,time_efit_s,efit_lineage,efit_quality,normalized_beta\n39915,0.319,magnetics,good,0.05\n")
    _write(root, "lane_v/MANIFEST.json", json.dumps({"units": {"normalized_beta": "-"}}))
    monkeypatch.setenv("VAFT_ATLAS_DIR", str(root))
    return root


def test_the_atlas_listing_names_every_table_and_what_is_present(atlas):
    result = _json(tools.list_atlas_tables())
    rows = {row["name"]: row for row in result["items"]}
    assert set(rows) == set(_atlas.BY_NAME)
    assert rows["state"]["present"] and rows["state"]["rows"] == 4
    assert not rows["kinetic_profiles"]["present"] and rows["kinetic_profiles"]["rows"] is None
    assert rows["stability"]["must_filter"] == ["n_tor"]
    assert rows["stability"]["lane"] == "N"


def test_each_schema_format_describes_its_columns(atlas):
    state = _json(tools.describe_atlas_table("state"))
    columns = {c["name"]: c for c in state["columns"]}
    assert columns["q95"]["definition"] == "edge safety factor"
    assert state["provenance"]["vaft_git"] == "abc123" and len(state["provenance"]["sha256"]) == 64
    assert "command" not in state["provenance"]  # build commands carry absolute paths
    assert any("censored" in caveat for caveat in state["caveats"])

    transport = {c["name"]: c for c in _json(tools.describe_atlas_table("transport"))["columns"]}
    assert transport["qe_gb"]["unit"] == "gB" and transport["qe_gb"]["category"] == "tglf_predicted"

    pairs = {c["name"]: c for c in _json(tools.describe_atlas_table("transport_sensitivity_pairs"))["columns"]}
    assert pairs["q_tot_gb_sat_spread_es"]["definition"] == "SAT spread"

    stability = _json(tools.describe_atlas_table("stability"))
    columns = {c["name"]: c for c in stability["columns"]}
    assert "efit_quality" in columns and "efit_label" not in columns
    assert columns["dcon_full_256_W_t"]["pattern"] == "dcon_{t}_{r}_W_t"
    assert "no cross-n combination" in stability["rules"]
    assert "Never combine values across n" in stability["readme"]

    base = {c["name"]: c for c in _json(tools.describe_atlas_table("op_space_base"))["columns"]}
    assert base["normalized_beta"]["unit"] == "-"


def test_old_spellings_come_back_in_contract_v1_spelling(atlas):
    result = _json(tools.query_atlas_table("stability", where=[{"column": "n_tor", "op": "==", "value": 1}]))
    assert {row["efit_lineage"] for row in result["rows"]} == {"magnetics", "electron_kinetic"}
    assert all("efit_quality" in row and "efit_label" not in row for row in result["rows"])
    # The old spellings are accepted as filter input too.
    old = _json(tools.query_atlas_table("stability", where=[
        {"column": "n_tor", "op": "==", "value": 1},
        {"column": "efit_label", "op": "==", "value": "good"},
        {"column": "efit_lineage", "op": "==", "value": "magnetics-only"}]))
    assert old["total_matched"] == 1 and old["rows"][0]["time_efit_s"] == 0.319
    assert old["provenance"]["atlas_version"] == ["1"]


def test_lane_rules_are_refusals(atlas):
    with pytest.raises(tools.ToolInputError, match="n_tor"):
        tools.query_atlas_table("stability")
    with pytest.raises(tools.ToolInputError, match="n_tor"):
        tools.query_atlas_table("stability", where=[{"column": "n_tor", "op": "in", "value": [1, 2]}])
    with pytest.raises(tools.ToolInputError, match="solver"):
        tools.query_atlas_table("stability_surfaces", where=[{"column": "n_tor", "op": "==", "value": 1}])
    with pytest.raises(tools.ToolInputError, match="tglf_config"):
        tools.query_atlas_table("transport_sensitivity")
    surfaces = _json(tools.query_atlas_table("stability_surfaces", where=[
        {"column": "n_tor", "op": "==", "value": 1}, {"column": "solver", "op": "==", "value": "stride"}]))
    assert [row["delta_prime"] for row in surfaces["rows"]] == [2.0]


def test_queries_filter_sort_select_and_count(atlas):
    result = _json(tools.query_atlas_table(
        "state",
        where=[{"column": "shot", "op": "==", "value": 39915}, {"column": "r_w", "op": ">", "value": 1.15}],
        columns=["r_w"], order_by=["-r_w"], count_by=["efit_lineage"]))
    assert result["total_matched"] == 2
    assert result["columns"] == ["shot", "time_efit_s", "efit_lineage", "r_w"]
    assert [row["r_w"] for row in result["rows"]] == [1.4, 1.2]
    assert {c["value"]: c["rows"] for c in result["counts"]["efit_lineage"]} == {"magnetics": 1, "electron_kinetic": 1}
    missing = _json(tools.query_atlas_table("state", where=[{"column": "r_w", "op": "isnull"}]))
    assert missing["rows"][0]["shot"] == 42929 and missing["rows"][0]["r_w"] is None
    top = _json(tools.query_atlas_table("transport_sensitivity_pairs", order_by=["-q_tot_gb_sat_spread_es"], limit=1))
    assert top["rows"][0]["r_over_a"] == 0.5 and top["total_matched"] == 3
    assert top["truncated"]["count"] >= 1  # 3 matched, 1 kept
    zeff = _json(tools.query_atlas_table("zeff_windows"))
    assert "conductivity_model" in zeff["columns"] and any("conductivity_model" in r for r in zeff["rules"])


def test_queries_refuse_unknown_columns_and_operators(atlas):
    for bad in ([{"column": "nope", "op": "==", "value": 1}], [{"column": "shot", "op": "~=", "value": 1}],
                [{"column": "shot", "op": "in", "value": 3}], ["shot == 1"]):
        with pytest.raises(tools.ToolInputError):
            tools.query_atlas_table("state", where=bad)
    with pytest.raises(tools.ToolInputError, match="unknown columns"):
        tools.query_atlas_table("state", columns=["shot", "__class__"])
    with pytest.raises(tools.ToolInputError, match="unknown atlas table"):
        tools.query_atlas_table("../../etc/passwd")


def test_the_atlas_reads_only_inside_its_directory(atlas, tmp_path):
    outside = tmp_path / "outside.csv"
    outside.write_text("shot,time_efit_s,efit_lineage\n1,0.1,magnetics\n", encoding="utf-8")
    (atlas / "v1" / "state.csv").unlink()
    try:
        (atlas / "v1" / "state.csv").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(tools.ToolInputError, match="not present"):
        tools.query_atlas_table("state")


def test_the_atlas_needs_its_directory(monkeypatch):
    monkeypatch.delenv("VAFT_ATLAS_DIR", raising=False)
    with pytest.raises(tools.ToolInputError, match="VAFT_ATLAS_DIR is not set"):
        tools.list_atlas_tables()


def test_atlas_results_stay_under_the_byte_cap(atlas):
    wide = ",".join(f"c{i}" for i in range(300))
    _write(atlas, "lane_v/efit_base.csv", "shot,time_efit_s,efit_lineage," + wide + "\n" +
           "\n".join(f"{i},0.3,magnetics," + ",".join(["x" * 150] * 300) for i in range(400)))
    result = tools.query_atlas_table("op_space_base", columns=[f"c{i}" for i in range(250)], limit=500)
    assert len(json.dumps(result, allow_nan=False)) < 60_000
    assert result["truncated"]["count"] >= 1


# -- datasets -------------------------------------------------------------------------


def test_a_packaged_shot_is_inspected_with_provenance():
    result = _json(tools.inspect_dataset(shot=39915))
    assert result["dataset"]["kind"] == "sample" and result["dataset"]["shot"] == 39915
    equilibrium = next(row for row in result["items"] if row["ids"] == "equilibrium")
    assert equilibrium["times"] >= 1 and equilibrium["time_first_s"] <= equilibrium["time_last_s"]


def test_equilibrium_summary_matches_by_time_and_equals_the_direct_extraction():
    from vaft.database._summary import extract_equilibrium_global
    from vaft.omas.sample import sample_ods

    direct = extract_equilibrium_global(sample_ods(39915), 39915)
    times = _json(tools.list_equilibrium_times(shot=39915))["times_s"]
    assert times == [row["time_s"] for row in direct]
    target = direct[3]
    result = _json(tools.get_equilibrium_summary(shot=39915, time=target["time_s"] + 1e-5))
    assert result["matched"]["time_s"] == target["time_s"]
    row = result["slices"][0]
    for key in ("q_95", "beta_normal", "li_3", "ip_kA"):
        assert row[key] == pytest.approx(target[key], rel=1e-12, nan_ok=True)
    with pytest.raises(tools.ToolInputError, match="no slice within"):
        tools.get_equilibrium_summary(shot=39915, time=target["time_s"] + 0.5)


def test_a_data_path_is_read_without_creating_it():
    from vaft.omas.sample import sample_ods

    result = _json(tools.inspect_data_path("equilibrium.time_slice.*.global_quantities.q_95", shot=39915,
                                           time=0.32, tolerance=1e-3))
    ods = sample_ods(39915)
    index = result["matched"]["slice"]
    assert float(ods["equilibrium.time"][index]) == pytest.approx(result["matched"]["time_s"])
    assert result["value"] == pytest.approx(float(ods[f"equilibrium.time_slice.{index}.global_quantities.q_95"]))
    assert _json(tools.inspect_data_path("summary.global_quantities.no_such_leaf", shot=39915))["present"] is False
    with pytest.raises(tools.ToolInputError, match="pass time"):
        tools.inspect_data_path("equilibrium.time_slice.*.global_quantities.q_95", shot=39915)


def test_a_shot_without_a_sample_needs_the_database_switch(monkeypatch):
    import vaft.database

    monkeypatch.delenv("VAFT_MCP_DATABASE", raising=False)
    monkeypatch.setattr(vaft.database, "load", lambda *a, **k: pytest.fail("database contacted"))
    with pytest.raises(ValueError, match="VAFT_MCP_DATABASE=1"):
        tools.inspect_dataset(shot=1)
    with pytest.raises(ValueError, match="VAFT_MCP_DATABASE=1"):
        tools.inspect_dataset(shot=39915, database_source="main")


def test_a_database_failure_reports_its_class_and_never_its_text(monkeypatch):
    import vaft.database

    def refuse(*args, **kwargs):
        raise PermissionError("403 for user hs_user at http://hsds.internal:5101 password=hunter22")

    monkeypatch.setenv("VAFT_MCP_DATABASE", "1")
    monkeypatch.setattr(vaft.database, "load", refuse)
    with pytest.raises(ValueError) as caught:
        tools.inspect_dataset(shot=1, database_source="main")
    text = str(caught.value)
    assert "PermissionError" in text and "hunter22" not in text and "hsds.internal" not in text


def test_artifacts_resolve_only_inside_their_directory(tmp_path, monkeypatch):
    from vaft.data.resources import data_path, require_repository_sample

    root = tmp_path / "artifacts"
    root.mkdir()
    gfile = require_repository_sample(data_path("efit/g039915.00319"))
    (root / "g039915").write_bytes(Path(gfile).read_bytes())
    monkeypatch.setenv("VAFT_ARTIFACT_DIR", str(root))
    result = _json(tools.list_equilibrium_times(artifact="g039915"))
    assert result["dataset"]["kind"] == "artifact" and result["dataset"]["artifact"] == "g039915"
    assert result["count"] == 1
    assert str(tmp_path) not in json.dumps(result)
    for path in ("../outside", "/etc/passwd", "~/x", "C:\\x", "missing"):
        with pytest.raises(tools.ToolInputError, match="no readable artifact"):
            tools.inspect_dataset(artifact=path)
    monkeypatch.delenv("VAFT_ARTIFACT_DIR")
    with pytest.raises(tools.ToolInputError, match="VAFT_ARTIFACT_DIR is not set"):
        tools.inspect_dataset(artifact="g039915")


def test_exactly_one_dataset_is_named():
    with pytest.raises(ValueError, match="exactly one"):
        tools.inspect_dataset()
    with pytest.raises(ValueError, match="exactly one"):
        tools.inspect_dataset(shot=39915, artifact="x")
