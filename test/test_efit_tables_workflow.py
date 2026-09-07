"""The Green-table workflow tools (issue #194): comparison and A/B bookkeeping.

The tools are path-run scripts; they are imported here by file so their
pure parts stay covered without EFIT or EFUND.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lfs_assets import skip_unless_materialized
from vaft.data.resources import data_path

WORKFLOW = Path(__file__).resolve().parents[1] / "workflow" / "efit_tables"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"efit_tables_{name}", WORKFLOW / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def compare_tables():
    return _load("compare_tables")


@pytest.fixture(scope="module")
def ab_efit_table():
    return _load("ab_efit_table")


def test_a_table_compared_with_itself_is_identical_everywhere(compare_tables):
    bundled = Path(data_path("efit"))
    skip_unless_materialized(*sorted(bundled.glob("*.ddd")))
    report = compare_tables.compare(bundled, bundled)
    assert all(row["same"] for row in report["mhdin"]["counts"].values())
    for block in report["mhdin"]["geometry"].values():
        for delta in block.values():
            if isinstance(delta, dict) and "identical" in delta:
                assert delta["identical"], delta
    assert all(row["turns_diff"] == 0 for row in report["mhdin"]["pf_groups"]["groups"])
    tables = report["tables"]["129129"]
    for file_key in ("ep", "ec", "rfcoil", "rv"):
        for label, delta in tables[file_key]["by_record"].items():
            assert delta.get("identical"), (file_key, label)
    assert report["a"]["identity"]["provenance"] == "unrecorded"
    text = compare_tables.markdown(report)
    assert "## Tables at 129x129" in text and "| ep | rsilpc |" in text


def test_kfile_diff_accepts_only_the_two_directory_lines(ab_efit_table, tmp_path):
    a = tmp_path / "A"
    b = tmp_path / "B"
    a.mkdir()
    b.mkdir()
    body = " &IN1\n ISHOT = 39915\n INPUT_DIR = '/tables/a/' \n TABLE_DIR = '/tables/a/' \n PLASMA = 80000.0\n /\n"
    (a / "k039915.00319").write_text(body)
    (b / "k039915.00319").write_text(body.replace("/tables/a/", "/tables/b/"))
    report = ab_efit_table.kfile_diff(a, b, 39915)
    assert report["only_directory_lines_differ"] is True
    assert report["files"]["k039915.00319"]["changed_lines"] == 2

    (b / "k039915.00319").write_text(body.replace("/tables/a/", "/tables/b/").replace("80000.0", "80001.0"))
    report = ab_efit_table.kfile_diff(a, b, 39915)
    assert report["only_directory_lines_differ"] is False
    assert any("PLASMA" in line for line in report["files"]["k039915.00319"]["non_directory_changes"])

    (b / "k039915.00323").write_text(body)
    assert ab_efit_table.kfile_diff(a, b, 39915)["names_equal"] is False


def test_log_parser_splits_slices_on_the_iteration_counter(ab_efit_table):
    text = (
        " t= 316 it=  1 chi2= 1.0E+03 ... err= 1.0E-01\n"
        " t= 316 it=  2 chi2= 2.0E+02 ... err= 1.0E-02\n"
        " ERROR in bound\n"
        " t= 319 it=  1 chi2= 9.0E+01 ... err= 3.0E-02\n"
        " t= 319 it=  2 chi2= 7.4E+01 ... err= 8.5E-03\n"
        " t= 319 it=  3 chi2= 7.4E+01 ... err= 8.5E-03\n"
    )
    blocks = ab_efit_table.iterations_from_log(text)
    assert [block["iterations_n"] for block in blocks] == [2, 3]
    assert blocks[0]["bound_error"] is True and blocks[1]["bound_error"] is False
    assert blocks[1]["chi2_log"] == 74.0 and blocks[1]["gs_error_log"] == 8.5e-3


def test_reference_rows_read_the_stored_39915_reference(ab_efit_table):
    rows = ab_efit_table.reference_rows(Path(data_path("efit")), 39915)
    assert "00319" in rows
    assert rows["00319"]["jflag"] == 1
    assert rows["00319"]["chisq"] == pytest.approx(77.62, abs=0.05)
    assert rows["00319"]["rmaxis"] > 0 and rows["00319"]["r_lcfs_max"] > rows["00319"]["r_lcfs_min"]
