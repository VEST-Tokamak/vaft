"""#1331 campaign shot selection: inventory, quality join and tiers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from vaft.machine_mapping.thomson_scattering import thomson_file_shot

SCRIPT = Path(__file__).resolve().parents[1] / "workflow" / "efit_campaign" / "select_shots.py"


@pytest.fixture(scope="module")
def selection():
    spec = importlib.util.spec_from_file_location("select_shots", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name, shot", [
    ("41672_NeTe.mat", 41672), ("NeTe_48224.mat", 48224), ("NeTe_Shot40330_v9_rev.mat", 40330),
    ("Shot40330_v10.mat", 40330), ("IDS_48224.mat", None), ("readme.txt", None),
])
def test_thomson_file_shot_reads_the_layouts_the_resolver_ranks(name, shot):
    assert thomson_file_shot(name) == shot


def test_the_raw_inventory_finds_thomson_and_ion_files(tmp_path, selection):
    for name in ("41672_NeTe.mat", "IDS_48224.mat", "CES_47514.mat", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "thomson_scattering").mkdir()
    (tmp_path / "thomson_scattering" / "NeTe_48224.mat").write_bytes(b"")
    thomson, ions = selection.raw_inventory(tmp_path)
    assert thomson == {41672: "41672_NeTe.mat", 48224: "NeTe_48224.mat"}
    assert ions == {48224: "IDS_48224.mat", 47514: "CES_47514.mat"}


def _quality(shot, verdict="degraded", condemned=2, outboard=20, status=None):
    if status == "absent":
        return {"shot": shot, "status": "absent"}
    return {"shot": shot, "verdict": verdict, "condemned": [f"p{i}" for i in range(condemned)],
            "families": {"outboard": {"usable": outboard}}}


def test_each_shot_lands_in_the_first_tier_that_applies(selection):
    t = selection.Thresholds()
    overview = [
        {"shot": 1, "max_ip_kA": 100.0, "pulse_duration_s": 0.02},   # A: Thomson + 39915-like
        {"shot": 2, "max_ip_kA": 280.0, "pulse_duration_s": 0.02},   # B: Thomson + record Ip, weak magnetics
        {"shot": 3, "max_ip_kA": 120.0, "pulse_duration_s": 0.02},   # C: Thomson, no magnetics
        {"shot": 4, "max_ip_kA": 290.0, "pulse_duration_s": 0.02},   # D: no Thomson, important + good
        {"shot": 5, "max_ip_kA": 100.0, "pulse_duration_s": 0.036},  # long pulse, unfit: nothing
        {"shot": 6, "max_ip_kA": 100.0, "pulse_duration_s": 0.02},   # ordinary, no Thomson: nothing
    ]
    quality = {1: _quality(1), 2: _quality(2, condemned=17, outboard=18), 3: _quality(3, status="absent"),
               4: _quality(4), 5: _quality(5, verdict="unfit", condemned=40, outboard=7), 6: _quality(6)}
    thomson = {1: "1_NeTe.mat", 2: "2_NeTe.mat", 3: "NeTe_3.mat", 5: "5_NeTe.mat"}
    rows = {r["shot"]: r for r in selection.select(overview, quality, thomson, {}, t)}
    assert [rows[s]["tier"] for s in range(1, 7)] == ["A", "B", "C", "D", None, None]
    assert rows[5]["important"] and not rows[5]["good_magnetics"]


def test_a_shot_never_scanned_is_not_called_good(selection):
    rows = selection.select([{"shot": 7, "max_ip_kA": 90.0, "pulse_duration_s": 0.02}], {},
                            {7: "7_NeTe.mat"}, {}, selection.Thresholds())
    assert rows[0]["magnetics"] == "not_scanned" and rows[0]["tier"] is None


def test_a_later_quality_table_wins_and_directories_are_read(tmp_path, selection):
    (tmp_path / "a.json").write_text(json.dumps({"rows": [_quality(1, condemned=9)]}))
    (tmp_path / "b.json").write_text(json.dumps({"rows": [_quality(1, condemned=1)]}))
    assert len(selection.quality_by_shot([tmp_path])[1]["condemned"]) == 1


def test_the_command_line_writes_json_and_markdown(tmp_path, selection):
    overview = tmp_path / "overview.csv"
    overview.write_text("shot,max_ip_kA,pulse_duration_s,shot_class\n1,100,0.02,Plasma\n")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "1_NeTe.mat").write_bytes(b"")
    table = tmp_path / "mq.json"
    table.write_text(json.dumps({"rows": [_quality(1)]}))
    out, md = tmp_path / "sel.json", tmp_path / "sel.md"
    assert selection.main(["--overview", str(overview), "--quality", str(table), "--data-root", str(tmp_path / "raw"),
                           "--output", str(out), "--markdown", str(md)]) == 0
    payload = json.loads(out.read_text())
    assert payload["rows"][0]["tier"] == "A" and payload["thresholds"]["max_condemned"] == 4
    assert "## Tier A (1)" in md.read_text()
