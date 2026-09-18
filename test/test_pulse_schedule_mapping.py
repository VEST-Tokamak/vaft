"""ShotLog records -> pulse_schedule.event, and the shotlog stage product (#995)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from omas import ODS
import pytest

from _shotlog_fixtures import legacy_workbook, modern_workbook
from vaft.database import sources
from vaft.database.filedb import FileDB, OMASStage
from vaft.database.shotlog import build_shot_records, convert_directory, write_extraction
from vaft.machine_mapping.pulse_schedule import (
    PulseScheduleUnavailableError,
    map_pulse_schedule,
    pulse_schedule,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def filedb(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    modern_workbook(
        source / "ShotLog_2025_09 #46590-46599.xlsx",
        {"250915": [
            {"shot": 46590, "diagnostics": {"IF": "490-510", "SXR1": 485},
             "ech": "100-450", "nbi_t0": 494, "nbi_dt": 3, "gas": ("90V(H2)", "300-302")},
            {"shot": 46591},
        ]},
    )
    legacy_workbook(source / "ShotLog_2014_01.xlsx")
    db = FileDB(tmp_path / "filedb")
    sessions, manifest = convert_directory(source)
    write_extraction(sessions, build_shot_records(sessions), manifest, db)
    return db


def _events(ods):
    return {
        ods[f"pulse_schedule.event.{index}.identifier"]: ods[f"pulse_schedule.event.{index}"]
        for index in range(len(ods["pulse_schedule.event"]))
    }


def test_logged_triggers_become_events_on_the_daq_clock(filedb):
    ods = pulse_schedule(ODS(), 46590, data_root=filedb.root / "legacy/shotlog")
    events = _events(ods)
    interferometer = events["diagnostic:IF"]
    assert interferometer["time_stamp"] == pytest.approx(0.290)
    assert interferometer["duration"] == pytest.approx(0.020)
    assert list(interferometer["listeners"]) == ["interferometer"]
    assert interferometer["type.name"] == "diagnostic_trigger"
    assert interferometer["type.index"] < 0  # private: the DD has no event enumeration
    assert events["ec:ECH (2.45 GHz)"]["time_stamp"] == pytest.approx(-0.100)
    assert events["ec:ECH (2.45 GHz)"]["duration"] == 0.35
    assert events["nbi:NBI T0 (ms)"]["duration"] == pytest.approx(0.003)
    assert events["gas:LFS"]["time_stamp"] == pytest.approx(0.100)
    assert ods["pulse_schedule.ids_properties.homogeneous_time"] == 2
    parameters = json.loads(ods["pulse_schedule.code.parameters"])
    assert parameters["daq_offset_ms"] == -200
    assert parameters["schema_version"] == "modern_v4"


def test_every_event_names_the_cell_it_came_from(filedb):
    ods = pulse_schedule(ODS(), 46590, data_root=filedb.root / "legacy/shotlog")
    sources_ = [
        ods[f"pulse_schedule.ids_properties.provenance.node.{index}.sources"][0]
        for index in range(len(ods["pulse_schedule.event"]))
    ]
    assert all("sha256=" in text and "cell=" in text for text in sources_)
    gas = next(text for text in sources_ if "valve_voltage_V" in text)
    assert "valve_voltage_V=90" in gas and "species=H2" in gas


def test_a_carried_forward_trigger_is_marked_inherited(filedb):
    ods = pulse_schedule(ODS(), 46591, data_root=filedb.root / "legacy/shotlog")
    assert "diagnostic:IF" in _events(ods)
    text = ods["pulse_schedule.ids_properties.provenance.node.0.sources"][0]
    assert "inherited=true" in text and "source_shot=46590" in text


def test_a_shot_the_shotlog_never_mentions_is_unavailable(filedb):
    with pytest.raises(FileNotFoundError):
        pulse_schedule(ODS(), 1, data_root=filedb.root / "legacy/shotlog")


def test_a_pre_card_shot_has_a_record_but_nothing_to_schedule(filedb):
    with pytest.raises(PulseScheduleUnavailableError):
        pulse_schedule(ODS(), 7344, data_root=filedb.root / "legacy/shotlog")


def test_unreadable_triggers_are_not_scheduled():
    record = {
        "shot": 1, "source": {"workbook": "w", "sha256": "0", "sheet": "s"},
        "effective_timing": [{"system": "gas", "identifier": "LFS", "raw": "300-303.5 490-",
                              "window_ms": None, "cell": "U5"}],
    }
    with pytest.raises(PulseScheduleUnavailableError):
        map_pulse_schedule(ODS(), record)


def test_shotlog_is_an_optional_corrective_stage_owning_pulse_schedule():
    assert OMASStage("shotlog")
    entry = sources.replication_for_stage("shotlog")
    assert entry.ids == ("pulse_schedule",)
    assert entry.optional and entry.produced_by == "corrective"


def _ingest_module():
    script = ROOT / "workflow/automatic_pipeline_2_corrective_data_update/ingest_external_diagnostics.py"
    spec = importlib.util.spec_from_file_location("ingest_external_diagnostics_shotlog", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_corrective_ingest_builds_products_and_records_the_rest(filedb):
    ingest = _ingest_module()
    summary = ingest.ingest(filedb.root, ["shotlog"])
    assert (summary["succeeded"], summary["unavailable"], summary["failed"]) == (2, 1, 0)
    product = filedb.omas_product("shotlog", shot=46590)
    assert product.name == "shotlog.json.gz" and product.exists()
    manifest = json.loads(filedb.omas_manifest("shotlog", shot=46590).read_text())
    assert manifest["measured"]["count"] >= 4
    assert manifest["provenance"]["record"] == "legacy/shotlog/46590/metadata/shotlog.json"
    assert len(manifest["provenance"]["record_sha256"]) == 64
    # A shot with nothing to schedule is not retried on every run...
    assert ingest.ingest(filedb.root, ["shotlog"])["skipped"] == 3
    # ...until its record changes (the card filled in later, a parser fix).
    record_file = filedb.root / "legacy/shotlog/46590/metadata/shotlog.json"
    record = json.loads(record_file.read_text())
    record["effective_timing"] = record["effective_timing"][:1]
    record_file.write_text(json.dumps(record))
    rerun = ingest.ingest(filedb.root, ["shotlog"])
    assert (rerun["succeeded"], rerun["skipped"]) == (1, 2)
