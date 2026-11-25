"""``summary(..., provenance=True)`` records how every value was obtained.

One row per (row key, column) with a status -- computed, copied, fallback,
failed, missing, unrecorded -- the step, a reason and the decisive inputs. The
record changes no value, and the branches that change values without saying so
(a rejected R0, a stored B0, an updater that raises, a slice with no time) are
the ones it must name.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy

import numpy as np
import pandas as pd
import pytest

import vaft
from vaft import database
from vaft.database import _summary_provenance as provenance

SHOT = 42962


@pytest.fixture(scope="module")
def sample():
    return vaft.omas.sample_ods(SHOT)


@pytest.fixture(scope="module")
def recorded(sample):
    return database.summary_from_ods({SHOT: sample}, preset="equilibrium_global", provenance=True)


def _run(ods):
    return database.summary_from_ods({SHOT: ods}, preset="equilibrium_global", provenance=True)


def _entry(table: pd.DataFrame, column: str, eq_index: int = 1) -> pd.Series:
    record = table.attrs["provenance"].table
    rows = record[(record["eq_index"] == eq_index) & (record["column"] == column)]
    assert len(rows) == 1, (column, rows)
    return rows.iloc[0]


def test_the_record_changes_no_value(sample, recorded):
    plain = database.summary_from_ods({SHOT: sample}, preset="equilibrium_global")
    assert "provenance" not in plain.attrs
    pd.testing.assert_frame_equal(recorded, plain)


def test_every_row_and_column_has_a_status(recorded):
    preset = database.get_summary_preset("equilibrium_global")
    record = recorded.attrs["provenance"].table
    expected = len(recorded) * (len(preset.columns) - len(preset.key_columns) + 1)  # + time_s
    assert len(record) == expected
    assert set(record["status"]) <= set(provenance.STATUSES)
    assert "unrecorded" not in set(record["status"])  # this extractor records everything
    assert not record.duplicated(["shot", "eq_index", "column"]).any()
    assert (record["reason"].str.len() > 0)[record["status"] != "computed"].all()


def test_status_agrees_with_the_value(recorded):
    record = recorded.attrs["provenance"].table
    values = recorded.melt(id_vars=["shot", "eq_index"], var_name="column", value_name="value")
    joined = record.merge(values, on=["shot", "eq_index", "column"], validate="one_to_one")
    numeric = pd.to_numeric(joined["value"], errors="coerce")
    nan_status = joined["status"].isin(["failed", "missing"])
    assert numeric[nan_status].isna().all()
    assert joined.loc[joined["status"].isin(["computed", "copied", "fallback"]), "value"].notna().all()


def test_recomputed_and_stored_columns_are_told_apart(recorded):
    li3 = _entry(recorded, "li_3")
    assert li3["status"] == "computed"
    assert li3["step"] == "update_equilibrium_global_quantities_beta_li"
    assert li3["inputs"].startswith("R0: ")
    ip = _entry(recorded, "ip_kA")
    assert ip["status"] == "copied" and ip["reason"] == "stored global_quantities.ip"
    assert _entry(recorded, "time_s")["status"] == "copied"


def test_the_packaged_tf_check_does_not_count_as_a_substitution(recorded):
    """42962 stores r0 = 0.4 = tf.r0: the tf branch runs but substitutes nothing."""
    li3 = _entry(recorded, "li_3")
    assert "tf.r0" in li3["inputs"] and li3["status"] == "computed"


def test_a_rejected_r0_is_a_fallback(sample):
    ods = copy.deepcopy(sample)
    ods["equilibrium.vacuum_toroidal_field.r0"] = 0.25
    table = _run(ods)
    plain = database.summary_from_ods({SHOT: ods}, preset="equilibrium_global")
    pd.testing.assert_frame_equal(table, plain)
    for column in ("beta_pol", "li_3", "beta_tor"):
        entry = _entry(table, column)
        assert entry["status"] == "fallback", column
        assert "stored r0 = 0.2500 m was rejected" in entry["inputs"], column
    # b0 rebuilt as |f|/R0 and reported as b0*R0/0.4: the radius cancels
    b0 = _entry(table, "vacuum_b0_T")
    assert b0["status"] == "computed" and "R0: " not in b0["inputs"]
    assert _entry(table, "ip_kA")["status"] == "copied"


def test_a_value_recomputed_to_the_stored_number_is_computed(sample):
    """A product already processed at ingest stores what the updaters recompute."""
    from vaft.database._summary import _preset_view, extract_equilibrium_global

    processed = _preset_view(sample, database.get_summary_preset("equilibrium_global"))
    extract_equilibrium_global(processed, SHOT)
    table = _run(processed)
    for column in ("li_3", "volume_m3", "energy_mhd_J"):
        entry = _entry(table, column)
        assert entry["status"] == "computed", (column, entry["reason"])
        assert entry["reason"].startswith("recomputed by "), column


def test_a_stored_b0_is_a_fallback(sample):
    ods = copy.deepcopy(sample)
    for index in range(len(ods["equilibrium.time_slice"])):
        del ods["equilibrium.time_slice"][index]["profiles_1d"]["f"]
    table = _run(ods)
    pd.testing.assert_frame_equal(table, database.summary_from_ods({SHOT: ods}, preset="equilibrium_global"))
    for column in ("beta_tor", "normalized_plasma_current", "vacuum_b0_T"):
        entry = _entry(table, column)
        assert entry["status"] == "fallback", column
        assert "no finite profiles_1d.f at the boundary" in entry["inputs"], column


def test_a_failing_updater_is_failed_with_its_reason(sample, monkeypatch):
    def broken(ods, time_slice=None):
        raise RuntimeError("synthetic failure")

    broken.__name__ = "update_equilibrium_stored_energy"
    monkeypatch.setattr(vaft.omas, "update_equilibrium_stored_energy", broken)
    entry = _entry(_run(copy.deepcopy(sample)), "energy_mhd_J")
    assert entry["status"] == "failed"
    assert entry["reason"] == "update_equilibrium_stored_energy raised RuntimeError: synthetic failure"


def test_a_slice_without_time_is_a_fallback(sample):
    ods = copy.deepcopy(sample)
    for index in range(len(ods["equilibrium.time_slice"])):
        del ods["equilibrium.time_slice"][index]["time"]
    del ods["equilibrium"]["time"]
    entry = _entry(_run(ods), "time_s")
    assert entry["status"] == "fallback" and "slice index" in entry["reason"]


def test_a_skipped_slice_says_why():
    """The updater's own warning about a slice becomes that slice's reason (39915 slice 8)."""
    table = database.summary_from_ods({39915: vaft.omas.sample_ods(39915)},
                                      preset="equilibrium_global", provenance=True)
    record = table.attrs["provenance"].table
    for column, step in (("li_3", "update_equilibrium_global_quantities_beta_li"),
                         ("volume_m3", "update_equilibrium_global_quantities_volume"),
                         ("energy_mhd_J", "update_equilibrium_stored_energy")):
        missing = record[(record["column"] == column) & (record["status"] == "missing")]
        assert len(missing), column
        assert missing["reason"].str.contains(f"{step}: .*time.slice", regex=True).all(), column


def test_a_volume_beta_failure_keeps_its_exception(sample, monkeypatch):
    import vaft.omas.process_wrapper as wrapper

    def broken(*args, **kwargs):
        raise ValueError("synthetic field integral")

    monkeypatch.setattr(wrapper, "compute_magnetic_energy", broken)
    entry = _entry(_run(copy.deepcopy(sample)), "beta_volume_B2")
    assert entry["status"] == "missing"
    assert entry["reason"] == "volume beta raised ValueError: synthetic field integral"


def test_warnings_from_another_recorder_are_not_taken():
    """The capture handler sits on the shared logger; it keeps only its own run's records."""
    import logging

    mine, other = provenance.RunRecorder(), provenance.RunRecorder()
    with provenance.recording() as active:
        with provenance.capture_step_warnings(other, "step"):
            logging.getLogger("vaft.test").warning("time slice 3 is someone else's")
        with provenance.capture_step_warnings(active, "step"):
            logging.getLogger("vaft.test").warning("time slice 3 is ours")
    assert other.step_warnings == {} and mine.step_warnings == {}
    assert active.step_warnings == {"step": {3: ["time slice 3 is ours"]}}


def test_a_preset_that_records_nothing_still_reports_every_column():
    table = database.summary_from_ods({48224: vaft.omas.sample_ods(48224)},
                                      preset="core_profiles", provenance=True)
    record = table.attrs["provenance"].table
    assert set(record["status"]) <= {"unrecorded", "missing"}
    assert (record.loc[record["status"] == "unrecorded", "reason"]
            == "this preset's extractor does not record provenance").all()


def test_the_record_rides_along_in_attrs(recorded):
    record = recorded.attrs["provenance"]
    assert set(record.code) == {"vaft_version", "commit", "dirty"}
    # pandas compares attrs on concat/merge; a bare DataFrame there raises
    pd.concat([recorded, recorded])
    recorded.merge(recorded, on=["shot", "eq_index", "time_s"])
    assert recorded.copy().attrs["provenance"] is record
    assert set(record.counts().columns) <= set(provenance.STATUSES)


def test_the_database_path_records_the_same(sample, monkeypatch, recorded):
    @contextmanager
    def fake_open(shot, *, source=None, paths=None, **kwargs):
        yield copy.deepcopy(sample)

    monkeypatch.setattr(database, "open", fake_open)
    table = database.summary((SHOT, SHOT), preset="equilibrium_global", source="public",
                             provenance=True)
    pd.testing.assert_frame_equal(
        table.attrs["provenance"].table,
        recorded.attrs["provenance"].table,
    )


def test_without_the_flag_no_recorder_is_active():
    assert provenance.active() is None
    with provenance.recording(False) as recorder:
        assert recorder is None and provenance.active() is None
    with provenance.recording() as recorder:
        assert provenance.active() is recorder
    assert provenance.active() is None
    assert np.isnan(provenance._value_at_row(provenance.RunRecorder(), "b0", 0))
