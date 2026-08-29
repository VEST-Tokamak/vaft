import gzip
import json

import numpy as np
import pytest

from vaft.database import raw
from vaft.machine_mapping.pf_active import PF_COIL_COUNT, vfit_pf


def _write_raw_dump(path, shot: int, fields: dict[int, list[float]] | None = None) -> None:
    fields = fields or {13: [1.0, 2.0, 3.0]}
    payload = {
        "shot": shot,
        "fields": {
            str(field): {"data": data, "type": "slow"}
            for field, data in fields.items()
        },
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_explicit_raw_source_template_loads_without_sql(tmp_path, monkeypatch):
    dump = tmp_path / "shot_123.json.gz"
    _write_raw_dump(dump, 123)

    def unexpected_sql_initialization():
        pytest.fail("an explicit raw source must not initialize live SQL")

    monkeypatch.setattr(raw, "init_pool", unexpected_sql_initialization)
    time, data = raw.vest_load(
        123,
        13,
        sample_opt=tmp_path / "shot_{shot}.json.gz",
    )

    np.testing.assert_allclose(time, [0.0, raw.SLOW_DT, 2 * raw.SLOW_DT])
    np.testing.assert_allclose(data, [1.0, 2.0, 3.0])


def test_missing_explicit_raw_source_does_not_fallback_to_sql(tmp_path, monkeypatch):
    def unexpected_sql_initialization():
        pytest.fail("a missing explicit source must not fall back to live SQL")

    monkeypatch.setattr(raw, "init_pool", unexpected_sql_initialization)

    with pytest.raises(FileNotFoundError, match="Archived raw source not found"):
        raw.load_raw(123, 13, sample_opt=tmp_path / "missing_{shot}.json.gz")


def test_pf_mapping_does_not_require_optional_reference_archive(tmp_path):
    requested_shot = 41672
    _write_raw_dump(
        tmp_path / f"shot_{requested_shot}.json.gz",
        requested_shot,
        {field: [1.0, 2.0, 3.0] for field in (5, 59, 62, 65)},
    )

    time, currents = vfit_pf(
        requested_shot,
        raw_source=tmp_path / "shot_{shot}.json.gz",
    )

    assert time.size > 0
    assert len(currents) == PF_COIL_COUNT


def test_archive_honours_a_per_field_dt(tmp_path):
    """A fast field with an explicit ``dt`` reconstructs its native timebase.

    The two-rate archive format collapsed every fast channel to FAST_DT, which
    silently stretched a 2 MHz outboard-Mirnov record eightfold in time.  An
    entry-level ``dt`` overrides the class default; fields without one keep the
    historical behaviour bit for bit.
    """
    shot = 45531  # >= 41660 -> 0.26 s fast-DAQ trigger correction
    native_dt = 5e-7
    payload = {
        "shot": shot,
        "fields": {
            "286": {"type": "fast", "dt": native_dt, "data": [1.0, 2.0, 3.0, 4.0]},
            "172": {"type": "fast", "data": [5.0, 6.0, 7.0]},
            "1": {"type": "slow", "data": [8.0, 9.0]},
        },
    }
    path = tmp_path / f"shot_{shot}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    time_native, _ = raw.load_raw(shot, 286, sample_opt=path)
    np.testing.assert_allclose(time_native, 0.26 + native_dt * np.arange(4))

    time_default, _ = raw.load_raw(shot, 172, sample_opt=path)
    np.testing.assert_allclose(time_default, 0.26 + raw.FAST_DT * np.arange(3))

    time_slow, _ = raw.load_raw(shot, 1, sample_opt=path)
    np.testing.assert_allclose(time_slow, raw.SLOW_DT * np.arange(2))


def test_self_describing_entries_reproduce_the_stored_timebase(tmp_path):
    """``t0`` + ``dt`` entries are authoritative: no class default, no trigger table."""
    shot = 45531
    payload = {
        "shot": shot,
        "fields": {
            # a 2 MHz fast channel with its corrected absolute start time
            "286": {"type": "fast", "t0": 0.26, "dt": 5.000025e-7,
                    "data": [1.0, 2.0, 3.0]},
            # a slow channel starting at t=0 with the DB's measured cadence
            "109": {"type": "slow", "t0": 0.0, "dt": 4.00016e-5,
                    "data": [4.0, 5.0]},
        },
    }
    path = tmp_path / f"shot_{shot}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    time_fast, _ = raw.load_raw(shot, 286, sample_opt=path)
    np.testing.assert_allclose(time_fast, 0.26 + 5.000025e-7 * np.arange(3))

    time_slow, _ = raw.load_raw(shot, 109, sample_opt=path)
    np.testing.assert_allclose(time_slow, 4.00016e-5 * np.arange(2))


def test_dump_writes_a_self_describing_timebase(tmp_path, monkeypatch):
    """Every dumped field records t0 and the measured span/(n-1) cadence."""
    native_dt = 5e-7
    times = 0.26 + native_dt * np.arange(5)

    monkeypatch.setattr(raw, "get_all_field_codes_for_shot", lambda shot, max_retries=3: [286])
    monkeypatch.setattr(
        raw, "load_raw",
        lambda shot, fcode, max_retries=3, daq_type=0, sample_opt=False: (
            times, np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ),
    )
    monkeypatch.setattr(raw, "date_from_shot", lambda shot: ("2026-01-01", None))

    output = tmp_path / "dump.json.gz"
    assert raw.dump_all_raw_signals_for_shot(shot=45531, output_path=str(output))

    with gzip.open(output, "rt", encoding="utf-8") as handle:
        entry = json.load(handle)["fields"]["286"]
    assert entry["t0"] == pytest.approx(0.26)
    assert entry["dt"] == pytest.approx(native_dt)

    # And the dump round-trips through the loader bit for bit.
    reloaded_time, reloaded_data = raw.load_raw(45531, 286, sample_opt=output)
    np.testing.assert_allclose(reloaded_time, times)
    np.testing.assert_allclose(reloaded_data, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_dump_field_subset_restricts_the_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(raw, "get_all_field_codes_for_shot",
                        lambda shot, max_retries=3: [1, 2, 3])
    monkeypatch.setattr(
        raw, "load_raw",
        lambda shot, fcode, max_retries=3, daq_type=0, sample_opt=False: (
            np.array([0.0, 4e-5]), np.array([float(fcode), float(fcode)])
        ),
    )
    monkeypatch.setattr(raw, "date_from_shot", lambda shot: ("2026-01-01", None))

    output = tmp_path / "subset.json.gz"
    assert raw.dump_all_raw_signals_for_shot(shot=1234, output_path=str(output), fields=[1, 3])

    with gzip.open(output, "rt", encoding="utf-8") as handle:
        stored = json.load(handle)["fields"]
    assert sorted(stored) == ["1", "3"]


def test_multi_field_loads_refuse_mixed_cadences(tmp_path):
    """Stacking a 2 MHz channel against a slow channel must fail loudly.

    The multi-field path returns the first field's time axis for every column;
    with mixed cadences that silently misaligns the data.
    """
    shot = 45531
    payload = {
        "shot": shot,
        "fields": {
            "286": {"type": "fast", "t0": 0.26, "dt": 5e-7, "data": [1.0] * 10},
            "109": {"type": "slow", "t0": 0.0, "dt": 4e-5, "data": [2.0] * 10},
            "287": {"type": "fast", "t0": 0.26, "dt": 5e-7, "data": [3.0] * 10},
        },
    }
    path = tmp_path / f"shot_{shot}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    with pytest.raises(ValueError, match="mix sampling cadences"):
        raw.load_raw(shot, [286, 109], sample_opt=path)

    # Same-cadence batches still stack.
    time_ok, data_ok = raw.load_raw(shot, [286, 287], sample_opt=path)
    assert data_ok.shape == (10, 2)
