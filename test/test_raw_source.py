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


class TestArchiveTimebaseUpgrade:
    """upgrade_archive_timebase infers the era-correct cadence offline."""

    def test_v3_era_infers_the_linspace_cadence(self):
        payload = {
            "shot": 43100,
            "fields": {
                "275": {"type": "fast", "data": [0.0] * 200000},   # 2 MHz
                "66": {"type": "fast", "data": [0.0] * 25000},     # 250 kHz
                "1": {"type": "slow", "data": [0.0] * 25000},      # 25 kHz
            },
        }
        report = raw.upgrade_archive_timebase(payload)

        assert report["upgraded"] == 3
        assert report["non_nominal"] == [275]
        assert payload["fields"]["275"]["dt"] == pytest.approx(0.1 / 199999)
        assert payload["fields"]["275"]["t0"] == pytest.approx(0.26)
        assert payload["fields"]["66"]["dt"] == pytest.approx(0.1 / 24999)
        assert payload["fields"]["1"]["dt"] == pytest.approx(1.0 / 24999)
        assert payload["fields"]["1"]["t0"] == 0.0

    def test_v2_era_keeps_the_exact_nominal_cadence(self):
        # shotDataWaveform_2 stores arange-convention times at exactly the
        # nominal rates, so the inference must NOT apply the linspace formula.
        payload = {
            "shot": 39915,
            "fields": {
                "66": {"type": "fast", "data": [0.0] * 25000},
                "1": {"type": "slow", "data": [0.0] * 25000},
            },
        }
        raw.upgrade_archive_timebase(payload)

        assert payload["fields"]["66"]["dt"] == raw.FAST_DT
        assert payload["fields"]["66"]["t0"] == pytest.approx(0.24)  # pre-41446 trigger
        assert payload["fields"]["1"]["dt"] == raw.SLOW_DT

    def test_upgrade_is_idempotent_and_leaves_uninferable_entries_alone(self):
        payload = {
            "shot": 43100,
            "fields": {
                "275": {"type": "fast", "t0": 0.26, "dt": 5e-7, "data": [0.0] * 4},
                "9": {"type": "unknown", "data": [0.0] * 10},
                "10": {"type": "fast", "data": [1.0]},
            },
        }
        report = raw.upgrade_archive_timebase(payload)

        assert report == {"upgraded": 0, "already": 1, "skipped": 2, "non_nominal": []}
        assert payload["fields"]["275"]["dt"] == 5e-7          # untouched
        assert "dt" not in payload["fields"]["9"]

    def test_upgraded_entry_loads_like_a_fresh_dump(self, tmp_path):
        # End to end: legacy archive -> upgrade -> loader reproduces the
        # timebase a new-schema dump of the same data would produce.
        shot, n = 43100, 200000
        legacy = {"shot": shot,
                  "fields": {"275": {"type": "fast", "data": [0.0] * n}}}
        raw.upgrade_archive_timebase(legacy)
        path = tmp_path / f"shot_{shot}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(legacy, handle)

        time, _ = raw.load_raw(shot, 275, sample_opt=path)

        np.testing.assert_allclose(time, 0.26 + (0.1 / (n - 1)) * np.arange(n))


def test_cadence_error_names_the_loaded_fields_not_the_requested_ones(tmp_path):
    """A skipped (missing) field must not shift the labels in the error message."""
    shot = 45531
    payload = {
        "shot": shot,
        "fields": {
            # field 100 is requested but absent; 286 (fast) and 109 (slow) load.
            "286": {"type": "fast", "t0": 0.26, "dt": 5e-7, "data": [1.0] * 10},
            "109": {"type": "slow", "t0": 0.0, "dt": 4e-5, "data": [2.0] * 10},
        },
    }
    path = tmp_path / f"shot_{shot}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    with pytest.raises(ValueError) as caught:
        raw.load_raw(shot, [100, 286, 109], sample_opt=path)

    message = str(caught.value)
    assert "field 286: dt=5e-07" in message
    assert "field 109: dt=4e-05" in message
    # The missing field must not be attributed a cadence it never had.
    assert "field 100: dt" not in message


def test_an_archive_is_parsed_once_however_many_fields_are_asked_for(tmp_path, monkeypatch):
    """One archive read per file, not per field (issue #444).

    `_safe_vest_load_cached` memoises on the field code, so before this was
    fixed a build that wanted every magnetics channel re-decoded the whole
    archive once per field: 114 full parses of a 13 MB dump for a single
    magnetics build, 96.5 s where 1.7 s was the real work.
    """
    raw._parse_sample_archive.cache_clear()
    path = tmp_path / "dump.json.gz"
    _write_raw_dump(path, 12345, {code: [1.0, 2.0, 3.0] for code in range(10, 30)})

    parses = []
    real_open = gzip.open

    def counting_open(*args, **kwargs):
        parses.append(args[0] if args else kwargs.get("filename"))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(gzip, "open", counting_open)
    for code in range(10, 30):
        raw._load_from_sample_file(12345, [code], str(path))

    assert len(parses) == 1, f"archive re-parsed {len(parses)} times for 20 fields"


def test_a_rewritten_archive_is_not_served_from_cache(tmp_path):
    """The cache key carries mtime and size, so editing a dump in place is seen.

    A path-only key would hand back the previous contents here -- the failure
    mode that makes caching a correctness question rather than a speed one.
    """
    raw._parse_sample_archive.cache_clear()
    path = tmp_path / "dump.json.gz"

    _write_raw_dump(path, 777, {13: [1.0, 2.0, 3.0]})
    first = raw._load_from_sample_file(777, [13], str(path))
    assert first is not None
    np.testing.assert_allclose(first[1], [1.0, 2.0, 3.0])

    _write_raw_dump(path, 777, {13: [9.0, 9.0, 9.0, 9.0]})
    second = raw._load_from_sample_file(777, [13], str(path))
    assert second is not None
    np.testing.assert_allclose(second[1], [9.0, 9.0, 9.0, 9.0])
