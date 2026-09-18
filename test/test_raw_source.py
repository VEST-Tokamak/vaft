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

    def test_a_long_record_at_the_nominal_rate_still_upgrades(self):
        """#770: an implied-span guard would refuse this, and it is correct data.

        Field 138 in the low-30000s is 250 kHz -- the nominal rate -- recorded
        for 0.2 s, so it holds 50000 samples and spans twice the 0.1 s class
        window. Measured in 16 of 200 sampled shots, all in the v2 branch.
        """
        payload = {
            "shot": 32878,
            "fields": {"138": {"type": "fast", "data": [0.0] * 50000}},
        }
        report = raw.upgrade_archive_timebase(payload)

        assert report["upgraded"] == 1
        assert payload["fields"]["138"]["dt"] == raw.FAST_DT

    def test_the_sample_count_cannot_tell_a_long_record_from_a_fast_one(self):
        """Why #770 has no code fix: the two cases are the same number.

        250 kHz for 0.2 s and 500 kHz for 0.1 s both hold 50000 samples and both
        carry the `fast` label, so no rule over (n, label) separates them. What
        keeps the v2 branch correct is that its shot range has no channel above
        250 kHz -- measured, not assumed.
        """
        long_at_nominal = round(0.2 / raw.FAST_DT)          # 250 kHz, 0.2 s
        normal_at_double = round(0.1 / (raw.FAST_DT / 2))   # 500 kHz, 0.1 s
        assert long_at_nominal == normal_at_double == 50000

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


class _FakeCursor:
    """Answers the field-count query from a per-table mapping."""

    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts
        self._value = 0

    def execute(self, query: str, params=None) -> None:
        for table, count in self._counts.items():
            if table in query:
                self._value = count
                return
        self._value = 0

    def fetchone(self):
        return (self._value,)

    def close(self) -> None:
        pass


class _FakeConnection:
    def __init__(self, counts: dict[str, int], error: Exception | None = None) -> None:
        self._counts = counts
        self._error = error
        self.cursors = 0

    def cursor(self):
        self.cursors += 1
        if self._error is not None:
            raise self._error
        return _FakeCursor(self._counts)


def test_the_table_that_holds_the_shot_wins_over_the_shot_number():
    """Issue #761: `_2` runs past the nominal boundary, so the range rule lies.

    42194 sits above 42190 and exists only in `shotDataWaveform_2`. Choosing by
    shot number queried the empty `_3` and the export failed outright.
    """
    raw._GENERATION_CACHE.clear()
    conn = _FakeConnection({"shotDataWaveform_2": 114, "shotDataWaveform_3": 0})
    assert raw.waveform_generation_for_shot(conn, 42194) == 2


def test_a_stub_does_not_outrank_the_full_record():
    """The truncating half of #761, which reported success.

    19 shots exist in both tables with a two-to-four field stub on the `_3`
    side. Reading it produced a product carrying 2 of shot 42647's 127 fields
    while the manifest said the export succeeded.
    """
    raw._GENERATION_CACHE.clear()
    conn = _FakeConnection({"shotDataWaveform_2": 127, "shotDataWaveform_3": 2})
    assert raw.waveform_generation_for_shot(conn, 42647) == 2


def test_the_newer_table_still_wins_where_it_is_the_real_one():
    raw._GENERATION_CACHE.clear()
    conn = _FakeConnection({"shotDataWaveform_2": 0, "shotDataWaveform_3": 145})
    assert raw.waveform_generation_for_shot(conn, 43000) == 3


def test_a_shot_in_neither_table_resolves_to_nothing():
    """Registered in the shot list but never acquired -- 2321 of these exist."""
    raw._GENERATION_CACHE.clear()
    conn = _FakeConnection({"shotDataWaveform_2": 0, "shotDataWaveform_3": 0})
    assert raw.waveform_generation_for_shot(conn, 29400) is None


def test_a_shot_not_written_yet_is_asked_about_again():
    """Cold review data F15: `None` was cached for the life of the process.

    On the day of a shot, a poller that asks before the DAQ has written the
    waveforms must see them once they arrive.
    """
    raw._GENERATION_CACHE.clear()
    early = _FakeConnection({"shotDataWaveform_2": 0, "shotDataWaveform_3": 0})
    assert raw.waveform_generation_for_shot(early, 48900) is None

    written = _FakeConnection({"shotDataWaveform_2": 0, "shotDataWaveform_3": 5})
    assert raw.waveform_generation_for_shot(written, 48900) == 3

    # A real answer is still cached: the next call does not touch the database.
    assert raw.waveform_generation_for_shot(early, 48900) == 3
    assert early.cursors == 2


def test_vest_check_table_asks_the_database_before_guessing():
    raw._GENERATION_CACHE.clear()
    conn = _FakeConnection({"shotDataWaveform_2": 141, "shotDataWaveform_3": 2})
    assert raw.vest_check_table(conn, 42823) == 2
    # Without a connection there is nothing to ask, so the ranges are all there is.
    assert raw.vest_check_table(None, 40000) == 2
    assert raw.vest_check_table(None, 48000) == 3


def test_a_read_error_is_not_reported_as_an_empty_shot():
    """#446's mistake, not repeated: a failed read is not an absence.

    Both callers retry on MysqlError. Recording the failure as "0 fields" would
    resolve to "no table holds this shot", and each caller returns immediately
    on that -- skipping its own retry loop and handing back a shot that merely
    looks empty.
    """
    raw._GENERATION_CACHE.clear()
    conn = _FakeConnection({}, error=raw.MysqlError("server has gone away"))
    with pytest.raises(raw.MysqlError):
        raw.waveform_generation_for_shot(conn, 42194)
    # and nothing was learned, so nothing was remembered
    assert 42194 not in raw._GENERATION_CACHE


def test_the_resolution_is_remembered_per_shot():
    """load_raw runs once per field; without this each one costs two COUNTs."""
    raw._GENERATION_CACHE.clear()
    conn = _FakeConnection({"shotDataWaveform_2": 141, "shotDataWaveform_3": 0})
    assert raw.waveform_generation_for_shot(conn, 42823) == 2
    first = conn.cursors
    for _ in range(20):
        assert raw.waveform_generation_for_shot(conn, 42823) == 2
    assert conn.cursors == first, "the tables were re-queried for a known shot"
