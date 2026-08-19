import gzip
import json

import numpy as np
import pytest

from vaft.database import raw


def _write_raw_dump(path, shot: int) -> None:
    payload = {
        "shot": shot,
        "fields": {
            "13": {
                "data": [1.0, 2.0, 3.0],
                "type": "slow",
            }
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
