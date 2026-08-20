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
