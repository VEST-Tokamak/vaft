"""Behaviour for repository-only sample data."""

from pathlib import Path

import pytest

from vaft.data.resources import require_repository_sample


def test_require_repository_sample_explains_pypi_exclusion(tmp_path: Path):
    missing = tmp_path / "archived-sample.mat"

    with pytest.raises(FileNotFoundError, match="not included in the PyPI distribution"):
        require_repository_sample(missing)


def test_require_repository_sample_returns_existing_path(tmp_path: Path):
    sample = tmp_path / "sample.dat"
    sample.write_text("sample", encoding="utf-8")

    assert require_repository_sample(sample) == sample


def test_missing_packaged_digitizer_explains_how_to_supply_it(tmp_path: Path):
    from vaft.machine_mapping.soft_x_rays import _resolve_digitizer_file

    with pytest.raises(FileNotFoundError, match="not included in the PyPI distribution"):
        _resolve_digitizer_file(45531, "17592", data_root=tmp_path)
