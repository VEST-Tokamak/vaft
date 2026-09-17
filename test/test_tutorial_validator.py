"""Regression tests for the standalone tutorial validator."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "verify_tutorial_under_test", ROOT / "test" / "verify_tutorial.py"
)
VALIDATOR = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATOR)


def test_machine_path_pattern_allows_urls_but_rejects_absolute_paths():
    for value in ("https://example.test/data", "http://127.0.0.1:5101"):
        assert VALIDATOR.MACHINE_PATH.search(value) is None

    for value in ("/Users/name/checkout", "/home/name/checkout", "C:/checkout"):
        assert VALIDATOR.MACHINE_PATH.search(value) is not None


def test_inventory_ignores_runtime_output_data(monkeypatch, tmp_path):
    tutorial = tmp_path / "tutorial"
    for name in ("common", "01", "02", "03", "04", "05", "06"):
        (tutorial / "figures" / name).mkdir(parents=True)
    scratch = tutorial / "outputs" / "scratch.csv"
    scratch.parent.mkdir()
    scratch.write_text("value\n1\n", encoding="utf-8")

    monkeypatch.setattr(VALIDATOR, "ROOT", tmp_path)
    monkeypatch.setattr(VALIDATOR, "TUTORIAL", tutorial)
    failures = []
    VALIDATOR._validate_inventory(failures)

    assert all("scratch.csv" not in failure for failure in failures)
