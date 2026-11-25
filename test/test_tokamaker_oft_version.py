"""The OFT version guard for F0 < 0 (upstream OpenFUSIONToolkit #321/#322); no solver needed."""

from __future__ import annotations

import sys
import types

import pytest

from vaft.code.tokamaker import _oft


def _fake_oft(monkeypatch, version):
    module = types.ModuleType("OpenFUSIONToolkit")
    if version is not None:
        module.__version__ = version
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit", module)


@pytest.mark.parametrize(
    "raw, parsed",
    [("26.9", (26, 9)), ("26.9.dev412", (26, 9)), ("27.0", (27, 0)), ("26.6", (26, 6)),
     ("1.0.0b6", (1, 0)), ("@CMAKE_PROJECT_VERSION@@OFT_PYPI_VER_SUFFIX@", None), (None, None)],
)
def test_oft_version_keeps_the_release_numbers(monkeypatch, raw, parsed):
    _fake_oft(monkeypatch, raw)
    assert _oft.oft_version() == parsed


@pytest.mark.parametrize("version", ["26.9", "26.9.dev12", "27.3"])
def test_negative_f0_runs_on_a_fixed_release(monkeypatch, version):
    _fake_oft(monkeypatch, version)
    _oft.require_negative_f0_support(-0.48)


@pytest.mark.parametrize("version", [None, "26.6", "1.0.0b7", "@CMAKE_PROJECT_VERSION@"])
def test_positive_or_unset_f0_never_needs_the_fix(monkeypatch, version):
    _fake_oft(monkeypatch, version)
    _oft.require_negative_f0_support(0.48)
    _oft.require_negative_f0_support(None)


@pytest.mark.parametrize("version", ["26.6", "1.0.0b6"])
def test_negative_f0_is_refused_on_an_old_release(monkeypatch, version):
    _fake_oft(monkeypatch, version)
    monkeypatch.setenv(_oft.ALLOW_UNVERSIONED_OFT_ENV, "1")  # does not unlock a known-old version
    with pytest.raises(RuntimeError, match="#321"):
        _oft.require_negative_f0_support(-0.48)


def test_unversioned_oft_needs_the_explicit_override(monkeypatch):
    _fake_oft(monkeypatch, None)
    monkeypatch.delenv(_oft.ALLOW_UNVERSIONED_OFT_ENV, raising=False)
    with pytest.raises(RuntimeError, match=_oft.ALLOW_UNVERSIONED_OFT_ENV):
        _oft.require_negative_f0_support(-0.48)
    monkeypatch.setenv(_oft.ALLOW_UNVERSIONED_OFT_ENV, "1")
    _oft.require_negative_f0_support(-0.48)


def test_extra_pins_the_first_fixed_release():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert 'tokamaker = ["openfusiontoolkit>=%d.%d"]' % _oft.MIN_OFT_VERSION in text
