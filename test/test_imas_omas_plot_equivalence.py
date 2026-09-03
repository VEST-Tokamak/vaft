"""Equivalence at the data boundary (issue #63, decision 1).

The OMAS and the IMAS adapters must turn the same shot into the same view
models before anything reaches a renderer.  Shot 39915 is packaged in both
representations, so every canonical plot is built from both and compared
field by field; a plot neither side can draw must be refused for the same
reason.  Differences a representation legitimately introduces are listed in
``KNOWN_DIFFERENCES`` as strict expected failures, so a fix is noticed.
"""

import contextlib
import dataclasses
import io
import warnings
from collections.abc import Mapping

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import vaft
import vaft.imas
import vaft.omas
from vaft.plot.backend import recipes
from vaft.plot.registry import canonical_names

imas = pytest.importorskip("imas")

#: Plots whose models legitimately differ between the two representations of
#: 39915, with the reason.  Empty means the two adapters agree on everything.
KNOWN_DIFFERENCES: dict[str, str] = {}


def _load_ods():
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path("samples/39915/omas.json.gz")))


@pytest.fixture(scope="module")
def ods():
    return _load_ods()


@pytest.fixture(scope="module")
def entry():
    return imas.DBEntry(str(vaft.data.data_path("samples/39915/imas.nc")), "r", dd_version="3.41.0")


def assert_models_equal(a, b, where=""):
    if dataclasses.is_dataclass(a):
        assert type(a) is type(b), f"{where}: {type(a).__name__} vs {type(b).__name__}"
        for f in dataclasses.fields(a):
            assert_models_equal(getattr(a, f.name), getattr(b, f.name), f"{where}.{f.name}")
    elif isinstance(a, np.ndarray):
        try:
            from uncertainties import unumpy
            if a.dtype == object:
                a, b = unumpy.nominal_values(a), unumpy.nominal_values(b)
        except ImportError:
            pass
        assert np.shape(a) == np.shape(b), f"{where}: shapes {np.shape(a)} vs {np.shape(b)}"
        if np.asarray(a).dtype.kind in "fiu":
            assert np.allclose(a, b, equal_nan=True), where
        else:
            assert np.array_equal(a, b), where
    elif isinstance(a, Mapping):
        assert dict(a) == dict(b), where
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), f"{where}: {len(a)} vs {len(b)} items"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_models_equal(x, y, f"{where}[{i}]")
    else:
        assert a == b, f"{where}: {a!r} vs {b!r}"


@pytest.mark.parametrize("name", canonical_names())
def test_omas_and_imas_build_the_same_model(name, ods, entry):
    if name in KNOWN_DIFFERENCES:
        pytest.xfail(KNOWN_DIFFERENCES[name])
    omas_entries = vaft.omas.normalize_entries(ods)
    imas_entries = vaft.imas.normalize_entries(entry)
    missing_omas = recipes.missing_required_path(omas_entries[0][1], name)
    missing_imas = recipes.missing_required_path(imas_entries[0][1], name)
    assert missing_omas == missing_imas, "the two adapters must refuse for the same reason"
    if missing_omas is not None:
        return
    try:
        from_ods = recipes.build_model(name, omas_entries)
    except Exception as exc:  # a code-backed plot this sample cannot serve
        with pytest.raises(type(exc)):
            recipes.build_model(name, imas_entries)
        return
    from_ids = recipes.build_model(name, imas_entries)
    assert_models_equal(from_ods, from_ids, name)


def test_discovery_agrees(ods, entry):
    assert vaft.omas.available_plots(ods).names() == vaft.imas.available_plots(entry).names()


def test_both_label_the_shot(ods, entry):
    assert [l for l, _ in vaft.omas.normalize_entries(ods)] == ["39915"]
    assert [l for l, _ in vaft.imas.normalize_entries(entry)] == ["39915"]
