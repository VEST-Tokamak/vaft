"""The backend-neutral extraction layer (issue #63, H·0).

Recipes read every value through ``vaft.plot.backend.access``, which chooses
the accessor for the object; the OMAS namespace keeps its behaviour to the
byte, a native IMAS object is refused loudly rather than read as empty, and a
plot's declared IDS are exactly what it needs.
"""

import contextlib
import io
import warnings

import numpy as np
import omas
import pytest

import vaft
import vaft.omas
from vaft.plot.backend import access, recipes
from vaft.plot.backend.entries import label_entries
from vaft.plot.registry import canonical_names, get_spec


def _load(rel):
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path(rel)))


@pytest.fixture(scope="module")
def shot():
    return _load("samples/39915/omas.json.gz")


# ---------------------------------------------------------------------------
# accessor dispatch
# ---------------------------------------------------------------------------

def test_ods_and_mappings_read_through_the_omas_accessor(shot):
    assert access.accessor_for(shot) is access.ODS_ACCESSOR
    assert access.accessor_for({"a": {"b": 1}}) is access.ODS_ACCESSOR
    assert access.get({"a": {"b": 1}}, "a.b") == 1
    assert access.has(shot, "magnetics.ip.0.data") and not access.has(shot, "magnetics.no_such")
    assert access.count(shot, "magnetics.flux_loop") == 11
    assert access.array(shot, "magnetics.ip.0.data").dtype == float
    assert access.array(shot, "magnetics.no_such") is None
    assert access.array({"x": []}, "x") is None


def test_a_native_imas_object_is_refused_not_read_as_empty():
    imas = pytest.importorskip("imas")
    ids = imas.IDSFactory("3.41.0").new("magnetics")
    with pytest.raises(TypeError, match="vaft.imas"):
        access.get(ids, "flux_loop.0.flux.data")
    from vaft.ods_access import path_exists, path_value
    with pytest.raises(TypeError, match="vaft.imas"):
        path_value(ids, "flux_loop.0.flux.data")
    with pytest.raises(TypeError, match="vaft.imas"):
        path_exists(ids, "flux_loop")
    with pytest.raises(TypeError, match="vaft.imas"):
        vaft.omas.normalize_entries(ids)


def test_a_registered_accessor_wins_for_its_objects():
    class Box:
        def __init__(self, data):
            self.data = data

    class BoxAccessor:
        def get(self, obj, path, default=None):
            return obj.data.get(path, default)

        def count(self, obj, path):
            return len(obj.data.get(path, ()))

        def has(self, obj, path):
            return path in obj.data

    accessor = BoxAccessor()
    access.register_accessor(lambda obj: isinstance(obj, Box), accessor)
    access.register_accessor(lambda obj: isinstance(obj, Box), accessor)  # idempotent
    try:
        box = Box({"a.b": [1, 2, 3]})
        assert access.accessor_for(box) is accessor
        assert access.count(box, "a.b") == 3 and access.has(box, "a.b")
        assert np.array_equal(access.array(box, "a.b"), [1.0, 2.0, 3.0])
    finally:
        access._REGISTERED[:] = [pair for pair in access._REGISTERED if pair[1] is not accessor]


def test_label_entries_is_the_one_labelling_rule(shot):
    assert label_entries([("0", shot)], "shot") == (("39915", shot),)
    assert label_entries([("k", shot)], "key") == (("k", shot),)
    assert label_entries([("k", {})], "shot") == (("k", {}),)
    assert label_entries([("0", shot)], ["mine"]) == (("mine", shot),)
    with pytest.raises(ValueError, match="received 2 labels for 1 entries"):
        label_entries([("0", shot)], ["a", "b"])


# ---------------------------------------------------------------------------
# the shim and the declared IDS
# ---------------------------------------------------------------------------

def test_the_omas_shim_exposes_the_backend_objects():
    from vaft.omas import _plot_recipes as shim
    assert shim.RECIPES is recipes.RECIPES
    assert shim._wall_layers is recipes._wall_layers
    assert shim.build_model is recipes.build_model
    assert shim.normalize_entries is vaft.omas.normalize_entries


def test_required_ids_unions_a_composite_and_keeps_declaration_order():
    assert recipes.required_ids("plasma_current_time") == ("magnetics",)
    assert recipes.required_ids("magnetics_overview") == ("magnetics", "pf_active", "tf", "equilibrium")
    assert "wall" in recipes.required_ids("soft_x_rays_overview")
    assert "wall" in recipes.required_ids("equilibrium_overview")
    for name in canonical_names():
        assert set(get_spec(name).ids) <= set(recipes.required_ids(name)), name


def _pruned(ods, roots):
    pruned = omas.ODS()
    for root in roots:
        if root in ods:
            pruned[root] = ods[root]
    return pruned


def _assert_models_equal(a, b, where=""):
    import dataclasses
    from collections.abc import Mapping
    if dataclasses.is_dataclass(a):
        assert type(a) is type(b), where
        for f in dataclasses.fields(a):
            _assert_models_equal(getattr(a, f.name), getattr(b, f.name), f"{where}.{f.name}")
    elif isinstance(a, np.ndarray):
        assert a.shape == b.shape, where
        assert np.allclose(a, b, equal_nan=True), where
    elif isinstance(a, Mapping):
        assert dict(a) == dict(b), where
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_models_equal(x, y, f"{where}[{i}]")
    else:
        assert a == b, where


def test_a_plot_needs_no_more_than_its_declared_ids(shot):
    """The guard behind selective loading: pruning to required_ids changes nothing."""
    offered = [row.name for row in vaft.omas.available_plots(shot)]
    entries_full = vaft.omas.normalize_entries(shot)
    checked = 0
    for name in offered + [n for n in canonical_names() if n not in offered and recipes.diagnoses_itself(n)]:
        # Code-backed plots are included when the full sample can build them:
        # their builders read whatever they like, and the declared IDS must
        # cover it, or a selectively loaded shot will build something else.
        roots = ("dataset_description", *recipes.required_ids(name))
        pruned = _pruned(shot, roots)
        try:
            full = recipes.build_model(name, entries_full)
        except Exception:
            continue
        partial = recipes.build_model(name, vaft.omas.normalize_entries(pruned))
        _assert_models_equal(full, partial, name)
        checked += 1
    assert checked >= 40, checked
