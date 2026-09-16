"""OMAS-bound computed views on inputs that cannot hand over a whole ODS (#434).

A lazily loaded IMAS handle cannot convert an IDS, and a lazy OMAS store's
unfetched leaves are lost by the deep copy an OMAS-bound builder takes; for
both, :func:`vaft.plot.backend.recipes.materialise_reads` reads exactly the
paths the recipe declares in ``reads`` into a private ODS and the builder
runs on that.  The proof here is the same model as from a plain ODS.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import warnings
from pathlib import Path

import numpy as np
import pytest

import vaft
import vaft.omas
from vaft.omas.entries import normalize_entries
from vaft.plot.backend import recipes as R
from vaft.plot.backend.recipes import (
    build_model,
    expand_template,
    is_lazy_ods,
    materialise_reads,
    materialises_for_builder,
    missing_required_path,
)

from _read_recorder import assert_models_equal

#: OMAS-bound views that deep-copy a whole IDS: a lazy IMAS handle cannot
#: supply one, and the refusal names the IDS.
ROOT_DECLARING = {
    name for name, recipe in R.RECIPES.items()
    if isinstance(recipe, R.CallableRecipe) and any("." not in t for t in recipe.reads)
}


@pytest.fixture(scope="module")
def sample():
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(vaft.data.sample(39915, representation="omas"))


@pytest.fixture(scope="module")
def lazy_entry():
    """The packaged IMAS entry, wrapped so that no whole-IDS conversion is possible."""
    imas = pytest.importorskip("imas")
    from vaft.imas.access import IDSEntry

    class Lazy(IDSEntry):
        def as_ods_for(self, ids_names):
            raise NotImplementedError("a lazily loaded handle cannot supply a full copy (test double)")

    entry = imas.DBEntry(str(vaft.data.data_path("samples/39915/imas.nc")), "r", dd_version="3.41.0")
    yield Lazy(entry)
    entry.close()


def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


# ---------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------


def test_a_template_expands_over_the_input_indices(sample):
    assert expand_template(sample, "magnetics.ip.{i}.data") == ["magnetics.ip.0.data"]
    coils = expand_template(sample, "pf_active.coil.{i}.name")
    assert coils == [f"pf_active.coil.{i}.name" for i in range(len(sample["pf_active.coil"]))]
    elements = expand_template(sample, "pf_active.coil.{i}.element.{j}.geometry.rectangle.r")
    assert len(elements) == sum(len(sample[f"pf_active.coil.{i}.element"]) for i in range(len(coils)))
    assert expand_template(sample, "core_profiles.profiles_1d.{i}.electrons.temperature") == []
    assert expand_template(sample, "equilibrium.time") == ["equilibrium.time"]


def test_materialised_reads_keep_the_arrays_of_structures_whole(sample):
    private = materialise_reads(sample, "passive_structure_overview_wall_reduction")
    # Every probe exists at its own index, whether or not it holds a field.
    probes = len(sample["magnetics.b_field_pol_probe"])
    assert len(private["magnetics.b_field_pol_probe"]) == probes
    with_field = next(i for i in range(probes) if f"magnetics.b_field_pol_probe.{i}.field.data" in sample)
    assert f"magnetics.b_field_pol_probe.{with_field}.field.data" in private
    # Only the declared leaves travel: the probe's voltage is not one of them here.
    assert f"magnetics.b_field_pol_probe.{with_field}.voltage.data" in sample
    assert f"magnetics.b_field_pol_probe.{with_field}.voltage.data" not in private
    assert private["dataset_description.data_entry.pulse"] == 39915
    # The private copy is a plain ODS the builder may write into; the input is untouched.
    assert type(private).__name__ == "ODS" and private is not sample


def test_only_a_lazy_store_is_a_lazy_ods(sample):
    assert not is_lazy_ods(sample)
    assert not materialises_for_builder(sample, "equilibrium_overview")


# ---------------------------------------------------------------------------
# the proof: the same model from a lazy-like IMAS entry as from the ODS
# ---------------------------------------------------------------------------


def _omas_bound_on_sample(sample):
    return sorted(
        name for name, recipe in R.RECIPES.items()
        if isinstance(recipe, R.CallableRecipe) and recipe.backend == R.OMAS_BOUND
        and missing_required_path(sample, name) is None
    )


def test_the_sample_supports_enough_omas_bound_views_for_this_to_mean_something(sample):
    names = _omas_bound_on_sample(sample)
    assert len(names) >= 10 and len([n for n in names if n not in ROOT_DECLARING]) >= 8


@pytest.mark.parametrize(
    "name",
    sorted(
        name for name, recipe in R.RECIPES.items()
        if isinstance(recipe, R.CallableRecipe) and recipe.backend == R.OMAS_BOUND
    ),
)
def test_an_omas_bound_view_builds_from_its_declared_reads_on_a_lazy_entry(name, sample, lazy_entry):
    if missing_required_path(sample, name) is not None:
        pytest.skip("the packaged sample cannot build this view")
    if name in ROOT_DECLARING:
        root = next(t for t in R.RECIPES[name].reads if "." not in t)
        with pytest.raises(NotImplementedError, match=f"{name!r} deep-copies the whole {root!r} IDS"):
            _quiet(build_model, name, [("39915", lazy_entry)])
        return
    expected = _quiet(build_model, name, normalize_entries(sample, label=["39915"]))
    got = _quiet(build_model, name, [("39915", lazy_entry)])
    assert_models_equal(got, expected, where=name)


# ---------------------------------------------------------------------------
# a lazy OMAS store: leaves by template, a root by walking the store's index
# ---------------------------------------------------------------------------


def _lazy_ods_fixtures():
    spec = importlib.util.spec_from_file_location("_lazy_ods_fixtures", Path(__file__).with_name("test_lazy_ods.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_equilibrium_store():
    fx = _lazy_ods_fixtures()
    from vaft.database.lazy_ods import HSDSODS, HSDSStore

    module = fx.FakeH5pyd({
        "hdf5://main/39915/equilibrium.h5": fx.FakeFile("equilibrium", fx.FakeGroup({
            "time": fx.FakeDataset([0.1, 0.2]),
            "time_slice[]&AOS_SHAPE": fx.FakeDataset([2]),
            "time_slice[]&time": fx.FakeDataset([0.1, 0.2]),
            "time_slice[]&global_quantities&ip": fx.FakeDataset([10.0, 20.0]),
            "time_slice[]&global_quantities&q_axis": fx.FakeDataset([1.1, 1.2]),
        })),
        "hdf5://main/39915/dataset_description.h5": fx.FakeFile("dataset_description", fx.FakeGroup({
            "data_entry&pulse": fx.FakeDataset(39915),
        })),
    })
    store = HSDSStore(39915, ids=["equilibrium", "dataset_description"], h5pyd_module=module)
    return HSDSODS(store=store, consistency_check=False), store


def test_a_lazy_store_is_read_by_template_and_walked_for_a_root():
    lazy, store = _fake_equilibrium_store()
    assert is_lazy_ods(lazy)
    assert materialises_for_builder(lazy, "equilibrium_overview")
    assert not materialises_for_builder(lazy, "equilibrium_geometry_topview")  # neutral: no conversion at all

    # A leaf template fetches exactly the leaves it names, index by index.
    assert expand_template(lazy, "equilibrium.time_slice.{i}.global_quantities.ip") == [
        "equilibrium.time_slice.0.global_quantities.ip", "equilibrium.time_slice.1.global_quantities.ip",
    ]
    # A bare root walks the store's index: every leaf beneath, no more.
    from omas import ODS

    private = ODS(consistency_check=False)
    R._copy_root(lazy, "equilibrium", private, "equilibrium_overview")
    assert list(private["equilibrium.time"]) == [0.1, 0.2]
    assert private["equilibrium.time_slice.1.global_quantities.q_axis"] == 1.2
    assert len(private["equilibrium.time_slice"]) == 2
    store.close()


def test_discovery_says_a_lazy_store_is_read_by_its_declared_paths():
    from vaft.plot.backend.discovery import describe_one

    lazy, store = _fake_equilibrium_store()
    record = describe_one("equilibrium_overview", [("39915", lazy)])
    assert "declared reads on this lazy input" in record.reason
    assert "update_equilibrium_derived_profiles" in record.reason
    store.close()
