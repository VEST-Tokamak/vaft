"""``vaft.omas`` declares what it publishes, and the declaration keeps up with the wrappers.

The generated API reference reads ``__all__`` and nothing else, so before
``vaft.omas`` declared one, ``compute_magnetic_energy`` and every other
function of ``vaft.omas.process_wrapper`` / ``formula_wrapper`` / ``update`` /
``general`` was absent from ``/reference/api/omas/`` while being the way every
guide calls them (cold review 0.8.0 docs-and-tutorials F4).  Those submodules
still declare nothing (they are star-imported), so the package list is the one
statement of the surface -- and a wrapper function added without a line here
would vanish from the reference again.  This module makes that a failure.
"""

from __future__ import annotations

import inspect

import pytest

import vaft
import vaft.omas


def _defined_in_vaft_omas(name: str, obj) -> bool:
    """A wrapper the star imports bound; not a plotting adapter ``__getattr__`` cached on first use."""
    if inspect.ismodule(obj) or name.startswith("_") or vaft.omas._is_plotting_export(name):
        return False
    target = inspect.unwrap(obj) if callable(obj) else obj
    module = getattr(target, "__module__", None) or ""
    return module.startswith("vaft.omas") and not module.startswith("vaft.omas.plotting")


def test_every_wrapper_function_in_the_namespace_is_declared():
    """A function the star imports bind on ``vaft.omas`` is published there."""
    bound = {name for name, obj in vars(vaft.omas).items() if _defined_in_vaft_omas(name, obj)}
    undeclared = sorted(bound - set(vaft.omas.__all__))
    assert not undeclared, (
        f"vaft.omas binds {undeclared} from its own submodules but vaft/omas/__init__.py "
        "does not list them in __all__; the generated API reference would omit them"
    )


def test_every_declared_name_resolves_and_is_no_star_import_leak():
    """``__all__`` names real attributes, and none of them is OMAS's or numpy's."""
    for name in vaft.omas.__all__:
        obj = getattr(vaft.omas, name)  # AttributeError here is a dangling declaration
        module = getattr(inspect.unwrap(obj) if callable(obj) else obj, "__module__", None)
        if module is not None:
            assert module.startswith("vaft."), f"{name} is {module}.{name}, not VAFT API"
    assert "ODS" not in vaft.omas.__all__ and "np" not in vaft.omas.__all__


@pytest.mark.parametrize("name", [
    "compute_magnetic_energy",                       # process_wrapper
    "update_equilibrium_global_quantities_beta_li",  # update
    "compute_tau_E_scaling",                         # formula_wrapper
    "ods_cocos",                                     # general
    "sample_ods",                                    # sample
    "compare_ods",                                   # comparison, lazy
    "load_reference_manifest",                       # reference, lazy
])
def test_the_names_the_guides_call_are_published(name):
    assert name in vaft.omas.__all__
    assert callable(getattr(vaft.omas, name))


def test_the_generated_reference_lists_the_wrapper_functions():
    from vaft import _api_catalog

    ids = {entry["id"] for entry in _api_catalog.documentation_snapshot()["entries"]}
    assert "vaft.omas.compute_magnetic_energy" in ids
    assert "vaft.omas.update_equilibrium_global_quantities_beta_li" in ids
    modules = {row["name"]: row for row in _api_catalog.documentation_snapshot()["modules"]}
    assert modules["vaft.omas"]["declared"] is True
