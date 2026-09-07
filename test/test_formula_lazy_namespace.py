"""`vaft.formula` resolves its submodules lazily without changing what it exposes.

The package used to star-import all seven submodules at load time, so importing
one pure module -- `vaft.formula.statistics`, which needs nothing but numpy --
dragged in scipy by way of `.green` and `.equilibrium`, along with those
modules' import-time failure surface.

Resolution is now deferred, which makes the *order* of resolution load-bearing:
eighteen public names are defined by more than one submodule, and under the old
star imports the last one imported won.  These tests pin both halves -- that the
lazy path imports less, and that every name it hands back is the identical
object the eager path did.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import vaft.formula


#: The submodules, in the order they were star-imported when the package was
#: eager.  Later entries shadowed earlier ones.
_IMPORT_ORDER = (
    "constants",
    "utils",
    "equilibrium",
    "stability",
    "green",
    "atomic",
    "statistics",
    "magnetics",
)


def _eager_namespace():
    """What the old `from .<mod> import *` sequence left in the package."""
    namespace: dict[str, object] = {}
    for name in _IMPORT_ORDER:
        exec(f"from vaft.formula.{name} import *", namespace)  # noqa: S102
    return namespace


def _import_in_subprocess(statement: str) -> set[str]:
    """The `vaft.formula.*` submodules loaded by running ``statement`` alone."""
    code = (
        "import sys\n"
        f"{statement}\n"
        "print(' '.join(sorted(m for m in sys.modules "
        "if m.startswith('vaft.formula.'))))\n"
    )
    output = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(output.stdout.split())


# ---------------------------------------------------------------------------
# Laziness
# ---------------------------------------------------------------------------

def test_importing_the_package_alone_imports_no_submodule():
    assert _import_in_subprocess("import vaft.formula") == set()


def test_importing_one_submodule_does_not_drag_in_its_siblings():
    loaded = _import_in_subprocess("import vaft.formula.statistics")

    assert loaded == {"vaft.formula.statistics"}


def test_a_pure_submodule_stays_reachable_without_scipy_heavy_siblings():
    """The point of the exercise: numpy-only kernels cost only numpy."""
    loaded = _import_in_subprocess(
        "from vaft.formula.statistics import rms; rms([3.0, 4.0])"
    )

    assert "vaft.formula.green" not in loaded
    assert "vaft.formula.equilibrium" not in loaded


# ---------------------------------------------------------------------------
# Unchanged public surface
# ---------------------------------------------------------------------------

def test_all_covers_every_submodule_export_and_every_submodule_name():
    expected = set(vaft.formula._SUBMODULES)
    for name in _IMPORT_ORDER:
        module = vaft.formula._submodule(name)
        expected |= set(vaft.formula._exported(module))
    expected = {name for name in expected if not name.startswith("_")}

    assert set(vaft.formula.__all__) == expected
    assert vaft.formula.__all__ == sorted(vaft.formula.__all__)


def test_every_exported_name_is_the_object_the_eager_star_imports_bound():
    """The whole compatibility claim, checked name by name rather than spot-checked."""
    eager = _eager_namespace()

    mismatched = []
    for name in vaft.formula.__all__:
        if name in vaft.formula._SUBMODULES:
            continue
        if getattr(vaft.formula, name) is not eager[name]:
            mismatched.append(name)

    assert not mismatched


def test_star_import_still_binds_the_full_surface():
    namespace: dict[str, object] = {}
    exec("from vaft.formula import *", namespace)  # noqa: S102

    for name in vaft.formula.__all__:
        assert name in namespace, name


# ---------------------------------------------------------------------------
# Shadowing, which a naive first-match search would get wrong
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("name", "winner", "also_defined_in"),
    [
        ("trapz_integral", "green", ("utils",)),
    ],
)
def test_a_shadowed_name_resolves_to_the_last_submodule_that_defined_it(
    name, winner, also_defined_in
):
    resolved = getattr(vaft.formula, name)

    assert resolved is getattr(vaft.formula._submodule(winner), name)
    # The losers really do define it -- otherwise this test proves nothing.
    for loser in also_defined_in:
        assert name in vaft.formula._exported(vaft.formula._submodule(loser))


def test_only_one_name_is_exported_by_more_than_one_submodule():
    """Shadowing used to be accidental; now it is a single deliberate case.

    Before #368 no submodule declared ``__all__``, so every re-imported name --
    ``gradient`` reaching stability from utils, ``MU0``/``QE``/``ME`` reaching
    green and stability from constants -- was re-exported by each module that
    imported it, and resolution depended on import order rather than on where
    the name is defined. The values were identical objects, so nothing was
    wrong; it just meant ``vaft.formula.MU0`` was answered by ``green``.

    ``trapz_integral`` is the one real collision: ``green`` keeps its own copy
    (see the note on ``vaft.formula.green.trapz_integral``) and both define it.
    """
    owners: dict[str, list[str]] = {}
    for category in _IMPORT_ORDER:
        module = vaft.formula._submodule(category)
        for name in vaft.formula._exported(module):
            owners.setdefault(name, []).append(category)

    shared = {name: where for name, where in owners.items() if len(where) > 1}
    assert shared == {"trapz_integral": ["utils", "green"]}


@pytest.mark.parametrize(
    ("name", "defining_module"),
    [
        ("gradient", "utils"),
        ("COLLISIONALITY_COEF", "constants"),
        ("ME", "constants"),
        ("MI_P", "constants"),
        ("QE", "constants"),
        ("MU0", "constants"),
        ("C_B", "constants"),
    ],
)
def test_a_name_now_resolves_to_the_module_that_defines_it(name, defining_module):
    """What ``__all__`` bought: the answer no longer depends on import order."""
    module = vaft.formula._submodule(defining_module)
    assert name in vaft.formula._exported(module)
    assert getattr(vaft.formula, name) is getattr(module, name)


def test_a_name_a_submodule_declines_to_export_is_not_attributed_to_it():
    """`__all__` gates the star surface, so `hasattr` is not the right question."""
    statistics = vaft.formula._submodule("statistics")

    assert hasattr(statistics, "math")
    assert "math" not in vaft.formula._exported(statistics)


def test_an_unknown_name_still_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'not_a_formula'"):
        vaft.formula.not_a_formula


def test_dir_lists_the_public_surface():
    listed = dir(vaft.formula)

    assert set(vaft.formula.__all__) <= set(listed)
    assert listed == sorted(listed)


def test_importing_the_catalog_alone_loads_only_its_parser():
    # The discovery layer (issue #248) is opt-in: reaching it must not drag a
    # single physics submodule in until a formula is actually looked up.
    assert _import_in_subprocess("import vaft.formula.catalog") == {
        "vaft.formula.catalog",
        "vaft.formula._docstring",
    }


# ---------------------------------------------------------------------------
# The star export publishes formulas, not the modules' own imports (issue #368)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    ["np", "warnings", "Union", "Tuple", "List", "Dict", "Optional",
     "curve_fit", "minimize", "cho_factor", "cho_solve", "cumulative_trapezoid",
     "ellipe", "ellipk", "trapz_compat"],
)
def test_a_re_imported_name_is_not_a_formula(name):
    """None of these is a formula, and all of them used to be exported."""
    assert name not in vaft.formula.__all__


@pytest.mark.parametrize(
    ("old_name", "replacement"),
    [("e", "QE"), ("eV_to_J", "QE"), ("m_e", "ME"), ("epsilon_0", "EPS0"),
     ("k_B", "K_BOLTZMANN"), ("c", "C_LIGHT"), ("h", "H_PLANCK")],
)
def test_a_leaked_duplicate_constant_deprecates_to_its_canonical_name(old_name, replacement):
    """`vaft.formula.e` was the elementary charge; say so rather than vanish."""
    assert old_name not in vaft.formula.__all__
    with pytest.warns(DeprecationWarning, match=replacement):
        value = getattr(vaft.formula, old_name)
    assert value == getattr(vaft.formula, replacement)


def test_the_plumbing_gets_no_shim():
    """A shim for `np` would only invite importing numpy through this package."""
    with pytest.raises(AttributeError, match="no attribute 'np'"):
        vaft.formula.np


def test_every_submodule_declares_what_it_exports():
    for category in _IMPORT_ORDER:
        module = vaft.formula._submodule(category)
        assert hasattr(module, "__all__"), f"{category} still falls back to vars()"
