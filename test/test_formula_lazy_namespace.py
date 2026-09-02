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
        ("gradient", "stability", ("utils", "equilibrium")),
        ("COLLISIONALITY_COEF", "stability", ("constants",)),
        ("ME", "stability", ("constants", "equilibrium")),
        ("MI_P", "stability", ("constants", "equilibrium")),
        ("QE", "stability", ("constants", "equilibrium")),
        ("MU0", "green", ("constants", "equilibrium", "stability")),
        ("trapz_integral", "green", ("utils", "equilibrium")),
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
