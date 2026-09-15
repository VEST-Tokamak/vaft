"""Every name a vaft.formula function uses has to exist (issue #753).

`cyclotron_synchrotron_power_density_scaling_from_n_e_B_t_T_e` referenced `e`,
`epsilon_0`, `m_e` and `c` -- module-level constants that #368 had moved into
`constants.py` under their upper-case names. It was the one caller left behind,
so it raised `NameError` on every call, which is another way of saying nothing
had ever executed it.

The check that catches this is ruff's F821, and it was silently off: a star
import makes ruff abandon undefined-name analysis for the whole module, and
#711's `from .virial import *` compatibility shim had switched it off for the
4000-line formula module. Both halves are pinned here -- the physics, so the
correction cannot be reverted to a plausible-looking wrong constant, and the
coverage, so the next shim cannot turn the check off again.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from vaft.formula.constants import C_LIGHT, EPS0, ME, QE
from vaft.formula.equilibrium import (
    cyclotron_synchrotron_power_density_scaling_from_n_e_B_t_T_e as p_cyc,
)


REPO = Path(__file__).resolve().parents[1]


pytestmark = pytest.mark.core


def test_the_synchrotron_power_density_is_callable_and_matches_its_own_docstring():
    """The docstring states the scaling this reduces to; hold it to that.

    `p_cyc ~= 6.2e-17 B_t^2 n_e T_keV` W/m^3. A wrong constant -- the electron
    charge where the mass belongs, say -- still returns a float, so the guard
    has to be the documented magnitude rather than merely "it does not raise".
    """
    n_e, B_t, T_e_eV = 1.0e19, 0.2, 100.0

    value = p_cyc(n_e_m3=n_e, B_t_T=B_t, T_e_eV=T_e_eV)

    documented = 6.2e-17 * B_t**2 * n_e * (T_e_eV / 1.0e3)
    assert value == pytest.approx(documented, rel=0.01)

    # And it is that expression, not a coincidence at one operating point.
    expected = (QE**4 / (3.0 * 3.141592653589793 * EPS0 * ME**3 * C_LIGHT**3)) * (
        n_e * B_t**2 * T_e_eV * QE
    )
    assert value == pytest.approx(expected, rel=1e-12)


def test_the_power_density_scales_as_the_formula_says():
    """Linear in n_e and T_e, quadratic in B_t."""
    base = dict(n_e_m3=1.0e19, B_t_T=0.2, T_e_eV=100.0)
    reference = p_cyc(**base)

    assert p_cyc(**{**base, "n_e_m3": 2.0e19}) == pytest.approx(2 * reference)
    assert p_cyc(**{**base, "T_e_eV": 200.0}) == pytest.approx(2 * reference)
    assert p_cyc(**{**base, "B_t_T": 0.4}) == pytest.approx(4 * reference)


@pytest.mark.skipif(
    importlib.util.find_spec("ruff") is None, reason="ruff is not installed"
)
def test_no_formula_module_references_an_undefined_name():
    """F821 over `vaft/formula/`, which a star import would silently disable.

    Run as a subprocess rather than through ruff's Python API: the API is not a
    supported interface, and the command here is exactly what CI and a developer
    would run by hand.
    """
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "F821",
         "--output-format", "concise", "vaft/formula/"],
        cwd=REPO, capture_output=True, text=True,
    )

    assert result.returncode == 0, (
        "vaft/formula/ references names that do not exist:\n"
        f"{result.stdout}{result.stderr}\n\n"
        "If this appeared after adding `from .x import *`, that is the cause: "
        "ruff abandons F821 for a module containing a star import. Spell the "
        "re-exports out instead (issue #753)."
    )


def test_the_virial_compatibility_shim_still_re_exports_without_a_star_import():
    """#711's shim kept working when the star import became explicit.

    `from vaft.formula.equilibrium import virial_*` has to keep resolving, and
    the names must stay out of this module's `__all__` so the catalog attributes
    them to `vaft.formula.virial`, which defines them.
    """
    from vaft.formula import equilibrium, virial

    for name in virial.__all__:
        assert hasattr(equilibrium, name), f"{name} no longer re-exported"
        assert name not in equilibrium.__all__, f"{name} leaked into equilibrium.__all__"

    # An actual star-import statement, not the substring: the prose explaining
    # why there isn't one contains it.
    import ast

    module = ast.parse((REPO / "vaft" / "formula" / "equilibrium.py").read_text(encoding="utf-8"))
    starred = [
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "*" for alias in node.names)
    ]
    assert not starred, (
        f"star import from {starred} switches F821 off for this module again"
    )
