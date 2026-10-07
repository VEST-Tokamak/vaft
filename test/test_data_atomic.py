"""Atomic and spectral identity under ``vaft.data`` (#1711), and the deprecated ``vaft.spectroscopy`` alias."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from vaft.data import atomic, spectroscopy

LEGACY = "vaft.spectroscopy"


def test_one_table_owns_the_nuclear_charge_and_the_weight():
    from vaft.data.synthetic_kinetic_profiles import ION_SPECIES
    from vaft.process import atomic as process_atomic
    from vaft.process.impurity import _element_table

    assert set(atomic.STANDARD_ATOMIC_WEIGHTS) == set(atomic.ATOMIC_NUMBERS)
    for symbol, z in atomic.ATOMIC_NUMBERS.items():
        weight = atomic.STANDARD_ATOMIC_WEIGHTS[symbol]
        assert ION_SPECIES[symbol] == (z, weight)
        assert _element_table()[symbol] == (z, weight)
    assert process_atomic._ATOMIC_NUMBERS is atomic.ATOMIC_NUMBERS
    assert process_atomic._ELEMENT_NAMES is atomic.ELEMENT_NAMES
    # the values the impurity preset and the #1565 reference numbers rest on
    assert ION_SPECIES["C"] == (6, 12.011) and ION_SPECIES["O"] == (8, 15.999) and ION_SPECIES["D"][0] == 1


def test_the_atomic_parser_reads_species_not_lines():
    assert atomic.parse_species("C III") == atomic.AtomicSpecies("C", None, 3)
    assert atomic.parse_species("H") == atomic.AtomicSpecies("H", 1)            # a selector: protium
    assert atomic.parse_species("deuterium") == atomic.AtomicSpecies("H", 2)
    assert atomic.parse_species("H_alpha") is None                              # a line, not a species
    assert spectroscopy.parse_emission_term("H_alpha").species == atomic.AtomicSpecies("H", 1)


def test_identity_types_carry_no_population():
    fields = set(atomic.AtomicSpecies.__dataclass_fields__)
    assert fields == {"element", "mass_number", "ionization_stage"}


def test_the_data_modules_import_nothing_heavy():
    code = ("import sys; import vaft.data.atomic, vaft.data.spectroscopy; "
            "bad = [m for m in ('omas', 'imas', 'matplotlib', 'vaft.plot', 'vaft.process') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-W", "error::DeprecationWarning", "-c", code],
                   check=True, capture_output=True, text=True)


def test_the_old_module_warns_and_reexports_the_same_objects():
    sys.modules.pop(LEGACY, None)
    with pytest.warns(DeprecationWarning, match="vaft.data.atomic"):
        legacy = importlib.import_module(LEGACY)
    assert legacy.Species is atomic.AtomicSpecies
    assert legacy.LineIdentity is spectroscopy.SpectralLineIdentity
    for name in legacy.__all__:
        if name in ("Species", "LineIdentity", "parse_species"):
            continue
        owner = atomic if name in atomic.__all__ else spectroscopy
        assert getattr(legacy, name) is getattr(owner, name), name
    # the legacy parse_species keeps reading series terms as their species
    assert legacy.parse_species("H_alpha") == atomic.AtomicSpecies("H", 1)
    assert legacy.parse_species("CIII") == atomic.parse_species("CIII")
    assert legacy.parse_species("C III alpha") is None


def test_nothing_in_vaft_or_its_tests_imports_the_old_module():
    # the test tree too: a test importing the alias becomes a collection error under
    # -W error::DeprecationWarning today and when the alias is removed (cold review 0.8.0)
    import ast
    from pathlib import Path

    import vaft

    package = Path(vaft.__file__).parent
    tests = Path(__file__).resolve().parent
    offenders = []
    for root in (package, tests):
        for path in sorted(root.rglob("*.py")):
            if path == package / "spectroscopy.py":
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.module == LEGACY:
                    offenders.append(f"{path.relative_to(root.parent)}:{node.lineno}")
                elif isinstance(node, ast.Import) and any(alias.name == LEGACY for alias in node.names):
                    offenders.append(f"{path.relative_to(root.parent)}:{node.lineno}")
                elif (isinstance(node, ast.ImportFrom) and node.module == "vaft"
                      and any(alias.name == "spectroscopy" for alias in node.names)):
                    offenders.append(f"{path.relative_to(root.parent)}:{node.lineno}")
    assert not offenders, f"import vaft.data.atomic / vaft.data.spectroscopy instead: {offenders}"
