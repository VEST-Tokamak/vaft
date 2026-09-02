"""The formula catalog: discovery parity, resolution rules and laziness (issue #248)."""

from __future__ import annotations

import inspect
import re
import subprocess
import sys

import pytest
import yaml

import vaft.formula
from vaft.formula import catalog
from vaft.formula._docstring import SECTION_VOCABULARY


def _run(code: str, *flags: str) -> str:
    """stdout of ``code`` run in a fresh interpreter."""
    result = subprocess.run(
        [sys.executable, *flags, "-c", code], capture_output=True, text=True, check=True
    )
    return result.stdout


def _loaded_after(statement: str) -> set[str]:
    """The ``vaft.formula.*`` modules present after running ``statement`` alone."""
    code = (
        "import sys\n"
        f"{statement}\n"
        "print(' '.join(sorted(m for m in sys.modules if m.startswith('vaft.formula.'))))\n"
    )
    return set(_run(code).split())


# --- coverage and resolution -------------------------------------------------


@pytest.mark.parametrize("category", catalog.CATEGORIES)
def test_catalog_covers_every_function_defined_in_the_submodule(category):
    module = vaft.formula._submodule(category)
    defined = {
        name
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and inspect.isfunction(obj)
        and obj.__module__ == module.__name__
    }
    specs = catalog._specs_for(category)
    covered = set(specs) | {alias for spec in specs.values() for alias in spec.aliases}
    assert covered == defined
    assert all(spec.category == category for spec in specs.values())


def test_the_catalog_counts_the_known_public_surface():
    counts = {doc.name: doc.count for doc in catalog.categories()}
    assert counts == {
        "constants": 0,
        "utils": 9,
        "equilibrium": 89,
        "stability": 19,
        "green": 16,
        "atomic": 3,
        "statistics": 21,
    }
    assert len(catalog.list_formulas()) == sum(counts.values())


@pytest.mark.parametrize(
    "alias, canonical",
    [
        ("magnetic_shear", "shear_from_r_q"),
        ("alpha_heating_power", "alpha_heating_power_from_n_D_n_T_T_keV_V"),
        ("coulomb_logarithm", "coulomb_logarithm_from_n_T"),
        ("calc_rho_star", "rho_star_from_M_T_B_R_epsilon"),
        ("calc_beta_t", "beta_t_from_n_T_B"),
        ("calc_q_cyl", "q_cyl_from_B_R_epsilon_kappa_I"),
        ("calc_nu_star", "nu_star_from_n_T_B_R_epsilon_kappa_I"),
        ("calc_omega_i_tau_E", "omega_i_tau_E_from_B_tau_E_M"),
    ],
)
def test_aliases_resolve_to_their_canonical_spec(alias, canonical):
    spec = catalog.describe(alias)
    assert spec.name == canonical
    assert alias in spec.aliases
    assert catalog.describe(f"equilibrium.{alias}") is spec


def test_bare_name_lookup_agrees_with_package_attribute_resolution():
    for spec in catalog.list_formulas():
        resolved = catalog.describe(spec.name)
        assert resolved.module == getattr(vaft.formula, spec.name).__module__, spec.name


def test_the_one_colliding_function_name_reports_who_wins():
    assert catalog.describe("trapz_integral").category == "green"
    assert catalog.describe("utils.trapz_integral").shadowed_by == "green"
    assert catalog.describe("green.trapz_integral").shadowed_by is None
    assert catalog.describe("green.trapz_integral").qualname == "green.trapz_integral"


def test_list_formulas_is_sorted_by_category_order_then_name():
    specs = catalog.list_formulas()
    keys = [(catalog.CATEGORIES.index(s.category), s.name) for s in specs]
    assert keys == sorted(keys)
    only = catalog.list_formulas(category="stability")
    assert {s.category for s in only} == {"stability"}
    assert [s.name for s in only] == sorted(s.name for s in only)


def test_describe_of_an_unknown_name_points_at_the_discovery_helper():
    with pytest.raises(KeyError, match="list_formulas"):
        catalog.describe("no_such_formula")
    with pytest.raises(KeyError, match="list_formulas"):
        catalog.describe("nowhere.greenwald_density")
    with pytest.raises(KeyError, match="list_formulas"):
        catalog.describe("stability.no_such_formula")


def test_search_matches_reference_and_summary_text_case_insensitively():
    names = {spec.qualname for spec in catalog.search("sauter")}
    assert "equilibrium.poloidal_field_factor" in names
    assert catalog.search("") == catalog.list_formulas()
    assert {s.category for s in catalog.search("", category="green")} == {"green"}


def test_render_shows_signature_units_and_parameters():
    text = str(catalog.describe("stability.greenwald_density"))
    assert text.startswith("stability.greenwald_density(I_p, a)")
    assert "[MA]" in text
    assert "Parameters" in text and "Returns" in text


# --- namespace hygiene --------------------------------------------------------


def test_catalog_names_are_reachable_on_the_package_but_never_exported():
    for name in sorted(vaft.formula._CATALOG_NAMES):
        assert name not in vaft.formula.__all__
        assert name in dir(vaft.formula)
    assert vaft.formula.describe is catalog.describe
    assert vaft.formula.catalog is catalog


# --- laziness (each check in its own interpreter) -----------------------------


def test_importing_the_package_loads_neither_catalog_nor_parser():
    assert _loaded_after("import vaft.formula") == set()


def test_importing_a_physics_submodule_does_not_load_the_catalog():
    loaded = _loaded_after("import vaft.formula.stability")
    assert "vaft.formula.catalog" not in loaded
    assert "vaft.formula._docstring" not in loaded


def test_the_star_import_neither_loads_nor_binds_the_catalog():
    output = _run(
        "import sys\n"
        "from vaft.formula import *\n"
        "print('describe' in dir(), 'vaft.formula.catalog' in sys.modules)\n"
    )
    assert output.split() == ["False", "False"]


def test_touching_describe_loads_the_catalog_and_nothing_physical():
    loaded = _loaded_after("import vaft.formula; vaft.formula.describe")
    assert loaded == {"vaft.formula.catalog", "vaft.formula._docstring"}


def test_describing_one_formula_imports_only_its_category():
    loaded = _loaded_after("import vaft.formula; vaft.formula.describe('greenwald_density')")
    # stability itself pulls constants and utils; that is its own import graph.
    assert loaded == {
        "vaft.formula.catalog",
        "vaft.formula._docstring",
        "vaft.formula.constants",
        "vaft.formula.utils",
        "vaft.formula.stability",
    }


def test_listing_one_category_imports_only_that_submodule():
    loaded = _loaded_after(
        "from vaft.formula.catalog import list_formulas; list_formulas(category='atomic')"
    )
    assert "vaft.formula.green" not in loaded
    assert "vaft.formula.equilibrium" not in loaded
    assert "vaft.formula.atomic" in loaded


def test_the_catalog_refuses_to_run_without_docstrings():
    result = subprocess.run(
        [sys.executable, "-OO", "-c", "import vaft.formula as F; F.describe('greenwald_density')"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "RuntimeError" in result.stderr and "-OO" in result.stderr


@pytest.mark.perf
def test_the_normal_import_path_stays_cheap():
    """``-X importtime`` self-times of the formula modules, catalog absent."""
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import vaft.formula.stability"],
        capture_output=True,
        text=True,
        check=True,
    )
    self_us: dict[str, int] = {}
    for line in result.stderr.splitlines():
        match = re.match(r"import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s*(\S+)", line)
        if match:
            self_us[match.group(3).strip()] = int(match.group(1))
    assert "vaft.formula.catalog" not in self_us
    assert "vaft.formula._docstring" not in self_us
    for module in ("vaft.formula", "vaft.formula.constants", "vaft.formula.stability"):
        assert self_us[module] < 50_000, (module, self_us[module])


# --- snapshot ------------------------------------------------------------------

_ROW_KEYS = {
    "id", "name", "category", "module", "signature", "summary", "description",
    "parameters", "returns", "sections", "references", "empirical",
    "convention_sensitive", "deprecated", "aliases", "shadowed_by",
}


def test_snapshot_schema():
    snapshot = catalog.documentation_snapshot()
    assert snapshot["schema_version"] == catalog.SCHEMA_VERSION
    assert set(snapshot) == {"schema_version", "generator", "source", "categories", "formulas"}
    assert [entry["path"] for entry in snapshot["source"]] == [
        f"vaft/formula/{key}.py" for key in vaft.formula._IMPORT_ORDER
    ]
    for entry in snapshot["source"]:
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
    category_names = [doc["name"] for doc in snapshot["categories"]]
    assert category_names == list(vaft.formula._IMPORT_ORDER)
    ids = [row["id"] for row in snapshot["formulas"]]
    assert len(ids) == len(set(ids))
    for row in snapshot["formulas"]:
        assert set(row) == _ROW_KEYS, row["id"]
        assert row["category"] in category_names
        assert row["id"] == f"{row['category']}.{row['name']}"
        for section in row["sections"]:
            assert section["title"] in SECTION_VOCABULARY, row["id"]
        assert ":func:" not in yaml.safe_dump(row), row["id"]


def test_cli_round_trip(tmp_path):
    output = tmp_path / "formula_catalog.yml"
    subprocess.run(
        [sys.executable, "-m", "vaft.formula.catalog", "--output", str(output)], check=True
    )
    assert yaml.safe_load(output.read_text(encoding="utf-8")) == catalog.documentation_snapshot()


def test_cli_can_restrict_to_one_category(tmp_path):
    output = tmp_path / "atomic.yml"
    catalog.main(["--output", str(output), "--category", "atomic"])
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert {row["category"] for row in data["formulas"]} == {"atomic"}
