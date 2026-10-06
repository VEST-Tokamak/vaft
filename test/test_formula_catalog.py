"""The formula catalog: discovery parity, resolution rules and laziness (issue #248)."""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path

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
        # +gp_fit, the scipy Gaussian process (#426); +a/L, the normalised
        # gradient scale length the tutorial's kinetic state reports: 12.
        "utils": 12,
        # #711 split the virial closures out of equilibrium: 110 = 77 + 33.
        # #365 added the two IMAS extremity triangularities and the sub-vertex
        # extremum helper they share with vaft.process: 77 + 3 = 80.
        # #760 renamed the first-principles bremsstrahlung form to state its
        # real argument order; the deprecated spelling is a distinct function
        # object, so it counts: 80 + 1 = 81.
        # The electron and ion thermal pressures p = n T e (#952): 82 + 2 = 84.
        # #782 added the dimensional internal inductance and its li_3
        # conversions: 84 + 3 = 87.
        # +4 psi_N profile kernels and their derivatives (#552): 93 + 4 = 97.
        # 0.8.0 removed the two deprecated shims promised gone in it (#355, #760):
        # current_density_from_psi and the Z_eff-first bremsstrahlung spelling, 99 - 2 = 97.
        # #351 added the dimensionless-to-engineering inverse map: 97 + 1 = 98.
        "equilibrium": 101,  # +estimated_q95, q_star_cylindrical, q_star_kink (#1583), +SFL toroidal shift nu (#1074 part 2), +miller_surface, vacuum_toroidal_field (#1145), +shafranov_shift (#1073), +generalized SFL angle (#1074), +GS source and J_phi(p', FF') (#1052), +flux freezing (#1209)
        "virial": 33,
        "stability": 37,  # +s-alpha ballooning eigenmode and k_x(theta) (#1075 part 2), +shear Alfven frequency, magnetosonic speeds (#1063), +kadomtsev_mixing_radius (#1209)
        "green": 16,
        "atomic": 11,  # +mean charge and Z_eff (#783 3.10), +single-impurity inversion (#952), +hydrogenic levels and wavelengths (#1046), +mean square charge, transient abundances, coronal relaxation time (#1565)
        "statistics": 22,
        "magnetics": 2,
        # #781 child A: Romero's exact transformer identities.
        "transformer": 8,   # +Romero first-order closure (#781 child C)
        "neoclassical": 18,  # +orbit scales and regime orderings (#1111)
        # #783 first slice: the prefill -> Townsend -> Lloyd breakdown chain.
        # #783 comment 1 added the post-avalanche equilibrium-field and
        # flux-closure kernels and the limiter-aperture geometry, comment 2
        # the generic Townsend inversion, and #676 the Ejiri mirror proxy:
        # 5 + 7 = 12.  #888 added the Lloyd figure of merit E_phi B_phi / B_p
        # the tutorial's empirical thresholds are stated against: 17 + 1 = 18.
        # #783 3.2/3.6-3.8 closed the lumped plasma circuit -- resistivity,
        # ring resistance, circular inductance, dIp/dt and the L/R time:
        # 18 + 5 = 23.  #783 3.9, the burn-through barrier of a depleting
        # fill: 23 + 5 = 28.  The Townsend gas catalogue, coefficients and
        # the gas-keyed threshold: 28 + 2 = 30.  #782's boundary-voltage and
        # internal inductive-voltage splits: 30 + 2 = 32.
        "startup": 32,
        "particle": 13,  # +gyration_offset (#1145), +mirror (#1070), +invariants and P_phi (#1092)
        "geometry": 12,  # slab / cylinder / local reduction (#1062), +Ampere and peaked-current q (#1072), +Harris sheet, X-point (#1063)
        "ripple": 6,  # TF ripple field and orbit consequences (#1070)
        "disruption": 11,  # TQ/CQ, induced field, runaway reference relations (#1041)
        "nbi": 6,  # beam rate, attenuation, birth density, shine-through, momentum rate (#1136)
        "waves": 7,  # cold-plasma frequencies, Stix parameters, dielectric tensor, n^2 roots, CMA, regime (#1113)
        "ntv": 2,  # precession frequency and flux-torque relation (#1111)
        "sol": 19,  # sound speed, sheath fluxes, Spitzer-Harm, two-point conduction, Eich profile (#951), MARFE (#1209), blobs (#1211)
        "vde": 6,  # vertical motion, thin-wall time, halo descriptors (#1042)
        "pwi": 4,  # collision kinematics, reflection/recycling definitions, Bohdansky threshold (#1047)
        "boundaries": 13,  # operational-boundary data model: value, margin, window, curve, registry (#1067), +Hugill coordinates (#1068), +threshold line and quantity identity (#1425), +Freidberg kink coordinates (#1456), +Menard q*, ITER and START q95 estimates (#1580)
        "impurity": 9,  # mixture moments, target-Z_eff solver, reduce/expand pseudo-impurity, dilution (#1565)
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
    # _taxonomy is the Reduction vocabulary (#1626): pure tuples, no physics
    assert loaded == {"vaft.formula.catalog", "vaft.formula._docstring", "vaft.formula._taxonomy"}


def test_describing_one_formula_imports_only_its_category():
    loaded = _loaded_after("import vaft.formula; vaft.formula.describe('greenwald_density')")
    # stability itself pulls constants and utils; that is its own import graph.
    assert loaded == {
        "vaft.formula.catalog",
        "vaft.formula._docstring",
        "vaft.formula._taxonomy",
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
    """``-X importtime`` costs of the formula modules, catalog absent.

    Both columns, deliberately. This test read only the self-time and checked
    only three modules, so it passed all the way through #426: importing
    `vaft.formula.stability` self-timed at 0.55 ms while costing 2.57 s
    cumulatively, because `vaft.formula.utils` -- not in the checked list --
    imported scikit-learn at module scope. Adding `utils` without reading the
    cumulative column would still have passed, and so would the reverse.
    """
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import vaft.formula.stability"],
        capture_output=True,
        text=True,
        check=True,
    )
    self_us: dict[str, int] = {}
    cumulative_us: dict[str, int] = {}
    for line in result.stderr.splitlines():
        match = re.match(r"import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s*(\S+)", line)
        if match:
            name = match.group(3).strip()
            self_us[name] = int(match.group(1))
            cumulative_us[name] = int(match.group(2))
    assert "vaft.formula.catalog" not in self_us
    assert "vaft.formula._docstring" not in self_us
    assert "sklearn" not in cumulative_us, "scikit-learn is optional since #426"

    checked = (
        "vaft.formula",
        "vaft.formula.constants",
        "vaft.formula.utils",
        "vaft.formula.stability",
    )
    for module in checked:
        assert self_us[module] < 50_000, (module, self_us[module])

    # Cumulative, which is what a caller actually waits for. Measured at
    # ~0.81 s for stability and ~0.35 ms for utils once sklearn left the import
    # path. The budgets are deliberately loose -- a wall-clock assertion on a
    # shared runner is the flaky half of this test, and the `sklearn` check
    # above is the timing-independent guard that actually catches the
    # regression this exists for. These only catch something an order of
    # magnitude worse.
    assert cumulative_us["vaft.formula.utils"] < 1_000_000, cumulative_us["vaft.formula.utils"]
    assert cumulative_us["vaft.formula.stability"] < 3_000_000, (
        cumulative_us["vaft.formula.stability"]
    )


# --- snapshot ------------------------------------------------------------------

_ROW_KEYS = {
    "id", "name", "category", "module", "signature", "summary", "description",
    "parameters", "returns", "sections", "references", "empirical",
    "convention_sensitive", "deprecated", "aliases", "shadowed_by", "raises", "source",
    "definitions", "reduction",
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
        # the inline source (#1069) is the code as written, roles and all
        prose = {**row, "source": {k: v for k, v in row["source"].items() if k != "code"}}
        assert ":func:" not in yaml.safe_dump(prose), row["id"]
        assert "Raises" not in [section["title"] for section in row["sections"]], row["id"]
        assert row["source"]["path"] == f"vaft/formula/{row['category']}.py", row["id"]
        for item in row["raises"]:
            assert set(item) == {"type", "description"} and item["type"], row["id"]


def test_snapshot_rows_point_at_their_definition():
    """The site links each entry to ``source.path#L<line>``; that line must be the ``def``."""
    root = Path(vaft.formula.__file__).resolve().parents[2]
    for row in catalog.documentation_snapshot()["formulas"]:
        lines = (root / row["source"]["path"]).read_text(encoding="utf-8").splitlines()
        start = lines[row["source"]["line"] - 1].lstrip()
        assert start.startswith(("def ", "@")), (row["id"], start)


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


# --- definitions and the Markdown card (issue #889) ---------------------------------


def test_the_definition_is_the_docstrings_display_equation():
    spec = catalog.describe("greenwald_density")
    assert spec.definitions == (
        r"n_G\,[10^{20}\,\mathrm{m^{-3}}] = \frac{I_p\,[\mathrm{MA}]}{\pi a^2\,[\mathrm{m^2}]}",
    )
    assert spec.definition == f"$${spec.definitions[0]}$$"
    several = catalog.describe("startup.plasma_external_inductance_hirshman_from_R_eps_kappa")
    assert len(several.definitions) == 3 and several.definition.count("$$") == 6
    assert catalog.describe("equilibrium.poloidal_field_magnitude").definition == ""


@pytest.mark.parametrize("category", catalog.CATEGORIES)
def test_every_definition_is_read_out_of_the_description(category):
    for spec in catalog.list_formulas(category):
        rest = spec.description
        for equation in spec.definitions:
            assert equation and "$$" not in equation, spec.qualname
            assert equation in rest, spec.qualname
            rest = rest.replace(equation, "", 1)
        assert re.sub(r"\$\$\s*\$\$", "", rest).count("$$") == 0, spec.qualname  # none left behind


def test_the_snapshot_carries_the_same_definitions_the_card_renders():
    for row in catalog.documentation_snapshot()["formulas"]:
        spec = catalog.describe(row["id"])
        assert row["definitions"] == list(spec.definitions), row["id"]
        for equation in row["definitions"]:
            assert equation in row["description"], row["id"]  # what the reference page shows
            assert equation in spec.to_markdown(), row["id"]


@pytest.mark.parametrize("category", catalog.CATEGORIES)
def test_every_formula_renders_a_markdown_card(category):
    for spec in catalog.list_formulas(category):
        card = spec.to_markdown()
        assert card == spec._repr_markdown_()
        assert card.startswith(f"**`{spec.qualname}")
        assert ":func:" not in card and ":class:" not in card, spec.qualname
        for item in spec.parameters:
            assert f"`{item.name}` : " in card, (spec.qualname, item.name)
            if item.unit:
                assert f"[{item.unit}]" in card, (spec.qualname, item.name)
        for title, _ in spec.sections:
            if title not in ("Parameters", "Returns", "Yields", "Raises", "References"):
                assert f"**{title}**" in card, (spec.qualname, title)
        for ref in spec.references:
            assert f"- [{ref.label}]" in card, spec.qualname


def test_the_card_follows_the_reference_page_order():
    card = catalog.describe("greenwald_density").to_markdown()
    marks = ["**`stability.greenwald_density(I_p, a)`**", "Greenwald density limit", "$$n_G",
             "**Parameters**", "**Returns**", "**Convention**", "**Validity**", "**References**"]
    positions = [card.index(mark) for mark in marks]
    assert positions == sorted(positions)
    assert "*empirical fit · convention-sensitive*" in card


def test_selected_parts_render_alone_and_in_the_order_asked():
    spec = catalog.describe("greenwald_density")
    assert spec.to_markdown(["definition"]) == spec.definition
    text = spec.to_markdown(["validity", "definition"])
    assert text.index("**Validity**") < text.index("$$") and "**Parameters**" not in text
    assert spec.to_markdown(["Physical_interpretation"]) == spec.to_markdown(["physical interpretation"])
    # the description shows the equation in its prose already
    both = spec.to_markdown(["definition", "description"])
    assert both == spec.to_markdown(["description"])


def test_asking_for_what_a_formula_lacks_or_does_not_exist_fails():
    spec = catalog.describe("equilibrium.poloidal_field_magnitude")
    with pytest.raises(ValueError, match="documents no definition"):
        spec.to_markdown(["definition"])
    with pytest.raises(ValueError, match="documents no validity"):
        catalog.describe("exb_drift_velocity").to_markdown(["validity"])
    with pytest.raises(ValueError, match="unknown part"):
        spec.to_markdown(["equation"])
    with pytest.raises(TypeError):
        spec.to_markdown("definition")
    with pytest.raises(ValueError, match="empty"):
        spec.to_markdown([])


def test_a_part_asked_twice_renders_once():
    spec = catalog.describe("greenwald_density")
    assert spec.to_markdown(["definition", "definition"]) == spec.definition
    assert spec.to_markdown(["description", "definition", "definition"]) == spec.to_markdown(["description"])


def test_the_card_carries_the_alias_and_shadowing_notes_the_page_shows():
    aliased = [spec for spec in catalog.list_formulas() if spec.aliases]
    shadowed = [spec for spec in catalog.list_formulas() if spec.shadowed_by]
    assert aliased and shadowed
    for spec in aliased:
        assert all(f"`{alias}`" in spec.to_markdown(["signature"]) for alias in spec.aliases), spec.qualname
    for spec in shadowed:
        assert f"resolves to the `{spec.shadowed_by}` copy" in spec.to_markdown(), spec.qualname


def test_examples_render_as_code():
    for spec in catalog.list_formulas():
        if spec.section("Examples"):
            assert "**Examples**\n\n```python\n" in spec.to_markdown(), spec.qualname


def test_the_terminal_text_is_not_markdown():
    spec = catalog.describe("greenwald_density")
    assert str(spec) == spec.render()
    assert spec.render().startswith("stability.greenwald_density(I_p, a)")
    assert "**" not in spec.render()


def test_show_renders_in_jupyter():
    formatters = pytest.importorskip("IPython.core.formatters")
    shown = vaft.formula.show("greenwald_density", sections=["definition"])
    data, _ = formatters.DisplayFormatter().format(shown)
    assert data["text/markdown"] == catalog.describe("greenwald_density").definition
    data, _ = formatters.DisplayFormatter().format(vaft.formula.describe("greenwald_density"))
    assert data["text/markdown"] == catalog.describe("greenwald_density").to_markdown()
    assert str(vaft.formula.show("greenwald_density")) == data["text/markdown"]
