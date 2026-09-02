"""Every public formula documents itself under the issue #248 contract.

The contract is parsed by :mod:`vaft.formula._docstring`; this file enforces
it and holds the three policy lists that are physics judgement rather than
content: which functions are definitional (no literature reference needed),
which are convention-sensitive (must carry a ``Convention`` section) and
which are empirical fits (``Validity`` must open with ``Empirical fit.`` and
a source must be cited).
"""

from __future__ import annotations

import inspect

import pytest

import vaft.formula
from vaft.formula import catalog
from vaft.formula._docstring import (
    CUSTOM_SECTIONS,
    EMPIRICAL_MARKER,
    SECTION_VOCABULARY,
    parse_docstring,
    parse_module_docstring,
)

#: Identities and bookkeeping: no literature source adds anything.
DEFINITIONAL = frozenset({
    "aspect_ratio_from_a_R",
    "inverse_aspect_ratio_from_a_R",
    "calc_inverse_aspect_ratio",
    "psi_normalised",
    "greenwald_fraction",
    "heating_power_from_p_ohm_p_aux",
    "auxiliary_heating_power",
    "loss_power_from_p_heat_dWdt_p_rad",
    "nbi_heating_power_from_I_nbi_V_nbi",
    "ec_heating_power_from_I_ec_V_ec",
    "ohmic_heating_power_from_I_p_V_res",
    "stored_energy_from_p_V",
    "inductive_voltage_from_dW_magdt_I_p",
    "confinement_time_from_P_loss_W_th",
    "confinement_factor_ITER89P",
    "line_to_volume_avg_density",
    "eK_from_K",
    "peaking_factor",
    "calculate_distance",
    "trapz_integral",
    "greens_integral_2d",
    "greens_integral_3d",
    # utils: numerical helpers, not physics
    "gradient",
    "normalize_profile",
    "calculate_peaking_factor",
    "calculate_volume_weighted_average",
    "calculate_poloidal_flux",
    "calculate_toroidal_flux",
    "make_fit_function",
    "fit_profile",
})

#: Sign, normalisation, COCOS or engineering-unit choices change the number.
CONVENTION_SENSITIVE = frozenset({
    # psi / B / j / q / flux
    "poloidal_field_factor",
    "radial_magnetic_field_from_psi",
    "vertical_magnetic_field_from_psi",
    "current_density_from_psi",
    "current_density_from_B",
    "psi_from_RBtheta",
    "phi_from_Bphi",
    "rhoN_from_phi",
    "q_from_phi",
    "q_from_rhoN",
    "rhoN_from_qpsiN",
    "shear_from_r_q",
    "surface_poloidal_flux_from_psi_boundary",
    "loop_voltage_from_total_flux",
    "calculate_poloidal_flux",
    "calculate_toroidal_flux",
    # beta / q in engineering units
    "beta_N_from_beta_a_B0_Ip",
    "normalized_plasma_current",
    "kink_safety_factor",
    "cylindrical_safety_factor_from_R_B_epsilon_I_f_kappa_delta",
    "q_cyl_from_B_R_epsilon_kappa_I",
    "beta_t_from_n_T_B",
    "ballooning_alpha_from_p_B_R",
    "greenwald_density",
    "greenwald_fraction",
    "confinement_time_from_engineering_parameters",
    # nu* and rho* families
    "collisionality_from_n_T_B_R",
    "normalized_collisionality_from_nu_ii_T_i_M_i_R_a_q",
    "normalized_collisionality_from_a_n_q_epsilon_T",
    "nu_star_from_n_T_B_R_epsilon_kappa_I",
    "rhostar_from_Te_a_Bt",
    "normalized_larmor_radius_from_M_T_a_Bt",
    "rho_star_from_M_T_B_R_epsilon",
    # virial closures
    "virial_S1_approx",
    "virial_S2_approx_from_D0_a_R0",
    "virial_S3_approx_from_eK_d",
    "virial_muihat_from_Bt_R0_dphi",
    # Green's functions: full weber
    "greens_function_2d",
    "green_br_bz",
    "greens_function_exact",
    "green_psi_exact",
    "green_br_bz_exact",
    "green_r",
    # atomic: ADF11 table units
    "interpolate_adf11",
    "fractional_abundances",
    "line_cooling_coefficient",
})

#: Fitted coefficients or scalings: the source dataset must be named.
EMPIRICAL = frozenset({
    "greenwald_density",
    "confinement_time_from_engineering_parameters",
    "empirical_li_qa",
    "li_from_qa_empirical",
    "kink_stability_criterion",
    "beta_stability_boundary",
    "ballooning_stability_criterion",
    "sawtooth_stability_criterion",
    "current_drive_efficiency",
    "bootstrap_current_fraction",
    "alpha_heating_power_from_n_D_n_T_T_keV_V",
})

SPECS = catalog.list_formulas()
IDS = [spec.qualname for spec in SPECS]
_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _function(spec):
    return getattr(vaft.formula._submodule(spec.category), spec.name)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_has_a_summary_line(spec):
    assert spec.summary, "missing docstring or summary"
    assert not spec.summary.startswith("#"), "legacy comment-style docstring"
    assert spec.summary.endswith("."), spec.summary


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_parser_reports_no_contract_violations(spec):
    assert not spec.errors, "\n".join(spec.errors)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_parameters_cover_the_signature_in_order(spec):
    if spec.deprecated:
        pytest.skip("deprecated shim: summary and See Also only")
    signature = inspect.signature(_function(spec))
    expected = [p.name for p in signature.parameters.values() if p.kind not in _VARIADIC]
    documented = [name.strip() for item in spec.parameters for name in item.name.split(",")]
    assert documented == expected


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_every_parameter_and_return_carries_a_unit_tag(spec):
    if spec.deprecated:
        pytest.skip("deprecated shim")
    for item in spec.parameters:
        assert item.unit, f"parameter {item.name} lacks a [unit] tag"
    assert spec.returns, "missing Returns section"
    for item in spec.returns:
        assert item.unit, f"return {item.name or item.type} lacks a [unit] tag"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_physics_formulas_cite_a_reference(spec):
    if spec.deprecated or spec.name in DEFINITIONAL:
        pytest.skip("definitional or deprecated")
    assert spec.references, "physics formula without a References section"
    for ref in spec.references:
        assert ref.text, f"empty reference [{ref.label}]"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_convention_sensitive_formulas_state_their_convention(spec):
    if spec.name not in CONVENTION_SENSITIVE:
        pytest.skip("not convention-sensitive")
    assert spec.convention_sensitive, "missing Convention section"
    assert len(spec.section("Convention") or "") > 20


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_empirical_formulas_say_so_and_name_their_source(spec):
    if spec.name not in EMPIRICAL:
        assert not spec.empirical, "flagged empirical but not in the EMPIRICAL policy list"
        pytest.skip("not empirical")
    assert spec.empirical, f"Validity must open with {EMPIRICAL_MARKER!r}"
    assert spec.references, "empirical fit without a cited source"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_documented_bugs_are_tracked_by_issue(spec):
    text = spec.section("Limitations") or ""
    if "tracked in" in text:
        assert "#" in text, "a tracked limitation must name its GitHub issue number"


def test_policy_lists_name_real_functions():
    known = {spec.name for spec in SPECS} | {alias for spec in SPECS for alias in spec.aliases}
    for name, policy in (
        ("DEFINITIONAL", DEFINITIONAL),
        ("CONVENTION_SENSITIVE", CONVENTION_SENSITIVE),
        ("EMPIRICAL", EMPIRICAL),
    ):
        assert policy <= known, (name, sorted(policy - known))


@pytest.mark.parametrize("doc", catalog.categories(), ids=lambda d: d.name)
def test_module_docstrings_carry_a_title_and_overview(doc):
    assert doc.title.endswith("."), doc.title
    assert doc.overview or doc.notation or doc.conventions


# --- parser unit tests ---------------------------------------------------------

_WELL_FORMED = '''Greenwald density limit $n_G$.

$$n_G = I_p / (\\pi a^2)$$

Parameters
----------
I_p : float
    Plasma current [MA].
a : float
    Minor radius [m].
    Second line of prose.

Returns
-------
n_G : float
    Greenwald density limit [1e19 m^-3].

Convention
----------
Engineering units -- MA, m -- note the dashes.

Validity
--------
Empirical fit. Greenwald et al. 1988 [1]_.

References
----------
.. [1] M. Greenwald et al., Nucl. Fusion 28 (1988) 2199,
       Eq. (1).
'''


def test_parser_reads_a_well_formed_docstring():
    parsed = parse_docstring(_WELL_FORMED)
    assert parsed.errors == ()
    assert parsed.summary == "Greenwald density limit $n_G$."
    assert parsed.description == "$$n_G = I_p / (\\pi a^2)$$"
    assert [(p.name, p.type, p.unit) for p in parsed.parameters] == [
        ("I_p", "float", "MA"),
        ("a", "float", "m"),
    ]
    assert parsed.parameters[1].description == "Minor radius. Second line of prose."
    assert parsed.returns == (parsed.returns[0],)
    assert (parsed.returns[0].name, parsed.returns[0].unit) == ("n_G", "1e19 m^-3")
    assert parsed.empirical and parsed.convention_sensitive and not parsed.deprecated
    assert parsed.references[0].text.endswith("2199, Eq. (1).")
    assert [title for title, _ in parsed.sections] == [
        "Parameters", "Returns", "Convention", "Validity", "References",
    ]


def test_parser_treats_raw_and_plain_docstrings_alike():
    assert parse_docstring(r"""Summary.""") == parse_docstring("""Summary.""")


def test_parser_handles_a_one_line_docstring():
    parsed = parse_docstring("Just a summary.")
    assert parsed.summary == "Just a summary."
    assert parsed.errors == () and parsed.parameters == () and parsed.returns == ()


def test_parser_reads_bare_and_named_tuple_returns():
    parsed = parse_docstring(
        "S.\n\nReturns\n-------\nnp.ndarray\n    First [T].\nq : float\n    Second [-].\n"
    )
    assert [(r.name, r.type, r.unit) for r in parsed.returns] == [
        (None, "np.ndarray", "T"),
        ("q", "float", "-"),
    ]


def test_parser_flags_unknown_headers_missing_units_and_bad_references():
    parsed = parse_docstring(
        "S.\n\nParameters\n----------\nx : float\n    No unit here.\n\n"
        "Caveats\n-------\nNot a vocabulary word.\n\n"
        "References\n----------\nWesson, Tokamaks.\n"
    )
    joined = "\n".join(parsed.errors)
    assert "unknown section header 'Caveats'" in joined
    assert "x: the first description paragraph must end with a unit tag" in joined
    assert "'.. [1] text'" in joined
    assert "Caveats" not in dict(parsed.sections)


def test_parser_does_not_mistake_a_dashed_prose_line_for_an_underline():
    parsed = parse_docstring("S.\n\nNotes\n-----\nRange is -1 -- 1, with --- inside.\n")
    assert parsed.errors == ()
    assert parsed.section("Notes") == "Range is -1 -- 1, with --- inside."


def test_deprecated_shims_are_recognised():
    parsed = parse_docstring("Deprecated: use psi_normalised.\n\nSee Also\n--------\npsi_normalised\n")
    assert parsed.deprecated and parsed.errors == ()


def test_vocabulary_lists_every_custom_section():
    assert set(CUSTOM_SECTIONS) <= set(SECTION_VOCABULARY)


def test_module_docstring_parser_reads_a_notation_table():
    doc = parse_module_docstring(
        "Title line.\n\nOverview prose.\n\nNotation\n--------\n"
        "ψ      : poloidal flux                 [Wb]\n"
        "q      : safety factor                 [-]\n"
        "x      : no unit\n"
    )
    assert doc.title == "Title line." and doc.overview == "Overview prose."
    assert [(r.symbol, r.description, r.unit) for r in doc.notation] == [
        ("ψ", "poloidal flux", "Wb"),
        ("q", "safety factor", "-"),
        ("x", "no unit", ""),
    ]
    assert doc.errors == ()
