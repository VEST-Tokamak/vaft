"""Every public processing function documents itself under the issue #252 contract.

The contract is parsed by :mod:`vaft.process._docstring` and its structural
half -- parameters match the signature and carry units, there is a Returns
with units, Applicability declares a machine scope -- is checked by the
catalog itself, so a function's ``conforming`` flag means the same thing on
the documentation site as it does here.  This file adds the half that is
physics judgement rather than parseable content: which functions need no
Provenance because they are pure numerics, which are multi-stage and must
list their Processing steps, which change the processing state and must say
so, and which are convention-sensitive.

``PENDING`` names the submodules not yet brought under the contract.  Each
of #418-#421 removes its modules from it, and a test below refuses to let a
module leave early or stay once it conforms, so the split is enforceable
rather than aspirational.
"""

from __future__ import annotations

import pytest

import vaft.process
from vaft.process import catalog
from vaft.process._docstring import (
    CUSTOM_SECTIONS,
    MACHINE_INDEPENDENT,
    SECTION_VOCABULARY,
    VEST_SPECIFIC,
)

#: Submodules whose functions are not yet documented under the contract.
#: Sub-issue B (#418): magnetics, electromagnetics, fluctuation -- done.
#: Every submodule is now under the contract, so PENDING is empty and the
#: gate below holds it that way: a module that stops conforming fails.
#: Sub-issue C (#419): equilibrium and cocos -- done.
#: Sub-issue D (#420): profile, atomic -- done.
#: Sub-issue E (#421): impa, soft_x_rays, langmuir, camera_geometry -- done.
#: Sub-issue F (#571) removed ``onset`` and ``wall_modes``, which landed
#: after #252 was scoped -- done.
PENDING: frozenset[str] = frozenset()

#: Pure numerics and bookkeeping: no source adds anything.
DEFINITIONAL = frozenset({
    "line_average_density",
    "linear_baseline",
    "quadratic_baseline",
    "exp_baseline",
    "describe_time_grid",
    "detrend_moving_average",
    "butterworth_lowpass",
    "butterworth_bandpass",
    "fir_filter",
    "fir_filter_coefficients",
    "is_signal_active",
    "signal_on_offset",
    "process_signal",
    "time_derivative",
    "filter_dataframe",
    "log_transform",
    "analyze_significance",
    "compute_metrics",
    "get_residuals",
    "get_correlation_matrix",
    "get_individual_correlations",
    "confinement_time_histogram",
    # profile / atomic (#420): pure numerics, bookkeeping, or synthetic constructions;
    # the mappers are a definition (normalize psi, derive rho_tor from q), not a ported method
    "equilibrium_mapping_thomson_scattering",
    "equilibrium_mapping_charge_exchange",
    "export_electron_profile_txt",
    "core_profiles_from_eq",
    "core_profiles_from_eq_ratio",
    "compute_time_match_atol",
    "find_time_match_index",
    "normalize_atomic_symbol",
    "integrate_emissivity_profile",
})

#: Multi-stage routines: the order of operations decides what the output means.
PIPELINE = frozenset({
    # magnetics / electromagnetics / fluctuation (#418)
    "analyze_fluctuation_spectrum",
    "b_field_pol_probe_field",
    "flux_loop_flux",
    "mirnov_preprocess_signal",
    "rogowski_coil_ip",
    "toroidal_mode_analysis",
    "toroidal_phase_fit_at_time",
    "vest_b_field_pol_probe_legacy",
    "vest_equilibrium_magnetics_detailed",
    "vest_flux_loop_flux_from_voltage",
    "vest_flux_loop_legacy",
    # impa (#421)
    "find_tf_calibration_window",
    "fit_impa_geometry",
    "grade_impa_quality",
    "impa_calibrate_signals",
    "legacy_impa_compensation",
    "legacy_impa_position",
    "process_impa",
    # soft_x_rays / langmuir (#421)
    "process_triple_probe",
    "sxr_band_signals",
    "sxr_electron_temperature",
    # equilibrium (#419)
    "calculate_reconstructed_diamagnetic_flux",
    "convert_cocos",
    "derive_global_descriptors",
    "derive_radial_coordinates",
    "fit_miller_sequence",
    "fit_miller_surface",
    "flux_surface_quantities",
    "prepare_boundary_for_shafranov",
    "psi_to_radial",
    "psi_to_rz",
    "solve_solovev_constraints",
    "trace_field_line",
    # cocos (#419)
    "validate_cocos",
    "identify_flux_exponent",
    "identify_convention",
    "cocos_consistency_signs",
    # onset / wall_modes (#571)
    "active_window",
    "allocate_per_segment",
    "build_wall_mode_basis",
    "principal_pulse_onset",
    "robust_peak",
    "segment_eigenmodes",
    "sustained_excess_onset",
    "zero_crossing_after_excursion",
    "repair_clipped_interval",
    "vest_coil_current_noise_reduction",
    "anti_alias_filter",
    "resample_to_time",
    "process_signal",
    "subtract_baseline",
    "signal_on_offset",
    "time_derivative",
    "filter_dataframe",
    "generate_core_profiles_history_dataframe",
    "perform_ols_regression",
    "compute_metrics",
    # profile / atomic (#420)
    "fit_ti_te_ratio",
    "equilibrium_mapping_thomson_scattering",
    "equilibrium_mapping_charge_exchange",
    "profile_fitting_thomson_scattering",
    "profile_fitting_charge_exchange",
    "core_profiles",
    "core_profiles_from_eq",
    "core_profiles_from_eq_ratio",
    "integrate_emissivity_profile",
    "compute_line_radiation_power_series",
})

#: Routines whose output sits at a different place in the processing chain
#: from their input, and must say so in Input/Output semantics.  Sub-issues
#: C and D add the equilibrium mappers, the profile fitters and the
#: reconstructions.
STATEFUL = frozenset({
    # wall_modes (#571): element space <-> mode space
    "combined_operators",
    "project",
    "reconstruct",
    "reduce_response",
    "reduced_operators",
    "solve_reduced_eddy",
    "repair_clipped_interval",
    # profile / atomic (#420): measured -> mapped -> fitted -> stored; reconstructed -> synthetic
    "equilibrium_mapping_thomson_scattering",
    "equilibrium_mapping_charge_exchange",
    "profile_fitting_thomson_scattering",
    "profile_fitting_charge_exchange",
    "core_profiles",
    "core_profiles_from_eq",
    "core_profiles_from_eq_ratio",
    "compute_line_radiation_power_series",
})

#: Sign, phase, coordinate or normalisation choices change the number.
CONVENTION_SENSITIVE = frozenset({
    # magnetics / electromagnetics / fluctuation (#418): integration sign,
    # shot-era baselines, per-unit-current responses, and the two mode-number
    # entry points that disagree on the sign of n (#638)
    "analyze_fluctuation_spectrum",
    "b_field_pol_probe_field",
    "calc_grid",
    "compute_band_power",
    "compute_br_bz_phi",
    "compute_impedance_matrices",
    "compute_mutual_passive_active",
    "compute_point_response_matrices",
    "compute_psd",
    "compute_response_matrix",
    "compute_response_vector",
    "compute_spectrogram",
    "compute_vacuum_fields_1d",
    "find_spectral_break",
    "fit_power_law_spectrum",
    "flux_loop_flux",
    "magnetics_sensor_centre",
    "magnetics_sensor_poloidal_angle",
    "mirnov_preprocess_signal",
    "mirnov_spectrogram",
    "rogowski_coil_ip",
    "solve_eddy_currents",
    "toroidal_mode_analysis",
    "toroidal_phase_fit_at_time",
    "vest_b_field_pol_probe_legacy",
    "vest_equilibrium_magnetics_detailed",
    "vest_equilibrium_magnetics_signals",
    "vest_flux_loop_flux_from_voltage",
    "vest_flux_loop_legacy",
    "vest_flux_loop_voltage",
    "vest_magnetics_time_window",
    "wall_propagator",
    # impa (#421): calibration polarity, the geometry/coupling degeneracy,
    # filter phase, and the volts-not-tesla crosstalk slope
    "find_tf_calibration_window",
    "fit_impa_crosstalk",
    "fit_impa_geometry",
    "fit_impa_tf_coupling",
    "grade_impa_quality",
    "impa_calibrate_signals",
    "impa_lowpass",
    "legacy_impa_compensation",
    "legacy_impa_position",
    "process_impa",
    "remove_bz_crosstalk",
    "remove_tf_pickup",
    "toroidal_field",
    # soft_x_rays / langmuir / camera_geometry (#421): filter phase, pixel and
    # world-frame order, probe geometry, and the mode-number degeneracy
    "electron_density",
    "hilbert_instantaneous_phase",
    "load_te_ratio_calibration",
    "median_filter_signal",
    "probe_surface_area",
    "process_triple_probe",
    "project_points",
    "rank_toroidal_mode_numbers",
    "remove_offset",
    "solve_electron_temperature",
    "sweep_toroidal",
    "sxr_band_signals",
    "sxr_baseline_correction",
    "sxr_cwt_spectrogram",
    "sxr_electron_temperature",
    "sxr_subtract_vacuum_reference",
    "sxr_te_pairs_from_ods",
    "toroidal_ring",
    "trajectory_world_points",
    # equilibrium (#419): every function states a flux unit, a COCOS, a radial
    # coordinate or a contour orientation; the module is convention work
    "as_equilibrium",
    "calculate_average_boundary_poloidal_field",
    "calculate_diamagnetism",
    "calculate_reconstructed_diamagnetic_flux",
    "check_equilibrium_requirements",
    "computed_diamagnetism_from_phi",
    "contour_shape_parameters",
    "convert_cocos",
    "derive_boundary_representation",
    "derive_global_descriptors",
    "derive_radial_coordinates",
    "efit_virial_volume_integrals",
    "evaluate_miller",
    "evaluate_solovev",
    "extract_flux_surface_contours",
    "fit_miller_sequence",
    "fit_miller_surface",
    "flux_surface_quantities",
    "fractional_cell_weights_from_boundary",
    "grad_shafranov_operator",
    "grad_shafranov_residual",
    "equilibrium_field_on_grid",
    "make_equilibrium_field_interpolator",
    "parallel_current_from_toroidal",
    "poloidal_field_at_boundary",
    "prepare_boundary_for_shafranov",
    "psi_to_radial",
    "psi_to_rho",
    "psi_to_rz",
    "r_at_z_extremum",
    "radial_to_psi",
    "rho_to_psi",
    "scale_boundary_conformal",
    "shafranov_integrals",
    "virial_alpha_conformal_annulus",
    "virial_alpha_thin_annulus",
    "solovev_to_equilibrium",
    "solve_solovev_constraints",
    "trace_field_line",
    "volume_average",
    # cocos (#419): the module exists to reason about conventions
    "cocos_consistency_signs",
    "validate_cocos",
    "identify_flux_exponent",
    "identify_convention",
    # onset / wall_modes (#571)
    "active_window",
    "allocate_per_segment",
    "build_wall_mode_basis",
    "canonical_sign",
    "combined_operators",
    "global_time_constants",
    "median_smooth",
    "moment_patterns",
    "orthonormalize_r",
    "pickup_scale",
    "principal_pulse_onset",
    "project",
    "reconstruction_error",
    "reduced_operators",
    "robust_peak",
    "run_features",
    "segment_eigenmodes",
    "select_by_score",
    "subspace_angles_r",
    "sustained_excess_onset",
    "zero_crossing_after_excursion",
    "zero_phase_lowpass",
    "line_average_density",
    "smooth",
    "butterworth_lowpass",
    "butterworth_bandpass",
    "fir_filter",
    "fir_filter_coefficients",
    "detrend_moving_average",
    "vest_coil_current_noise_reduction",
    "describe_time_grid",
    "anti_alias_filter",
    "resample_to_time",
    "process_signal",
    "signal_on_offset",
    "infer_signal_orientation",
    "time_derivative",
    "filter_dataframe",
    "log_transform",
    "perform_ols_regression",
    "compute_metrics",
    # profile / atomic (#420): three radial coordinates, none interchangeable
    "equilibrium_mapping_thomson_scattering",
    "equilibrium_mapping_charge_exchange",
    "profile_fitting_thomson_scattering",
    "profile_fitting_charge_exchange",
    "core_profiles",
    "core_profiles_from_eq",
    "core_profiles_from_eq_ratio",
    "integrate_emissivity_profile",
    "compute_line_radiation_power_series",
    "export_electron_profile_txt",
    "toroidal_mode_decomposition",
    "biot_savart_filaments",
})

SPECS = [spec for spec in catalog.list_processes() if spec.category not in PENDING]
IDS = [spec.qualname for spec in SPECS]


# --- the split ------------------------------------------------------------------


def test_pending_names_real_categories():
    assert PENDING <= set(catalog.CATEGORIES), sorted(PENDING - set(catalog.CATEGORIES))


def test_pending_is_exactly_the_set_of_non_conforming_categories():
    """A module may not leave PENDING early, nor linger once it conforms."""
    actual = {doc.name for doc in catalog.categories() if not doc.conforming}
    assert actual == PENDING, {
        "should be pending": sorted(actual - PENDING),
        "should be removed from PENDING": sorted(PENDING - actual),
    }


def test_something_is_under_the_contract():
    assert SPECS, "every category is pending; the contract enforces nothing"


# --- structural, per function ------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_has_a_summary_line(spec):
    assert spec.summary, "missing docstring or summary"
    assert spec.summary.endswith("."), spec.summary


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_conforms_structurally(spec):
    assert spec.conforming, "\n".join(spec.errors)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_every_section_is_in_the_vocabulary(spec):
    for title, _ in spec.sections:
        assert title in SECTION_VOCABULARY, title


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_applicability_declares_exactly_one_scope(spec):
    if spec.deprecated:
        pytest.skip("deprecated shim: summary only")
    text = spec.section("Applicability") or ""
    assert text.startswith((MACHINE_INDEPENDENT, VEST_SPECIFIC)), text[:60]
    assert not (text.startswith(MACHINE_INDEPENDENT) and VEST_SPECIFIC in text[:40])
    assert spec.machine_scope in ("independent", "vest")


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_a_vest_specific_routine_says_which_data(spec):
    """`VEST-specific.` alone is not applicability; it must say for what."""
    if spec.machine_scope != "vest":
        pytest.skip("not VEST-specific")
    text = spec.section("Applicability") or ""
    assert len(text) > len(VEST_SPECIFIC) + 20, text


# --- policy, per function ------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_ported_and_empirical_routines_record_their_provenance(spec):
    if spec.deprecated or spec.name in DEFINITIONAL:
        pytest.skip("definitional or deprecated")
    assert spec.references, "no Provenance section; a ported or empirical routine must name its source"
    for ref in spec.references:
        assert ref.text, f"empty provenance entry [{ref.label}]"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_multi_stage_routines_list_their_processing_steps(spec):
    if spec.name not in PIPELINE:
        pytest.skip("single-step")
    steps = spec.section("Processing steps") or ""
    assert steps, "missing Processing steps"
    assert "1." in steps and "2." in steps, "Processing steps must be an ordered list"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_stateful_routines_describe_input_and_output_semantics(spec):
    if spec.name not in STATEFUL:
        pytest.skip("state unchanged")
    assert spec.section("Input semantics"), "missing Input semantics"
    assert spec.section("Output semantics"), "missing Output semantics"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_convention_sensitive_routines_state_their_convention(spec):
    if spec.name not in CONVENTION_SENSITIVE:
        assert not spec.convention_sensitive, (
            "carries a Convention section but is not in the CONVENTION_SENSITIVE policy list"
        )
        pytest.skip("not convention-sensitive")
    assert spec.convention_sensitive, "missing Convention section"
    assert len(spec.section("Convention") or "") > 20


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_defaults_that_matter_are_classified(spec):
    """A Defaults section must say what kind of value each default is."""
    text = spec.section("Defaults")
    if text is None:
        pytest.skip("no Defaults section")
    kinds = (
        "physical constant", "literature value", "diagnostic calibration",
        "empirical", "validated-workflow default", "validated workflow default",
        "machine-specific", "acquisition-era", "legacy compatibility",
        "numerical convenience", "assumed value", "hard-coded", "conventional",
    )
    assert any(kind in text for kind in kinds), text[:120]


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_documented_defects_are_tracked_by_issue(spec):
    text = spec.section("Limitations") or ""
    if "tracked in" in text.lower():
        assert "#" in text, "a tracked limitation must name its GitHub issue number"


def test_policy_lists_name_real_functions():
    known = {spec.name for spec in catalog.list_processes()}
    known |= {alias for spec in catalog.list_processes() for alias in spec.aliases}
    for name, policy in (
        ("DEFINITIONAL", DEFINITIONAL),
        ("PIPELINE", PIPELINE),
        ("STATEFUL", STATEFUL),
        ("CONVENTION_SENSITIVE", CONVENTION_SENSITIVE),
    ):
        assert policy <= known, (name, sorted(policy - known))


def test_policy_lists_only_govern_functions_under_the_contract():
    """A name on a policy list for a pending module is a promise nobody checks."""
    governed = {spec.name for spec in SPECS}
    for name, policy in (
        ("DEFINITIONAL", DEFINITIONAL),
        ("PIPELINE", PIPELINE),
        ("STATEFUL", STATEFUL),
        ("CONVENTION_SENSITIVE", CONVENTION_SENSITIVE),
    ):
        assert policy <= governed, (name, sorted(policy - governed))


# --- module docstrings ----------------------------------------------------------


@pytest.mark.parametrize(
    "doc", [d for d in catalog.categories() if d.name not in PENDING], ids=lambda d: d.name
)
def test_module_docstrings_carry_a_title_and_overview(doc):
    assert doc.title.endswith("."), doc.title
    assert doc.overview or doc.notation or doc.conventions


def test_vocabulary_lists_every_custom_section():
    assert set(CUSTOM_SECTIONS) <= set(SECTION_VOCABULARY)
