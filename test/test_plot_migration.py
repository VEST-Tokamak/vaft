"""The pre-redesign ``vaft.plot`` surface must stay accounted for and warn."""

import warnings

import matplotlib

matplotlib.use("Agg")

import pytest

import vaft.plot
from vaft.plot import registry
from vaft.plot._migration import (
    DEPRECATED,
    PRESERVED,
    LEGACY,
    RELOCATED,
    REMOVAL_RELEASE,
    REMOVED,
    RENAMED,
    RENAMED_REMOVAL_RELEASE,
    legacy_surface,
)

#: The exact public callables ``vaft.plot`` exported before the redesign, as
#: measured on the wildcard-exporting package.  Every one must resolve to a
#: canonical name, a relocation, a legacy renderer, or a documented removal.
HISTORICAL_SURFACE = (
    "analysis_diagnostics",
    "barometry_time_pressure",
    "charge_exchange_radial",
    "charge_exchange_rho_profile",
    "charge_exchange_rho_profiles",
    "charge_exchange_time",
    "check_and_update_equilibrium_data",
    "compute_confinement_scaling_metrics",
    "compute_mirnov_spectrogram",
    "compute_point_vacuum_fields_ods",
    "confinement_time_exp_vs_scaling",
    "ec_launchers_CX_topview",
    "electromagnetics_time_current",
    "equilibrium_1d_radial",
    "equilibrium_2d_profiles",
    "equilibrium_CX_topview",
    "equilibrium_psi_norm_f",
    "equilibrium_psi_norm_ffprime",
    "equilibrium_psi_norm_j_tor",
    "equilibrium_psi_norm_pprime",
    "equilibrium_psi_norm_pressure",
    "equilibrium_psi_norm_q",
    "equilibrium_r_major_f",
    "equilibrium_r_major_ffprime",
    "equilibrium_r_major_j_tor",
    "equilibrium_r_major_pprime",
    "equilibrium_r_major_pressure",
    "equilibrium_r_major_q",
    "equilibrium_r_minor_f",
    "equilibrium_r_minor_ffprime",
    "equilibrium_r_minor_j_tor",
    "equilibrium_r_minor_pprime",
    "equilibrium_r_minor_pressure",
    "equilibrium_r_minor_q",
    "equilibrium_rho_tor_norm_f",
    "equilibrium_rho_tor_norm_ffprime",
    "equilibrium_rho_tor_norm_j_tor",
    "equilibrium_rho_tor_norm_pprime",
    "equilibrium_rho_tor_norm_pressure",
    "equilibrium_rho_tor_norm_q",
    "equilibrium_time_beta_n",
    "equilibrium_time_beta_pol",
    "equilibrium_time_beta_tor",
    "equilibrium_time_li",
    "equilibrium_time_major_radius",
    "equilibrium_time_plasma_current",
    "equilibrium_time_q0",
    "equilibrium_time_q95",
    "equilibrium_time_qa",
    "equilibrium_time_w_mag",
    "equilibrium_time_w_mhd",
    "equilibrium_time_w_tot",
    "extract_labels_from_odc",
    "find_time_slice_for_max_ip",
    "format_equilibrium_title",
    "format_legend_label",
    "get_1d_profile_data",
    "get_equilibrium_parameters",
    "get_from_path",
    "get_path",
    "handle_labels",
    "handle_xlim",
    "is_signal_active",
    "lh_antennas_CX_topview",
    "magnetics_time_b_field_pol_probe_field",
    "magnetics_time_diamagnetic_flux",
    "magnetics_time_flux_loop_flux",
    "magnetics_time_flux_loop_voltage",
    "magnetics_time_ip",
    "make_plot_func",
    "mirnov_preprocess_signal",
    "mirnov_signal",
    "mirnov_spectrogram",
    "odc_or_ods_check",
    "overlay_all",
    "overlay_all_with_vacuum_psi_contour",
    "pellets_trajectory_CX_topview",
    "pf_active_time_current",
    "pf_active_time_current_turns",
    "pf_passive_overlay",
    "plot_H_factor_distribution",
    "plot_H_factor_vs_greenwald_fraction",
    "plot_H_factor_vs_parameters",
    "plot_TeNe_from_eq",
    "plot_bremsstrahlung_power_scaling_vs_fundamental_method",
    "plot_ces_profile",
    "plot_confinement_time_exp_vs_scaling",
    "plot_core_profiles_ne",
    "plot_core_profiles_te",
    "plot_core_profiles_time_volume_averaged",
    "plot_correlation_heatmap",
    "plot_ec_launchers_CX_topview",
    "plot_electron_2d_profile",
    "plot_electron_profile_with_thomson",
    "plot_electron_psi_profile",
    "plot_electron_time_volume_averaged",
    "plot_equilibrium_CX_topview",
    "plot_equilibrium_and_core_profiles_pressure",
    "plot_equilibrium_pressure",
    "plot_equilibrium_q",
    "plot_individual_parameter_effects",
    "plot_lh_antennas_CX_topview",
    "plot_ohmic_power_flux_vs_dissipation_method",
    "plot_onedim_profile",
    "plot_onedim_profile_interactive",
    "plot_pellets_trajectory_CX_topview",
    "plot_pressure_profile_with_geqdsk",
    "plot_regression_summary",
    "plot_scaling_fit",
    "plot_scaling_metrics_bars",
    "plot_soft_x_ray_los",
    "plot_soft_x_ray_overview",
    "plot_soft_x_ray_pattern",
    "plot_soft_x_ray_signal",
    "plot_soft_x_ray_spectrogram",
    "plot_tauE_exp_vs_scaling_loglog",
    "plot_thomson_profiles",
    "plot_thomson_radial_position",
    "plot_thomson_time_series",
    "plot_topview",
    "set_xlim_time",
    "signal_on_offset",
    "soft_x_rays_los",
    "soft_x_rays_overview",
    "soft_x_rays_pattern",
    "soft_x_rays_signal",
    "soft_x_rays_spectrogram",
    "soft_x_rays_time",
    "spectrometer_uv_time_intensity",
    "tf_time_b_field_tor",
    "tf_time_b_field_tor_vacuum_r",
    "tf_time_coil_current",
    "thomson_scattering_radial",
    "thomson_scattering_radial_profiles",
    "thomson_scattering_time",
    "time_barometry_pressure",
    "time_beta",
    "time_diamagnetic_flux",
    "time_electromagnetics_current",
    "time_energy",
    "time_equilibrium_analysis",
    "time_equilibrium_beta_n",
    "time_equilibrium_beta_pol",
    "time_equilibrium_beta_tor",
    "time_equilibrium_li",
    "time_equilibrium_major_radius",
    "time_equilibrium_plasma_current",
    "time_equilibrium_q0",
    "time_equilibrium_q95",
    "time_equilibrium_qa",
    "time_equilibrium_w_mag",
    "time_equilibrium_w_mhd",
    "time_equilibrium_w_tot",
    "time_impurity_effect",
    "time_magnetics_b_field_pol_probe_field",
    "time_magnetics_diamagnetic_flux",
    "time_magnetics_flux_loop_flux",
    "time_magnetics_flux_loop_voltage",
    "time_magnetics_ip",
    "time_pf_active_current",
    "time_pf_active_current_turns",
    "time_power_balance",
    "time_spectrometer_uv_intensity",
    "time_tf_b_field_tor",
    "time_tf_b_field_tor_vacuum_r",
    "time_tf_coil_current",
    "time_virial_equilibrium_quantities",
    "time_voltage_consumption",
    "toroidal_mode_analysis",
    "toroidal_mode_spectrum",
    "toroidal_phase_fit_at_time",
    "toroidal_phase_mode_fit",
    "twodim_geometry_all",
    "vacuum_psi_contour",
)


def test_every_historical_name_is_accounted_for():
    covered = legacy_surface() | set(registry.canonical_names()) | set(RENAMED)
    missing = sorted(set(HISTORICAL_SURFACE) - covered)
    assert not missing, missing


def test_classifications_do_not_overlap():
    groups = (set(DEPRECATED), set(RELOCATED), set(LEGACY), set(REMOVED),
              set(RENAMED))
    for index, first in enumerate(groups):
        for second in groups[index + 1:]:
            assert not first & second, sorted(first & second)


def test_the_table_does_not_claim_names_that_never_existed():
    # ``confinement_time_histogram`` moved in from vaft.process during this
    # redesign, so it is the one legacy entry outside the historical surface.
    invented = sorted(legacy_surface() - set(HISTORICAL_SURFACE))
    assert invented == ["confinement_time_histogram"]


@pytest.mark.parametrize("name", sorted(DEPRECATED))
def test_deprecated_names_resolve_and_name_their_replacement(name):
    vaft.plot.__dict__.pop(name, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = getattr(vaft.plot, name)

    assert callable(value)
    assert any(item.category is DeprecationWarning for item in caught)
    message = str(caught[0].message)
    assert DEPRECATED[name] in message
    assert REMOVAL_RELEASE in message


@pytest.mark.parametrize("name", sorted(RELOCATED))
def test_relocated_names_resolve_and_point_at_their_home(name):
    vaft.plot.__dict__.pop(name, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = getattr(vaft.plot, name)

    assert callable(value)
    assert any(item.category is DeprecationWarning for item in caught)
    assert RELOCATED[name] in str(caught[0].message)


@pytest.mark.parametrize("name", sorted(LEGACY))
def test_legacy_statistics_still_resolve(name):
    assert callable(getattr(vaft.plot, name))


@pytest.mark.parametrize("name", sorted(REMOVED))
def test_removed_helpers_raise_with_guidance(name):
    with pytest.raises(AttributeError, match="migration_table"):
        getattr(vaft.plot, name)


def test_every_deprecated_name_points_at_a_registered_plot():
    canonical = set(registry.canonical_names())
    for name, replacement in DEPRECATED.items():
        assert replacement in canonical, (name, replacement)


def test_unknown_names_still_raise_attribute_error():
    with pytest.raises(AttributeError):
        vaft.plot.definitely_not_a_plot


def test_rendered_table_covers_the_whole_surface():
    text = vaft.plot.migration_table()
    for name in sorted(legacy_surface() | set(PRESERVED) | set(RENAMED)):
        assert f"`{name}`" in text, name
    assert REMOVAL_RELEASE in text
    assert RENAMED_REMOVAL_RELEASE in text
    for heading in ("Deprecated renderers", "Renamed canonical stems", "Relocated",
                    "Removed internal helpers", "Already canonical"):
        assert heading in text, heading


def test_preserved_names_are_canonical_and_carry_no_deprecation():
    canonical = set(registry.canonical_names())
    for name in PRESERVED:
        assert name in canonical, name
        assert name not in DEPRECATED, name
        assert name not in RENAMED, name


@pytest.mark.parametrize("name", sorted(RENAMED))
def test_renamed_stems_warn_and_delegate_from_vaft_plot(name):
    vaft.plot.__dict__.pop(name, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = getattr(vaft.plot, name)

    assert value is getattr(vaft.plot, RENAMED[name]), name
    assert any(item.category is DeprecationWarning for item in caught)
    message = str(caught[0].message)
    assert RENAMED[name] in message
    assert RENAMED_REMOVAL_RELEASE in message


def test_every_renamed_stem_points_at_a_registered_plot():
    canonical = set(registry.canonical_names())
    for old, new in RENAMED.items():
        assert new in canonical, (old, new)
        assert old not in canonical, old


@pytest.mark.parametrize("name", sorted(RENAMED))
def test_renamed_omas_adapters_warn_and_delegate(name):
    import vaft.omas.plotting as omas_plotting

    adapter = getattr(omas_plotting, f"plot_{name}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError):
            adapter(object())
    assert any(item.category is DeprecationWarning for item in caught)
    assert f"plot_{RENAMED[name]}" in str(caught[0].message)
