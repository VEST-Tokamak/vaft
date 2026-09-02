"""Legacy ``vaft.plot`` names and where they went.

Issue #62 replaced the wildcard-exported ``vaft.plot`` surface with an explicit,
registry-driven one.  This module holds the machine-readable migration table that
drives three things:

* the deprecation ``__getattr__`` in :mod:`vaft.plot`;
* the Markdown rendering from :func:`render_markdown_table`, exported as
  ``vaft.plot.migration_table``;
* the regression tests that keep both in sync.

Every name that ``vaft.plot`` exported before the redesign appears in exactly one
of the mappings below.

Classifications
---------------

``DEPRECATED``
    A real renderer whose canonical replacement exists.  The old name keeps
    working and warns until the removal release.  Renderers now take view models,
    so a legacy call routed through ``vaft.omas.plot_<stem>`` is the direct
    equivalent; the mapping records the canonical stem.

``RENAMED``
    A canonical renderer whose stem changed in the issue #251 subject-taxonomy
    redesign (``magnetics_time_ip`` -> ``plasma_current_time``).  The old stem
    keeps working from ``vaft.plot`` and ``vaft.omas.plot_<stem>`` and warns
    until ``RENAMED_REMOVAL_RELEASE``.

``RELOCATED``
    Still supported, but it belongs to another namespace.  It keeps working from
    ``vaft.plot`` and warns, pointing at its real home.

``LEGACY``
    Cross-shot statistics that have no canonical destination yet.  Still
    exported, registered with ``status="legacy"``, and excluded from the
    canonical ``available_plots()`` listing.

``REMOVED``
    Internal helpers and wildcard leakage that were never intended as API.  These
    raise ``AttributeError`` immediately; the table records where the behavior
    lives now so callers can adapt.
"""

from __future__ import annotations

__all__ = [
    "COMPATIBILITY_NOTES",
    "DEPRECATED",
    "GENERATED_EQUILIBRIUM_PROFILES",
    "LEGACY_MODULES",
    "PRESERVED",
    "render_markdown_table",
    "INTRODUCED_IN",
    "LEGACY",
    "RENAMED",
    "RENAMED_IN",
    "RENAMED_REMOVAL_RELEASE",
    "RELOCATED",
    "REMOVAL_RELEASE",
    "REMOVED",
    "legacy_surface",
]

#: Release that introduced the deprecations, and the release that removes them.
INTRODUCED_IN = "0.5.0"
REMOVAL_RELEASE = "0.7.0"

#: Legacy renderer name -> canonical ``vaft.plot`` stem.
DEPRECATED: dict[str, str] = {
    # --- vaft.plot.time: view-first names replaced by domain-first ones -------
    "time_magnetics_ip": "plasma_current_time",
    "time_magnetics_diamagnetic_flux": "diamagnetic_flux_time",
    "time_magnetics_flux_loop_flux": "flux_loop_time_flux",
    "time_magnetics_flux_loop_voltage": "flux_loop_time_voltage",
    "time_magnetics_b_field_pol_probe_field": "b_field_probe_time_field",
    "time_pf_active_current": "pf_coil_time_current",
    "time_pf_active_current_turns": "pf_coil_time_current_turns",
    "time_equilibrium_plasma_current": "equilibrium_time_plasma_current",
    "time_equilibrium_li": "equilibrium_time_li",
    "time_equilibrium_beta_pol": "equilibrium_time_beta_p",
    "time_equilibrium_beta_tor": "equilibrium_time_beta_t",
    "time_equilibrium_beta_n": "equilibrium_time_beta_n",
    "time_equilibrium_w_mhd": "equilibrium_time_w_mhd",
    "time_equilibrium_w_mag": "equilibrium_time_w_mag",
    "time_equilibrium_w_tot": "equilibrium_time_w_tot",
    "time_equilibrium_q0": "equilibrium_time_q0",
    "time_equilibrium_q95": "equilibrium_time_q95",
    "time_equilibrium_qa": "equilibrium_time_qa",
    "time_equilibrium_major_radius": "equilibrium_time_major_radius",
    "time_tf_b_field_tor": "tf_coil_time_b_t",
    "time_tf_b_field_tor_vacuum_r": "tf_coil_time_b_t_vacuum_r",
    "time_tf_coil_current": "tf_coil_time_current",
    "time_spectrometer_uv_intensity": "spectrometer_uv_time_intensity",
    "time_barometry_pressure": "barometry_time_pressure",
    "time_diamagnetic_flux": "equilibrium_time_diamagnetic_flux",
    # --- vaft.plot.time: composites ------------------------------------------
    "time_energy": "summary_time_energy",
    "time_beta": "equilibrium_time_beta",
    "time_power_balance": "summary_time_power_balance",
    "time_voltage_consumption": "summary_time_voltage_consumption",
    "time_virial_equilibrium_quantities": "equilibrium_time_virial",
    "time_electromagnetics_current": "current_overview",
    "time_impurity_effect": "spectrometer_uv_time_impurity",
    "plot_core_profiles_time_volume_averaged": "core_profiles_time_volume_averaged",
    # --- vaft.plot.analysis ---------------------------------------------------
    "analysis_diagnostics": "magnetics_overview",
    "time_equilibrium_analysis": "equilibrium_overview",
    # --- vaft.plot.onedim: generic and coordinate-specific profiles -----------
    # The radial coordinate is now a model/adapter option rather than part of the
    # name, so the 24 generated ``equilibrium_<coord>_<quantity>`` globals map
    # onto six canonical renderers.
    "plot_onedim_profile": "equilibrium_profile_pressure",
    "plot_onedim_profile_interactive": "equilibrium_profile_pressure",
    "equilibrium_1d_radial": "equilibrium_profile_pressure",
    "plot_equilibrium_pressure": "equilibrium_profile_pressure",
    "plot_equilibrium_q": "equilibrium_profile_q",
    "plot_core_profiles_ne": "electron_density_profile",
    "plot_core_profiles_te": "electron_temperature_profile",
    # --- vaft.plot.profile ----------------------------------------------------
    "plot_thomson_radial_position": "thomson_scattering_geometry_poloidal",
    "thomson_scattering_radial": "thomson_scattering_profile_electron_temperature",
    "thomson_scattering_radial_profiles": "thomson_scattering_profile_electron_temperature",
    "plot_thomson_profiles": "thomson_scattering_profile_electron_temperature",
    "plot_electron_profile_with_thomson": "electron_temperature_profile",
    "plot_thomson_time_series": "thomson_scattering_time_electron_temperature",
    "thomson_scattering_time": "thomson_scattering_time_electron_temperature",
    "charge_exchange_radial": "charge_exchange_profile_ion_temperature",
    "charge_exchange_rho_profile": "charge_exchange_profile_ion_temperature",
    "charge_exchange_rho_profiles": "charge_exchange_profile_ion_temperature",
    "plot_ces_profile": "charge_exchange_profile_ion_temperature",
    "charge_exchange_time": "charge_exchange_time_ion_temperature",
    "plot_electron_psi_profile": "electron_temperature_profile",
    "plot_electron_2d_profile": "electron_temperature_field",
    "plot_TeNe_from_eq": "electron_temperature_profile",
    "plot_electron_time_volume_averaged": "core_profiles_time_volume_averaged",
    "plot_equilibrium_and_core_profiles_pressure": "equilibrium_profile_pressure",
    "plot_pressure_profile_with_geqdsk": "equilibrium_profile_pressure",
    # --- vaft.plot.twodim -----------------------------------------------------
    "equilibrium_2d_profiles": "equilibrium_field_psi",
    "vacuum_psi_contour": "equilibrium_field_psi_vacuum",
    "overlay_all_with_vacuum_psi_contour": "equilibrium_field_psi_vacuum",
    "pf_passive_overlay": "passive_structure_geometry_poloidal",
    "overlay_all": "machine_geometry_poloidal",
    "twodim_geometry_all": "machine_geometry_poloidal",
    # --- vaft.plot.topview ----------------------------------------------------
    "equilibrium_CX_topview": "equilibrium_geometry_topview",
    "plot_equilibrium_CX_topview": "equilibrium_geometry_topview",
    "lh_antennas_CX_topview": "machine_geometry_topview",
    "plot_lh_antennas_CX_topview": "machine_geometry_topview",
    "ec_launchers_CX_topview": "machine_geometry_topview",
    "plot_ec_launchers_CX_topview": "machine_geometry_topview",
    "pellets_trajectory_CX_topview": "machine_geometry_topview",
    "plot_pellets_trajectory_CX_topview": "machine_geometry_topview",
    "plot_topview": "machine_geometry_topview",
    # --- vaft.plot.soft_x_rays ------------------------------------------------
    "plot_soft_x_ray_los": "soft_x_rays_geometry_lines_of_sight",
    "soft_x_rays_los": "soft_x_rays_geometry_lines_of_sight",
    "plot_soft_x_ray_signal": "soft_x_rays_time_power",
    "soft_x_rays_signal": "soft_x_rays_time_power",
    "soft_x_rays_time": "soft_x_rays_time_power",
    "plot_soft_x_ray_spectrogram": "soft_x_rays_spectrogram",
    "plot_soft_x_ray_pattern": "soft_x_rays_overview",
    "soft_x_rays_pattern": "soft_x_rays_overview",
    "plot_soft_x_ray_overview": "soft_x_rays_overview",
    # --- vaft.plot.mirnov -----------------------------------------------------
    "mirnov_signal": "mirnov_time_voltage",
    # NOTE: the legacy ``mirnov_spectrogram`` entry was retired in 0.6.0 when
    # the canonical spectrogram renderer took that exact name (issue #251);
    # the canonical attribute now shadows the legacy one.
    "toroidal_mode_spectrum": "mirnov_spectrogram",
    "toroidal_phase_mode_fit": "mirnov_time_voltage",
}

# The 24 coordinate-specific equilibrium profile globals that ``onedim`` used to
# generate. Each maps to the canonical renderer for its quantity; the coordinate
# becomes the adapter's ``coordinate=`` argument.
#: The coordinate-specific equilibrium profile names, in generation order.
GENERATED_EQUILIBRIUM_PROFILES: tuple[tuple[str, str, str], ...] = tuple(
    (f"equilibrium_{coordinate}_{quantity}", coordinate, canonical)
    for coordinate in ("psi_norm", "rho_tor_norm", "r_major", "r_minor")
    for quantity, canonical in (
        ("pressure", "equilibrium_profile_pressure"),
        ("q", "equilibrium_profile_q"),
        ("j_tor", "equilibrium_profile_j_tor"),
        ("pprime", "equilibrium_profile_pprime"),
        ("f", "equilibrium_profile_f"),
        ("ffprime", "equilibrium_profile_ffprime"),
    )
)

for _name, _coordinate, _canonical in GENERATED_EQUILIBRIUM_PROFILES:
    DEPRECATED[_name] = _canonical
del _name, _coordinate, _canonical

#: Old canonical stem -> new subject-centered canonical stem (issue #251).
#: Unlike ``DEPRECATED`` (pre-redesign legacy names), these were canonical
#: names between the issue #62 redesign and the issue #251 subject taxonomy.
#: ``vaft.plot.<old>`` and ``vaft.omas.plot_<old>`` keep working with a
#: ``DeprecationWarning`` until ``RENAMED_REMOVAL_RELEASE``.
RENAMED_IN = "0.6.0"
RENAMED_REMOVAL_RELEASE = "0.8.0"

RENAMED: dict[str, str] = {
    "magnetics_time_ip": "plasma_current_time",
    "magnetics_time_diamagnetic_flux": "diamagnetic_flux_time",
    "magnetics_time_flux_loop_flux": "flux_loop_time_flux",
    "magnetics_time_flux_loop_voltage": "flux_loop_time_voltage",
    "magnetics_time_b_field_pol_probe_field": "b_field_probe_time_field",
    "magnetics_time_mirnov_voltage": "mirnov_time_voltage",
    "magnetics_spectrum_mirnov": "mirnov_spectrum",
    "magnetics_spectrogram_mirnov": "mirnov_spectrogram",
    "magnetics_time_limiter_current": "limiter_current_time",
    "magnetics_time_impa_field": "impa_time_field",
    "magnetics_time_impa_voltage": "impa_time_voltage",
    "magnetics_overview_impa": "impa_overview",
    "magnetics_profile_impa_tf": "impa_profile_field",
    "pf_active_time_current": "pf_coil_time_current",
    "pf_active_time_current_turns": "pf_coil_time_current_turns",
    "pf_active_geometry_poloidal": "pf_coil_geometry_poloidal",
    "pf_passive_geometry_poloidal": "passive_structure_geometry_poloidal",
    "tf_time_coil_current": "tf_coil_time_current",
    "tf_time_b_field_tor": "tf_coil_time_b_t",
    "tf_time_b_field_tor_vacuum_r": "tf_coil_time_b_t_vacuum_r",
    "electromagnetics_time_current": "current_overview",
    "equilibrium_time_beta_pol": "equilibrium_time_beta_p",
    "equilibrium_time_beta_tor": "equilibrium_time_beta_t",
    "summary_time_beta": "equilibrium_time_beta",
    "core_profiles_time_electron_density": "electron_density_time",
    "core_profiles_time_electron_temperature": "electron_temperature_time",
    "core_profiles_profile_electron_density": "electron_density_profile",
    "core_profiles_profile_electron_temperature": "electron_temperature_profile",
    "core_profiles_profile_ion_temperature": "ion_temperature_profile",
    "core_profiles_profile_pressure": "thermal_pressure_profile",
    "core_profiles_field_electron_density": "electron_density_field",
    "core_profiles_field_electron_temperature": "electron_temperature_field",
    "coils_non_axisymmetric_geometry3d": "coil_3d_geometry3d",
    "coils_non_axisymmetric_geometry_topview": "coil_3d_geometry_topview",
}


#: Legacy name -> the namespace that actually owns it.
RELOCATED: dict[str, str] = {
    "odc_or_ods_check": "vaft.omas.odc_or_ods_check",
    "extract_labels_from_odc": "vaft.omas.extract_labels_from_odc",
    "get_path": "vaft.machine_mapping.utils.get_path",
    "compute_point_vacuum_fields_ods": "vaft.omas.compute_point_vacuum_fields_ods",
    "compute_mirnov_spectrogram": "vaft.process.mirnov_spectrogram",
    "mirnov_preprocess_signal": "vaft.process.mirnov_preprocess_signal",
    "toroidal_mode_analysis": "vaft.process.toroidal_mode_analysis",
    "toroidal_phase_fit_at_time": "vaft.process.toroidal_phase_fit_at_time",
    "is_signal_active": "vaft.process.is_signal_active",
    "signal_on_offset": "vaft.process.signal_on_offset",
}

#: Cross-shot statistics kept as-is until a later pass gives them a home.
LEGACY: tuple[str, ...] = (
    "compute_confinement_scaling_metrics",
    "confinement_time_histogram",
    "confinement_time_exp_vs_scaling",
    "plot_H_factor_distribution",
    "plot_H_factor_vs_greenwald_fraction",
    "plot_H_factor_vs_parameters",
    "plot_bremsstrahlung_power_scaling_vs_fundamental_method",
    "plot_confinement_time_exp_vs_scaling",
    "plot_correlation_heatmap",
    "plot_individual_parameter_effects",
    "plot_ohmic_power_flux_vs_dissipation_method",
    "plot_regression_summary",
    "plot_scaling_fit",
    "plot_scaling_metrics_bars",
    "plot_tauE_exp_vs_scaling_loglog",
)

#: Internal helper -> where the behavior lives now.  Raises ``AttributeError``.
REMOVED: dict[str, str] = {
    "handle_xlim": "internal helper; use LineSeries.x_limits",
    "handle_labels": "internal helper; use vaft.omas label extraction",
    "set_xlim_time": "internal helper; use LineSeries.x_limits",
    "get_from_path": "internal helper; use ODS/IDS indexing directly",
    "make_plot_func": "import-time factory removed with the generated globals",
    "get_1d_profile_data": "internal helper; adapters build Profile1D instead",
    "check_and_update_equilibrium_data": "internal helper of the old onedim module",
    "get_equilibrium_parameters": "internal helper of the old onedim module",
    "format_equilibrium_title": "internal helper; set the model's title field",
    "format_legend_label": "internal helper; set Series.label",
    "find_time_slice_for_max_ip": "internal helper of the old onedim module",
}


#: Names that were already spelled ``<domain>_<view>_<quantity>`` before the
#: redesign.  They keep working without a warning, but their signature changed:
#: they now take a view model and default to ``show=False``, so callers holding
#: an ODS should move to ``vaft.omas.plot_<name>``.
PRESERVED: tuple[str, ...] = (
    "barometry_time_pressure",
    "equilibrium_time_beta_n",
    "equilibrium_time_li",
    "equilibrium_time_major_radius",
    "equilibrium_time_plasma_current",
    "equilibrium_time_q0",
    "equilibrium_time_q95",
    "equilibrium_time_qa",
    "equilibrium_time_w_mag",
    "equilibrium_time_w_mhd",
    "equilibrium_time_w_tot",
    "soft_x_rays_overview",
    "soft_x_rays_spectrogram",
    "spectrometer_uv_time_intensity",
)


def legacy_surface() -> frozenset[str]:
    """Every name the pre-redesign ``vaft.plot`` exported."""
    return frozenset(DEPRECATED) | frozenset(RELOCATED) | frozenset(LEGACY) | frozenset(REMOVED)


COMPATIBILITY_NOTES = (
    f"Deprecated vaft.plot names were introduced in {INTRODUCED_IN} and are "
    f"removed in {REMOVAL_RELEASE}."
)


#: Legacy name -> the ``vaft.plot`` submodule that still implements it during the
#: compatibility window.  ``vaft.plot.__getattr__`` uses this to resolve the old
#: names without importing every legacy module at package import.
LEGACY_MODULES: dict[str, str] = {}

for _module, _names in {
    "analysis": ("analysis_diagnostics", "time_equilibrium_analysis"),
    "history": LEGACY,
    "mirnov": (
        "mirnov_signal",
        "toroidal_mode_spectrum",
        "toroidal_phase_mode_fit",
    ),
    "onedim": (
        "check_and_update_equilibrium_data",
        "equilibrium_1d_radial",
        "find_time_slice_for_max_ip",
        "format_equilibrium_title",
        "format_legend_label",
        "get_1d_profile_data",
        "get_equilibrium_parameters",
        "make_plot_func",
        "plot_core_profiles_ne",
        "plot_core_profiles_te",
        "plot_equilibrium_pressure",
        "plot_equilibrium_q",
        "plot_onedim_profile",
        "plot_onedim_profile_interactive",
    )
    + tuple(name for name, _, _ in GENERATED_EQUILIBRIUM_PROFILES),
    "profile": (
        "charge_exchange_radial",
        "charge_exchange_rho_profile",
        "charge_exchange_rho_profiles",
        "charge_exchange_time",
        "plot_TeNe_from_eq",
        "plot_ces_profile",
        "plot_electron_2d_profile",
        "plot_electron_profile_with_thomson",
        "plot_electron_psi_profile",
        "plot_electron_time_volume_averaged",
        "plot_equilibrium_and_core_profiles_pressure",
        "plot_pressure_profile_with_geqdsk",
        "plot_thomson_profiles",
        "plot_thomson_radial_position",
        "plot_thomson_time_series",
        "thomson_scattering_radial",
        "thomson_scattering_radial_profiles",
        "thomson_scattering_time",
    ),
    "soft_x_rays": (
        "plot_soft_x_ray_los",
        "plot_soft_x_ray_overview",
        "plot_soft_x_ray_pattern",
        "plot_soft_x_ray_signal",
        "plot_soft_x_ray_spectrogram",
        "soft_x_rays_los",
        "soft_x_rays_pattern",
        "soft_x_rays_signal",
        "soft_x_rays_time",
    ),
    "topview": (
        "ec_launchers_CX_topview",
        "equilibrium_CX_topview",
        "lh_antennas_CX_topview",
        "pellets_trajectory_CX_topview",
        "plot_ec_launchers_CX_topview",
        "plot_equilibrium_CX_topview",
        "plot_lh_antennas_CX_topview",
        "plot_pellets_trajectory_CX_topview",
        "plot_topview",
    ),
    "twodim": (
        "equilibrium_2d_profiles",
        "overlay_all",
        "overlay_all_with_vacuum_psi_contour",
        "pf_passive_overlay",
        "twodim_geometry_all",
        "vacuum_psi_contour",
    ),
}.items():
    for _legacy_name in _names:
        LEGACY_MODULES[_legacy_name] = _module
del _module, _names, _legacy_name

# Anything left is a time-series renderer that still lives in vaft.plot.time.
for _legacy_name in DEPRECATED:
    LEGACY_MODULES.setdefault(_legacy_name, "time")
del _legacy_name


def render_markdown_table() -> str:
    """Render the full migration table as Markdown.

    Exported as :func:`vaft.plot.migration_table`, so the per-symbol table can be
    read without opening this file::

        python -c "import vaft.plot; print(vaft.plot.migration_table())"
    """
    lines = [
        "# `vaft.plot` migration table",
        "",
        "Issue #62 replaced the wildcard-exported `vaft.plot` surface with an",
        "explicit, registry-driven one. This table covers every symbol the package",
        "exported before that change.",
        "",
        "Measured on the pre-redesign package, `vaft.plot` exposed 318 public",
        "attributes: 174 VAFT callables, roughly 76 unrelated OMAS functions, and 42",
        "modules (`np`, `plt`, `sns`, `omas_h5`, ...). The modules and the OMAS",
        "functions were accidental leakage from `from omas import *` in",
        "`vaft/plot/utils.py` and are gone without a deprecation period -- they were",
        "never part of this package's API.",
        "",
        "## Compatibility window",
        "",
        f"Deprecations were introduced in **{INTRODUCED_IN}** and are removed in",
        f"**{REMOVAL_RELEASE}**. Until then, every name in the *Deprecated* and",
        "*Relocated* sections keeps working from `vaft.plot` and emits a",
        "`DeprecationWarning` that names its replacement.",
        "",
        "## How to read the replacements",
        "",
        "Canonical renderers take typed view models, so a deprecated call does not",
        "map one-to-one onto its canonical renderer. The practical replacement for a",
        "call that used to take an ODS is the adapter:",
        "",
        "```python",
        "vaft.plot.time_magnetics_ip(ods)              # before",
        "vaft.omas.plot_plasma_current_time(ods)         # after",
        "```",
        "",
        "Use `vaft.plot.<canonical>` directly when you already hold a view model.",
        "",
        "## Deprecated renderers",
        "",
        "These still work and warn. Several legacy names collapse onto one canonical",
        "renderer: the 24 generated `equilibrium_<coordinate>_<quantity>` globals",
        "become six `equilibrium_profile_*` renderers plus a `coordinate=` argument,",
        "and the four top-view wrappers become `machine_geometry_topview`.",
        "",
        "| Legacy name | Canonical `vaft.plot` name | Adapter |",
        "| --- | --- | --- |",
    ]
    for name in sorted(DEPRECATED):
        canonical = DEPRECATED[name]
        lines.append(f"| `{name}` | `{canonical}` | `vaft.omas.plot_{canonical}` |")

    lines += [
        "",
        "## Renamed canonical stems (issue #251)",
        "",
        "The issue #251 subject taxonomy renamed these canonical stems from",
        "IDS-oriented to physical-subject-oriented names. The old stems keep",
        f"working and warn from **{RENAMED_IN}** until **{RENAMED_REMOVAL_RELEASE}**,",
        "from both `vaft.plot.<stem>` and `vaft.omas.plot_<stem>`.",
        "",
        "| Old canonical stem | New canonical stem |",
        "| --- | --- |",
    ]
    for name in sorted(RENAMED):
        lines.append(f"| `{name}` | `{RENAMED[name]}` |")

    lines += [
        "",
        "## Relocated",
        "",
        "These were never plotting functions. They were visible from `vaft.plot`",
        "only because of wildcard imports. They still resolve and warn, pointing at",
        "the namespace that owns them.",
        "",
        "| Legacy `vaft.plot` name | Real home |",
        "| --- | --- |",
    ]
    for name in sorted(RELOCATED):
        lines.append(f"| `{name}` | `{RELOCATED[name]}` |")

    lines += [
        "",
        "## Legacy -- no canonical destination yet",
        "",
        "Cross-shot statistics over pandas DataFrames. They do not fit the five",
        "array-shaped view models, so they keep their current names, do not warn,",
        "and are excluded from the canonical `available_plots()` listing. A later",
        "pass gives them a home. `confinement_time_histogram` moved here from",
        "`vaft.process.statistical_analysis` so that processing stays free of",
        "Matplotlib; the old call site still works.",
        "",
        "| Name | Still implemented in |",
        "| --- | --- |",
    ]
    for name in sorted(LEGACY):
        lines.append(f"| `{name}` | `vaft.plot.{LEGACY_MODULES.get(name, 'history')}` |")

    lines += [
        "",
        "## Removed internal helpers",
        "",
        "These leaked out of implementation modules through wildcard imports and",
        "were never intended as API. They raise `AttributeError` immediately. Where",
        "the behavior survives, it is in the view models or the adapters.",
        "",
        "| Removed name | Why, and what to use |",
        "| --- | --- |",
    ]
    for name in sorted(REMOVED):
        lines.append(f"| `{name}` | {REMOVED[name]} |")

    lines += [
        "",
        "## Already canonical -- unchanged",
        "",
        "These names were already spelled `<domain>_<view>_<quantity>` before the",
        "redesign and keep working without a warning. Their reverse-order twins",
        "(`time_magnetics_ip` and friends) are in the *Deprecated* table above. Note",
        "that the signature changed: they now take a view model and default to",
        "`show=False`, so ODS callers should move to `vaft.omas.plot_<name>`.",
        "",
        "| Name |",
        "| --- |",
    ]
    for name in PRESERVED:
        lines.append(f"| `{name}` |")

    return "\n".join(lines) + "\n"
