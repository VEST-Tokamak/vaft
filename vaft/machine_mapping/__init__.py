"""Canonical and compatibility machine_mapping namespace for VEST."""

import warnings
from importlib import import_module


_LEGACY_REPLACEMENTS = {
    "VEST_DiamagneticFlux": "vest_diamagnetic_flux",
    "vfit_barometry_dynamic": None,
    "vfit_barometry_static": None,
    "vfit_camera_visible_dynamic": None,
    "vfit_camera_visible_static": None,
    "vfit_charge_exchange": "charge_exchange",
    "vfit_dataset_description": None,
    "vfit_filterscope": "spectrometer_uv",
    "vfit_ion_doppler_spectroscophy": "charge_exchange",
    "vfit_magnetics_dynamic": None,
    "vfit_magnetics_for_shot": "magnetics",
    "vfit_magnetics_static": None,
    "vfit_mirnov_raw_dynamic": None,
    "vfit_pf_active_dynamic": None,
    "vfit_pf_active_for_shot": "pf_active",
    "vfit_pf_active_static": None,
    "vfit_soft_x_rays_dynamic": None,
    "vfit_soft_x_rays_static": None,
    "vfit_tf_dynamic": None,
    "vfit_tf_static": None,
    "vfit_thomson_scattering_dynamic": None,
    "vfit_thomson_scattering_static": None,
    # These remain supported VEST source-policy functions while their canonical
    # physical-data replacements are designed in later #103 phases.
    "vfit_md": None,
    "vfit_PlasmaCurrent": None,
    "vfit_plasma_current": None,
    "vfit_pf": None,
    "vfit_plasmaMGods_startend": None,
    "vfit_plasma_mgods_startend": None,
    "vfit_tf_btR": None,
    "vfit_tf_bt_r": None,
    "vfit_tf_current": None,
    "vest_md_channel_definitions": "vest_equilibrium_magnetics_channel_definitions",
}

__all__ = [
    "DEFAULT_CONSTRAINT_UNCERTAINTIES",
    "DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR",
    "VEST_DiamagneticFlux",
    "apply_default_constraint_uncertainties",
    "apply_magnetics_uncertainties",
    "apply_pf_active_current_uncertainties",
    "apply_tf_uncertainties",
    "b_field_pol_probe_from_raw_database",
    "barometry_from_raw_database",
    "calibrate_vest_signal",
    "calculate_em_coupling_from_raw_database",
    "CameraFrameSelectionError",
    "camera_visible_from_frame_dir",
    "camera_visible_from_raw_database",
    "dataset_description_from_raw_database",
    "diamagnetic_flux_rogowski_coil_from_raw_database",
    "filterscope_from_raw_database",
    "find_valid_frame_interval",
    "flux_loop_from_raw_database",
    "frame_time_ms",
    "get_metadata",
    "impa_from_raw_database",
    "impa_probe_indices",
    "interferometer_94ghz",
    "interferometer_282ghz",
    "ip_rogowski_coil_from_raw_database",
    "is_near_black",
    "apply_langmuir_probe_measured_positions",
    "langmuir_probes_from_raw_database",
    "magnetics_from_raw_database",
    "normalize_constraint_uncertainties",
    "pf_active_from_raw_database",
    "pf_geometry_version_for_shot",
    "vest_processing_provenance",
    "read_doppler_profile",
    "read_doppler_single",
    "resolve_vest_diagnostic",
    "resolve_geometry_asset",
    "vfit_soft_x_rays_static",
    "vfit_soft_x_rays_dynamic",
    "soft_x_rays_from_raw_database",
    "soft_x_rays_from_digitizer_csv",
    "save_camera_visible_ods",
    "save_soft_x_rays_ods",
    "tf_from_raw_database",
    "raw_database_info",
    "vfit_barometry_dynamic",
    "vfit_barometry_static",
    "vfit_camera_visible_dynamic",
    "vfit_camera_visible_static",
    "vfit_charge_exchange",
    "vfit_dataset_description",
    "vfit_filterscope",
    "vfit_ion_doppler_spectroscophy",
    "vfit_md",
    "vest_md_channel_definitions",
    "vfit_magnetics_dynamic",
    "vfit_magnetics_for_shot",
    "vfit_magnetics_static",
    "vfit_mirnov_raw_dynamic",
    "vfit_PlasmaCurrent",
    "vfit_plasma_current",
    "vfit_pf",
    "vfit_pf_active_dynamic",
    "vfit_pf_active_for_shot",
    "vfit_pf_active_static",
    "vfit_plasmaMGods_startend",
    "vfit_plasma_mgods_startend",
    "diamagnetic_saturation_report",
    "vest_diamagnetic_flux",
    "vest_diamagnetic_flux_detailed",
    "vest_equilibrium_magnetics_channel_definitions",
    "vfit_tf_btR",
    "vfit_tf_bt_r",
    "vfit_tf_current",
    "vfit_tf_dynamic",
    "vfit_tf_static",
    "vfit_thomson_scattering_dynamic",
    "vfit_thomson_scattering_static",
]

_EXPORT_MAP = {
    "DEFAULT_CONSTRAINT_UNCERTAINTIES": (".utils", "DEFAULT_CONSTRAINT_UNCERTAINTIES"),
    "DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR": (".utils", "DEFAULT_CONSTRAINT_UNCERTAINTY_VECTOR"),
    "VEST_DiamagneticFlux": (".magnetics", "VEST_DiamagneticFlux"),
    "apply_default_constraint_uncertainties": (".utils", "apply_default_constraint_uncertainties"),
    "apply_magnetics_uncertainties": (".utils", "apply_magnetics_uncertainties"),
    "apply_pf_active_current_uncertainties": (".utils", "apply_pf_active_current_uncertainties"),
    "apply_tf_uncertainties": (".utils", "apply_tf_uncertainties"),
    "b_field_pol_probe_from_raw_database": (".magnetics", "b_field_pol_probe_from_raw_database"),
    "barometry_from_raw_database": (".barometry", "barometry_from_raw_database"),
    "calculate_em_coupling_from_raw_database": (".em_coupling", "calculate_em_coupling_from_raw_database"),
    "calibrate_vest_signal": (".utils", "calibrate_vest_signal"),
    "CameraFrameSelectionError": (".camera_visible", "CameraFrameSelectionError"),
    "camera_visible_from_frame_dir": (".camera_visible", "camera_visible_from_frame_dir"),
    "camera_visible_from_raw_database": (".camera_visible", "camera_visible_from_raw_database"),
    "find_valid_frame_interval": (".camera_visible", "find_valid_frame_interval"),
    "frame_time_ms": (".camera_visible", "frame_time_ms"),
    "is_near_black": (".camera_visible", "is_near_black"),
    "save_camera_visible_ods": (".camera_visible", "save_camera_visible_ods"),
    "vfit_camera_visible_dynamic": (".camera_visible", "vfit_camera_visible_dynamic"),
    "vfit_camera_visible_static": (".camera_visible", "vfit_camera_visible_static"),
    "dataset_description_from_raw_database": (".dataset_description", "dataset_description_from_raw_database"),
    "diamagnetic_flux_rogowski_coil_from_raw_database": (".magnetics", "diamagnetic_flux_rogowski_coil_from_raw_database"),
    "filterscope_from_raw_database": (".spectrometer_uv", "filterscope_from_raw_database"),
    "flux_loop_from_raw_database": (".magnetics", "flux_loop_from_raw_database"),
    "get_metadata": (".utils", "get_metadata"),
    "impa_from_raw_database": (".impa", "impa_from_raw_database"),
    "impa_probe_indices": (".impa", "impa_probe_indices"),
    "interferometer_94ghz": (".interferometer", "interferometer_94ghz"),
    "interferometer_282ghz": (".interferometer", "interferometer_282ghz"),
    "ip_rogowski_coil_from_raw_database": (".magnetics", "ip_rogowski_coil_from_raw_database"),
    "apply_langmuir_probe_measured_positions": (".langmuir_probes", "apply_langmuir_probe_measured_positions"),
    "langmuir_probes_from_raw_database": (".langmuir_probes", "langmuir_probes_from_raw_database"),
    "magnetics_from_raw_database": (".magnetics", "magnetics_from_raw_database"),
    "normalize_constraint_uncertainties": (".utils", "normalize_constraint_uncertainties"),
    "pf_active_from_raw_database": (".pf_active", "pf_active_from_raw_database"),
    "pf_geometry_version_for_shot": (".pf_active", "pf_geometry_version_for_shot"),
    "vest_processing_provenance": (".provenance", "vest_processing_provenance"),
    "raw_database_info": (".utils", "raw_database_info"),
    "resolve_vest_diagnostic": (".utils", "resolve_vest_diagnostic"),
    "read_doppler_profile": (".charge_exchange", "read_doppler_profile"),
    "read_doppler_single": (".charge_exchange", "read_doppler_single"),
    "resolve_geometry_asset": (".pf_active", "resolve_geometry_asset"),
    "vfit_soft_x_rays_static": (".soft_x_rays", "vfit_soft_x_rays_static"),
    "vfit_soft_x_rays_dynamic": (".soft_x_rays", "vfit_soft_x_rays_dynamic"),
    "soft_x_rays_from_raw_database": (".soft_x_rays", "soft_x_rays_from_raw_database"),
    "soft_x_rays_from_digitizer_csv": (".soft_x_rays", "soft_x_rays_from_digitizer_csv"),
    "save_soft_x_rays_ods": (".soft_x_rays", "save_soft_x_rays_ods"),
    "tf_from_raw_database": (".tf", "tf_from_raw_database"),
    "vfit_barometry_dynamic": (".barometry", "vfit_barometry_dynamic"),
    "vfit_barometry_static": (".barometry", "vfit_barometry_static"),
    "vfit_charge_exchange": (".charge_exchange", "vfit_charge_exchange"),
    "vfit_dataset_description": (".dataset_description", "vfit_dataset_description"),
    "vfit_filterscope": (".spectrometer_uv", "vfit_filterscope"),
    "vfit_ion_doppler_spectroscophy": (".charge_exchange", "vfit_ion_doppler_spectroscophy"),
    "vfit_md": (".magnetics", "vfit_equilibrium_magnetics"),
    "vest_md_channel_definitions": (".magnetics", "vest_equilibrium_magnetics_channel_definitions"),
    "vfit_magnetics_dynamic": (".magnetics", "vfit_magnetics_dynamic"),
    "vfit_magnetics_for_shot": (".magnetics", "vfit_magnetics_for_shot"),
    "vfit_magnetics_static": (".magnetics", "vfit_magnetics_static"),
    "vfit_mirnov_raw_dynamic": (".magnetics", "vfit_mirnov_raw_dynamic"),
    "vfit_PlasmaCurrent": (".magnetics", "vfit_PlasmaCurrent"),
    "vfit_plasma_current": (".magnetics", "vfit_plasma_current"),
    "vfit_pf": (".pf_active", "vfit_pf"),
    "vfit_pf_active_dynamic": (".pf_active", "vfit_pf_active_dynamic"),
    "vfit_pf_active_for_shot": (".pf_active", "vfit_pf_active_for_shot"),
    "vfit_pf_active_static": (".pf_active", "vfit_pf_active_static"),
    "vfit_plasmaMGods_startend": (".magnetics", "vfit_plasmaMGods_startend"),
    "vfit_plasma_mgods_startend": (".magnetics", "vfit_plasma_mgods_startend"),
    "vest_diamagnetic_flux": (".magnetics", "vest_diamagnetic_flux"),
    "vest_diamagnetic_flux_detailed": (".magnetics", "vest_diamagnetic_flux_detailed"),
    "diamagnetic_saturation_report": (".magnetics", "diamagnetic_saturation_report"),
    "vest_equilibrium_magnetics_channel_definitions": (".magnetics", "vest_equilibrium_magnetics_channel_definitions"),
    "vfit_tf_btR": (".tf", "vfit_tf_btR"),
    "vfit_tf_bt_r": (".tf", "vfit_tf_bt_r"),
    "vfit_tf_current": (".tf", "vfit_tf_current"),
    "vfit_tf_dynamic": (".tf", "vfit_tf_dynamic"),
    "vfit_tf_static": (".tf", "vfit_tf_static"),
    "vfit_thomson_scattering_dynamic": (".thomson_scattering", "vfit_thomson_scattering_dynamic"),
    "vfit_thomson_scattering_static": (".thomson_scattering", "vfit_thomson_scattering_static"),
}

_LEGACY_EXPORT_MAP = {
    name: _EXPORT_MAP.pop(name) for name in _LEGACY_REPLACEMENTS
}
__all__ = [name for name in __all__ if name in _EXPORT_MAP]

# Canonical IDS entry points whose function name equals their module name.
# These are deliberately NOT exported from the package: a module's __getattr__
# only runs when normal lookup fails, so once anything imports the submodule
# the module object is bound here and would shadow the function -- silently,
# and depending only on which import happened first.  `vaft.machine_mapping.tf`
# therefore means the tf *module*, always; import the entry point explicitly.
_ENTRYPOINT_MODULES = frozenset(
    {
        "barometry",
        "camera_visible",
        "charge_exchange",
        "coils_non_axisymmetric",
        "dataset_description",
        "em_coupling",
        "equilibrium",
        "filterscope",
        "gpec_ideal",
        "impa",
        "interferometer",
        "langmuir_probes",
        "magnetics",
        "mhd_linear",
        "pf_active",
        "pf_passive",
        "soft_x_rays",
        "spectrometer_uv",
        "summary",
        "tf",
        "thomson_scattering",
        "wall",
    }
)

_collisions = set(_EXPORT_MAP) & _ENTRYPOINT_MODULES
assert not _collisions, (
    f"machine_mapping: {sorted(_collisions)} are both lazily exported and "
    "entry-point submodules; a submodule import would silently shadow the "
    "export. Remove them from _EXPORT_MAP."
)


def __getattr__(name: str):
    if name in _EXPORT_MAP:
        module_name, attribute = _EXPORT_MAP[name]
    elif name in _LEGACY_EXPORT_MAP:
        replacement = _LEGACY_REPLACEMENTS[name]
        guidance = (
            f" use vaft.machine_mapping.{replacement}() instead."
            if replacement is not None
            else " import it from its diagnostic module while migration is in progress."
        )
        warnings.warn(
            f"vaft.machine_mapping.{name} is a legacy compatibility API;{guidance}",
            DeprecationWarning,
            stacklevel=2,
        )
        module_name, attribute = _LEGACY_EXPORT_MAP[name]
    elif name in _ENTRYPOINT_MODULES:
        raise AttributeError(
            f"vaft.machine_mapping.{name} is the {name} submodule, not the mapping "
            f"entry point. Import it explicitly: "
            f"from vaft.machine_mapping.{name} import {name}"
        )
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
