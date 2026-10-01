from os import fspath

from omas import load_omas_json as _omas_load_omas_json

from .general import *
from .process_wrapper import *
from .formula_wrapper import *
from .update import *
from .sample import *
from .startup_summary import NULL_FIELD_THRESHOLD_T, startup_summary
from .fluctuation import (
    VerticalPositionHistory,
    fluctuation_bandwidths,
    vertical_position_history,
)

#: Plotting adapters live in ``.plotting`` and are resolved lazily so that
#: importing ``vaft.omas`` does not pull in Matplotlib.
def _plotting_exports() -> frozenset:
    from .plotting import __all__ as names

    return frozenset(names)


_REFERENCE_EXPORTS = {
    "ArtifactVerification",
    "ReferenceManifestError",
    "load_reference_manifest",
    "sha256_file",
    "verify_reference_artifacts",
}
_COMPARISON_EXPORTS = {
    "ComparisonEntry",
    "DifferenceKind",
    "ODSComparison",
    "ParityClassification",
    "Tolerance",
    "TolerancePolicy",
    "ToleranceRule",
    "compare_ods",
    "load_tolerance_policy",
    "write_comparison_reports",
}


_PLOTTING_EXPORTS: frozenset | None = None


def _is_plotting_export(name: str) -> bool:
    global _PLOTTING_EXPORTS
    if name != "plotting" and not (
        name.startswith(("plot_", "dd_", "extract_"))
        or name in {
            "available_plots",
            "compose",
            "disable_overlay_methods",
            "disable_plot_methods",
            "enable_overlay_methods",
            "enable_plot_methods",
            "extract_labels_from_odc",
            "normalize_entries",
            "render_plot",
        }
    ):
        return False
    if _PLOTTING_EXPORTS is None:
        _PLOTTING_EXPORTS = _plotting_exports()
    return name == "plotting" or name in _PLOTTING_EXPORTS or name == "render_plot"


def __getattr__(name):
    if _is_plotting_export(name):
        from . import plotting

        value = plotting if name == "plotting" else getattr(
            plotting, "render" if name == "render_plot" else name
        )
        globals()[name] = value
        return value
    if name in _REFERENCE_EXPORTS:
        from . import reference

        value = getattr(reference, name)
        globals()[name] = value
        return value
    if name in _COMPARISON_EXPORTS:
        from . import comparison

        value = getattr(comparison, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        set(globals())
        | _REFERENCE_EXPORTS
        | _COMPARISON_EXPORTS
        | _plotting_exports()
        | {"plotting", "render_plot"}
    )


def load_omas_json(source, *args, **kwargs):
    """Load OMAS JSON from a string or any :class:`os.PathLike` path."""
    return _omas_load_omas_json(fspath(source), *args, **kwargs)


def load(source, *, imas_version=None):
    """Read any supported local artifact and return a normalized OMAS ODS.

    ``source`` may be OMAS JSON/HDF5, an IMAS netCDF file, an IMAS HDF5
    directory/image set, a GEQDSK file, or a sequence of GEQDSK files.
    """
    from ..database._local import load_ods

    ods, _info = load_ods(source, imas_version=imas_version)
    return ods


def save(ods, target, *, compression=None):
    """Save an OMAS ODS as JSON, HDF5 or netCDF, chosen from ``target``'s suffix.

    ``.nc`` is OMAS's own flat netCDF serialization (``omas.save_omas_nc``),
    not the IMAS netCDF convention -- write that with :func:`vaft.imas.save`.

    ``compression`` applies to ``.h5``/``.hdf5`` targets only and names an
    h5py filter (``"gzip"``, ``"lzf"``). OMAS's own ``save_omas_h5`` is
    ``dict2hdf5(filename, ods, lists_as_dicts=True)`` with no way to pass a
    filter through, so a compressed target calls ``dict2hdf5`` directly with
    the same arguments plus ``compression``. The dataset contents are
    identical either way -- HDF5 compression is lossless and transparent to
    readers, so a compressed product loads through the ordinary loader with
    no flag on the reading side.
    """
    import gzip
    from pathlib import Path
    import shutil

    from ..compat import reopenable_temporary_file

    target_path = Path(target).expanduser()
    suffixes = target_path.suffixes
    is_hdf5 = target_path.suffix.lower() in {".h5", ".hdf5"}
    if (
        target_path.suffix.lower() not in {".h5", ".hdf5", ".json", ".nc"}
        and suffixes[-2:] != [".json", ".gz"]
    ):
        raise ValueError("vaft.omas.save target must end in .json, .json.gz, .h5, .hdf5, or .nc")
    if compression is not None and not is_hdf5:
        raise ValueError(
            "vaft.omas.save compression applies to .h5/.hdf5 targets only; "
            f"got {target_path.name!r}. JSON.GZ is already gzip-compressed."
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if compression is not None:
        from omas.omas_h5 import dict2hdf5

        dict2hdf5(str(target_path), ods, lists_as_dicts=True, compression=compression)
    elif target_path.suffix.lower() == ".nc":
        from omas import save_omas_nc

        save_omas_nc(ods, str(target_path))
    elif suffixes[-2:] == [".json", ".gz"]:
        # `ODS.save` takes a path and opens it itself, so the staging file has
        # to be openable by name -- which a NamedTemporaryFile is not on
        # Windows. See vaft.compat.reopenable_temporary_file.
        with reopenable_temporary_file(suffix=".json") as plain_path:
            ods.save(str(plain_path))
            with plain_path.open("rb") as plain:
                with target_path.open("wb") as target_handle:
                    with gzip.GzipFile(
                        filename="",
                        mode="wb",
                        fileobj=target_handle,
                        compresslevel=9,
                        mtime=0,
                    ) as compressed:
                        shutil.copyfileobj(plain, compressed)
    else:
        ods.save(str(target_path))
    return target_path


def to_equilibrium(ods, *, time_index=0, profile_index=0, convention=None):
    """Adapt one ODS equilibrium slice to the lightweight scientific model."""
    from vaft.process.equilibrium import as_equilibrium

    return as_equilibrium(
        ods, time_index=time_index, profile_index=profile_index, convention=convention
    )

#: What ``vaft.omas`` publishes: the wrapper, update, sample and helper functions
#: its star-imported submodules define, plus the reference and comparison exports
#: resolved by ``__getattr__``.  The plotting adapters are published by
#: ``vaft.omas.plotting.__all__`` (they are lazy here) and every other name the
#: star imports drag in -- ``ODS``, ``np``, the ``vaft.process`` and
#: ``vaft.formula`` functions the wrappers call -- stays reachable as an
#: attribute but is not this package's API.  The generated API reference reads
#: this list (cold review 0.8.0 docs-and-tutorials F4); a new public function
#: of a wrapper module is added here (test/test_docs_omas_surface.py).
__all__ = [
    "ArtifactVerification",
    "ComparisonEntry",
    "DifferenceKind",
    "NULL_FIELD_THRESHOLD_T",
    "ODSComparison",
    "ParityClassification",
    "ReferenceManifestError",
    "Tolerance",
    "TolerancePolicy",
    "ToleranceRule",
    "VerticalPositionHistory",
    "camera_projection_for",
    "change_time_convention",
    "classify_shot",
    "clear_vacuum_field_cache",
    "combine_ods",
    "compare_ods",
    "compute_bremsstrahlung_power",
    "compute_camera_visible_efit_overlay",
    "compute_camera_visible_field_line_overlay",
    "compute_camera_visible_vacuum_field_lines",
    "compute_confiment_time_paramters",
    "compute_connection_length_map_ods",
    "compute_core_profile_2d",
    "compute_core_profile_psi",
    "compute_decay_index_ods",
    "compute_diamagnetic_flux_measured_vs_computed",
    "compute_diamagnetism",
    "compute_eddy_currents",
    "compute_ejiri_mirror_proxy_ods",
    "compute_field_line_trace",
    "compute_grad_shafranov_residual",
    "compute_grid_ods",
    "compute_grid_response_ods",
    "compute_impedance_matrices_ods",
    "compute_magnetic_energy",
    "compute_magnetic_shear",
    "compute_null_ods",
    "compute_ohmic_heating_power_from_core_profiles",
    "compute_parallel_current_from_toroidal",
    "compute_point_response_matrices_ods",
    "compute_point_response_ods",
    "compute_point_vacuum_fields_ods",
    "compute_power_balance",
    "compute_prefill_pressure_ods",
    "compute_reconstructed_diamagnetic_flux",
    "compute_romero_flux_balance_ods",
    "compute_startup_loop_voltage_ods",
    "compute_startup_proxies_ods",
    "compute_tau_E_engineering_parameters",
    "compute_tau_E_exp",
    "compute_tau_E_scaling",
    "compute_vacuum_field_map",
    "compute_vacuum_midplane_profiles_ods",
    "compute_virial_equilibrium_quantities_ods",
    "compute_voltage_consumption",
    "compute_volume_averaged_pressure",
    "compute_wall_mode_basis_ods",
    "ensure_em_coupling",
    "equilibrium_psi_to_weber",
    "find_breakdown_onset",
    "find_bt",
    "find_chamber_boundary",
    "find_ip_onset",
    "find_major_radius",
    "find_matching_time_indices",
    "find_max_ip",
    "find_pf_active_onset",
    "find_pulse_duration",
    "find_shotclass",
    "find_shotnumber",
    "find_vloop_onset",
    "fluctuation_bandwidths",
    "load",
    "load_omas_json",
    "load_reference_manifest",
    "load_tolerance_policy",
    "odc_or_ods_check",
    "ods_cocos",
    "print_info",
    "resolve_reference_major_radius",
    "sample_equilibria",
    "sample_gfile",
    "sample_ods",
    "save",
    "set_ods_cocos",
    "sha256_file",
    "shift_time",
    "signal_time",
    "startup_summary",
    "to_equilibrium",
    "update_core_profiles_global_quantities_volume_average",
    "update_equilibrium_boundary",
    "update_equilibrium_constraints_diamagnetic_flux",
    "update_equilibrium_coordinates",
    "update_equilibrium_derived_profiles",
    "update_equilibrium_global_quantities_area",
    "update_equilibrium_global_quantities_beta_li",
    "update_equilibrium_global_quantities_q_min",
    "update_equilibrium_global_quantities_volume",
    "update_equilibrium_profiles_1d_geometry",
    "update_equilibrium_profiles_1d_j_tor",
    "update_equilibrium_profiles_1d_normalized_psi",
    "update_equilibrium_profiles_1d_radial_coordinates",
    "update_equilibrium_profiles_1d_toroidal_flux",
    "update_equilibrium_profiles_2d_b_field",
    "update_equilibrium_profiles_2d_j_tor",
    "update_equilibrium_profiles_2d_sfl_coordinates",
    "update_equilibrium_stored_energy",
    "verify_reference_artifacts",
    "vertical_position_history",
    "write_comparison_reports",
]
