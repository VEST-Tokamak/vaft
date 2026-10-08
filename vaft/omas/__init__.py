from os import fspath

from omas import load_omas_json as _omas_load_omas_json

from .general import *
from .process_wrapper import *
from .formula_wrapper import *
from .update import *
from .sample import *
from .startup_summary import NULL_FIELD_THRESHOLD_T, startup_summary
from .edge_q import EdgeQEstimate, edge_q_estimate
from .equilibrium_state import EQUILIBRIUM_STATE_UNITS, equilibrium_state_rows, equilibrium_state_table
from .fluctuation import (
    DiagnosticSelection,
    SelectedDiagnostics,
    VerticalPositionHistory,
    fluctuation_bandwidths,
    select_fluctuation_records,
    vertical_position_history,
)
from . import formula_wrapper as _formula_wrapper
from . import general as _general
from . import process_wrapper as _process_wrapper
from . import sample as _sample
from . import update as _update

# Before the submodules above declared ``__all__`` (#1382), their star imports
# also bound every VAFT function they had imported themselves, so these names
# have always been reachable as ``vaft.omas.<name>``.  They stay bound here, but
# they are not published by this package: each belongs to the module that
# defines it, and ``from vaft.omas import *`` no longer binds them.  The
# ``omas``, NumPy, SciPy, typing and standard-library names the submodules
# imported (and their ``logger`` and the ``vaft`` module) are not kept.
from vaft.compat import trapz_compat
from vaft.data.eqdsk import ods_psi_to_wb_per_radian_factor
from vaft.formula.constants import MU0
from vaft.formula.equilibrium import (
    bremsstrahlung_power_density_from_n_e_T_e_Z_eff,
    bremsstrahlung_power_density_from_T_e_p_Z_eff,
    confinement_factor_ITER89P,
    confinement_time_from_engineering_parameters,
    confinement_time_from_P_loss_W_th,
    cyclotron_synchrotron_power_density_scaling_from_n_e_B_t_T_e,
    elongation_from_RZ_boundary,
    heating_power_from_p_ohm_p_aux,
    inductive_voltage_from_dW_magdt_I_p,
    inverse_aspect_ratio_from_a_R,
    kinetic_energy_from_beta_p_B_pa_V_p,
    loop_voltage_from_total_flux,
    loss_power_from_p_heat_dWdt_p_rad,
    magnetic_energy_from_li_B_pa_V_p,
    magnetic_shear,
    normalize_psi,
    ohmic_heating_power_from_I_p_V_res,
    poloidal_field_factor,
    spitzer_resistivity_from_T_e_Z_eff_ln_Lambda,
    stored_energy_from_p_V,
)
from vaft.formula.virial import (
    virial_alpha_approx_from_kappa,
    virial_beta_p_from_volume,
    virial_beta_pd_from_S_mu_rt,
    virial_bongard_from_S_alpha_mu,
    virial_closure_denominators,
    virial_full_123_from_S_alpha_rt,
    virial_identity_residuals,
    virial_lao_from_S_alpha_mu_rt,
    virial_li_from_volume,
    virial_muihat_from_Bt_R0_dphi,
    virial_normalized_residual,
    virial_pair_12_from_S_mu_rt,
    virial_pair_13_from_S_alpha_mu,
    virial_pair_23_from_S_alpha_mu_rt,
    virial_residual_rms,
)
from vaft.process.atomic import compute_line_radiation_power_series
from vaft.process.camera_geometry import (
    project_points,
    sweep_toroidal,
    toroidal_ring,
    trajectory_world_points,
)
from vaft.process.electromagnetics import (
    calc_grid,
    compute_br_bz_phi,
    compute_impedance_matrices,
    compute_response_matrix,
    compute_vacuum_fields_1d,
    solve_eddy_currents,
)
from vaft.process.equilibrium import (
    calculate_average_boundary_poloidal_field,
    calculate_diamagnetism,
    calculate_reconstructed_diamagnetic_flux,
    computed_diamagnetism_from_phi,
    efit_virial_volume_integrals,
    extract_flux_surface_contours,
    fractional_cell_weights_from_boundary,
    make_equilibrium_field_interpolator,
    parallel_current_from_toroidal,
    poloidal_field_at_boundary,
    prepare_boundary_for_shafranov,
    psi_to_rz,
    shafranov_integrals,
    trace_field_line,
    virial_alpha_conformal_annulus,
    virial_alpha_thin_annulus,
    volume_average,
)
from vaft.process.numerical import time_derivative

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

#: What ``vaft.omas`` publishes: everything its star-imported submodules
#: publish (each declares its own ``__all__`` since #1382), the functions
#: defined or imported by name here, and the reference and comparison exports
#: resolved by ``__getattr__``.  The plotting adapters are published by
#: ``vaft.omas.plotting.__all__`` and stay out of this list so that
#: ``from vaft.omas import *`` does not load Matplotlib.  Every other name bound
#: here -- the ``vaft.process`` and ``vaft.formula`` functions imported above --
#: stays reachable as an attribute but is not this package's API.  The generated
#: API reference reads this list (cold review 0.8.0 docs-and-tutorials F4;
#: test/test_docs_omas_surface.py).
__all__ = [
    *_general.__all__,
    *_process_wrapper.__all__,
    *_formula_wrapper.__all__,
    *_update.__all__,
    *_sample.__all__,
    "NULL_FIELD_THRESHOLD_T",
    "startup_summary",
    "EdgeQEstimate",
    "edge_q_estimate",
    "EQUILIBRIUM_STATE_UNITS",
    "equilibrium_state_rows",
    "equilibrium_state_table",
    "VerticalPositionHistory",
    "fluctuation_bandwidths",
    "DiagnosticSelection",
    "SelectedDiagnostics",
    "select_fluctuation_records",
    "vertical_position_history",
    "load_omas_json",
    "load",
    "save",
    "to_equilibrium",
    *sorted(_REFERENCE_EXPORTS),
    *sorted(_COMPARISON_EXPORTS),
]
