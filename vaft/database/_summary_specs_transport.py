"""Column specs of the volume-averaged, EFIT-reliability and transport summary presets.

One :class:`~vaft.database._summary_contract.ColumnSpec` per column, in the order
of the column tuples in :mod:`vaft.database._summary`. Every unit and definition
is read off the extractor that fills the column, not off the column name; where
the two disagree the definition says what is stored.
"""

from __future__ import annotations

from vaft.database._summary_contract import ColumnSpec

_CP1D = "core_profiles.profiles_1d[:]"
_EQ = "equilibrium.time_slice[:]"
_CONSTRAINT = f"{_EQ}.constraints.<measurement_type>[:]"
_CT1D = "core_transport.model[:].profiles_1d[:]"

#: The unit of ``measured``/``reconstructed``/``residual``/``uncertainty`` by
#: constraint family, as the IMAS data dictionary defines ``measured``.
_FAMILY_UNITS = (
    "bpol_probe T, flux_loop Wb, pf_current A, ip A, diamagnetic_flux Wb, pressure Pa, "
    "pressure_rotational Pa, j_tor A m^-2, mse_polarisation_angle rad"
)

_SOURCE = ("database source the row was read from (the resolved source name of summary(), "
           "or the source argument of summary_from_ods, default 'local')")


VOLUME_AVERAGED_SPECS = (
    ColumnSpec("shot", "", "VEST shot number."),
    ColumnSpec("cp_index", "", "Index of the core_profiles.profiles_1d slice this row describes (one row per slice).",
               ids_path=_CP1D),
    ColumnSpec("eq_index", "",
               "Index of the equilibrium.time_slice nearest in time to the core_profiles slice "
               "(argmin |t_eq - t_core|, no tolerance applied).",
               ids_path=_EQ),
    ColumnSpec("time_core_s", "s",
               "Time of the core_profiles slice: profiles_1d[:].time, else core_profiles.time[cp_index], "
               "else the slice index itself.",
               ids_path=f"{_CP1D}.time"),
    ColumnSpec("time_equilibrium_s", "s",
               "Time of the matched equilibrium slice: time_slice[:].time, else equilibrium.time[eq_index], "
               "else the slice index itself.",
               ids_path=f"{_EQ}.time"),
    ColumnSpec("time_diff_s", "s",
               "Absolute offset |time_equilibrium_s - time_core_s| of the nearest-time match; nothing rejects a "
               "large offset.",
               ids_path=None),
    ColumnSpec("ne_volume_m3", "m^-3",
               "Plasma-volume average of core_profiles electron density inside the LCFS, mapped rho_tor_norm -> "
               "psi_N -> (R,Z) on the nearest equilibrium slice; stored in m^-3 (not m^3 as the suffix reads).",
               formula="vaft.omas.update.update_core_profiles_global_quantities_volume_average",
               ids_path="core_profiles.global_quantities.n_e_volume_average"),
    ColumnSpec("te_volume_eV", "eV",
               "Plasma-volume average of core_profiles electron temperature inside the LCFS, same mapping and "
               "equilibrium slice as ne_volume_m3.",
               formula="vaft.omas.update.update_core_profiles_global_quantities_volume_average",
               ids_path="core_profiles.global_quantities.t_e_volume_average"),
    ColumnSpec("electron_pressure_volume_Pa", "Pa",
               "2 e <n_e>_V <T_e>_V: product of the two volume averages (not the average of n_e T_e), with the "
               "factor 2 assuming n_i = n_e and T_i = T_e, so a total thermal pressure estimate, not p_e alone.",
               ids_path="core_profiles.global_quantities"),
    ColumnSpec("equilibrium_pressure_volume_Pa", "Pa",
               "Plasma-volume average of the equilibrium pressure profile (profiles_1d.pressure on psi_norm, "
               "mapped onto profiles_2d psi) of the matched equilibrium slice.",
               formula="vaft.omas.process_wrapper.compute_volume_averaged_pressure",
               ids_path=f"{_EQ}.profiles_1d.pressure"),
    ColumnSpec("electron_to_equilibrium_pressure_ratio", "-",
               "electron_pressure_volume_Pa / equilibrium_pressure_volume_Pa; NaN when either is non-finite or "
               "the equilibrium value is zero."),
)


EFIT_RELIABILITY_SPECS = (
    ColumnSpec("shot", "", "VEST shot number."),
    ColumnSpec("eq_index", "", "Index of the equilibrium.time_slice the constraint belongs to.", ids_path=_EQ),
    ColumnSpec("time_s", "s",
               "Equilibrium slice time: time_slice[:].time, else equilibrium.time[eq_index], else the slice "
               "index itself.",
               ids_path=f"{_EQ}.time"),
    ColumnSpec("ip_kA", "kA",
               "Reconstructed plasma current of the slice, global_quantities.ip / 1e3, signed in the IMAS "
               "convention (repeated on every constraint row of the slice).",
               ids_path=f"{_EQ}.global_quantities.ip"),
    ColumnSpec("measurement_index", "",
               "Index of the channel within the constraint family array; 0 for the scalar families ip and "
               "diamagnetic_flux.",
               ids_path=_CONSTRAINT),
    ColumnSpec("measurement_type", "",
               "Constraint family name: bpol_probe, flux_loop, pf_current, ip, diamagnetic_flux (magnetic "
               "presets) or pressure, pressure_rotational, j_tor, mse_polarisation_angle (kinetic preset).",
               ids_path=f"{_EQ}.constraints"),
    ColumnSpec("identifier", "", "Constraint identifier text as stored, '' when absent.",
               ids_path=f"{_CONSTRAINT}.identifier"),
    ColumnSpec("constraint_source", "", "Constraint source text (the diagnostic path it came from), '' when absent.",
               ids_path=f"{_CONSTRAINT}.source"),
    ColumnSpec("measured", "",
               f"Measured value EFIT was asked to fit, in the unit of its family ({_FAMILY_UNITS}); rows where "
               "measured or reconstructed is non-finite are dropped.",
               ids_path=f"{_CONSTRAINT}.measured"),
    ColumnSpec("reconstructed", "",
               "Value of the same channel in the reconstructed equilibrium, in the unit of measured for that "
               "family.",
               ids_path=f"{_CONSTRAINT}.reconstructed"),
    ColumnSpec("residual", "",
               "reconstructed - measured (signed), in the unit of measured for that family."),
    ColumnSpec("absolute_residual", "", "|reconstructed - measured|, in the unit of measured for that family."),
    ColumnSpec("uncertainty", "",
               "Upper measurement error the constraint carries (measured_error_upper), in the unit of measured "
               "for that family.",
               ids_path=f"{_CONSTRAINT}.measured_error_upper"),
    ColumnSpec("normalized_residual", "-",
               "(reconstructed - measured) / uncertainty, signed; NaN when the uncertainty is zero or "
               "non-finite."),
    ColumnSpec("weight", "-", "Weight EFIT gave the constraint, as stored; 0 marks a disabled channel.",
               ids_path=f"{_CONSTRAINT}.weight"),
    ColumnSpec("chi_squared", "-",
               "Per-constraint chi-square as stored, except for ip, where it is recomputed from the plasma-only "
               "residual ((measured - reconstructed)/measured_error_upper)^2 without the vessel-current term.",
               formula="vaft.omas.efit_quality.ip_chi_squared_record",
               ids_path=f"{_CONSTRAINT}.chi_squared"),
    ColumnSpec("exact", "",
               "Boolean flag: the constraint was imposed exactly (DD exact = 1) rather than fitted; NaN when not "
               "stored.",
               ids_path=f"{_CONSTRAINT}.exact"),
    ColumnSpec("disabled", "", "Boolean flag: weight is finite and exactly 0."),
    ColumnSpec("chi_squared_reduced", "-",
               "Slice aggregate chi-square per degree of freedom; when the ip vessel-current term is nonzero it "
               "is reduced by that term / freedom_degrees_n, and NaN if the degree count is missing.",
               ids_path=f"{_EQ}.constraints.chi_squared_reduced"),
    ColumnSpec("convergence_iterations_n", "", "Number of iterations EFIT ran for the slice, as stored.",
               ids_path=f"{_EQ}.convergence.iterations_n"),
    ColumnSpec("convergence_deviation", "",
               "Grad-Shafranov residual deviation of the slice, as stored; its unit and form are set by "
               "convergence.grad_shafranov_deviation_expression (the DD declares the unit 'mixed').",
               ids_path=f"{_EQ}.convergence.grad_shafranov_deviation_value"),
    ColumnSpec("equilibrium_source", "", f"The {_SOURCE}."),
    ColumnSpec("equilibrium_lineage", "",
               "Sorted-key JSON text of the slice lineage: data_entry machine/pulse/run, equilibrium comment and "
               "code name/version, mapping_source_revision and the g/m-file SHA-256 from the EFIT collection "
               "payload in equilibrium.code.parameters.",
               formula="vaft.database._summary._equilibrium_lineage",
               ids_path="equilibrium.code.parameters"),
    ColumnSpec("efit_configuration", "",
               "EFIT configuration label 'name@sha12' from the efit_preset record (efit_collection, else "
               "kinetic_efit) in equilibrium.code.parameters, or 'unrecorded'.",
               formula="vaft.database._summary._efit_configuration",
               ids_path="equilibrium.code.parameters"),
)


NEOCLASSICAL_SPECS = (
    ColumnSpec("shot", "", "VEST shot number."),
    ColumnSpec("cp_index", "",
               "Index of the core_profiles.profiles_1d slice; only slices with a finite j_bootstrap on "
               "grid.rho_tor_norm give a row.",
               ids_path=_CP1D),
    ColumnSpec("time_s", "s",
               "core_profiles.time[cp_index]; NaN when that vector is absent (the transport match below uses "
               "profiles_1d[:].time first when stored).",
               ids_path="core_profiles.time"),
    ColumnSpec("i_bootstrap_kA", "kA",
               "Integral of core_profiles j_bootstrap (DD: <J_bs.B>/B0) over the equilibrium cross-section "
               "area(rho_tor_norm) of the time-matched equilibrium slice, trapezoid over solved points only, "
               "/1e3; NaN without an area.",
               formula="vaft.database._summary._bootstrap_current_kA",
               ids_path=f"{_CP1D}.j_bootstrap"),
    ColumnSpec("j_bootstrap_peak_A_m2", "A m^-2",
               "Signed j_bootstrap at the solved point of largest |j_bootstrap| (the value keeps its sign).",
               ids_path=f"{_CP1D}.j_bootstrap"),
    ColumnSpec("rho_at_peak", "-", "core_profiles grid.rho_tor_norm at that peak point.",
               ids_path=f"{_CP1D}.grid.rho_tor_norm"),
    ColumnSpec("rho_solved_min", "-", "Smallest rho_tor_norm at which j_bootstrap is finite (solved).",
               ids_path=f"{_CP1D}.grid.rho_tor_norm"),
    ColumnSpec("rho_solved_max", "-", "Largest rho_tor_norm at which j_bootstrap is finite (solved).",
               ids_path=f"{_CP1D}.grid.rho_tor_norm"),
    ColumnSpec("points_solved", "", "Number of grid points with a finite j_bootstrap.",
               ids_path=f"{_CP1D}.j_bootstrap"),
    ColumnSpec("b0_T", "T",
               "core_profiles.vacuum_toroidal_field.b0 for the slice, signed as stored (the single value when "
               "only one is stored, else entry cp_index), at vacuum_toroidal_field.r0; NaN otherwise.",
               ids_path="core_profiles.vacuum_toroidal_field.b0"),
    ColumnSpec("has_core_transport", "", "Boolean flag: core_transport.model holds at least one model.",
               ids_path="core_transport.model"),
    ColumnSpec("source", "", f"The {_SOURCE}."),
    ColumnSpec("transport_match_status", "",
               "Why the row has (matched) or lacks a NEO core_transport slice: bootstrap_time_unknown, "
               "no_neoclassical_model, ambiguous_models, no_slice_at_time, ambiguous_slices, producer_mismatch, "
               "no_flux_grid; the match needs exactly one model with identifier 5 and one slice within 1 us.",
               formula="vaft.database._summary._neoclassical_transport"),
    ColumnSpec("transport_model_index", "", "Index of the matched neoclassical core_transport model; NaN unmatched.",
               ids_path="core_transport.model[:]"),
    ColumnSpec("transport_profile_index", "",
               "Index of the matched profiles_1d slice of that model; NaN unmatched.",
               ids_path=_CT1D),
    ColumnSpec("transport_code_version", "",
               "Producer version of the matched fluxes: the model's code.version, else core_transport.code.version "
               "when the names agree; None unmatched.",
               ids_path="core_transport.model[:].code.version"),
    ColumnSpec("transport_rho_grid_min", "-", "Smallest finite grid_flux.rho_tor_norm of the matched slice.",
               ids_path=f"{_CT1D}.grid_flux.rho_tor_norm"),
    ColumnSpec("transport_rho_grid_max", "-", "Largest finite grid_flux.rho_tor_norm of the matched slice.",
               ids_path=f"{_CT1D}.grid_flux.rho_tor_norm"),
    ColumnSpec("transport_ion_count", "", "Number of ion entries in the matched slice.",
               ids_path=f"{_CT1D}.ion"),
    ColumnSpec("q_e_points", "",
               "Grid_flux points with a finite electron energy flux in the matched slice (0: channel absent; "
               "NaN: unmatched).",
               ids_path=f"{_CT1D}.electrons.energy.flux"),
    ColumnSpec("q_i_points", "",
               "Grid_flux points with a finite total ion energy flux (sum over all ion species, only when every "
               "species stores one); 0 when absent, NaN unmatched.",
               ids_path=f"{_CT1D}.ion[:].energy.flux"),
    ColumnSpec("gamma_e_points", "",
               "Grid_flux points with a finite electron particle flux; 0 when absent, NaN unmatched.",
               ids_path=f"{_CT1D}.electrons.particles.flux"),
    ColumnSpec("rho_q_e_min", "-", "Smallest grid_flux rho_tor_norm with a finite electron energy flux.",
               ids_path=f"{_CT1D}.electrons.energy.flux"),
    ColumnSpec("rho_q_e_max", "-", "Largest grid_flux rho_tor_norm with a finite electron energy flux.",
               ids_path=f"{_CT1D}.electrons.energy.flux"),
    ColumnSpec("rho_q_i_min", "-", "Smallest grid_flux rho_tor_norm with a finite total ion energy flux.",
               ids_path=f"{_CT1D}.ion[:].energy.flux"),
    ColumnSpec("rho_q_i_max", "-", "Largest grid_flux rho_tor_norm with a finite total ion energy flux.",
               ids_path=f"{_CT1D}.ion[:].energy.flux"),
    ColumnSpec("rho_gamma_e_min", "-", "Smallest grid_flux rho_tor_norm with a finite electron particle flux.",
               ids_path=f"{_CT1D}.electrons.particles.flux"),
    ColumnSpec("rho_gamma_e_max", "-", "Largest grid_flux rho_tor_norm with a finite electron particle flux.",
               ids_path=f"{_CT1D}.electrons.particles.flux"),
    ColumnSpec("q_e_peak_abs_W_m2", "W m^-2",
               "Largest |electron energy flux| over finite grid_flux points of the matched NEO slice.",
               formula="vaft.database._summary._peak_abs",
               ids_path=f"{_CT1D}.electrons.energy.flux"),
    ColumnSpec("q_i_peak_abs_W_m2", "W m^-2",
               "Largest |sum over ion species of ion energy flux| over finite grid_flux points of the matched slice.",
               formula="vaft.database._summary._peak_abs",
               ids_path=f"{_CT1D}.ion[:].energy.flux"),
    ColumnSpec("gamma_e_peak_abs_m2_s", "m^-2 s^-1",
               "Largest |electron particle flux| over finite grid_flux points of the matched slice (unit is "
               "m^-2 s^-1, not m^2 s^-1 as the suffix reads).",
               formula="vaft.database._summary._peak_abs",
               ids_path=f"{_CT1D}.electrons.particles.flux"),
)


TURBULENT_TRANSPORT_SPECS = (
    ColumnSpec("shot", "", "VEST shot number."),
    ColumnSpec("model_index", "",
               "Index of the core_transport model; only models with identifier.index 6 (anomalous) give rows.",
               ids_path="core_transport.model[:]"),
    ColumnSpec("profile_index", "", "Index of the profiles_1d slice of that model.", ids_path=_CT1D),
    ColumnSpec("time_s", "s",
               "The slice's own core_transport.model[:].profiles_1d[:].time (no fallback; slices without one "
               "are dropped).",
               ids_path=f"{_CT1D}.time"),
    ColumnSpec("source", "", f"The {_SOURCE}."),
    ColumnSpec("model_name", "", "The model's identifier.name as stored.",
               ids_path="core_transport.model[:].identifier.name"),
    ColumnSpec("code_name", "", "The model's code.name as stored.", ids_path="core_transport.model[:].code.name"),
    ColumnSpec("code_version", "", "The model's code.version as stored.",
               ids_path="core_transport.model[:].code.version"),
    ColumnSpec("configuration_status", "",
               "Always 'not_standardized': code.parameters has no configuration-only schema, so its hash is not a "
               "configuration identity."),
    ColumnSpec("parameters_text_sha256", "",
               "SHA-256 hex of the model's code.parameters text (UTF-8); None when absent or empty.",
               ids_path="core_transport.model[:].code.parameters"),
    ColumnSpec("rho_grid_min", "-", "Smallest finite grid_flux.rho_tor_norm of the slice.",
               ids_path=f"{_CT1D}.grid_flux.rho_tor_norm"),
    ColumnSpec("rho_grid_max", "-", "Largest finite grid_flux.rho_tor_norm of the slice.",
               ids_path=f"{_CT1D}.grid_flux.rho_tor_norm"),
    ColumnSpec("points_on_grid", "", "Number of finite grid_flux.rho_tor_norm points.",
               ids_path=f"{_CT1D}.grid_flux.rho_tor_norm"),
    ColumnSpec("ion_count", "", "Number of ion entries in the slice.", ids_path=f"{_CT1D}.ion"),
    ColumnSpec("q_e_points", "", "Grid_flux points with a finite electron energy flux (0 when absent).",
               ids_path=f"{_CT1D}.electrons.energy.flux"),
    ColumnSpec("q_i_points", "",
               "Grid_flux points with a finite total ion energy flux (sum over all ion species, only when every "
               "species stores one; 0 otherwise).",
               ids_path=f"{_CT1D}.ion[:].energy.flux"),
    ColumnSpec("gamma_e_points", "", "Grid_flux points with a finite electron particle flux (0 when absent).",
               ids_path=f"{_CT1D}.electrons.particles.flux"),
    ColumnSpec("rho_q_e_min", "-", "Smallest grid_flux rho_tor_norm with a finite electron energy flux.",
               ids_path=f"{_CT1D}.electrons.energy.flux"),
    ColumnSpec("rho_q_e_max", "-", "Largest grid_flux rho_tor_norm with a finite electron energy flux.",
               ids_path=f"{_CT1D}.electrons.energy.flux"),
    ColumnSpec("rho_q_i_min", "-", "Smallest grid_flux rho_tor_norm with a finite total ion energy flux.",
               ids_path=f"{_CT1D}.ion[:].energy.flux"),
    ColumnSpec("rho_q_i_max", "-", "Largest grid_flux rho_tor_norm with a finite total ion energy flux.",
               ids_path=f"{_CT1D}.ion[:].energy.flux"),
    ColumnSpec("rho_gamma_e_min", "-", "Smallest grid_flux rho_tor_norm with a finite electron particle flux.",
               ids_path=f"{_CT1D}.electrons.particles.flux"),
    ColumnSpec("rho_gamma_e_max", "-", "Largest grid_flux rho_tor_norm with a finite electron particle flux.",
               ids_path=f"{_CT1D}.electrons.particles.flux"),
    ColumnSpec("q_e_peak_abs_W_m2", "W m^-2", "Largest |electron energy flux| over finite grid_flux points.",
               formula="vaft.database._summary._peak_abs",
               ids_path=f"{_CT1D}.electrons.energy.flux"),
    ColumnSpec("q_i_peak_abs_W_m2", "W m^-2",
               "Largest |sum over ion species of ion energy flux| over finite grid_flux points.",
               formula="vaft.database._summary._peak_abs",
               ids_path=f"{_CT1D}.ion[:].energy.flux"),
    ColumnSpec("gamma_e_peak_abs_m2_s", "m^-2 s^-1",
               "Largest |electron particle flux| over finite grid_flux points (unit is m^-2 s^-1, not m^2 s^-1 "
               "as the suffix reads).",
               formula="vaft.database._summary._peak_abs",
               ids_path=f"{_CT1D}.electrons.particles.flux"),
)


CLASSICAL_TRANSPORT_SPECS = (
    ColumnSpec("shot", "", "VEST shot number."),
    ColumnSpec("model_index", "",
               "Index of the core_transport model; only models with identifier.index -1 and identifier.name "
               "'classical' give rows.",
               ids_path="core_transport.model[:]"),
    ColumnSpec("profile_index", "", "Index of the profiles_1d slice of that model.", ids_path=_CT1D),
    ColumnSpec("time_s", "s",
               "The slice's own core_transport.model[:].profiles_1d[:].time (no fallback; slices without one "
               "are dropped).",
               ids_path=f"{_CT1D}.time"),
    ColumnSpec("source", "", f"The {_SOURCE}."),
    ColumnSpec("state_identity", "",
               "Resolved-state identity of the slice from the code.parameters classical envelope, the one slice "
               "entry within 1 us of time_s; None otherwise.",
               formula="vaft.database._summary._classical_parameters",
               ids_path="core_transport.model[:].code.parameters"),
    ColumnSpec("efit_lineage", "",
               "EFIT lineage text recorded in the code.parameters classical envelope; None if absent.",
               formula="vaft.database._summary._classical_parameters",
               ids_path="core_transport.model[:].code.parameters"),
    ColumnSpec("model_name", "", "The model's identifier.name as stored ('classical').",
               ids_path="core_transport.model[:].identifier.name"),
    ColumnSpec("code_name", "", "The model's code.name as stored.", ids_path="core_transport.model[:].code.name"),
    ColumnSpec("code_version", "", "The model's code.version as stored.",
               ids_path="core_transport.model[:].code.version"),
    ColumnSpec("configuration_status", "",
               "'recorded' (state identity found), 'state_unrecorded' (envelope present, slice time not listed "
               "once) or 'unrecorded' (no classical envelope in code.parameters).",
               ids_path="core_transport.model[:].code.parameters"),
    ColumnSpec("formulation_sha256", "",
               "Formulation hash recorded in the code.parameters classical envelope; None if absent.",
               formula="vaft.database._summary._classical_parameters",
               ids_path="core_transport.model[:].code.parameters"),
    ColumnSpec("rho_grid_min", "-", "Smallest finite grid_flux.rho_tor_norm of the slice.",
               ids_path=f"{_CT1D}.grid_flux.rho_tor_norm"),
    ColumnSpec("rho_grid_max", "-", "Largest finite grid_flux.rho_tor_norm of the slice.",
               ids_path=f"{_CT1D}.grid_flux.rho_tor_norm"),
    ColumnSpec("points_on_grid", "", "Number of finite grid_flux.rho_tor_norm points.",
               ids_path=f"{_CT1D}.grid_flux.rho_tor_norm"),
    ColumnSpec("main_ion_z", "-",
               "z_ion of ion 0 (charge in elementary charges) when the slice stores exactly one ion, else NaN.",
               ids_path=f"{_CT1D}.ion[0].z_ion"),
    ColumnSpec("q_e_points", "", "Grid_flux points with a finite electron energy flux (0 when absent).",
               ids_path=f"{_CT1D}.electrons.energy.flux"),
    ColumnSpec("q_i_points", "",
               "Grid_flux points with a finite ion-0 energy flux, only when exactly one ion is stored (0 otherwise).",
               ids_path=f"{_CT1D}.ion[0].energy.flux"),
    ColumnSpec("rho_q_e_min", "-", "Smallest grid_flux rho_tor_norm with a finite electron energy flux.",
               ids_path=f"{_CT1D}.electrons.energy.flux"),
    ColumnSpec("rho_q_e_max", "-", "Largest grid_flux rho_tor_norm with a finite electron energy flux.",
               ids_path=f"{_CT1D}.electrons.energy.flux"),
    ColumnSpec("rho_q_i_min", "-", "Smallest grid_flux rho_tor_norm with a finite ion-0 energy flux.",
               ids_path=f"{_CT1D}.ion[0].energy.flux"),
    ColumnSpec("rho_q_i_max", "-", "Largest grid_flux rho_tor_norm with a finite ion-0 energy flux.",
               ids_path=f"{_CT1D}.ion[0].energy.flux"),
    ColumnSpec("q_e_peak_abs_W_m2", "W m^-2", "Largest |electron energy flux| over finite grid_flux points.",
               formula="vaft.database._summary._peak_abs",
               ids_path=f"{_CT1D}.electrons.energy.flux"),
    ColumnSpec("q_i_peak_abs_W_m2", "W m^-2",
               "Largest |ion-0 energy flux| over finite grid_flux points, only when exactly one ion is stored.",
               formula="vaft.database._summary._peak_abs",
               ids_path=f"{_CT1D}.ion[0].energy.flux"),
    ColumnSpec("chi_e_max_m2_s", "m^2 s^-1",
               "Largest |electrons.energy.d| (the model's electron heat diffusivity) over finite grid_d points.",
               formula="vaft.database._summary._peak_abs",
               ids_path=f"{_CT1D}.electrons.energy.d"),
    ColumnSpec("chi_i_max_m2_s", "m^2 s^-1",
               "Largest |ion[0].energy.d| over finite grid_d points, only when exactly one ion is stored.",
               formula="vaft.database._summary._peak_abs",
               ids_path=f"{_CT1D}.ion[0].energy.d"),
    ColumnSpec("particle_flux_status", "",
               "'stored' when any electron or ion particles.flux is stored in the slice, else 'not_evaluated'.",
               ids_path=f"{_CT1D}.electrons.particles.flux"),
)


__all__ = [
    "CLASSICAL_TRANSPORT_SPECS",
    "EFIT_RELIABILITY_SPECS",
    "NEOCLASSICAL_SPECS",
    "TURBULENT_TRANSPORT_SPECS",
    "VOLUME_AVERAGED_SPECS",
]
