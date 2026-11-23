"""Column specs of the shot-overview, equilibrium-global and core-profile presets.

Each spec says what the stored value is, read from the extractor in
:mod:`vaft.database._summary` rather than from the column name. A
``quantity`` is set only where the column *is* that registered quantity in the
same unit; a column that is close to one (``beta_tor`` as a fraction, the
``toroidal_beta`` quantity in percent) carries none.
"""

from __future__ import annotations

from ._summary_contract import ColumnSpec as C

_EQ = "equilibrium.time_slice[:]"
_GQ = f"{_EQ}.global_quantities"
_BND = f"{_EQ}.boundary"
_SUMMARY = "vaft.database._summary"
_FORMULA = "vaft.formula.equilibrium"
_VIRIAL = "vaft.omas.process_wrapper.compute_virial_equilibrium_quantities_ods"
_B_PA = ("normalised with B_pa, the arc-length average of |B_p| on the boundary outline "
         "(mu0 I_p / L_pol when the outline is a flux surface)")

SHOT = C("shot", "", "VEST shot number.", ids_path="dataset_description.data_entry.pulse")
PULSE_TIME_BEGIN = C(
    "pulse_time_begin", "",
    "Acquisition timestamp of the shot as stored (ISO 8601 text, may be empty for older archives).",
    ids_path="dataset_description.data_entry.pulse_time_begin",
)


def _virial(name, unit, definition, key, quantity=None):
    return C(name, unit, definition, quantity=quantity, formula=_VIRIAL, ids_path=f"{_EQ}.profiles_2d[0].psi ({key})")


SHOT_OVERVIEW_SPECS = (
    SHOT,
    PULSE_TIME_BEGIN,
    C("plasma_onset_time_s", "s",
      "Start of the plasma window found by vaft.omas.plasma_timing (H-alpha by label, plasma current as fallback); "
      "NaN when no window is found.",
      formula="vaft.omas.plasma_timing.plasma_timing", ids_path="spectrometer_uv / magnetics"),
    C("plasma_onset_source", "",
      "Signal that set the plasma window ('none' when no window was found).",
      formula="vaft.omas.plasma_timing.plasma_timing"),
    C("pulse_duration_s", "s",
      "Window end minus window start; the window is the envelope of the detector's segments, so gaps inside it "
      "are counted.", formula="vaft.omas.plasma_timing.plasma_timing"),
    C("pulse_duty_cycle", "-",
      "Fraction of the window covered by the detected segments (1 for a single segment).",
      formula="vaft.omas.plasma_timing.plasma_timing"),
    C("max_ip_kA", "kA",
      "Representative peak plasma current inside the plasma window (measured, not reconstructed); NaN when no "
      "peak is found.", formula="vaft.omas.plasma_features.ip_peak", ids_path="magnetics.ip[0].data"),
    C("mean_b_t_T", "T",
      "Mean over the plasma window of the vacuum toroidal field at tf.r0, tf.b_field_tor_vacuum_r / tf.r0 "
      "(signed as stored).", ids_path="tf.b_field_tor_vacuum_r.data"),
    C("shot_class", "",
      "Shot class label from vaft.omas.shot_class, decided from the plasma timing and the gas response; says why "
      "when there is no plasma window.", formula="vaft.omas.shot_class.shot_class"),
)


EQUILIBRIUM_GLOBAL_SPECS = (
    SHOT,
    PULSE_TIME_BEGIN,
    C("eq_index", "", "Index of the equilibrium time slice in the ODS.", ids_path=_EQ),
    C("time_s", "s", "Time of the equilibrium slice (time_slice.time, else equilibrium.time).",
      ids_path=f"{_EQ}.time"),
    C("ip_kA", "kA", "Reconstructed plasma current, signed as stored (COCOS of the file).",
      ids_path=f"{_GQ}.ip"),
    C("psi_axis_Wb", "Wb",
      "Poloidal flux at the magnetic axis in Wb (not Wb/rad): legacy Wb/rad files are converted.",
      formula="vaft.data.eqdsk.ods_psi_to_wb_per_radian_factor", ids_path=f"{_GQ}.psi_axis"),
    C("q_axis", "-", "Safety factor on the magnetic axis.", ids_path=f"{_GQ}.q_axis"),
    C("q_95", "-", "Safety factor at the 95 % poloidal-flux surface, signed as stored (COCOS of the file).", quantity="edge_safety_factor_95",
      ids_path=f"{_GQ}.q_95"),
    C("q_min", "-", "Safety factor of smallest magnitude, sign kept.", ids_path=f"{_GQ}.q_min.value"),
    C("beta_pol", "-",
      "DD poloidal beta 4 int p dV / (R_0 mu0 I_p^2), R_0 the resolved reference major radius (not the database r0). Stored value kept on a slice the updater cannot recompute.",
      formula=f"{_FORMULA}.beta_poloidal_from_pressure_integral", ids_path=f"{_GQ}.beta_pol"),
    C("beta_pol_circumference", "-",
      f"EFIT/OMFIT poloidal beta 2 mu0 <p>_V / B_pa^2, {_B_PA}; differs from beta_pol by L_pol^2 R_0 / (2V).",
      formula=f"{_FORMULA}.beta_poloidal_from_circumference", ids_path=f"{_EQ}.profiles_1d.pressure"),
    C("beta_tor", "-",
      "Toroidal beta 2 mu0 <p>_V / B_0^2 as a fraction, B_0 the vacuum field at the reference radius rebuilt "
      "from f at the boundary. Stored value kept on a slice the updater cannot recompute.",
      formula=f"{_FORMULA}.beta_toroidal_from_p_B0", ids_path=f"{_GQ}.beta_tor"),
    C("beta_normal", "% m T/MA",
      "Normalised beta 100 beta_tor a |B_0| / |I_p[MA]|, a the boundary minor radius and B_0 the vacuum field at "
      "the reference radius. Stored value kept on a slice the updater cannot recompute.", quantity="normalized_beta",
      formula=f"{_FORMULA}.beta_normal_from_beta_tor", ids_path=f"{_GQ}.beta_normal"),
    C("beta_volume_B2", "-",
      "Volume beta 2 mu0 int p dV / int B^2 dV inside the boundary outline with the total field (B_phi = F/R), "
      "as a fraction; NaN without an outline.",
      formula=f"{_FORMULA}.beta_volume_from_p_B2", ids_path=f"{_EQ}.profiles_1d.pressure"),
    C("beta_normal_B2", "% m T/MA",
      "100 beta_volume_B2 a |B_0| / |I_p[MA]| with the same a and B_0 as beta_normal (Menard's <beta_N> form, "
      "B_0 at the reference radius, not at the geometric centre).", quantity="normalized_volume_beta",
      formula=f"{_FORMULA}.beta_normal_from_beta_volume"),
    C("normalized_plasma_current", "MA m^-1 T^-1",
      "|I_p[MA]| / (a |B_0|), a the boundary minor radius and B_0 the vacuum field at the reference radius.",
      quantity="normalized_current", formula=f"{_FORMULA}.normalized_plasma_current"),
    C("li_3", "-",
      "Internal inductance 2 int B_p^2 dV / (mu0^2 I_p^2 R_0), R_0 the resolved reference major radius (DD form). Stored value kept on a slice the updater cannot recompute.",
      quantity="internal_inductance_li3", formula=f"{_FORMULA}.li_3_from_Bp2_volume_integral",
      ids_path=f"{_GQ}.li_3"),
    C("energy_mhd_J", "J", "Stored thermal energy (3/2) int p dV of the reconstructed pressure.",
      formula="vaft.formula.equilibrium.thermal_energy_from_p_V", ids_path=f"{_GQ}.energy_mhd"),
    C("area_m2", "m^2", "Poloidal cross-section area inside the boundary.", ids_path=f"{_GQ}.area"),
    C("volume_m3", "m^3", "Plasma volume inside the boundary.", ids_path=f"{_GQ}.volume"),
    C("major_radius_m", "m", "Geometric major radius (R_max + R_min)/2 of the boundary.", quantity="major_radius",
      ids_path=f"{_BND}.geometric_axis.r"),
    C("minor_radius_m", "m", "Minor radius (R_max - R_min)/2 of the boundary.", quantity="minor_radius",
      ids_path=f"{_BND}.minor_radius"),
    C("aspect_ratio", "-", "major_radius_m / minor_radius_m.", quantity="aspect_ratio"),
    C("elongation", "-", "Boundary elongation.", quantity="elongation", ids_path=f"{_BND}.elongation"),
    C("triangularity", "-", "Boundary triangularity, mean of upper and lower.", quantity="triangularity",
      ids_path=f"{_BND}.triangularity"),
    C("triangularity_upper", "-", "Upper boundary triangularity.", ids_path=f"{_BND}.triangularity_upper"),
    C("triangularity_lower", "-", "Lower boundary triangularity.", ids_path=f"{_BND}.triangularity_lower"),
    C("magnetic_axis_r_m", "m", "Major radius of the magnetic axis.", ids_path=f"{_GQ}.magnetic_axis.r"),
    C("magnetic_axis_z_m", "m", "Height of the magnetic axis.", ids_path=f"{_GQ}.magnetic_axis.z"),
    C("magnetic_axis_btor_T", "T", "Total toroidal field on the magnetic axis (includes the plasma's diamagnetism).",
      ids_path=f"{_GQ}.magnetic_axis.b_field_tor"),
    C("vacuum_b0_T", "T",
      "Vacuum toroidal field referred to R = 0.4 m: b0 (rebuilt from f at the boundary) times r0 / 0.4, signed as "
      "stored.", ids_path="equilibrium.vacuum_toroidal_field.b0"),
    C("vacuum_r0_m", "m", "The radius vacuum_b0_T refers to: the constant 0.4 m.",
      ids_path="equilibrium.vacuum_toroidal_field.r0"),
    C("dia_flux_Wb", "Wb", "Reconstructed diamagnetic flux after the diamagnetic-constraint update.",
      formula="vaft.omas.update_equilibrium_constraints_diamagnetic_flux",
      ids_path=f"{_EQ}.constraints.diamagnetic_flux.reconstructed"),
    _virial("virial_s1", "-", "Shafranov integral S_1 over the boundary outline.", "s_1"),
    _virial("virial_s2", "-", "Shafranov integral S_2 over the boundary outline.", "s_2"),
    _virial("virial_s3", "-", "Shafranov integral S_3 over the boundary outline.", "s_3"),
    _virial("virial_alpha", "-", "Shape factor alpha of the virial closures (volume method).", "alpha"),
    _virial("virial_b_pa_T", "T", "Boundary-averaged poloidal field B_pa (line average over the outline).", "B_pa"),
    _virial("virial_mui", "-", "Diamagnetism mu_i used by the closures (approximated mu_hat_i when not measured).",
            "mui"),
    _virial("virial_rt_m", "m", "Current-centroid (virial) radius R_T; it enters the closures through R_T / R_0.", "rt"),
    _virial("virial_phi_dia_comp_Wb", "Wb", "Diamagnetic flux the virial mu_i is built on.", "phi_dia_comp"),
    _virial("virial_volume_m3", "m^3", "Plasma volume inside the virial boundary outline.", "V_p"),
    _virial("virial_beta_pd_vir", "-", f"Diamagnetic poloidal beta beta_pd from mu_i, {_B_PA}.", "beta_pd_vir"),
    _virial("virial_beta", "-", f"Alias of virial_beta_lao: poloidal beta, {_B_PA}.", "beta_p_vir_lao"),
    _virial("virial_li", "-", f"Alias of virial_li_lao: internal inductance, {_B_PA} (l_i(1) convention).",
            "li_vir_lao"),
    _virial("virial_beta_lao", "-", f"Lao (1985) closure poloidal beta from S_1, S_2 and mu_i, {_B_PA}.",
            "beta_p_vir_lao"),
    _virial("virial_li_lao", "-", f"Lao (1985) closure internal inductance, {_B_PA} (l_i(1) convention).",
            "li_vir_lao"),
    _virial("virial_beta_bongard", "-", f"Bongard (2016) closure poloidal beta, {_B_PA}.", "beta_p_vir_bongard"),
    _virial("virial_li_bongard", "-", f"Bongard (2016) closure internal inductance, {_B_PA}.", "li_vir_bongard"),
    _virial("virial_beta_pair_12", "-", f"Poloidal beta closed from identities 1 and 2, {_B_PA}.", "pair_12.beta_p"),
    _virial("virial_li_pair_12", "-", f"Internal inductance closed from identities 1 and 2, {_B_PA}.", "pair_12.li"),
    _virial("virial_beta_pair_13", "-", f"Poloidal beta closed from identities 1 and 3, {_B_PA}.", "pair_13.beta_p"),
    _virial("virial_li_pair_13", "-", f"Internal inductance closed from identities 1 and 3, {_B_PA}.", "pair_13.li"),
    _virial("virial_beta_pair_23", "-", f"Poloidal beta closed from identities 2 and 3, {_B_PA}.", "pair_23.beta_p"),
    _virial("virial_li_pair_23", "-", f"Internal inductance closed from identities 2 and 3, {_B_PA}.", "pair_23.li"),
    _virial("virial_beta_full_123", "-", f"Poloidal beta from all three identities together, {_B_PA}.",
            "full_123.beta_p"),
    _virial("virial_li_full_123", "-", f"Internal inductance from all three identities together, {_B_PA}.",
            "full_123.li"),
    _virial("virial_mui_full_123", "-", "Diamagnetism mu_i solved from all three identities together.",
            "full_123.mu_i"),
    _virial("virial_beta_volume", "-",
            f"Poloidal beta from the equilibrium's own volume integrals, {_B_PA}; close to beta_pol_circumference "
            "where the outline is a flux surface.", "volume.beta_p"),
    _virial("virial_li_volume", "-",
            f"Internal inductance <B_p^2>_V / B_pa^2 from the equilibrium's own volume integrals: l_i(1) where "
            "the outline is a flux surface.",
            "volume.li", quantity="internal_inductance_li1"),
    _virial("virial_mui_measured", "-", "Diamagnetism mu_i from the measured diamagnetic flux.",
            "mu_i_sources.measured"),
    _virial("virial_residual_e1", "-", "Normalised residual of virial identity 1 on the equilibrium's integrals.",
            "identity.e1_normalized"),
    _virial("virial_residual_e2", "-", "Normalised residual of virial identity 2 on the equilibrium's integrals.",
            "identity.e2_normalized"),
    _virial("virial_residual_e3", "-", "Normalised residual of virial identity 3 on the equilibrium's integrals.",
            "identity.e3_normalized"),
    _virial("virial_residual_rms", "-", "RMS of the three normalised identity residuals.", "identity.rms"),
    _virial("virial_omitted_residual_pair_12", "-",
            "Normalised residual of identity 3, left out of the 1-2 closure.", "pair_12.residual_normalized"),
    _virial("virial_omitted_residual_pair_13", "-",
            "Normalised residual of identity 2, left out of the 1-3 closure.", "pair_13.residual_normalized"),
    _virial("virial_omitted_residual_pair_23", "-",
            "Normalised residual of identity 1, left out of the 2-3 closure.", "pair_23.residual_normalized"),
    _virial("virial_rt_denominator_ratio", "-",
            "Conditioning of the R_T-dependent closures (denominator ratio); small values mean they are ill-posed.",
            "conditioning.rt_denominator_ratio"),
)


_CP = "core_profiles.profiles_1d[:]"
_TAU = "vaft.omas.formula_wrapper.compute_confiment_time_paramters"
_ENG = "vaft.omas.formula_wrapper.compute_tau_E_engineering_parameters"
_OHMIC = "vaft.omas.confinement.ohmic_confinement_ods"
_LEGACY_INPUTS = ("on the legacy engineering inputs (|I_p|, B_t at 0.4 m, R, epsilon, kappa, line-averaged n_e, "
                  "p_loss_MW, M = 1)")

CORE_PROFILES_SPECS = (
    SHOT,
    PULSE_TIME_BEGIN,
    C("cp_index", "", "Index of the core_profiles slice.", ids_path=_CP),
    C("eq_index", "", "Index of the equilibrium slice at the same time; the engineering inputs require the two times to "
      "agree to 1e-9 s and the row is skipped otherwise.",
      formula="vaft.omas.general.find_matching_time_indices", ids_path=_EQ),
    C("time_s", "s", "Time shared by the core_profiles slice and its equilibrium slice.",
      formula="vaft.omas.general.find_matching_time_indices", ids_path=f"{_CP}.time"),
    C("ip_kA", "kA", "|I_p| of the matched equilibrium slice (reconstructed, magnitude).", formula=_ENG,
      ids_path=f"{_GQ}.ip"),
    C("b_t_T", "T",
      "|B_phi R| on the magnetic axis referred to R = 0.4 m: |b_field_tor(axis) R_axis| / 0.4 (includes the "
      "plasma's diamagnetism).", formula=_ENG, ids_path=f"{_GQ}.magnetic_axis.b_field_tor"),
    C("p_loss_MW", "MW",
      "Legacy modelled loss power: Spitzer int eta J^2 dV (Z_eff = 2, ln Lambda = 17) - dW/dt - modelled P_rad, "
      "W = (3/2) <p> V with p = 2 n_e T_e.", formula=_ENG),
    C("tau_e_s", "s", "Legacy W_th / p_loss on that modelled budget.", formula=_TAU),
    C("tau_e_ipb89_s", "s", f"ITER89-P L-mode scaling prediction {_LEGACY_INPUTS}.", formula=_TAU),
    C("tau_e_h98y2_s", "s", f"IPB98(y,2) H-mode scaling prediction {_LEGACY_INPUTS}.", formula=_TAU),
    C("tau_e_nstx_s", "s", f"NSTX scaling prediction {_LEGACY_INPUTS}.", formula=_TAU),
    C("tau_e_nstx2006l_s", "s", f"NSTX 2006 L-mode scaling prediction {_LEGACY_INPUTS}.", formula=_TAU),
    C("tau_e_kurskiev2022_s", "s", f"Kurskiev (2022) spherical-tokamak scaling prediction {_LEGACY_INPUTS}.",
      formula=_TAU),
    C("v_res_V", "V",
      "Inferred resistive loop voltage: measured inboard-midplane flux-loop voltage minus the internal-inductive "
      "voltage.", formula=_OHMIC),
    C("p_ohm_MW", "MW", "Ohmic power: measured I_p times v_res_V.", formula=_OHMIC),
    C("p_net_MW", "MW", "p_ohm_MW - dW_MHD/dt; radiation not subtracted.", formula=_OHMIC),
    C("w_mhd_kJ", "kJ", "(3/2) int p_EFIT dV of the matched equilibrium slice.", formula=_OHMIC),
    C("tau_e_net_s", "s",
      "w_mhd_kJ / p_net_MW; dW/dt from a 3 ms local linear fit over every slice with positive W; NaN without a "
      "rate.", formula=_OHMIC),
    C("ne_line_1e19_m3", "1e19 m^-3",
      "Line-averaged electron density along the z = 0 midplane chord through the matched equilibrium.",
      quantity="line_average_density", formula=_ENG, ids_path=f"{_CP}.electrons.density"),
    C("ne_volume_1e19_m3", "1e19 m^-3", "Volume-averaged electron density inside the boundary.", formula=_ENG,
      ids_path=f"{_CP}.electrons.density"),
    C("te_mean_eV", "eV",
      "Unweighted mean of the electron-temperature profile over its grid points (not a volume average).",
      ids_path=f"{_CP}.electrons.temperature"),
    C("major_radius_m", "m", "Geometric major radius of the matched equilibrium boundary.", quantity="major_radius",
      formula=_ENG, ids_path=f"{_BND}.geometric_axis.r"),
    C("inverse_aspect_ratio", "-", "Boundary minor radius over geometric major radius.", formula=_ENG),
    C("elongation", "-", "Boundary elongation of the matched equilibrium slice.", quantity="elongation",
      formula=_ENG, ids_path=f"{_BND}.elongation"),
)


__all__ = ["CORE_PROFILES_SPECS", "EQUILIBRIUM_GLOBAL_SPECS", "PULSE_TIME_BEGIN", "SHOT", "SHOT_OVERVIEW_SPECS"]
