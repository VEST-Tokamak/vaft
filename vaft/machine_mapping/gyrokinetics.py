"""Project CGYRO results into IMAS ``gyrokinetics_local`` and ``core_transport`` (#1354).

The rule is the one :mod:`vaft.machine_mapping.neoclassical` and
:mod:`~vaft.machine_mapping.turbulence` follow: a quantity reaches an IDS only when the
correspondence survives an audit by physical definition. Each one is classified in
:data:`MAPPING_AUDIT` as ``exact``, ``unit`` (a rescaling by a known factor),
``coordinate`` (a change of coordinate), ``derived`` (computed from several native
quantities), ``convention`` (exact up to a sign or phase convention that is stated) or
``unsupported`` (kept in the native result only, with the reason). Nothing absent is
read back out of an ODS to fill a gap.

**The IDS is ``gyrokinetics_local``.** DD 3.41 -- omas's last -- calls it that, not
``gyrokinetics``, and it holds *one* flux-tube simulation: no surface axis, no state
axis. One ODS per (state, surface, field model) is the shape, which is why
:func:`gyrokinetics_local_from_cgyro` writes a whole IDS rather than one entry.

**Its normalisation is not CGYRO's.** The IDS follows the GKDB convention: lengths by
``L_ref = R0`` (``normalizing_quantities.r``, the surface's ``(Rmax+Rmin)/2``), field by
``B_ref = B_tor(R0)``, ``T_ref = T_e``, ``n_ref = n_e``, ``m_ref = m_D``,
``v_thref = sqrt(2 T_ref/m_ref)`` and ``rho_ref = m_ref v_thref/(e B_ref)``. CGYRO uses
``a``, ``B_unit`` and ``c_s = sqrt(T_e/m_D)``. With ``RMAJ = R0/a`` and
``b_gs2 = B_tor(R0)/B_unit`` -- which CGYRO computes itself from its geometry
(``cgyro_equilibrium.F90``: ``b_gs2 = geo_f/rmaj``) and writes to
``out.cgyro.equilibrium`` -- every conversion is one of::

    rate:       x_ref  = x_cgyro * RMAJ / sqrt(2)                 (gamma, omega)
    time:       t_ref  = t_cgyro * sqrt(2) / RMAJ
    wavenumber: k rho_ref = k rho_s * sqrt(2) / b_gs2             (ky)
    beta:       beta_ref = BETAE_UNIT / b_gs2**2
    gradient:   R0/L = a/L * RMAJ
    GB flux:    F_ref = F_cgyro * RMAJ**2 * b_gs2**2 / (2*sqrt(2))

**The frequency sign is a stated convention, not a DD one.** The DD 3.41 schema does not
document the sign of ``frequency_norm``. CGYRO's native sign depends on the field
orientation (``cgyro_make_profiles.F90`` prints which); the value written here has the
**ion diamagnetic direction negative**, TGLF's convention, and that statement is in the
eigenmode's ``code.parameters``.

What stays native, and why, is listed in :data:`MAPPING_AUDIT`; the main cases are the
Miller triangularity and squareness (the DD carries a Fourier shape instead, which needs
a fit), the collision matrix (the DD's species-pair definition is not CGYRO's
``nu_ee`` scaling) and the quasilinear weights (their amplitude normalisation is not
reproduced here).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional
from xml.sax.saxutils import escape

import numpy as np
from omas import ODS

from vaft.ods_access import path_count

IDS = "gyrokinetics_local"
CGYRO_REPOSITORY = "https://github.com/gafusion/gacode"

#: The sign convention of every frequency this module writes.
FREQUENCY_SIGN_CONVENTION = "ion_diamagnetic_negative"

#: The fields CGYRO's N_FIELD selects, and their DD names.
_FIELD_DD = ("phi_potential", "a_field_parallel", "b_field_parallel")

#: core_transport model enumeration: anomalous (turbulent) transport.
ANOMALOUS_MODEL_INDEX = 6
ANOMALOUS_MODEL_NAME = "anomalous"
CGYRO_MODEL_DESCRIPTION = (
    "Turbulent transport, from a nonlinear local CGYRO gyrokinetic simulation"
)

#: How each quantity reaches IMAS, keyed by the DD path below ``gyrokinetics_local``.
MAPPING_AUDIT: Mapping[str, tuple[str, str]] = {
    "model.include_a_field_parallel": ("exact", "N_FIELD >= 2"),
    "model.include_b_field_parallel": ("exact", "N_FIELD >= 3"),
    "model.adiabatic_electrons": ("exact", "kinetic electrons (AE_FLAG=0)"),
    "model.include_full_curvature_drift": ("exact", "CGYRO's drift is the full one"),
    "model.include_coriolis_drift": ("exact", "no rotation in the run (MACH=0)"),
    "model.include_centrifugal_effects": ("exact", "no rotation in the run (MACH=0)"),
    "model.collisions_*": (
        "exact", "COLLISION_MODEL=4 (Sugama, momentum+energy restoring, k_perp); "
        "other operators are left unwritten"),
    "species[:].charge_norm": ("exact", "Z"),
    "species[:].mass_norm": ("exact", "MASS, already in deuterium units"),
    "species[:].density_norm": ("exact", "DENS, by n_e"),
    "species[:].temperature_norm": ("exact", "TEMP, by T_e"),
    "species[:].density_log_gradient_norm": ("unit", "a/L_n * RMAJ"),
    "species[:].temperature_log_gradient_norm": ("unit", "a/L_T * RMAJ"),
    "species[:].velocity_tor_gradient_norm": ("exact", "0: no rotation in the run"),
    "species_all.beta_reference": ("unit", "BETAE_UNIT / b_gs2**2"),
    "species_all.debye_length_norm": ("unit", "LAMBDA_STAR * b_gs2 / sqrt(2)"),
    "species_all.shearing_rate_norm": ("exact", "0: GAMMA_E=0 in the run"),
    "species_all.velocity_tor_norm": ("exact", "0: MACH=0 in the run"),
    "flux_surface.r_minor_norm": ("unit", "RMIN / RMAJ"),
    "flux_surface.q": ("exact", "|q|; the orientation is in ip_sign/b_field_tor_sign"),
    "flux_surface.magnetic_shear_r_minor": ("exact", "S = (r/q) dq/dr"),
    "flux_surface.elongation": ("exact", "Miller kappa = (Zmax-Zmin)/(Rmax-Rmin)"),
    "flux_surface.delongation_dr_minor_norm": ("derived", "KAPPA * S_KAPPA / r_minor_norm"),
    "flux_surface.dgeometric_axis_r_dr_minor": ("exact", "SHIFT = dR0/dr"),
    "flux_surface.dgeometric_axis_z_dr_minor": ("exact", "DZMAG = dZ0/dr"),
    "flux_surface.ip_sign": ("convention", "IPCCW (counter-clockwise from above = +phi, COCOS 11)"),
    "flux_surface.b_field_tor_sign": ("convention", "BTCCW (as ip_sign)"),
    "flux_surface.shape_coefficients_*": (
        "unsupported", "the run uses Miller DELTA/ZETA; the DD's Fourier shape needs a fit"),
    "flux_surface.pressure_gradient_norm": (
        "unsupported", "the DD's pressure normalisation is not reproduced here; CGYRO "
        "rebuilds beta_star from BETAE_UNIT and the gradients"),
    "collisions.collisionality_norm": (
        "unsupported", "the DD's species-pair collision frequency is not CGYRO's nu_ee "
        "scaling; NU_EE stays native"),
    "normalizing_quantities.r": ("derived", "RMAJ * a [m]"),
    "normalizing_quantities.b_field_tor": ("derived", "|b_gs2 * B_unit| [T]"),
    "normalizing_quantities.t_e": ("unit", "T_e [eV]"),
    "normalizing_quantities.n_e": ("unit", "n_e [m^-3]"),
    "linear.wavevector[:].binormal_wavevector_norm": ("unit", "ky rho_s * sqrt(2) / b_gs2"),
    "linear.wavevector[:].radial_wavevector_norm": ("exact", "0: ballooning mode at theta_0 = 0"),
    "linear.wavevector[:].eigenmode[:].growth_rate_norm": ("unit", "gamma * RMAJ / sqrt(2)"),
    "linear.wavevector[:].eigenmode[:].frequency_norm": (
        "convention", "omega * RMAJ / sqrt(2), ion diamagnetic direction negative"),
    "linear.wavevector[:].eigenmode[:].growth_rate_tolerance": ("exact", "FREQ_TOL"),
    "linear.wavevector[:].eigenmode[:].angle_pol": (
        "coordinate", "Miller ballooning angle mapped to the DD's geometric angle about "
        "(R0, Z0), clockwise"),
    "linear.wavevector[:].eigenmode[:].fields.phi_potential_perturbed_norm": (
        "convention", "final-time phi, divided by its value at angle_pol=0 (amplitude "
        "and phase of a linear eigenmode are arbitrary)"),
    "linear.wavevector[:].eigenmode[:].fields.a_field_parallel_perturbed_norm": (
        "unsupported", "the A_par/phi unit ratio between the normalisations is not "
        "reproduced; kept native"),
    "linear.wavevector[:].eigenmode[:].linear_weights": (
        "unsupported", "quasilinear weight amplitude normalisation not reproduced"),
    "non_linear.fluxes_1d.{particles,energy}_*": (
        "unit", "time-averaged GB flux * RMAJ**2 * b_gs2**2 / (2 sqrt 2), per field"),
    "non_linear.fluxes_2d_k_x_sum.{particles,energy}_*": (
        "unit", "the same time average per toroidal mode (ky), on "
        "non_linear.binormal_wavevector_norm"),
    "non_linear.fluxes_1d.momentum_*": (
        "unsupported", "CGYRO's toroidal momentum flux is not split parallel/perpendicular"),
}

__all__ = [
    "TGLF_MAPPING_AUDIT",
    "gyrokinetics_local_from_tglf",
    "ANOMALOUS_MODEL_INDEX",
    "CGYRO_MODEL_DESCRIPTION",
    "FREQUENCY_SIGN_CONVENTION",
    "IDS",
    "MAPPING_AUDIT",
    "conversion_factors",
    "core_transport_from_cgyro",
    "geometric_angle",
    "gyrokinetics_local_from_cgyro",
    "merge_linear_scan",
]


def conversion_factors(local: Any, outputs: Any) -> Optional[dict[str, float]]:
    """The CGYRO -> DD rescalings for one run, or ``None`` when they cannot be formed.

    ``RMAJ`` comes from the input; ``b_gs2`` only from CGYRO's own equilibrium file,
    because it is a flux-surface average of the geometry CGYRO built and recomputing it
    here would be a second, divergent geometry.
    """
    equilibrium = getattr(outputs, "equilibrium", None) or {}
    return _factors(float(local.geometry["RMAJ"]), equilibrium.get("b_gs2"))


def _factors(rmaj: float, b_gs2: Any) -> Optional[dict[str, float]]:
    """The rescalings for ``RMAJ = R0/a`` and ``b_gs2 = B_tor(R0)/B_unit``."""
    if b_gs2 is None or not np.isfinite(b_gs2) or b_gs2 == 0.0 or rmaj <= 0.0:
        return None
    b = abs(float(b_gs2))
    return {
        "rmaj": rmaj,
        "b_gs2": b,
        "rate": rmaj / np.sqrt(2.0),
        # a time is the inverse of a rate: t_ref = t * (a/c_s) * (v_thref/R0)
        "time": np.sqrt(2.0) / rmaj,
        "wavenumber": np.sqrt(2.0) / b,
        "beta": 1.0 / b**2,
        "debye": b / np.sqrt(2.0),
        "flux": rmaj**2 * b**2 / (2.0 * np.sqrt(2.0)),
    }


def geometric_angle(theta: Any, local: Any) -> np.ndarray:
    """Miller angle (extended) -> the DD's geometric poloidal angle, clockwise.

    Miller: ``R = R0 + r cos(theta + arcsin(delta) sin theta)``,
    ``Z = Z0 + kappa r sin(theta + zeta sin 2 theta)``. The DD angle is
    ``atan2(Z - Z0, R - R0)`` measured *clockwise* in the (R right, Z up) view -- the
    direction that makes ``(r, theta, phi)`` right-handed under COCOS 11 -- so it is the
    negative of the counter-clockwise ``atan2``. Each ``2 pi`` of the extended
    ballooning angle is carried over unchanged.
    """
    theta = np.asarray(theta, dtype=float)
    g = local.geometry
    r = float(g["RMIN"])
    turns = np.round(theta / (2.0 * np.pi))
    base = theta - 2.0 * np.pi * turns
    x = np.arcsin(np.clip(float(g["DELTA"]), -1.0, 1.0))
    dr = r * np.cos(base + x * np.sin(base))
    dz = float(g["KAPPA"]) * r * np.sin(base + float(g["ZETA"]) * np.sin(2.0 * base))
    return -(np.arctan2(dz, dr) + 2.0 * np.pi * turns)


def _xml_parameters(entries: Mapping[str, Any], tag: str = "cgyro") -> str:
    body = "".join(
        f"<{key}>{escape(str(value))}</{key}>" for key, value in entries.items()
        if value is not None
    )
    return f"<parameters><{tag}>{body}</{tag}></parameters>"


def _write_code(ods: ODS, base: str, provenance: Mapping[str, Any]) -> None:
    version = provenance.get("version") or {}
    ods[f"{base}.code.name"] = "CGYRO"
    ods[f"{base}.code.repository"] = CGYRO_REPOSITORY
    commit = provenance.get("gacode_commit")
    if commit:
        ods[f"{base}.code.commit"] = str(commit)
    revision = version.get("revision") if isinstance(version, Mapping) else None
    if revision:
        ods[f"{base}.code.version"] = str(revision).replace("<", "").replace(">", "")
    formalism = provenance.get("formalism") or {}
    resolution = provenance.get("resolution") or {}
    ods[f"{base}.code.parameters"] = _xml_parameters(
        {
            **{f"formalism_{k}": v for k, v in formalism.items()},
            **{f"resolution_{k}": v for k, v in resolution.items()},
            "input_sha256": provenance.get("input_sha256"),
            "state_key": provenance.get("state_key"),
            "frequency_sign_convention": FREQUENCY_SIGN_CONVENTION,
            "normalisation": "GKDB: L_ref=R0, B_ref=Btor(R0), v_thref=sqrt(2Te/mD)",
        }
    )


def gyrokinetics_local_from_cgyro(
    ods: ODS,
    local: Any,
    outputs: Any,
    *,
    provenance: Optional[Mapping[str, Any]] = None,
    time: Optional[float] = None,
    flux_window: Optional[tuple[float, float]] = None,
) -> dict[str, Any]:
    """Write one CGYRO run into ``ods['gyrokinetics_local']``.

    Parameters
    ----------
    local
        The :class:`~vaft.code.gacode.cgyro.inputs.CGYROInput` the run was staged from.
    outputs
        Its :class:`~vaft.code.gacode.cgyro.outputs.CgyroOutputs`.
    provenance
        ``CGYROResult.provenance``; feeds ``code.*`` and the stated conventions.
    time
        The equilibrium time [s] of the state, for ``gyrokinetics_local.time``.
    flux_window
        ``(t0, t1)`` in ``a/c_s`` over which a nonlinear run's fluxes are averaged. A
        nonlinear run without one writes no fluxes and says so: choosing the window is a
        judgement about saturation this layer does not make.

    Returns
    -------
    dict
        ``{"written": [paths], "skipped": [reasons]}``.
    """
    written: list[str] = []
    skipped: list[str] = []
    provenance = dict(provenance or {})
    base = IDS

    def put(path: str, value: Any) -> None:
        ods[f"{base}.{path}"] = value
        written.append(path)

    factors = conversion_factors(local, outputs)
    if factors is None:
        return {
            "written": written,
            "skipped": ["no b_gs2 in out.cgyro.equilibrium: the DD normalisation "
                        "cannot be formed, nothing written"],
        }

    ods[f"{base}.ids_properties.homogeneous_time"] = 1
    ods[f"{base}.ids_properties.comment"] = (
        "CGYRO local delta-f flux tube; normalised per GKDB (L_ref=R0); frequency sign: "
        + FREQUENCY_SIGN_CONVENTION
    )
    if time is not None:
        ods[f"{base}.time"] = np.asarray([float(time)])
    _write_code(ods, base, provenance)

    # -- model --------------------------------------------------------------
    parameters = provenance.get("parameters") or {}
    n_field = int(parameters.get("N_FIELD", (getattr(outputs, "grid", None) or {}).get("n_field", 1)))
    put("model.include_a_field_parallel", int(n_field >= 2))
    put("model.include_b_field_parallel", int(n_field >= 3))
    put("model.adiabatic_electrons", 0)
    put("model.include_full_curvature_drift", 1)
    put("model.include_coriolis_drift", 0)
    put("model.include_centrifugal_effects", 0)
    if int(parameters.get("COLLISION_MODEL", 4)) == 4:
        put("model.collisions_pitch_only", 0)
        put("model.collisions_momentum_conservation", 1)
        put("model.collisions_energy_conservation", 1)
        put("model.collisions_finite_larmor_radius", 1)
    else:
        skipped.append("model.collisions_*: only COLLISION_MODEL=4 is classified")

    _write_local_state(local, factors, put, skipped)

    # -- linear -------------------------------------------------------------------
    if "NONLINEAR_FLAG" in parameters:
        nonlinear = int(parameters["NONLINEAR_FLAG"]) == 1
    else:
        # No staged parameters: judge from the run itself rather than defaulting to linear.
        nonlinear = bool(getattr(outputs, "nonlinear", False))
    if not nonlinear:
        _write_linear(ods, base, local, outputs, factors, parameters, put, skipped)
    else:
        _write_nonlinear(ods, base, outputs, factors, n_field, flux_window, put, skipped)
    return {"written": written, "skipped": skipped}


def _write_local_state(local: Any, factors: Mapping[str, float], put, skipped: list) -> None:
    """Species, flux surface and normalising quantities of one local input (CGYRO form).

    Shared by the CGYRO and TGLF writers: TGLF's local input is renamed into this form
    (:func:`~vaft.code.gacode.cgyro.inputs.cgyro_input_from_tglf`) without new physics,
    so both codes' IDS describe the surface with the same numbers.
    """
    # -- species --------------------------------------------------------------
    species = local.species
    for index in range(local.n_species):
        prefix = f"species.{index}"
        put(f"{prefix}.charge_norm", float(species["Z"][index]))
        put(f"{prefix}.mass_norm", float(species["MASS"][index]))
        put(f"{prefix}.density_norm", float(species["DENS"][index]))
        put(f"{prefix}.temperature_norm", float(species["TEMP"][index]))
        put(f"{prefix}.density_log_gradient_norm",
            float(species["DLNNDR"][index]) * factors["rmaj"])
        put(f"{prefix}.temperature_log_gradient_norm",
            float(species["DLNTDR"][index]) * factors["rmaj"])
        put(f"{prefix}.velocity_tor_gradient_norm", 0.0)
    put("species_all.beta_reference", float(local.betae_unit) * factors["beta"])
    put("species_all.debye_length_norm", float(local.lambda_star) * factors["debye"])
    put("species_all.shearing_rate_norm", 0.0)
    put("species_all.velocity_tor_norm", 0.0)
    skipped.append("collisions.collisionality_norm: " + MAPPING_AUDIT["collisions.collisionality_norm"][1])

    # -- flux surface -----------------------------------------------------------
    g = local.geometry
    r_minor_norm = float(g["RMIN"]) / factors["rmaj"]
    put("flux_surface.r_minor_norm", r_minor_norm)
    put("flux_surface.q", float(g["Q"]))
    put("flux_surface.magnetic_shear_r_minor", float(g["S"]))
    put("flux_surface.elongation", float(g["KAPPA"]))
    put("flux_surface.delongation_dr_minor_norm",
        float(g["KAPPA"]) * float(g["S_KAPPA"]) / r_minor_norm)
    put("flux_surface.dgeometric_axis_r_dr_minor", float(g["SHIFT"]))
    put("flux_surface.dgeometric_axis_z_dr_minor", float(g["DZMAG"]))
    put("flux_surface.ip_sign", float(local.ipccw))
    put("flux_surface.b_field_tor_sign", float(local.btccw))
    skipped.append("flux_surface.shape_coefficients_*: "
                   + MAPPING_AUDIT["flux_surface.shape_coefficients_*"][1])
    skipped.append("flux_surface.pressure_gradient_norm: "
                   + MAPPING_AUDIT["flux_surface.pressure_gradient_norm"][1])

    # -- normalizing quantities ---------------------------------------------------
    norm = getattr(local, "normalisation", None)
    if norm is not None:
        put("normalizing_quantities.r", factors["rmaj"] * float(norm.minor_radius))
        put("normalizing_quantities.b_field_tor", factors["b_gs2"] * abs(float(norm.b_unit)))
        put("normalizing_quantities.t_e", float(norm.electron_temperature) / 1.602176634e-19)
        put("normalizing_quantities.n_e", float(norm.electron_density))
    else:
        skipped.append("normalizing_quantities: the local input carries no SI scales")


def _write_linear(ods, base, local, outputs, factors, parameters, put, skipped) -> None:
    gamma = outputs.final_growth_rate
    omega = outputs.frequency_ion_negative
    ky = outputs.ky
    if gamma is None or ky is None:
        skipped.append("linear: no frequency record in the run")
        return
    if omega is None:
        skipped.append("linear.frequency_norm: CGYRO did not state the ion direction, so "
                       "the sign convention cannot be applied; growth rate only")
    for k in range(ky.size):
        wave = f"linear.wavevector.{k}"
        put(f"{wave}.binormal_wavevector_norm", float(ky[k]) * factors["wavenumber"])
        put(f"{wave}.radial_wavevector_norm", 0.0)
        mode = f"{wave}.eigenmode.0"
        put(f"{mode}.growth_rate_norm", float(gamma[k]) * factors["rate"])
        if omega is not None:
            put(f"{mode}.frequency_norm", float(omega[k]) * factors["rate"])
        put(f"{mode}.initial_value_run", 1)
        # The DD field is the tolerance the eigenvalue *reached*; a run stopped at
        # MAX_TIME reached none, so it is written only for a converged run.
        if "FREQ_TOL" in parameters and outputs.converged:
            put(f"{mode}.growth_rate_tolerance", float(parameters["FREQ_TOL"]))
        ods[f"{base}.{mode}.code.parameters"] = _xml_parameters(
            {
                "exit_message": outputs.exit_message,
                "converged": int(bool(outputs.converged)),
                "ion_direction_native": outputs.ion_direction,
                "frequency_sign_convention": FREQUENCY_SIGN_CONVENTION,
                "field_normalisation": "phi / phi(angle_pol=0), final time",
            }
        )
        phi = outputs.ballooning.get("phi") if outputs.ballooning else None
        thetab = (outputs.grid or {}).get("thetab")
        if phi is None or thetab is None or np.size(thetab) != np.size(phi):
            skipped.append(f"{mode}.fields: no ballooning-space phi in the run")
            continue
        angle = geometric_angle(thetab, local)
        order = np.argsort(angle)
        angle, phi = angle[order], np.asarray(phi)[order]
        reference = phi[int(np.argmin(np.abs(angle)))]
        if reference == 0:
            skipped.append(f"{mode}.fields: phi vanishes at angle_pol=0")
            continue
        put(f"{mode}.angle_pol", angle)
        put(f"{mode}.time_norm", np.asarray([0.0]))
        put(f"{mode}.fields.phi_potential_perturbed_norm", (phi / reference)[:, None])
    skipped.append("linear.*.a_field_parallel_perturbed_norm: "
                   + MAPPING_AUDIT["linear.wavevector[:].eigenmode[:].fields.a_field_parallel_perturbed_norm"][1])


def _write_nonlinear(ods, base, outputs, factors, n_field, window, put, skipped) -> None:
    if window is None:
        skipped.append("non_linear.fluxes_1d: no averaging window was given; saturation "
                       "is the caller's judgement")
        return
    if outputs.flux is None or outputs.time is None:
        skipped.append("non_linear.fluxes_1d: no bin.cgyro.ky_flux in the run")
        return
    t = np.asarray(outputs.time, dtype=float)[: outputs.flux.shape[-1]]
    mask = (t >= window[0]) & (t <= window[1])
    if np.count_nonzero(mask) < 2:
        skipped.append(f"non_linear.fluxes_1d: fewer than two samples in {window}")
        return
    integrate = getattr(np, "trapezoid", None) or np.trapz
    # (species, moment, field) after summing toroidal modes and averaging in time
    summed = np.sum(outputs.flux[..., mask], axis=3)
    average = integrate(summed, t[mask], axis=-1) / (t[mask][-1] - t[mask][0])
    for f in range(min(n_field, average.shape[2])):
        put(f"non_linear.fluxes_1d.particles_{_FIELD_DD[f]}",
            average[:, 0, f] * factors["flux"])
        put(f"non_linear.fluxes_1d.energy_{_FIELD_DD[f]}",
            average[:, 1, f] * factors["flux"])
    # ky-resolved: the same time average without the sum over toroidal modes,
    # (species, moment, field, n); n is the binormal axis already written below.
    per_mode = integrate(outputs.flux[..., mask], t[mask], axis=-1) / (t[mask][-1] - t[mask][0])
    for f in range(min(n_field, per_mode.shape[2])):
        put(f"non_linear.fluxes_2d_k_x_sum.particles_{_FIELD_DD[f]}",
            per_mode[:, 0, f, :] * factors["flux"])
        put(f"non_linear.fluxes_2d_k_x_sum.energy_{_FIELD_DD[f]}",
            per_mode[:, 1, f, :] * factors["flux"])
    put("non_linear.time_norm", t * factors["time"])
    put("non_linear.time_interval_norm", np.asarray(window, dtype=float) * factors["time"])
    if outputs.ky is not None:
        put("non_linear.binormal_wavevector_norm", np.asarray(outputs.ky) * factors["wavenumber"])
    put("non_linear.quasi_linear", 0)
    skipped.append("non_linear.fluxes_1d.momentum_*: "
                   + MAPPING_AUDIT["non_linear.fluxes_1d.momentum_*"][1])


# -- TGLF -------------------------------------------------------------------------

TGLF_REPOSITORY = CGYRO_REPOSITORY

#: How each TGLF quantity reaches ``gyrokinetics_local``, keyed by DD path. Classes as in
#: :data:`MAPPING_AUDIT`. TGLF is a *quasilinear* model: the IDS says so
#: (``non_linear.quasi_linear = 1``), and only quantities whose definition survives that
#: are written. The local state (species, flux surface, normalisation) is the same as the
#: CGYRO mapping's -- the TGLF input renamed (:func:`cgyro_input_from_tglf`).
TGLF_MAPPING_AUDIT: Mapping[str, tuple[str, str]] = {
    "species[:], species_all, flux_surface, normalizing_quantities": (
        "unit", "the shared local-state writer on the renamed TGLF input (same numbers "
        "as a CGYRO run on that surface)"),
    "normalizing_quantities.b_field_tor / GKDB rescaling": (
        "derived", "b_gs2 = B_tor(R0)/B_unit from TGLF's own Bt0_out = f/Rmaj "
        "(scalar_saturation_parameters); identical to CGYRO's b_gs2 on the same input"),
    "model.include_a_field_parallel / include_b_field_parallel": ("exact", "USE_BPER / USE_BPAR"),
    "model.adiabatic_electrons": ("exact", "ADIABATIC_ELEC"),
    "model.collisions_*": (
        "unsupported", "TGLF's reduced collision model (XNU_MODEL) has no counterpart "
        "in the DD's operator flags"),
    "model.include_full_curvature_drift / coriolis / centrifugal": (
        "unsupported", "TGLF's drift model is a fitted reduced form, not a GK operator choice"),
    "linear.wavevector[:].binormal_wavevector_norm": ("unit", "KY * sqrt(2) / b_gs2"),
    "linear.wavevector[:].eigenmode[:].growth_rate_norm": (
        "unit", "gamma * RMAJ / sqrt(2), every mode TGLF found; a (0, 0) slot is an "
        "absent mode, not a marginal one"),
    "linear.wavevector[:].eigenmode[:].frequency_norm": (
        "convention", "omega * RMAJ / sqrt(2); TGLF already writes the ion diamagnetic "
        "direction negative"),
    "linear.wavevector[:].eigenmode[:].initial_value_run": ("exact", "0: TGLF is an eigenvalue solver"),
    "linear.wavevector[:].eigenmode[:].fields": (
        "unsupported", "out.tglf.wavefunction is a separate single-ky run, not parsed"),
    "linear.wavevector[:].eigenmode[:].linear_weights": (
        "unsupported", "QL weights' amplitude normalisation is not the DD's"),
    "non_linear.quasi_linear": ("exact", "1"),
    "non_linear.fluxes_1d.{particles,energy}_<field>": (
        "unit", "sum over ky of sum_flux_spectrum per field (saturated quasilinear flux), "
        "GB flux * RMAJ**2 * b_gs2**2 / (2 sqrt 2) -- the same rescaling as CGYRO"),
    "non_linear.fluxes_2d_k_x_sum.{particles,energy}_<field>": (
        "derived", "sum_flux_spectrum per ky bin (TGLF's ky-integration weight x flux, so "
        "the bins sum to the total), on non_linear.binormal_wavevector_norm; the same "
        "GB rescaling"),
    "non_linear.fluxes_1d.momentum_*": (
        "unsupported", "TGLF's toroidal/parallel stresses are not the DD's parallel/"
        "perpendicular momentum split"),
    "fluctuation amplitudes, cross phases, field intensities": (
        "unsupported", "no audited DD home (moments_norm are eigenmode-level complex "
        "fields); kept native"),
}


def gyrokinetics_local_from_tglf(
    ods: ODS,
    local: Any,
    outputs: Any,
    *,
    parameters: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    time: Optional[float] = None,
) -> dict[str, Any]:
    """Write one TGLF surface into ``ods['gyrokinetics_local']``, per :data:`TGLF_MAPPING_AUDIT`.

    Parameters
    ----------
    local
        The :class:`~vaft.code.gacode.tglf.inputs.TGLFInput` the run was given.
    outputs
        Its :class:`~vaft.code.gacode.tglf.outputs.TglfOutputs`. The GKDB rescaling needs
        ``Bt0_out`` from ``out.tglf.scalar_saturation_parameters``, which TGLF writes
        only on its transport-model path; a run without it writes nothing and says so.
    parameters
        The ``input.tglf`` settings (``TGLFInputs.parameters``): field model, adiabatic
        electrons, SAT rule.
    provenance
        ``TGLFResult.provenance`` (version), recorded in ``code``.
    """
    from vaft.code.gacode.cgyro.inputs import cgyro_input_from_tglf

    written: list[str] = []
    skipped: list[str] = []
    parameters = dict(parameters or {})
    provenance = dict(provenance or {})
    base = IDS

    def put(path: str, value: Any) -> None:
        ods[f"{base}.{path}"] = value
        written.append(path)

    saturation = getattr(outputs, "saturation_parameters", None) or {}
    renamed = cgyro_input_from_tglf(local)
    factors = _factors(float(renamed.geometry["RMAJ"]), saturation.get("Bt0_out"))
    if factors is None:
        return {"written": written, "skipped": [
            "no Bt0_out in out.tglf.scalar_saturation_parameters (single-ky or NN run): "
            "the DD normalisation cannot be formed, nothing written"]}

    ods[f"{base}.ids_properties.homogeneous_time"] = 1
    ods[f"{base}.ids_properties.comment"] = (
        "TGLF quasilinear local model; normalised per GKDB (L_ref=R0); frequency sign: "
        + FREQUENCY_SIGN_CONVENTION)
    if time is not None:
        ods[f"{base}.time"] = np.asarray([float(time)])
    version = provenance.get("version") or getattr(outputs, "version", None) or {}
    ods[f"{base}.code.name"] = "TGLF"
    ods[f"{base}.code.repository"] = TGLF_REPOSITORY
    if isinstance(version, Mapping) and version.get("revision"):
        ods[f"{base}.code.version"] = str(version["revision"]).replace("<", "").replace(">", "")
    ods[f"{base}.code.parameters"] = _xml_parameters({
        "sat_rule": saturation.get("SAT_RULE", parameters.get("SAT_RULE")),
        "units": saturation.get("UNITS"),
        "xnu_model": saturation.get("XNU_MODEL"),
        "preset_note": "SAT2/3 presets set XNU_MODEL=3 and WDIA_TRAPPED=1: the linear "
                       "model differs from SAT0/1 on the same input",
        "use_bper": parameters.get("USE_BPER"), "use_bpar": parameters.get("USE_BPAR"),
        "frequency_sign_convention": FREQUENCY_SIGN_CONVENTION,
        "normalisation": "GKDB: L_ref=R0, B_ref=Btor(R0), v_thref=sqrt(2Te/mD)",
        "quasi_linear": 1,
    }, tag="tglf")

    put("model.include_a_field_parallel", int(bool(parameters.get("USE_BPER", False))))
    put("model.include_b_field_parallel", int(bool(parameters.get("USE_BPAR", False))))
    put("model.adiabatic_electrons", int(bool(parameters.get("ADIABATIC_ELEC", False))))
    for key in ("model.collisions_*", "model.include_full_curvature_drift / coriolis / centrifugal"):
        skipped.append(f"{key}: {TGLF_MAPPING_AUDIT[key][1]}")

    _write_local_state(renamed, factors, put, skipped)

    ky = getattr(outputs, "ky_spectrum", None)
    gamma = getattr(outputs, "growth_rate", None)
    omega = getattr(outputs, "frequency", None)
    if ky is None or gamma is None:
        skipped.append("linear: no eigenvalue spectrum in the run")
    else:
        empty = 0
        for k in range(ky.size):
            wave = f"linear.wavevector.{k}"
            put(f"{wave}.binormal_wavevector_norm", float(ky[k]) * factors["wavenumber"])
            mode_index = 0
            for m in range(gamma.shape[1]):
                if gamma[k, m] == 0.0 and omega[k, m] == 0.0:
                    empty += 1
                    continue
                mode = f"{wave}.eigenmode.{mode_index}"
                put(f"{mode}.growth_rate_norm", float(gamma[k, m]) * factors["rate"])
                put(f"{mode}.frequency_norm", float(omega[k, m]) * factors["rate"])
                put(f"{mode}.initial_value_run", 0)
                mode_index += 1
        if empty:
            skipped.append(f"linear: {empty} (gamma, omega) = (0, 0) slots are absent modes, not written")
    skipped.append("linear.*.fields: " + TGLF_MAPPING_AUDIT["linear.wavevector[:].eigenmode[:].fields"][1])

    put("non_linear.quasi_linear", 1)
    spectrum = getattr(outputs, "sum_flux_spectrum", None)
    if spectrum is None:
        skipped.append("non_linear.fluxes_1d: no sum_flux_spectrum in the run")
    else:
        # (species, field, quantity), TGLF species order (electrons first) moved to the
        # order species[:] was written in (electrons last), since fluxes_1d is indexed by
        # gyrokinetics_local.species.
        zs = np.asarray(local.zs, dtype=float)
        electron = int(np.flatnonzero(zs < 0)[0])
        order = [i for i in range(zs.size) if i != electron] + [electron]
        per_ky = spectrum[order]                       # (species, field, ky, quantity)
        totals = np.nansum(per_ky, axis=2)
        # Field slots are named from the run's own flags, not by position: TGLF writes
        # fields 1..jflds, so with USE_BPAR but not USE_BPER slot 2 is not B_parallel.
        use_bper = bool(parameters.get("USE_BPER", False))
        use_bpar = bool(parameters.get("USE_BPAR", False))
        names = ["phi_potential"] + (["a_field_parallel"] if use_bper else [])
        if use_bpar and use_bper:
            names.append("b_field_parallel")
        elif use_bpar:
            skipped.append("non_linear.fluxes_1d.*_b_field_parallel: USE_BPAR without "
                           "USE_BPER, TGLF's field slots are ambiguous; only phi written")
        for f, name in enumerate(names[: totals.shape[1]]):
            put(f"non_linear.fluxes_1d.particles_{name}", totals[:, f, 0] * factors["flux"])
            put(f"non_linear.fluxes_1d.energy_{name}", totals[:, f, 1] * factors["flux"])
            put(f"non_linear.fluxes_2d_k_x_sum.particles_{name}",
                per_ky[:, f, :, 0] * factors["flux"])
            put(f"non_linear.fluxes_2d_k_x_sum.energy_{name}",
                per_ky[:, f, :, 1] * factors["flux"])
        if ky is not None:
            put("non_linear.binormal_wavevector_norm", np.asarray(ky, dtype=float) * factors["wavenumber"])
        skipped.append("non_linear.fluxes_1d.momentum_*: "
                       + TGLF_MAPPING_AUDIT["non_linear.fluxes_1d.momentum_*"][1])
    skipped.append("fluctuation spectra: "
                   + TGLF_MAPPING_AUDIT["fluctuation amplitudes, cross phases, field intensities"][1])
    return {"written": written, "skipped": skipped}


# -- core_transport ---------------------------------------------------------------


def _cgyro_model_position(ods: ODS) -> int:
    """The CGYRO anomalous entry, appended if new -- never TGLF's.

    TGLF's mapping reuses *any* index-6 entry; CGYRO must not land on it, or the two
    models would overwrite each other's fluxes in one ODS.
    """
    count = path_count(ods, "core_transport.model")
    for index in range(count):
        if (
            ods.get(f"core_transport.model.{index}.identifier.index", None) == ANOMALOUS_MODEL_INDEX
            and ods.get(f"core_transport.model.{index}.code.name", None) == "CGYRO"
        ):
            return index
    return count


def core_transport_from_cgyro(
    ods: ODS,
    surfaces: Iterable[tuple[Any, Any, tuple[float, float]]],
    profile: Any,
    *,
    time: float = 0.0,
    time_index: int = 0,
    provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Write saturated nonlinear CGYRO fluxes into ``core_transport``.

    Parameters
    ----------
    surfaces
        ``(CGYROInput, CgyroOutputs, window)`` per surface; ``window`` is the
        saturated averaging interval in ``a/c_s``.
    profile
        The ``GACODEProfile`` the inputs came from, for ``r/a -> rho_tor_norm``.

    The gyro-Bohm units are CGYRO's and TGLF's alike (``q_gb = n_e T_e c_s (rho_s/a)^2``
    with deuterium ``c_s`` and ``B_unit``), so the SI conversion reuses the
    :class:`~vaft.code.gacode.tglf.inputs.TGLFNormalisation` the shared local input
    already carries rather than deriving it a second time. The energy flux is CGYRO's
    total energy moment, so ``flux_multiplier`` is 0, as for TGLF and NEO.
    """
    from vaft.machine_mapping.turbulence import _rho_tor_norm  # shared r/a bridge

    rows = []
    skipped: list[str] = []
    for local, outputs, window in surfaces:
        average = None if outputs is None else outputs.time_average_flux(window)
        norm = getattr(local, "normalisation", None)
        if average is None or norm is None:
            skipped.append(f"r/a={getattr(local, 'r_over_a', '?')}: no averaged flux or no SI scale")
            continue
        rows.append((float(local.r_over_a), local, average, norm))
    if not rows:
        return {"written": [], "skipped": skipped}
    rows.sort(key=lambda row: row[0])
    rho = _rho_tor_norm(profile, np.asarray([row[0] for row in rows]))
    if rho is None:
        return {"written": [], "skipped": skipped + ["profile cannot map r/a to rho_tor_norm"]}

    m = _cgyro_model_position(ods)
    model = f"core_transport.model.{m}"
    ods[f"{model}.identifier.index"] = ANOMALOUS_MODEL_INDEX
    ods[f"{model}.identifier.name"] = ANOMALOUS_MODEL_NAME
    ods[f"{model}.identifier.description"] = CGYRO_MODEL_DESCRIPTION
    ods[f"{model}.flux_multiplier"] = 0.0
    _write_code(ods, model, dict(provenance or {}))
    base = f"{model}.profiles_1d.{time_index}"
    ods[f"{base}.time"] = float(time)
    ods[f"{base}.grid_flux.rho_tor_norm"] = np.asarray(rho, dtype=float)

    names = rows[0][1].names
    electron = int(np.flatnonzero(np.asarray(rows[0][1].species["Z"]) < 0)[0])
    particle = np.asarray([row[2][:, 0] * row[3].particle_flux for row in rows])
    energy = np.asarray([row[2][:, 1] * row[3].energy_flux for row in rows])
    ods[f"{base}.electrons.particles.flux"] = particle[:, electron]
    ods[f"{base}.electrons.energy.flux"] = energy[:, electron]
    ion = 0
    for index, name in enumerate(names):
        if index == electron:
            continue
        ods[f"{base}.ion.{ion}.label"] = str(name)
        ods[f"{base}.ion.{ion}.z_ion"] = float(rows[0][1].species["Z"][index])
        ods[f"{base}.ion.{ion}.element.0.z_n"] = float(rows[0][1].species["Z"][index])
        ods[f"{base}.ion.{ion}.element.0.a"] = float(rows[0][1].species["MASS"][index]) * 2.0
        ods[f"{base}.ion.{ion}.particles.flux"] = particle[:, index]
        ods[f"{base}.ion.{ion}.energy.flux"] = energy[:, index]
        ion += 1
    ods["core_transport.ids_properties.homogeneous_time"] = 1
    try:
        ods.set_time_array("core_transport.time", time_index, float(time))
    except Exception:
        ods["core_transport.time"] = np.asarray([float(time)])
    skipped.append("momentum_tor: CGYRO's momentum moment is not mapped (no rotation)")
    return {"written": [base], "skipped": skipped}


# -- linear k_y scans ---------------------------------------------------------------

# Subtrees that differ between the single-k_y runs of one scan by construction.
_SCAN_FREE = ("linear", "code", "ids_properties")


def merge_linear_scan(odss: Iterable[ODS], *, rtol: float = 1e-9) -> ODS:
    """Combine single-``k_y`` linear runs of one local problem into one IDS.

    A linear ``k_y`` scan run as separate initial-value runs (one CGYRO run per
    ``k_y``) is still one ``gyrokinetics_local`` simulation in the GKDB sense: one
    plasma, many wavevectors. Every leaf outside ``linear``, ``code`` and
    ``ids_properties`` -- species, flux surface, model flags, normalisation -- must
    agree between the runs (to ``rtol``), or the runs are not one scan and a
    ``ValueError`` names the first leaf that differs. Wavevectors are sorted by
    ``binormal_wavevector_norm``; ``code`` and ``ids_properties`` come from the first
    run.
    """
    import copy

    odss = list(odss)
    if not odss:
        raise ValueError("no runs to merge")
    first = odss[0][IDS]

    def shared(ids: Any) -> dict[str, Any]:
        return {path: value for path, value in ids.flat().items()
                if path.split(".")[0] not in _SCAN_FREE}

    reference = shared(first)
    waves: list[tuple[float, Any]] = []
    for position, ods in enumerate(odss):
        ids = ods[IDS]
        leaves = shared(ids)
        if set(leaves) != set(reference):
            extra = sorted(set(leaves) ^ set(reference))[0]
            raise ValueError(f"run {position} is not part of the scan: {extra} is not in every run")
        for path, value in leaves.items():
            other = reference[path]
            if isinstance(value, str) or isinstance(other, str):
                same = value == other
            else:
                same = np.shape(value) == np.shape(other) and np.allclose(
                    value, other, rtol=rtol, atol=0.0, equal_nan=True)
            if not same:
                raise ValueError(f"run {position} is not part of the scan: {path} differs")
        for k in range(path_count(ods, f"{IDS}.linear.wavevector")):
            wave = ids[f"linear.wavevector.{k}"]
            waves.append((float(wave["binormal_wavevector_norm"]), wave))

    merged = ODS()
    merged[IDS] = copy.deepcopy(first)
    del merged[f"{IDS}.linear.wavevector"]
    for k, (_ky, wave) in enumerate(sorted(waves, key=lambda pair: pair[0])):
        merged[f"{IDS}.linear.wavevector.{k}"] = copy.deepcopy(wave)
    return merged
