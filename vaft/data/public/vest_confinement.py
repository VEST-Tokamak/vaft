"""VEST side of the canonical confinement table.

VEST enters the multi-machine comparison through its canonical layers, not a
notebook re-derivation:

* :func:`vest_summary_to_confinement_table` converts rows of the
  ``core_profiles`` summary preset (:func:`vaft.database.summary`) -- the
  shot-scale path lane D fills from the database;
* :func:`vest_ods_to_confinement_rows` builds rows straight from an ODS with the
  equilibrium descriptors of :mod:`vaft.process.equilibrium`, for packaged
  samples that carry no magnetics and therefore no power balance.  Loss power
  and confinement time are then missing, and stay missing.

The VEST definitions differ from DB5's and are recorded per row: the summary
loss power subtracts radiation (DB5 ``PLTH`` does not), its toroidal field is
taken on the magnetic axis and quoted at R = 0.4 m, and its ion mass is not a
measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CONFINEMENT_COLUMNS, make_record_id, validate_confinement_table

__all__ = [
    "VEST_REFERENCE_RADIUS_M",
    "VEST_SUMMARY_DEFINITIONS",
    "vest_ods_to_confinement_rows",
    "vest_summary_to_confinement_table",
]

#: Radius at which the ``core_profiles`` summary quotes ``b_t_T``.
VEST_REFERENCE_RADIUS_M = 0.4

VEST_SUMMARY_DEFINITIONS: dict[str, str] = {
    "p_loss_definition": (
        "VAFT core_profiles summary: P_ohm (integral of eta J^2, Spitzer, Z_eff=2) "
        "- dW/dt - P_rad (line + bremsstrahlung + synchrotron); radiation IS "
        "subtracted; no auxiliary heating; dW/dt uses W = (2/3)<p>V (#1282)"
    ),
    "w_th_definition": "not carried by the summary",
    "tau_e_definition": (
        "VAFT summary tau_e_s = W_th / P_loss, W_th = 1.5 <p_kinetic> V with "
        "p = 2 n_e T_e (T_i = T_e, n_i = n_e)"
    ),
    "n_e_definition": (
        "VAFT summary: z = 0 chord through the core_profiles n_e mapped onto "
        "the equilibrium (synthetic, not an interferometer)"
    ),
    "b_t_definition": (
        "B_phi(magnetic axis) * R_axis / r_geo_m: total field on axis, "
        "rescaled by 1/R to the geometric radius"
    ),
}


def _labels(table: pd.DataFrame, database: str, release: str) -> pd.DataFrame:
    table["machine"] = "VEST"
    table["regime"] = "unlabelled"
    table["selected"] = False
    table["source_database"] = database
    table["source_release"] = release
    table["source_reference"] = "VEST, VAFT canonical summaries"
    return table


def vest_summary_to_confinement_table(
    summary: pd.DataFrame,
    *,
    effective_mass_amu: float,
    release: str = "",
) -> pd.DataFrame:
    """Convert ``core_profiles`` summary rows into the canonical confinement table.

    Parameters
    ----------
    summary : pandas.DataFrame
        Rows of ``vaft.database.summary(..., preset="core_profiles")``: at least
        ``shot, time_s, ip_kA, b_t_T, p_loss_MW, tau_e_s, ne_line_1e19_m3,
        major_radius_m, inverse_aspect_ratio, elongation`` [table].
    effective_mass_amu : float
        Effective ion mass of the plasma; there is no default because VEST does
        not measure it and the summary hard-codes 1 for its own scalings [amu].
    release : str, optional
        Label of the summary source (database snapshot, product version);
        default empty [str].

    Returns
    -------
    pandas.DataFrame
        Canonical confinement table, one row per summary row [table].

    Notes
    -----
    Conversions: kA -> A, 1e19 m^-3 -> m^-3, MW -> W, and
    ``b_t = b_t_T * 0.4 / major_radius_m`` so the field is quoted at the
    geometric radius as DB5's ``BT`` is.  ``a_m`` is ``epsilon *
    major_radius_m``.  ``kappa_area``, ``delta`` and ``w_th_J`` are not in the
    preset and stay ``NaN``; ``regime`` is ``"unlabelled"``.
    """
    mass = float(effective_mass_amu)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"effective_mass_amu must be finite and > 0, got {effective_mass_amu!r}")
    required = (
        "shot", "time_s", "ip_kA", "b_t_T", "p_loss_MW", "tau_e_s",
        "ne_line_1e19_m3", "major_radius_m", "inverse_aspect_ratio", "elongation",
    )
    missing = [c for c in required if c not in summary.columns]
    if missing:
        raise KeyError(f"summary lacks core_profiles columns {missing}")

    def col(name: str) -> np.ndarray:
        return pd.to_numeric(summary[name], errors="coerce").to_numpy(float)

    r_geo = col("major_radius_m")
    table = pd.DataFrame(index=summary.index)
    table["shot"] = pd.array(summary["shot"], dtype="Int64")
    table["time_s"] = col("time_s")
    table["record_id"] = [make_record_id("VEST", s, t) for s, t in zip(summary["shot"], table["time_s"])]
    table["i_p_A"] = np.abs(col("ip_kA")) * 1e3
    table["b_t_T"] = np.abs(col("b_t_T")) * VEST_REFERENCE_RADIUS_M / r_geo
    table["n_e_line_avg_m3"] = col("ne_line_1e19_m3") * 1e19
    table["p_loss_W"] = col("p_loss_MW") * 1e6
    table["w_th_J"] = np.nan
    table["tau_e_th_s"] = col("tau_e_s")
    table["r_geo_m"] = r_geo
    table["epsilon"] = col("inverse_aspect_ratio")
    table["a_m"] = table["epsilon"] * r_geo
    table["kappa"] = col("elongation")
    table["kappa_area"] = np.nan
    table["delta"] = np.nan
    table["m_eff_amu"] = mass
    for column, text in VEST_SUMMARY_DEFINITIONS.items():
        table[column] = text
    table["m_eff_source"] = "user-specified"
    # Non-positive loss power (dW/dt larger than the input) gives no confinement time.
    nonpositive = ~(table["p_loss_W"] > 0.0)
    table.loc[nonpositive, ["p_loss_W", "tau_e_th_s"]] = np.nan
    _labels(table, "VAFT core_profiles summary", release)
    return validate_confinement_table(table)


def _get(ods, path: str):
    """Read ``path`` without creating it (OMAS materialises missing reads)."""
    return ods[path] if path in ods else None


def _line_average_density_z0(ods, eq_index: int, equilibrium) -> float:
    """Line average of n_e along the horizontal chord z = 0 inside the LCFS."""
    from scipy.interpolate import RegularGridInterpolator

    eq_prefix = f"equilibrium.time_slice.{eq_index}"
    eq_time = float(ods["equilibrium.time"][eq_index])
    cp_times = _get(ods, "core_profiles.time")
    if cp_times is None:
        return np.nan
    match = np.flatnonzero(np.isclose(np.asarray(cp_times, float), eq_time, rtol=0.0, atol=1e-6))
    if match.size == 0:
        return np.nan
    cp_prefix = f"core_profiles.profiles_1d.{int(match[0])}"
    density = _get(ods, f"{cp_prefix}.electrons.density")
    rho_cp = _get(ods, f"{cp_prefix}.grid.rho_tor_norm")
    psi_1d = _get(ods, f"{eq_prefix}.profiles_1d.psi")
    rho_eq = _get(ods, f"{eq_prefix}.profiles_1d.rho_tor_norm")
    if any(v is None for v in (density, rho_cp, psi_1d, rho_eq)) or equilibrium.lcfs is None:
        return np.nan

    lcfs_r, lcfs_z = equilibrium.lcfs.r, equilibrium.lcfs.z
    if lcfs_r.size and (lcfs_r[0] != lcfs_r[-1] or lcfs_z[0] != lcfs_z[-1]):
        lcfs_r, lcfs_z = np.append(lcfs_r, lcfs_r[0]), np.append(lcfs_z, lcfs_z[0])
    crossings = []
    for i in range(lcfs_r.size - 1):
        z0, z1 = lcfs_z[i], lcfs_z[i + 1]
        if z0 == 0.0:
            crossings.append(lcfs_r[i])
        elif z0 * z1 < 0.0:
            crossings.append(lcfs_r[i] + (lcfs_r[i + 1] - lcfs_r[i]) * z0 / (z0 - z1))
    if len(crossings) < 2:
        return np.nan
    r_chord = np.linspace(min(crossings), max(crossings), 400)

    psi_2d = RegularGridInterpolator((equilibrium.r, equilibrium.z), equilibrium.psi)(
        np.column_stack((r_chord, np.zeros_like(r_chord)))
    )
    span = equilibrium.psi_boundary - equilibrium.psi_axis
    psi_norm = np.clip((psi_2d - equilibrium.psi_axis) / span, 0.0, 1.0)
    psi_norm_1d = (np.asarray(psi_1d, float) - equilibrium.psi_axis) / span
    order = np.argsort(psi_norm_1d)
    rho = np.interp(psi_norm, psi_norm_1d[order], np.asarray(rho_eq, float)[order])
    n_chord = np.interp(rho, np.asarray(rho_cp, float), np.asarray(density, float))
    return float(np.trapezoid(n_chord, r_chord) / (r_chord[-1] - r_chord[0]))


def vest_ods_to_confinement_rows(
    ods,
    *,
    effective_mass_amu: float,
    release: str = "",
) -> pd.DataFrame:
    """Canonical confinement rows from a VEST ODS, one per equilibrium slice.

    Parameters
    ----------
    ods : omas.ODS
        VEST ODS with ``equilibrium`` and, for the density, ``core_profiles``;
        it is read, never modified [ODS].
    effective_mass_amu : float
        Effective ion mass; no default, VEST does not measure it [amu].
    release : str, optional
        Label of the ODS source, e.g. a sample name; default empty [str].

    Returns
    -------
    pandas.DataFrame
        Canonical confinement table with engineering parameters from the
        equilibrium descriptors [table].

    Notes
    -----
    * ``i_p_A``, ``r_geo_m``, ``a_m``, ``epsilon``, ``kappa`` and ``delta``
      (mean of upper and lower) are the closed-polygon descriptors of
      :func:`vaft.process.equilibrium.derive_global_descriptors`.
    * ``kappa_area = V / (2 pi^2 a^2 R)`` from the descriptor volume -- the
      DB5 ``KAREA`` definition.
    * ``b_t_T = |b0 r0| / r_geo_m``: the vacuum field of
      ``equilibrium.vacuum_toroidal_field`` at the geometric radius, as DB5
      ``BT``.
    * ``n_e_line_avg_m3`` is the average of core-profile n_e along the z = 0
      chord inside the LCFS, mapped through rho_tor_norm; ``NaN`` without a
      core-profile slice at the equilibrium time.
    * ``w_th_J`` is the descriptor ``thermal_energy`` (1.5 x the equilibrium
      pressure integral).
    * ``p_loss_W`` and ``tau_e_th_s`` are ``NaN``: they need the power
      balance, i.e. magnetics, which this path does not evaluate.
    """
    from vaft.process.equilibrium import as_equilibrium, derive_global_descriptors

    mass = float(effective_mass_amu)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"effective_mass_amu must be finite and > 0, got {effective_mass_amu!r}")
    shot = _get(ods, "dataset_description.data_entry.pulse")
    b0 = _get(ods, "equilibrium.vacuum_toroidal_field.b0")
    r0 = _get(ods, "equilibrium.vacuum_toroidal_field.r0")
    n_slices = len(ods["equilibrium.time_slice"]) if "equilibrium.time_slice" in ods else 0

    rows = []
    for index in range(n_slices):
        equilibrium = as_equilibrium(ods, time_index=index)
        descriptors = derive_global_descriptors(equilibrium).values

        def value(name: str) -> float:
            item = descriptors.get(name)
            return float(item.value) if item is not None and item.value is not None else np.nan

        r_geo = value("major_radius")
        a = value("minor_radius")
        volume = value("volume")
        time_s = float(ods["equilibrium.time"][index])
        b0_series = np.ravel(b0) if b0 is not None else np.array([])
        # b0 is per equilibrium time; a length that does not match cannot be
        # paired with this slice by position, so it is not guessed.
        b_t = (abs(float(b0_series[index]) * float(r0)) / r_geo
               if b0_series.size == n_slices and r0 is not None else np.nan)
        rows.append({
            "shot": int(shot) if shot is not None else pd.NA,
            "time_s": time_s,
            "record_id": make_record_id("VEST", shot if shot is not None else -1, time_s),
            "i_p_A": abs(value("ip")),
            "b_t_T": b_t,
            "n_e_line_avg_m3": _line_average_density_z0(ods, index, equilibrium),
            "p_loss_W": np.nan,
            "w_th_J": value("thermal_energy"),
            "tau_e_th_s": np.nan,
            "r_geo_m": r_geo,
            "a_m": a,
            "epsilon": value("inverse_aspect_ratio"),
            "kappa": value("elongation"),
            "kappa_area": volume / (2.0 * np.pi**2 * a**2 * r_geo),
            "delta": 0.5 * (value("triangularity_upper") + value("triangularity_lower")),
            "m_eff_amu": mass,
            "p_loss_definition": "not evaluated (needs the magnetics power balance)",
            "w_th_definition": "1.5 x integral of equilibrium pressure over the plasma volume",
            "tau_e_definition": "not evaluated",
            "b_t_definition": "vacuum field |b0 r0| / r_geo_m, as DB5 BT",
            "n_e_definition": (
                "z = 0 chord inside the LCFS through core_profiles n_e(rho_tor_norm); "
                "synthetic, own implementation (differs ~2% from the summary's)"
            ),
            "m_eff_source": "user-specified",
        })
    table = pd.DataFrame(rows, columns=[c for c in CONFINEMENT_COLUMNS if c not in (
        "machine", "regime", "selected", "source_database", "source_release", "source_reference")])
    table["shot"] = table["shot"].astype("Int64")
    _labels(table, "VAFT ODS equilibrium descriptors", release)
    return validate_confinement_table(table)
