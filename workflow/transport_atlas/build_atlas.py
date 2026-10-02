"""Build the (shot, t, rho) transport atlas from stored TGLF and NEO records (issue #1427).

Reads what ``run_tglf.py`` and ``run_neo.py`` wrote and runs no solver. ``--tglf`` is
one configuration's tree (``tglf-sat<n>-<field>/``): an atlas never mixes TGLF
configurations, and the configuration is a column, not an implied default (#1482). Writes to
``--out``:

    atlas.csv              one row per (shot, time_efit_s, efit_lineage, r_over_a)
    radial_summary.csv     one row per state (shot, time_efit_s, efit_lineage)
    discharge_summary.csv  one row per (shot, efit_lineage)
    schema.json            unit, epistemic category and definition of every column

The descriptors are continuous. No row is labelled ITG/TEM/ETG/KBM/MTM, and the sign
of a real frequency is stored as a number, never turned into a mode name. Every row
carries the EFIT quality label and lineage, and every model quantity names the model
that predicted it. A TGLF or NEO number is a model prediction under the declared
assumptions (``ti_lineage``, ``composition_origin``), not a measurement.

Spectral conventions (TGLF native units):

* ``ky`` is ``ky rho_s``; growth rates and frequencies are in ``c_s/a``.
* ``gamma_max`` is the largest growth rate over every ky and mode, with its ky and
  real frequency. ``gamma_max_ion_scale`` repeats this for ``ky rho_s <= 1`` only.
* ``ky_q_mean`` is the energy-flux-weighted mean ky,
  ``sum_k ky_k |Q_k| / sum_k |Q_k|``, where Q_k is ``out.tglf.sum_flux_spectrum``
  summed (signed) over species and fields before its magnitude is taken. Each Q_k
  already carries its ky weight, so the sums are TGLF's own ky integral. This is a
  spectral moment, not an argmax: on a log ky grid, an argmax of weighted
  contributions would depend on the grid.
* Classical columns are the Braginskii perpendicular baseline (#1435), evaluated from
  the same TGLF local input. They join the partition as a third component, and only
  where the run recorded them.
* ``f_em = |Q_mag| / (|Q_phi| + |Q_mag|)``, where Q_phi and Q_mag are the signed
  energy fluxes (all species, all ky) of the electrostatic field and of the magnetic
  fields (A_par, B_par). It is empty when the run had only the electrostatic field.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

#: column -> (unit, category, definition). Categories: key, label, provenance,
#: measured_input, assumed_input, reconstructed_input, derived_input, tglf_predicted,
#: neo_predicted, classical_predicted, model_derived.
SCHEMA: dict[str, tuple[str, str, str]] = {
    "shot": ("-", "key", "VEST shot number"),
    "time_efit_s": ("s", "key", "equilibrium slice time; the state's time"),
    "efit_lineage": ("-", "key", "magnetics | electron_kinetic"),
    "r_over_a": ("-", "key", "requested and solved surface, GACODE rmin/rmin[-1] on the profile cut at rho_max"),
    "efit_quality": ("-", "label", "#1331 slice label: good | admissible"),
    "quality_source": ("-", "label", "criteria, or magnetic_slice_at_same_time for electron_kinetic rows"),
    "ti_lineage": ("-", "label", "how Ti was resolved, e.g. ti_eq_te_assumed (#1414) or measured"),
    "ti_te_ratio": ("-", "assumed_input", "policy Ti/Te when ti_lineage is a ratio, else empty"),
    "composition_origin": ("-", "label", "policy_assumption: H+ and C6+ at z_eff via quasi-neutrality"),
    "z_eff": ("-", "assumed_input", "effective charge the species list realizes"),
    "shape_kind": ("-", "provenance", "reconstructed: stored EFIT shape profiles; derived: traced in-boundary (#1458)"),
    "profile_mapped_on": ("-", "provenance", "equilibrium the core_profiles were mapped on"),
    "time_profile_s": ("s", "provenance", "paired core_profiles slice time"),
    "dt_s": ("s", "provenance", "time_profile_s - time_equilibrium; |dt| <= tolerance_s"),
    "tolerance_s": ("s", "provenance", "allowed pairing window"),
    "state_identity": ("-", "provenance", "sha256 of the resolved upstream state (shared by TGLF and NEO rows)"),
    "rho_tor_norm": ("-", "derived_input", "sqrt(Phi/Phi_b) of the surface: GACODEProfile.rho, which prepare_gacode_profile re-derives from phi or q when the stored value is the sqrt(psi_N) proxy, mapped through the profile's rmin<->rho"),
    "a_m": ("m", "reconstructed_input", "minor radius a of the cut profile"),
    "b_unit_T": ("T", "derived_input", "GACODE B_unit at the surface"),
    "q": ("-", "reconstructed_input", "safety factor"),
    "shear": ("-", "derived_input", "magnetic shear s = (r/q) dq/dr"),
    "kappa": ("-", "reconstructed_input", "elongation"),
    "delta": ("-", "reconstructed_input", "triangularity"),
    "betae": ("-", "derived_input", "TGLF BETAE (electron beta on B_unit)"),
    "xnue": ("c_s/a", "derived_input", "TGLF XNUE electron-ion collision frequency"),
    "zeff": ("-", "derived_input", "TGLF ZEFF at the surface"),
    "a_over_lne": ("-", "measured_input", "a/L_ne = -a dln(ne)/dr (Thomson fit)"),
    "a_over_lte": ("-", "measured_input", "a/L_Te (Thomson fit)"),
    "a_over_lti": ("-", "assumed_input", "a/L_Ti of the main ion (follows Te under a ratio policy)"),
    "a_over_lni": ("-", "derived_input", "a/L_ni of the main ion"),
    "ti_over_te": ("-", "assumed_input", "main-ion Ti/Te at the surface (TGLF TAUS_2)"),
    "q_gb_W_m2": ("W/m^2", "derived_input", "gyro-Bohm energy-flux unit ne Te c_s (rho_s/a)^2"),
    "tglf_status": ("-", "provenance", "solved | failed | not_ready (per surface)"),
    "tglf_reason": ("-", "provenance", "readiness reason or runtime status when not solved"),
    "tglf_run_identity": ("-", "provenance", "sha256 of state + TGLF physics settings + surface + revision"),
    "qe_gb": ("Q_GB", "tglf_predicted", "electron energy flux, gyro-Bohm units"),
    "qi_gb": ("Q_GB", "tglf_predicted", "ion energy flux summed over ion species, gyro-Bohm units"),
    "gamma_e_gb": ("Gamma_GB", "tglf_predicted", "electron particle flux, gyro-Bohm units"),
    "q_tot_gb": ("Q_GB", "tglf_predicted", "qe_gb + qi_gb"),
    "qe_tglf_W_m2": ("W/m^2", "tglf_predicted", "electron energy flux via core_transport_from_tglf"),
    "qi_tglf_W_m2": ("W/m^2", "tglf_predicted", "ion energy flux summed over species, SI"),
    "gamma_e_tglf_m2_s": ("m^-2 s^-1", "tglf_predicted", "electron particle flux, SI"),
    "f_e": ("-", "tglf_predicted", "|Qe| / (|Qe| + |Qi|), TGLF heat-channel fraction"),
    "gamma_max": ("c_s/a", "tglf_predicted", "largest linear growth rate over ky and modes"),
    "ky_at_gamma_max": ("-", "tglf_predicted", "ky rho_s at gamma_max"),
    "omega_at_gamma_max": ("c_s/a", "tglf_predicted", "real frequency at gamma_max (TGLF sign convention, unlabelled)"),
    "gamma_max_ion_scale": ("c_s/a", "tglf_predicted", "largest growth rate with ky rho_s <= 1"),
    "ky_at_gamma_max_ion_scale": ("-", "tglf_predicted", "ky rho_s at gamma_max_ion_scale"),
    "omega_at_gamma_max_ion_scale": ("c_s/a", "tglf_predicted", "real frequency at gamma_max_ion_scale"),
    "n_unstable_ky": ("-", "tglf_predicted", "ky points with a growing mode (gamma > 0)"),
    "ky_q_mean": ("-", "tglf_predicted", "energy-flux-weighted mean ky rho_s (see module docstring)"),
    "f_em": ("-", "tglf_predicted", "|Q_mag| / (|Q_phi| + |Q_mag|), signed sums over species and ky; empty when electrostatic only"),
    "tglf_config": ("-", "provenance", "tglf-sat<n>-<field>: the one configuration this atlas was built from (#1482)"),
    "tglf_sat_rule": ("-", "provenance", "TGLF SAT_RULE (no default: named by the run, #1482)"),
    "tglf_use_bper": ("-", "provenance", "TGLF USE_BPER"),
    "tglf_use_bpar": ("-", "provenance", "TGLF USE_BPAR"),
    "gacode_revision": ("-", "provenance", "GACODE tree the runs used"),
    "tglf_native_dir": ("-", "provenance", "native TGLF directory, relative to the TGLF run root"),
    "neo_status": ("-", "provenance", "solved | failed | not_ready | missing (per state)"),
    "qe_neo_W_m2": ("W/m^2", "neo_predicted", "NEO electron energy flux via core_transport_from_neo"),
    "qi_neo_W_m2": ("W/m^2", "neo_predicted", "NEO ion energy flux summed over species"),
    "gamma_e_neo_m2_s": ("m^-2 s^-1", "neo_predicted", "NEO electron particle flux"),
    "qe_classical_W_m2": ("W/m^2", "classical_predicted", "Braginskii perpendicular electron heat flux (#1435)"),
    "qi_classical_W_m2": ("W/m^2", "classical_predicted", "Braginskii perpendicular main-ion heat flux (#1435)"),
    "chi_e_classical_m2_s": ("m^2/s", "classical_predicted", "gamma_1'(Z_eff) T_e / (m_e Omega_e^2 tau_e), |B| = B_T at the surface"),
    "chi_i_classical_m2_s": ("m^2/s", "classical_predicted", "2 T_i / (m_i Omega_i^2 tau_i), tau_i against every ion species"),
    "classical_reason": ("-", "provenance", "why the classical baseline is absent, when it is"),
    "partition_status": ("-", "provenance", "available, or the reason the NEO/TGLF join was refused"),
    "f_classical_qe": ("-", "model_derived", "|Qe_cl| / (|Qe_cl| + |Qe_neo| + |Qe_tglf|)"),
    "f_classical_qi": ("-", "model_derived", "|Qi_cl,main| / (|Qi_cl,main| + |sum Qi_neo| + |sum Qi_tglf|)"),
    "f_neo_qe": ("-", "model_derived", "|Qe_neo| / sum of |Qe| over the components present"),
    "f_neo_qi": ("-", "model_derived", "|sum Qi_neo| / (|sum Qi_neo| + |sum Qi_tglf| + |Qi_cl,main|), ions summed per model"),
    "f_neo_gamma_e": ("-", "model_derived", "|Ge_neo| / (|Ge_neo| + |Ge_tglf|); classical has no particle flux"),
    "qe_model_W_m2": ("W/m^2", "model_derived", "Qe_neo + Qe_tglf (+ Qe_classical): modelled sum, not a measurement"),
    "qi_model_W_m2": ("W/m^2", "model_derived", "sum Qi_neo + sum Qi_tglf + Qi_cl,main: modelled sum, not a measurement"),
}

def _labels() -> dict[str, tuple[str, str]]:
    """Symbols and display units, from the renderer module: one source for every label."""
    from vaft.plot.transport_atlas import LABELS

    return LABELS


COLUMNS = tuple(SCHEMA)


def _f(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def spectral_descriptors(native: Any) -> dict[str, Any]:
    """Continuous TGLF descriptors from one native container (see module docstring)."""
    out: dict[str, Any] = {}
    gbflux = getattr(native, "gbflux", None) or {}
    energy = np.asarray(gbflux.get("energy", []), dtype=float)
    particle = np.asarray(gbflux.get("particle", []), dtype=float)
    if energy.size >= 2:
        out["qe_gb"] = float(energy[0])
        out["qi_gb"] = float(energy[1:].sum())
        out["q_tot_gb"] = out["qe_gb"] + out["qi_gb"]
    if particle.size >= 1:
        out["gamma_e_gb"] = float(particle[0])
    ky = getattr(native, "ky_spectrum", None)
    gamma = getattr(native, "growth_rate", None)
    omega = getattr(native, "frequency", None)
    if ky is not None and gamma is not None and omega is not None and gamma.shape[0] == ky.size:
        out["n_unstable_ky"] = int(np.sum(np.any(np.nan_to_num(gamma, nan=-1.0) > 0.0, axis=1)))
        for suffix, mask in (("", np.ones(ky.size, bool)), ("_ion_scale", ky <= 1.0)):
            if not mask.any():
                continue
            sub = np.where(mask[:, None] & np.isfinite(gamma), gamma, -np.inf)
            if not np.isfinite(sub).any():
                continue
            k, m = np.unravel_index(int(np.argmax(sub)), sub.shape)
            out[f"gamma_max{suffix}"] = _f(gamma[k, m])
            out[f"ky_at_gamma_max{suffix}"] = _f(ky[k])
            out[f"omega_at_gamma_max{suffix}"] = _f(omega[k, m])
    spectrum = getattr(native, "sum_flux_spectrum", None)
    if spectrum is not None and ky is not None and spectrum.shape[2] == ky.size:
        q = np.nan_to_num(spectrum[..., 1])               # (species, field, ky); NaN = absent
        # Signed sums first, then magnitudes: an inward and an outward species flux at
        # one ky cancel, as they do in the total TGLF reports.
        per_ky = np.abs(q.sum(axis=(0, 1)))
        if per_ky.sum() > 0:
            out["ky_q_mean"] = float(np.sum(ky * per_ky) / per_ky.sum())
        if q.shape[1] > 1:
            electrostatic = abs(float(q[:, 0, :].sum()))
            magnetic = abs(float(q[:, 1:, :].sum()))
            if electrostatic + magnetic > 0:
                out["f_em"] = magnetic / (electrostatic + magnetic)
    return out


def species_charges(names: Iterable[str]) -> Optional[dict[str, float]]:
    """Charge from GACODE species labels: ``H+`` -> 1, ``C6+`` -> 6 (electrons skipped).

    ``None`` when any ion label does not state its charge that way, so an unparsable
    species list refuses the NEO join rather than guessing it.
    """
    import re

    charges = {}
    for name in names:
        if name in ("e", "e-", "electron"):
            continue
        match = re.fullmatch(r"([A-Za-z]+)(\d*)\+", str(name))
        if match is None:
            return None
        charges[str(name)] = float(match.group(2) or 1)
    return charges or None


def _local_columns(local: Optional[dict]) -> dict[str, Any]:
    if not local:
        return {}
    norm = local.get("normalisation") or {}
    ln, lt, taus = local.get("a_over_ln") or [], local.get("a_over_lt") or [], local.get("taus") or []
    return {
        "q": local.get("q"), "shear": local.get("shear"), "kappa": local.get("kappa"),
        "delta": local.get("delta"), "betae": local.get("betae"), "xnue": local.get("xnue"),
        "zeff": local.get("zeff"),
        "a_over_lne": ln[0] if ln else None, "a_over_lte": lt[0] if lt else None,
        "a_over_lni": ln[1] if len(ln) > 1 else None, "a_over_lti": lt[1] if len(lt) > 1 else None,
        "ti_over_te": taus[1] if len(taus) > 1 else None,
        "a_m": norm.get("a_m"), "b_unit_T": norm.get("b_unit_T"), "q_gb_W_m2": norm.get("q_gb_W_m2"),
    }


def _neo_index(neo_root: Optional[Path]) -> dict[tuple, dict]:
    if neo_root is None:
        return {}
    index: dict[str, dict] = {}
    where: dict[str, Path] = {}
    for path in sorted(Path(neo_root).glob("*/*/*/state.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        identity = state["state_identity"]
        if identity in index:
            raise ValueError(f"two NEO records share state_identity {identity[:12]}: "
                             f"{where[identity]} and {path}")
        index[identity], where[identity] = state, path
    return index


def build_rows(tglf_root: Path, neo_root: Optional[Path] = None) -> list[dict[str, Any]]:
    """Every TGLF surface of every state as one atlas row, joined to NEO by state identity."""
    from vaft.code.gacode.tglf.outputs import TglfOutputs
    from vaft.process.transport_state import EFIT_QUALITIES, transport_partition

    neo_states = _neo_index(neo_root)
    rows = []
    for path in sorted(Path(tglf_root).glob("*/*/*/state.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["efit_quality"] not in EFIT_QUALITIES:  # never expected; refuse to publish one
            raise ValueError(f"{path}: efit_quality {state['efit_quality']!r} is not good/admissible")
        times, ti, settings = state.get("times", {}), state.get("ti", {}), state.get("settings", {})
        provenance = state.get("provenance") or {}
        base = {
            "shot": state["shot"], "time_efit_s": state["time_efit_s"],
            "efit_lineage": state["efit_lineage"], "efit_quality": state["efit_quality"],
            "quality_source": state["quality_source"], "ti_lineage": state.get("ti_lineage"),
            "ti_te_ratio": ti.get("ratio"),
            "composition_origin": (state.get("composition") or {}).get("origin"),
            "z_eff": settings.get("z_eff"),
            "shape_kind": (provenance.get("shape") or {}).get("kind"),
            "profile_mapped_on": (state.get("inputs") or {}).get("profile_mapped_on"),
            "time_profile_s": times.get("time_profile_s"), "dt_s": times.get("dt_s"),
            "tolerance_s": times.get("tolerance_s"), "state_identity": state["state_identity"],
            "tglf_sat_rule": (state.get("tglf_parameters") or {}).get("sat_rule"),
            "tglf_use_bper": (state.get("tglf_parameters") or {}).get("use_bper"),
            "tglf_use_bpar": (state.get("tglf_parameters") or {}).get("use_bpar"),
            "tglf_config": state.get("tglf_config"),
            "gacode_revision": state.get("gacode_revision"),
        }
        mapped = {round(s["r_over_a"], 4): s for s in ((state.get("core_transport") or {}).get("surfaces") or [])}
        neo = neo_states.get(state["state_identity"])
        neo_rows = {}
        if neo is not None and neo.get("status") in ("solved", "partial"):
            neo_rows = {round(s["r_over_a"], 4): s for s in (neo.get("core_transport") or {}).get("surfaces") or []}
        charges = None
        for surface in state["surfaces"]:
            r = float(surface["r_over_a"])
            row = dict(base, r_over_a=r, **_local_columns(surface.get("local")))
            status = surface.get("status")
            row["tglf_status"] = status
            row["tglf_reason"] = None if status == "solved" else (
                surface.get("runtime_status") if status == "failed" else surface.get("readiness"))
            row["tglf_run_identity"] = surface.get("run_identity")
            si = mapped.get(round(r, 4))
            if status == "solved":
                rel = path.parent.relative_to(tglf_root) / f"r{r:.2f}"
                row["tglf_native_dir"] = str(rel)
                native = TglfOutputs.read_json(Path(tglf_root) / rel / "outputs.json")
                row.update(spectral_descriptors(native))
            if si is not None:
                row["rho_tor_norm"] = si["rho_tor_norm"]
                row["qe_tglf_W_m2"] = si["electron_energy_flux_W_m2"]
                row["qi_tglf_W_m2"] = _f(sum(v for v in si["ion_energy_flux_W_m2"].values() if v is not None))
                row["gamma_e_tglf_m2_s"] = si["electron_particle_flux_m2_s"]
                qe, qi = _f(row["qe_tglf_W_m2"]), _f(row["qi_tglf_W_m2"])
                if qe is not None and qi is not None and abs(qe) + abs(qi) > 0:
                    row["f_e"] = abs(qe) / (abs(qe) + abs(qi))
                if charges is None:
                    charges = species_charges((surface.get("local") or {}).get("species") or [])
            # NEO join: same state identity, same surface
            row["neo_status"] = "missing" if neo is None else neo.get("status")
            ns = neo_rows.get(round(r, 4))
            if ns is not None:
                row["qe_neo_W_m2"] = ns["electron_energy_flux_W_m2"]
                row["qi_neo_W_m2"] = _f(sum(v for v in ns["ion_energy_flux_W_m2"].values() if v is not None))
                row["gamma_e_neo_m2_s"] = ns["electron_particle_flux_m2_s"]
            classical = surface.get("classical")
            row["classical_reason"] = surface.get("classical_error")
            if classical is not None:
                row["qe_classical_W_m2"] = classical["electron_energy_flux_W_m2"]
                row["qi_classical_W_m2"] = _f(sum(classical["ion_energy_flux_W_m2"].values()))
                row["chi_e_classical_m2_s"] = classical.get("chi_e_m2_s")
                row["chi_i_classical_m2_s"] = classical.get("chi_i_m2_s")
            if si is not None and ns is not None and charges is not None:
                part = transport_partition(ns, si, turbulent_charges=charges, classical=classical)
                row["partition_status"] = part["status"] if part["status"] == "available" else part["reason"]
                if part["status"] == "available":
                    ch = part["channels"]
                    row["f_neo_qe"] = ch.get("electron_energy", {}).get("f_neo")
                    row["f_neo_qi"] = ch.get("ion_energy_total", {}).get("f_neo")
                    row["f_neo_gamma_e"] = ch.get("electron_particle", {}).get("f_neo")
                    row["qe_model_W_m2"] = ch.get("electron_energy", {}).get("model")
                    row["f_classical_qe"] = ch.get("electron_energy", {}).get("f_classical")
                    # Ions: NEO and TGLF over every ion species, classical for the main
                    # ion only (its impurity heat flux is not modelled, #1435).
                    total = ch.get("ion_energy_total")
                    qi_cl = _f(row.get("qi_classical_W_m2"))
                    if total is not None:
                        parts = [total["neo"], total["turb"]] + ([qi_cl] if qi_cl is not None else [])
                        row["qi_model_W_m2"] = float(sum(parts))
                        denominator = sum(abs(v) for v in parts)
                        if denominator > 0:
                            row["f_neo_qi"] = abs(total["neo"]) / denominator
                            if qi_cl is not None:
                                row["f_classical_qi"] = abs(qi_cl) / denominator
            elif si is None:
                row["partition_status"] = "missing_turbulent_component"
            elif ns is None:
                row["partition_status"] = "missing_neoclassical_component"
            else:
                row["partition_status"] = "unparsable_species_labels"
            rows.append(row)
    return rows


def _median(values: Iterable[Any]) -> Optional[float]:
    clean = [v for v in (_f(x) for x in values) if v is not None]
    return float(np.median(clean)) if clean else None


def radial_summary(rows: list[dict]) -> list[dict]:
    """One row per state: how many surfaces solved and where the transport sits radially."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["shot"], row["time_efit_s"], row["efit_lineage"]), []).append(row)
    out = []
    for (shot, t, lineage), group in sorted(groups.items()):
        solved = [r for r in group if r.get("tglf_status") == "solved"]
        fe = [r["f_e"] for r in solved if _f(r.get("f_e")) is not None]
        gam = [(r["gamma_max_ion_scale"], r["r_over_a"]) for r in solved
               if _f(r.get("gamma_max_ion_scale")) is not None]
        qtot = [r.get("q_tot_gb") for r in solved]
        fneo = [r.get("f_neo_qi") for r in group]
        # The largest growth rate; on a tie the outer surface is reported.
        top = max(gam, key=lambda pair: (pair[0], pair[1])) if gam else (None, None)
        out.append({
            "shot": shot, "time_efit_s": t, "efit_lineage": lineage,
            "efit_quality": group[0]["efit_quality"], "ti_lineage": group[0]["ti_lineage"],
            "n_surfaces": len(group), "n_tglf_solved": len(solved),
            "fraction_electron_dominated": (sum(v > 0.5 for v in fe) / len(fe)) if fe else None,
            "gamma_max_ion_scale": top[0], "r_over_a_at_gamma_max_ion_scale": top[1],
            "median_q_tot_gb": _median(qtot), "median_f_neo_qi": _median(fneo),
            "neo_status": group[0].get("neo_status"),
        })
    return out


def discharge_summary(radial: list[dict]) -> list[dict]:
    """One row per (shot, lineage): the states' radial summaries aggregated."""
    groups: dict[tuple, list[dict]] = {}
    for row in radial:
        groups.setdefault((row["shot"], row["efit_lineage"]), []).append(row)
    out = []
    for (shot, lineage), group in sorted(groups.items()):
        times = [r["time_efit_s"] for r in group]
        out.append({
            "shot": shot, "efit_lineage": lineage, "n_states": len(group),
            "n_good": sum(r["efit_quality"] == "good" for r in group),
            "time_first_s": min(times), "time_last_s": max(times),
            "median_fraction_electron_dominated": _median(r["fraction_electron_dominated"] for r in group),
            "median_gamma_max_ion_scale": _median(r["gamma_max_ion_scale"] for r in group),
            "median_q_tot_gb": _median(r["median_q_tot_gb"] for r in group),
            "median_f_neo_qi": _median(r["median_f_neo_qi"] for r in group),
        })
    return out


def _write_csv(path: Path, rows: list[dict], columns: Optional[Iterable[str]] = None) -> None:
    import csv

    columns = list(columns or (rows[0].keys() if rows else []))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in columns})


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tglf", type=Path, required=True, help="run_tglf.py output root")
    parser.add_argument("--neo", type=Path, help="run_neo.py output root")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = build_rows(args.tglf, args.neo)
    if not rows:
        raise ValueError(f"no TGLF state.json under {args.tglf}")
    configs = sorted({row.get("tglf_config") or "" for row in rows})
    settings = sorted({(row.get("tglf_sat_rule"), row.get("tglf_use_bper"), row.get("tglf_use_bpar"))
                       for row in rows}, key=str)
    if len(configs) != 1 or not configs[0] or len(settings) != 1:
        # One atlas is one named TGLF configuration; comparing them is build_sensitivity.py's job.
        raise ValueError(f"{args.tglf} is not one named TGLF configuration: labels {configs}, "
                         f"(SAT_RULE, USE_BPER, USE_BPAR) {settings}; pass one tglf-sat*/ tree")
    unknown = sorted({k for row in rows for k in row} - set(COLUMNS))
    if unknown:
        raise ValueError(f"columns without a schema entry: {unknown}")
    for row in rows:  # the pairing window is part of the contract, so it is checked here
        if row.get("dt_s") is not None and (row.get("tolerance_s") is None
                                            or abs(row["dt_s"]) > row["tolerance_s"] + 1e-12):
            raise ValueError(f"{row['shot']} {row['time_efit_s']}: |dt| exceeds (or has no) tolerance")
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "atlas.csv", rows, COLUMNS)
    radial = radial_summary(rows)
    _write_csv(args.out / "radial_summary.csv", radial)
    _write_csv(args.out / "discharge_summary.csv", discharge_summary(radial))
    symbols = _labels()
    schema = {
        "description": "VAFT transport atlas (lane T, #1427): TGLF/NEO model predictions on "
                       "#1331 Tier A good/admissible states. Model output under declared "
                       "assumptions, not measurement.",
        "row_key": ["shot", "time_efit_s", "efit_lineage", "r_over_a"],
        "state_key": ["shot", "time_efit_s", "efit_lineage"],
        "join_key_across_models": "state_identity",
        "columns": {name: {"unit": u, "category": c, "definition": d,
                           **({"symbol": symbols[name][0]} if name in symbols else {})}
                    for name, (u, c, d) in SCHEMA.items()},
        "tglf_config": configs[0],
        "sources": {"tglf": str(args.tglf), "neo": None if args.neo is None else str(args.neo)},
        "counts": {"rows": len(rows), "states": len(radial)},
    }
    (args.out / "schema.json").write_text(json.dumps(schema, indent=1), encoding="utf-8")
    print(json.dumps(schema["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
