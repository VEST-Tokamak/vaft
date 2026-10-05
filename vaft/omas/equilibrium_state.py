"""Equilibrium-only plasma states as one canonical table, for operational-space plots (#1620).

:func:`equilibrium_state_table` turns ODS equilibria from any machine into one
row per time slice, with columns named by the quantity identities of
:mod:`vaft.formula.boundaries` (``edge_safety_factor_95``,
``internal_inductance_li3``, ``normalized_current``, ...) and their units in
``table.attrs["units"]``, which is the table
:func:`vaft.plot.operational_space.operational_space_population` reads. Nothing
here plots, and the renderer never reads an ODS.

Every value comes from the equilibrium alone:

* the convention is resolved and the slice converted to COCOS 11
  (:func:`vaft.process.equilibrium.as_equilibrium`,
  :func:`~vaft.process.equilibrium.convert_cocos`) before
  :func:`~vaft.process.equilibrium.derive_global_descriptors` reads it; a
  source whose COCOS is ambiguous takes the first candidate and says so in
  ``cocos_status``;
* ``internal_inductance_li3`` and ``normalized_beta`` are the IMAS DD forms of
  :func:`vaft.omas.update.update_equilibrium_global_quantities_beta_li`, run on a
  copy; where it writes nothing, ``li_3`` is the grid integral of
  :func:`vaft.formula.equilibrium.li_3_from_Bp2_volume_integral` and
  ``beta_N`` the descriptor, and ``li_beta_source`` says so. This is the
  derivation of the VEST atlas base table
  (``workflow/operational_space_atlas/build_efit_base.py``), so a VEST slice
  gives the atlas's numbers;
* shape-based coordinates (Freidberg and Menard kink safety factors, the ITER
  and START $q_{95}$ estimates, $1/q_{cyl}$) are evaluated by the registered
  coordinate functions on the slice's own shape, never by a restated formula.

A quantity the source cannot supply stays NaN; a slice that cannot be read
keeps its row with ``state_status = "failed"`` and the reason. A slice whose
psi map does not satisfy Ampere's law round the LCFS in the flux family it is
read in (:func:`vaft.process.cocos.identify_flux_exponent`) gets
``state_status = "flux_conflict"``: its ``li_3`` and $\\beta_p$ would be off by
the mismatch, so it is not a state, though its values stay in the row. The quantities
that are not equilibrium properties (density, loss power, ...) are not
columns, and neither are the straight-cylinder $q(a)$ and $l_i$ of Cheng et
al. (1987), which a toroidal equilibrium does not define.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = ["EQUILIBRIUM_STATE_UNITS", "equilibrium_state_rows", "equilibrium_state_table"]

#: Units of the quantity columns, as ``table.attrs["units"]`` declares them.
EQUILIBRIUM_STATE_UNITS = {
    "plasma_current": "MA",
    "toroidal_field": "T",
    "b0": "T",
    "reference_major_radius": "m",
    "major_radius": "m",
    "minor_radius": "m",
    "aspect_ratio": "-",
    "inverse_aspect_ratio": "-",
    "elongation": "-",
    "area_elongation": "-",
    "triangularity": "-",
    "triangularity_upper": "-",
    "triangularity_lower": "-",
    "normalized_shafranov_shift": "-",
    "plasma_surface_area": "m^2",
    "edge_safety_factor_95": "-",
    "edge_safety_factor": "-",
    "internal_inductance_li3": "-",
    "li3_grid_integral": "-",
    "normalized_beta": "% m T/MA",
    "toroidal_beta": "%",
    "poloidal_beta": "-",
    "normalized_current": "MA m^-1 T^-1",
    "inverse_cylindrical_q": "-",
    "kink_safety_factor_elliptic": "-",
    "kink_safety_factor_cylindrical": "-",
    "edge_safety_factor_95_estimate_iter": "-",
    "edge_safety_factor_95_estimate_start": "-",
}

#: Provenance columns, first in every table.
_PROVENANCE = ("machine", "machine_class", "dataset_source", "dataset_type", "source_format", "shot", "time_s",
               "time_index", "equilibrium_provenance", "cocos_source", "cocos_status", "cocos_target",
               "state_status", "state_reason", "li_beta_source", "li_beta_crosscheck")

#: Relative tolerance of the li_3 / beta_N cross-check, as in the atlas base table (grid-cell bias of a few %).
CROSSCHECK_RTOL = 0.10


def _leaf(ods, path):
    """A leaf, or None, without creating the path (omas reads create paths)."""
    return ods[path] if path in ods else None


def _bp2_volume_integral(ods, k: int) -> float:
    """int B_p^2 dV [T^2 m^3] over the grid cells inside the boundary outline of slice k."""
    from matplotlib.path import Path as _Path

    from vaft.data.eqdsk import ods_psi_to_wb_per_radian_factor

    ts = f"equilibrium.time_slice.{k}"
    grid = f"{ts}.profiles_2d.0"
    r = np.asarray(ods[f"{grid}.grid.dim1"], dtype=float)
    z = np.asarray(ods[f"{grid}.grid.dim2"], dtype=float)
    psi = np.asarray(ods[f"{grid}.psi"], dtype=float) * ods_psi_to_wb_per_radian_factor(ods)
    dpsi_dr, dpsi_dz = np.gradient(psi, r, z, edge_order=2)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    outline = np.c_[np.asarray(ods[f"{ts}.boundary.outline.r"]), np.asarray(ods[f"{ts}.boundary.outline.z"])]
    inside = _Path(outline).contains_points(np.c_[rr.ravel(), zz.ravel()]).reshape(rr.shape)
    dv = 2.0 * np.pi * rr * (r[1] - r[0]) * (z[1] - z[0])
    return float(np.sum((dpsi_dr**2 + dpsi_dz**2) / rr**2 * dv * inside))


def _dd_li_beta(ods, k: int):
    """li_3 and beta_normal of the DD routine on a copy; NaNs and a status when it raises or writes nothing."""
    from vaft.omas.update import update_equilibrium_global_quantities_beta_li

    work = copy.deepcopy(ods)
    gq = f"equilibrium.time_slice.{k}.global_quantities"
    for name in ("li_3", "beta_normal"):
        if f"{gq}.{name}" in work:
            del work[f"{gq}.{name}"]
    try:
        update_equilibrium_global_quantities_beta_li(work, time_slice=k)
    except Exception as exc:  # the independent path stands in; the status says why
        return math.nan, math.nan, f"raised {type(exc).__name__}"
    li3, bn = (_leaf(work, f"{gq}.{n}") for n in ("li_3", "beta_normal"))
    if li3 is None and bn is None:
        return math.nan, math.nan, "skipped"
    return (float(li3) if li3 is not None else math.nan, float(bn) if bn is not None else math.nan, "wrote")


def _crosscheck(pairs, routine_status: str) -> str:
    if routine_status != "wrote" or any(not (np.isfinite(a) and np.isfinite(b) and b != 0.0) for a, b in pairs):
        return "unavailable"
    return "agree" if all(abs(a / b - 1.0) <= CROSSCHECK_RTOL for a, b in pairs) else "disagree"


def _resolved(ods, k: int, cocos: Optional[int]):
    """The slice in COCOS 11, the source COCOS used and a status that keeps an ambiguity visible."""
    from vaft.process.equilibrium import as_equilibrium, convert_cocos

    stated = as_equilibrium(ods, time_index=k, convention=cocos)
    if cocos is not None:
        source, status = int(cocos), "asserted"
    else:
        candidates = tuple(stated.convention.identified or stated.convention.candidates or ())
        per_radian = stated.convention.psi_per_radian
        if per_radian is not None:   # the flux family is settled even when the signs leave the index open
            candidates = tuple(c for c in candidates if (c < 10) == per_radian) or candidates
        if not candidates:
            raise ValueError("COCOS could not be identified from the equilibrium's signs")
        source = int(candidates[0])
        status = "identified" if len(candidates) == 1 else f"ambiguous {candidates}: took {source}"
        stated = as_equilibrium(ods, time_index=k, convention=source)
    return convert_cocos(stated, 11), source, status


def _flux_consistency(ods, k: int) -> str:
    """Ampere's law round the LCFS against the flux family the ODS is read in; "" when they agree.

    ``li_3`` and ``beta_p`` scale with the poloidal field, so a psi map that does not carry ``mu0 |I_p|``
    round the boundary in the family it is read in (a g-file of the other family converted as if it
    were this one, or a file whose ``ip`` and psi disagree) would give them silently wrong.
    """
    from vaft.data.eqdsk import ods_psi_to_wb_per_radian_factor
    from vaft.process.cocos import identify_flux_exponent
    from vaft.process.equilibrium import as_equilibrium

    exponent, ratio = identify_flux_exponent(as_equilibrium(ods, time_index=k))
    stored = 1 if ods_psi_to_wb_per_radian_factor(ods, k) < 1.0 else 0
    if ratio is None:
        return ""   # nothing to measure: the check abstains
    if exponent is None:
        return f"I_p and the psi map disagree: loop integral of B_p over mu0 |I_p| is {ratio:.3g}, neither family"
    if exponent != stored:
        family = ("weber", "weber per radian")
        return (f"psi is read as {family[1 - stored]} but Ampere's law says {family[1 - exponent]} "
                f"(loop ratio {ratio:.3g})")
    return ""


def _slice_values(ods, k: int, cocos: Optional[int]) -> dict:
    from vaft.formula import boundaries as _b
    from vaft.formula.equilibrium import li_3_from_Bp2_volume_integral, q_cyl_from_B_R_epsilon_kappa_I
    from vaft.omas.update import resolve_reference_major_radius
    from vaft.process.equilibrium import derive_global_descriptors

    eq, source, status = _resolved(ods, k, cocos)
    d = derive_global_descriptors(eq).values
    pick = lambda n: float(d[n].value) if n in d and d[n].available else math.nan  # noqa: E731
    r_ref = float(resolve_reference_major_radius(ods))
    ip_a = abs(pick("ip"))
    li3_grid = li_3_from_Bp2_volume_integral(_bp2_volume_integral(ods, k), ip_a, r_ref)
    li3_dd, bn_dd, routine = _dd_li_beta(ods, k)
    bn_desc = pick("beta_n")
    use_dd = np.isfinite(li3_dd) and np.isfinite(bn_dd)
    ip_ma = ip_a * 1e-6
    a, r_geo = pick("minor_radius"), pick("major_radius")
    kappa, area = pick("elongation"), pick("cross_section_area")
    b0_all = np.atleast_1d(np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"], dtype=float))
    b0 = abs(float(b0_all[min(k, b0_all.size - 1)]))   # a constant field is stored once
    kappa_a = area / (math.pi * a * a)
    b_geo = b0 * r_ref / r_geo   # vacuum field at R_geo from the field at the reference radius
    delta = 0.5 * (pick("triangularity_upper") + pick("triangularity_lower"))
    shift = pick("shafranov_shift")
    return {
        "cocos_source": source,
        "cocos_status": status,
        "cocos_target": 11,
        "li_beta_source": "dd_update_routine" if use_dd else "grid_integral_and_descriptors",
        "li_beta_crosscheck": _crosscheck([(li3_dd, li3_grid), (bn_dd, bn_desc)], routine),
        "plasma_current": ip_ma,
        "toroidal_field": b_geo,
        "b0": b0,
        "reference_major_radius": r_ref,
        "major_radius": r_geo,
        "minor_radius": a,
        "aspect_ratio": r_geo / a,
        "inverse_aspect_ratio": pick("inverse_aspect_ratio"),
        "elongation": kappa,
        "area_elongation": kappa_a,
        "triangularity": delta,
        "triangularity_upper": pick("triangularity_upper"),
        "triangularity_lower": pick("triangularity_lower"),
        "normalized_shafranov_shift": shift / a,
        "plasma_surface_area": pick("surface_area"),
        # magnitudes: after COCOS 11 the sign of q is the helicity sgn(I_p B_T), not part of the op-space axis
        "edge_safety_factor_95": abs(pick("q95")),
        "edge_safety_factor": abs(pick("q_edge")),
        "internal_inductance_li3": li3_dd if use_dd else li3_grid,
        "li3_grid_integral": li3_grid,
        "normalized_beta": bn_dd if use_dd else bn_desc,
        "toroidal_beta": 100.0 * pick("beta_t"),
        "poloidal_beta": pick("beta_p_boundary_average"),
        "normalized_current": ip_ma / (a * b0),
        "inverse_cylindrical_q": 1.0 / q_cyl_from_B_R_epsilon_kappa_I(b_geo, r_geo, a / r_geo, kappa_a, ip_a),
        "kink_safety_factor_elliptic": float(_b.kink_coordinates(a, r_geo, b_geo, kappa, ip_ma)),
        "kink_safety_factor_cylindrical": float(_b.cylindrical_kink_coordinates(a, r_geo, b_geo, kappa, ip_ma)),
        "edge_safety_factor_95_estimate_iter": float(_b.iter_q95_coordinates(a, r_geo, b_geo, kappa, delta, ip_ma)),
        "edge_safety_factor_95_estimate_start": float(_b.start_q95_coordinates(a, r_geo, b_geo, kappa, delta,
                                                                               ip_ma)),
    }


def _equilibrium_provenance(ods) -> str:
    parts = []
    for path in ("equilibrium.code.name", "equilibrium.ids_properties.comment"):
        value = _leaf(ods, path)
        if value is not None and str(value).strip():
            parts.append(str(value).strip())
    return "; ".join(parts)


def equilibrium_state_rows(ods: Any, provenance: Optional[Mapping[str, Any]] = None, *,
                           time_indices: Optional[Sequence[int]] = None,
                           cocos: Optional[int] = None) -> list:
    """One row per equilibrium time slice of one ODS.

    Parameters
    ----------
    ods : omas.ODS
        Source with an ``equilibrium`` IDS; it is not modified.
    provenance : mapping, optional
        Columns copied into every row, e.g. ``machine``, ``machine_class``,
        ``dataset_source``, ``dataset_type``, ``source_format``, ``shot``.
    time_indices : sequence of int, optional
        Slices to read; default every slice.
    cocos : int, optional
        COCOS the caller asserts for the source; default: identified from its signs.

    Returns
    -------
    list of dict
        Provenance and quantity columns. A slice that cannot be read keeps its
        row with ``state_status = "failed"`` and ``state_reason``.
    """
    base = {name: None for name in _PROVENANCE}
    base.update(dict(provenance or {}))
    base["equilibrium_provenance"] = base.get("equilibrium_provenance") or _equilibrium_provenance(ods)
    if "equilibrium.time_slice" not in ods:
        return [{**base, "state_status": "failed", "state_reason": "no equilibrium.time_slice"}]
    n = len(ods["equilibrium.time_slice"])
    rows = []
    for k in (range(n) if time_indices is None else time_indices):
        row = dict(base, time_index=int(k))
        t = _leaf(ods, f"equilibrium.time_slice.{k}.time")
        if t is None and "equilibrium.time" in ods:
            times = np.atleast_1d(np.asarray(ods["equilibrium.time"], dtype=float))
            t = times[k] if k < times.size else None
        row["time_s"] = float(t) if t is not None else math.nan
        try:
            row.update(_slice_values(ods, int(k), cocos))
            conflict = _flux_consistency(ods, int(k))
            # the values stay in the row so the conflict can be inspected; only "valid" rows are states
            row.update(state_status="flux_conflict" if conflict else "valid", state_reason=conflict)
        except Exception as exc:  # recorded per row, never silently dropped
            row.update(state_status="failed", state_reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def equilibrium_state_table(entries: Iterable, *, cocos: Optional[Mapping[str, int]] = None):
    """The canonical equilibrium-state table of several sources.

    Parameters
    ----------
    entries : iterable of (ODS, mapping)
        Each source with its provenance columns (see :func:`equilibrium_state_rows`).
        A ``time_indices`` key in the mapping selects slices and is not copied.
    cocos : mapping of str to int, optional
        COCOS asserted per ``machine``; others are identified.

    Returns
    -------
    pandas.DataFrame
        One row per slice, failed slices included; ``attrs["units"]`` holds the
        quantity units (:data:`EQUILIBRIUM_STATE_UNITS`).
    """
    import pandas as pd

    rows = []
    for ods, provenance in entries:
        provenance = dict(provenance or {})
        indices = provenance.pop("time_indices", None)
        rows += equilibrium_state_rows(ods, provenance, time_indices=indices,
                                       cocos=(cocos or {}).get(provenance.get("machine")))
    columns = list(_PROVENANCE) + list(EQUILIBRIUM_STATE_UNITS)
    table = pd.DataFrame(rows)
    table = table.reindex(columns=columns + [c for c in table.columns if c not in columns])
    for name in EQUILIBRIUM_STATE_UNITS:
        table[name] = pd.to_numeric(table[name], errors="coerce")
    table.attrs["units"] = dict(EQUILIBRIUM_STATE_UNITS)
    return table
