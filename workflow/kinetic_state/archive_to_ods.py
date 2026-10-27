"""The matched kinetic-state archive snapshots as stored kinetic-state ODS.

A workflow-side reconstruction, not package code: the archive notebooks, the
plotting sample notebook and the tests import it by putting this directory on
``sys.path``.

The ``kinetic_overview_state`` plot reads stored IMAS paths only, under the
plot's stored-path contract: evidence roles come from the per-quantity
``*_fit.parameters`` records, ``T_i`` validity is carried by the data (NaN),
and the equilibrium lineage, occurrence and ``Z_eff`` target are ``key=value``
fields of those records.  The archived cases
(``archive_data/vest_40326_322ms.json`` and ``vest_39915_316ms.json``) are
not ODS files, so this converter rebuilds one, writing through the **real
writers** wherever one exists so that a drift in a writer breaks the tests:

* the C/O composition goes through
  :func:`vaft.process.impurity.populate_radial_impurity_profiles`, fed a
  :class:`~vaft.process.impurity.RadialImpurityComposition` assembled from the
  archived OpenADAS transient charge-state fractions and scale (the ADF11
  replay itself needs the tables, which tests must not download).  It writes
  the diluted main ion, one bundled entry per element with its charge-state
  densities, ``zeff`` and their ``origin=derived`` records;
* ``T_i`` is :func:`vaft.process.kinetic_closure.infer_ti_pressure_partition`,
  point by point, exactly as the notebooks call it, with its record spelled by
  :func:`vaft.machine_mapping.core_profiles.inferred_ti_text`; written into
  ``ion[0]`` before the composition writer, which carries it onto the
  impurity ions as the stages do.  On the electron-kinetic lineage the
  partition is refused (circular) and the slice carries that EFIT's own
  ``T_i = ratio T_e`` prior with a ``ti_te_ratio=...; status=assumed`` record.

Hand-built, because no writer exists yet (each says so in a comment below):

* the electron arrays and their ``coordinate=rho_tor_norm; method=external``
  fit records (the spelling ``vaft.process.profile`` writes for an external
  profile), ``electrons.pressure = e n_e T_e``;
* ``t_i_average`` / ``pressure_ion_total`` and the
  ``equilibrium_lineage=...; equilibrium_occurrence=...`` fields on the T_i
  records -- the stage writer
  (:func:`vaft.omas.kinetic_state_inference.kinetic_state_stages`) emits them,
  but it needs a multi-slice equilibrium that the archive does not keep --
  ``target_zeff=...`` on the ``zeff`` record now comes from the composition
  writer itself;
* the equilibrium slice and the Thomson channels.

**Thomson positions and the psi map are synthetic, and say so.**  The archive
keeps each channel's *mapped* coordinate, not the 2-D equilibrium it was mapped
through.  The plot maps every channel from its native ``(R, Z)`` through the
selected equilibrium, so the converter provides a map: the channels sit at the
archived VEST Thomson radii (``R_m``, ``Z = 0``) -- 39915 stores none and is
given the 40326 radii, the same system -- and ``psi_N(R, Z) = s(R)^2 + (Z/b)^2``
with ``s(R)`` the monotone (PCHIP) signed square root of the archived channel
``psi_N``.  Every channel maps back to its archived ``psi_N`` exactly (it sits on
a grid node); ``q = d(rho_tor^2)/d psi_N`` of the archived table reproduces the
equilibrium's ``rho_tor_norm(psi_N)``.  The map is not a Grad-Shafranov
solution; only its midplane and its 1-D tables are used.  For 39915 the
archive maps Thomson through the magnetic EFIT only, so its electron-kinetic
case places the channels at the *magnetic* ``psi_N``.

**Where the writer and the notebook differ.**  The composition writer fills a
point whose charge states are undefined (one of 129 in each case) from its
neighbours (``filled_points=1`` in the record), where the notebooks leave it
NaN.  The converter checks the written densities and ``Z_eff`` against the
notebook algebra at every *defined* point and raises if they differ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ARCHIVE = Path(__file__).resolve().parent / "archive_data"
CASES = {40326: "vest_40326_322ms.json", 39915: "vest_39915_316ms.json"}
LINEAGES = ("magnetics", "electron_kinetic")

#: Elementary charge [C], as the notebooks write it.
E = 1.602176634e-19

#: The VEST Thomson radii of the 40326 archive [m]; 39915 stores no positions.
_VEST_TS_R = (0.475, 0.425, 0.37, 0.31, 0.255)

#: Vertical scale of the synthetic psi map [m] (only the midplane is used).
_Z_SCALE = 0.6

#: The record ``vaft.process.profile`` writes beside a profile it did not fit itself.
_EXTERNAL_FIT_RECORD = "coordinate=rho_tor_norm; method=external"


@dataclass
class KineticStateCase:
    """The ODS and the notebook-algebra arrays it was built from."""

    ods: Any
    shot: int
    time: float
    lineage: str
    rho: np.ndarray
    p_eq: np.ndarray
    rho_eq: np.ndarray
    p_e: np.ndarray
    p_i: np.ndarray
    p_kin: np.ndarray
    n_e: np.ndarray
    n_h: np.ndarray
    n_imp: np.ndarray
    t_e: np.ndarray
    t_i: np.ndarray
    zeff: np.ndarray
    valid_charge: np.ndarray
    valid_ti: np.ndarray
    ts_rho: np.ndarray
    ts_psi_norm: np.ndarray
    ts_r: np.ndarray
    ts_p_e: np.ndarray
    ts_sigma_p_e: np.ndarray
    ts_n_e: np.ndarray
    ts_t_e: np.ndarray
    target_zeff: float
    notes: dict = field(default_factory=dict)


def load_archive(shot: int) -> dict:
    if shot not in CASES:
        raise ValueError(f"no archived kinetic state for shot {shot}; archived: {sorted(CASES)}")
    return json.loads((ARCHIVE / CASES[shot]).read_text())


# ---------------------------------------------------------------------------
# The notebook algebra, verbatim in substance
# ---------------------------------------------------------------------------


def _composition(s: dict, ne: np.ndarray, rho: np.ndarray, volume: np.ndarray) -> dict:
    """nH, charged C/O and Z_eff exactly as the archive notebooks compute them."""
    charge = s["openadas_charge_states"]
    model = s["impurity_model"]
    scale = float(model["scale"]) if "scale" in model else float(charge["scale"])
    fC = np.asarray(charge["C"], float)
    fO = np.asarray(charge["O"], float)
    qC, qO = np.arange(7), np.arange(9)
    valid_charge = np.isfinite(fC).all(axis=1) & np.isfinite(fO).all(axis=1)
    nC = nO = 0.5 * scale * ne
    nH = ne - nC * (fC @ qC) - nO * (fO @ qO)
    nC_ion = nC * (1 - fC[:, 0])
    nO_ion = nO * (1 - fO[:, 0])
    nimp = nC_ion + nO_ion
    nion = nH + nimp
    zeff = (nH + nC * (fC @ (qC * qC)) + nO * (fO @ (qO * qO))) / ne
    edges = np.r_[0, .5 * (rho[:-1] + rho[1:]), rho[-1]]
    dV = np.clip(np.diff(np.interp(edges, rho, volume)), 0, None)
    zeff_mean = np.nansum((ne * dV * zeff)[valid_charge]) / np.nansum((ne * dV)[valid_charge])
    return {"nH": nH, "nC_ion": nC_ion, "nO_ion": nO_ion, "nimp": nimp, "nion": nion,
            "zeff": zeff, "valid_charge": valid_charge, "zeff_mean": zeff_mean, "scale": scale,
            "fractions": (fC, fO), "target_zeff": float(model["target_zeff"]),
            "ionization": str(model.get("ionization", "transient")),
            "normalization": str(model.get("normalization", "ne_weighted_mean"))}


def _partition(peq, ne, te, comp, lineage):
    """T_i by pressure partition, point by point, as the notebooks call it."""
    from vaft.process.kinetic_closure import infer_ti_pressure_partition

    rho_size = ne.size
    ti = np.full(rho_size, np.nan)
    if lineage != "magnetics":
        refused = infer_ti_pressure_partition(
            peq[0], ne[0], te[0],
            composition={"ion_density_per_electron": float(comp["nion"][0] / ne[0]),
                         "composition_source": "transient C/O"},
            sigma_p_eq=np.nan, sigma_n_e=np.nan, sigma_t_e=np.nan, equilibrium_lineage=lineage)
        if refused["eligible"] or "circular" not in str(refused["reason"]):
            raise ValueError(f"the {lineage} partition was expected to be refused as circular: {refused!r}")
        return ti, np.zeros(rho_size, bool), str(refused["reason"])
    for j in np.flatnonzero(comp["valid_charge"]):
        result = infer_ti_pressure_partition(
            peq[j], ne[j], te[j],
            composition={"ion_density_per_electron": comp["nion"][j] / ne[j],
                         "composition_source": "transient C/O, equal elemental densities, mean Zeff=2"},
            sigma_p_eq=np.nan, sigma_n_e=np.nan, sigma_t_e=np.nan, equilibrium_lineage="magnetics")
        if not result["eligible"]:
            raise ValueError(f"the magnetic partition was refused at point {j}: {result.get('reason')!r}")
        ti[j] = float(np.asarray(result["t_i"]))
    valid_ti = comp["valid_charge"] & np.isfinite(ti) & (comp["nion"] > 0)
    return ti, valid_ti, ""


def _radial_composition(comp: dict, rho, te, ne, time, *, kind: str = "derived", normalization=None):
    """The archived charge states as the composition writer's input type."""
    from vaft.process.impurity import RadialImpurityComposition

    fC, fO = comp["fractions"]
    weights = np.array([0.5, 0.5])  # n_C = n_O, normalised as the resolver does
    mean = np.column_stack([fC @ np.arange(7), fO @ np.arange(9)])
    mean2 = np.column_stack([fC @ np.arange(7) ** 2, fO @ np.arange(9) ** 2])
    s1, s2 = mean @ weights, mean2 @ weights
    scale = comp["scale"]
    main = 1.0 - scale * s1
    nan = np.full(mean.shape, np.nan)
    return RadialImpurityComposition(
        kind=kind, rho=rho, te_eV=te, ne_m3=ne, elements=("C", "O"), weights=weights, scale=scale,
        elemental_fractions=np.broadcast_to(scale * weights, mean.shape).copy(),
        charge_state_fractions=(fC, fO), mean_charge=mean, mean_square_charge=mean2, S1=s1, S2=s2,
        effective_charge=s2 / s1, zeff=main + scale * s2, main_ion_fraction=main,
        dilution_fraction=1.0 - main, coronal_mean_charge=nan, relaxation_time_s=nan, coronal_valid=None,
        main_ion="H", time=time,
        normalization=normalization or {"method": comp["normalization"], "target_zeff": comp["target_zeff"]},
        provenance={"ionization": comp["ionization"], "source": "#1839 archive charge states"},
    )


# ---------------------------------------------------------------------------
# The synthetic psi map (see the module docstring)
# ---------------------------------------------------------------------------


def _signed_root(r: np.ndarray, psi_norm: np.ndarray) -> np.ndarray:
    """``+-sqrt(psi_N)`` per channel, signed so it rises with R as evenly as possible."""
    order = np.argsort(r)
    root = np.sqrt(np.clip(psi_norm[order], 0.0, None))
    best, best_spread = None, np.inf
    for split in range(root.size + 1):
        s = np.r_[-root[:split], root[split:]]
        slopes = np.diff(s) / np.diff(r[order])
        if np.any(slopes <= 0):
            continue
        spread = float(np.std(np.log(slopes)))
        if spread < best_spread:
            best, best_spread = s, spread
    if best is None:
        raise ValueError("no monotone signed root through the archived channel psi_N")
    out = np.empty_like(best)
    out[order] = best
    return out


def _psi_map(r_ts, psi_n_ts):
    from scipy.interpolate import PchipInterpolator

    s_ts = _signed_root(np.asarray(r_ts, float), np.asarray(psi_n_ts, float))
    order = np.argsort(r_ts)
    rr, ss = np.asarray(r_ts, float)[order], s_ts[order]
    interp = PchipInterpolator(rr, ss, extrapolate=False)
    lo_slope = (ss[1] - ss[0]) / (rr[1] - rr[0])
    hi_slope = (ss[-1] - ss[-2]) / (rr[-1] - rr[-2])
    # Grid nodes on every channel radius (multiples of 5 mm), so the bilinear
    # interpolation the mapper applies returns the node value exactly.
    r_grid = np.round(np.arange(0.05, 0.9 + 1e-9, 0.005), 6)
    z_grid = np.round(np.linspace(-0.8, 0.8, 161), 6)
    s = interp(r_grid)
    s = np.where(r_grid < rr[0], ss[0] + lo_slope * (r_grid - rr[0]), s)
    s = np.where(r_grid > rr[-1], ss[-1] + hi_slope * (r_grid - rr[-1]), s)
    psi_n = s[:, None] ** 2 + (z_grid[None, :] / _Z_SCALE) ** 2
    axis_r = float(np.interp(0.0, ss, rr)) if ss[0] < 0 < ss[-1] else float(rr[np.argmin(np.abs(ss))])
    return r_grid, z_grid, psi_n, axis_r


def _q_from_rho(psi_wb: np.ndarray, rho_tor: np.ndarray) -> np.ndarray:
    """A q profile whose integral reproduces ``rho_tor_norm(psi_N)`` (to trapezoid error)."""
    psi_n = (psi_wb - psi_wb[0]) / (psi_wb[-1] - psi_wb[0])
    return 3.0 * np.gradient(rho_tor**2, psi_n)


# ---------------------------------------------------------------------------
# The ODS
# ---------------------------------------------------------------------------


def _ti_te_ratio(s: dict) -> float:
    """The T_i/T_e ratio the archived electron-kinetic EFIT assumed."""
    block = s.get("state_assessment", {}).get("electron_kinetic") or s.get("electron_kinetic_assessment") or {}
    if "ti_te_ratio_assumed" not in block:
        raise ValueError("the archive records no T_i/T_e ratio for its electron-kinetic EFIT")
    return float(block["ti_te_ratio_assumed"])


def _check(name: str, written, expected, where) -> None:
    written = np.asarray(written, dtype=float)
    if not np.allclose(written[where], np.asarray(expected, dtype=float)[where], rtol=1e-10, atol=0.0):
        raise ValueError(f"the writer's {name} differs from the archive notebook algebra at defined points")


def kinetic_state_case(shot: int = 40326, lineage: str = "magnetics", *, occurrence: int | None = None,
                       records: bool = True, lineage_fields: bool = True) -> KineticStateCase:
    """One archived case as a plot-ready ODS plus the notebook arrays.

    ``lineage`` selects the equilibrium the ODS carries (``magnetics`` or
    ``electron_kinetic``); ``occurrence`` is recorded on the T_i records
    (default 0 for magnetics, 1 for electron-kinetic).  ``records=False``
    strips every ``*_fit.parameters`` record after writing, and
    ``lineage_fields=False`` leaves the lineage/occurrence fields off.
    """
    from omas import ODS

    from vaft.machine_mapping.core_profiles import inferred_ti_text
    from vaft.process.impurity import populate_radial_impurity_profiles

    if lineage not in LINEAGES:
        raise ValueError(f"lineage must be one of {LINEAGES}, got {lineage!r}")
    s = load_archive(shot)
    t = float(s["time_s"])
    cp = s["core_profiles"]
    eq_mag = s["magnetic_equilibrium"]
    eq = eq_mag if lineage == "magnetics" else s["electron_kinetic_equilibrium"]
    rho = np.asarray(cp["rho_tor_norm"], float)
    psi_cp = np.asarray(cp["psi_Wb"], float)
    ne = np.asarray(cp["electron_density_m3"], float)
    te = np.asarray(cp["electron_temperature_eV"], float)
    peq = np.asarray(eq["pressure_Pa"], float)
    rho_eq = np.asarray(eq["rho_tor_norm"], float)
    psi_eq = np.asarray(eq["psi_Wb"], float)
    volume = np.asarray(eq_mag["volume_m3"], float)  # the notebooks weight by the magnetic volume
    if occurrence is None:
        occurrence = 0 if lineage == "magnetics" else 1

    comp = _composition(s, ne, rho, volume)
    ti, valid_ti, refusal = _partition(peq, ne, te, comp, lineage)
    pe = E * ne * te
    pi = np.where(valid_ti, E * comp["nion"] * ti, np.nan)
    pkin = pe + pi

    # Thomson: the archived channels, at the archived (or 40326) radii.
    mapped = s["mapped_thomson"]  # 40326 keeps one mapping per equilibrium; 39915 one (magnetic) mapping
    ts = mapped[lineage] if isinstance(mapped.get(lineage), dict) else mapped
    ts_r = np.asarray(ts.get("R_m", _VEST_TS_R), float)
    ts_z = np.asarray(ts.get("Z_m", np.zeros(ts_r.size)), float)
    ts_rho = np.asarray(ts["rho_tor_norm"], float)
    if "psi_norm" in ts:
        ts_psi = np.asarray(ts["psi_norm"], float)
    else:  # 39915: invert the magnetic table the channels were mapped through
        psi_mag = np.asarray(eq_mag["psi_Wb"], float)
        ts_psi = np.interp(ts_rho, np.asarray(eq_mag["rho_tor_norm"], float),
                           (psi_mag - psi_mag[0]) / (psi_mag[-1] - psi_mag[0]))
    ne_ts = np.asarray(ts["electron_density_m3"], float)
    te_ts = np.asarray(ts["electron_temperature_eV"], float)
    sne_ts = np.asarray(ts["electron_density_error_m3"], float)
    ste_ts = np.asarray(ts["electron_temperature_error_eV"], float)
    p_ts = E * ne_ts * te_ts
    sp_ts = E * np.hypot(te_ts * sne_ts, ne_ts * ste_ts)

    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = shot
    ods["dataset_description.data_entry.machine"] = "VEST"

    # --- equilibrium (hand-built: the archive keeps 1-D tables only) ---------
    r_grid, z_grid, psi_n_map, axis_r = _psi_map(ts_r, ts_psi)
    psi_axis, psi_boundary = float(psi_eq[0]), float(psi_eq[-1])
    ods["equilibrium.ids_properties.homogeneous_time"] = 1
    ods["equilibrium.time"] = np.array([t])
    ods["equilibrium.vacuum_toroidal_field.r0"] = 0.4
    ods["equilibrium.vacuum_toroidal_field.b0"] = np.array([0.1])
    base = "equilibrium.time_slice.0"
    ods[f"{base}.time"] = t
    ods[f"{base}.global_quantities.psi_axis"] = psi_axis
    ods[f"{base}.global_quantities.psi_boundary"] = psi_boundary
    ods[f"{base}.global_quantities.magnetic_axis.r"] = axis_r
    ods[f"{base}.global_quantities.magnetic_axis.z"] = 0.0
    ods[f"{base}.profiles_1d.psi"] = psi_eq
    ods[f"{base}.profiles_1d.rho_tor_norm"] = rho_eq
    ods[f"{base}.profiles_1d.pressure"] = peq
    if "volume_m3" in eq:  # the 39915 kinetic snapshot keeps no volume
        ods[f"{base}.profiles_1d.volume"] = np.asarray(eq["volume_m3"], float)
    ods[f"{base}.profiles_1d.q"] = _q_from_rho(psi_eq, rho_eq)
    ods[f"{base}.profiles_2d.0.grid_type.index"] = 1
    ods[f"{base}.profiles_2d.0.grid.dim1"] = r_grid
    ods[f"{base}.profiles_2d.0.grid.dim2"] = z_grid
    ods[f"{base}.profiles_2d.0.psi"] = psi_axis + psi_n_map * (psi_boundary - psi_axis)
    ods["equilibrium.code.name"] = "efit"

    # --- core_profiles: the electron fit (hand-built; archived arrays) -------
    cpb = "core_profiles.profiles_1d.0"
    ods["core_profiles.ids_properties.homogeneous_time"] = 1
    ods["core_profiles.time"] = np.array([t])
    ods[f"{cpb}.time"] = t
    ods[f"{cpb}.grid.rho_tor_norm"] = rho
    ods[f"{cpb}.grid.psi"] = psi_cp
    ods[f"{cpb}.electrons.density"] = ne
    ods[f"{cpb}.electrons.density_thermal"] = ne
    ods[f"{cpb}.electrons.temperature"] = te
    ods[f"{cpb}.electrons.density_fit.parameters"] = _EXTERNAL_FIT_RECORD
    ods[f"{cpb}.electrons.temperature_fit.parameters"] = _EXTERNAL_FIT_RECORD
    ods[f"{cpb}.electrons.pressure"] = pe

    # --- the main-ion temperature, before the composition writer carries it --
    lineage_text = f"; equilibrium_lineage={lineage}; equilibrium_occurrence={occurrence}" if lineage_fields else ""
    if lineage == "magnetics":
        t_ion = ti
        t_record = inferred_ti_text("equilibrium_pressure_partition") + lineage_text
    else:
        ratio = _ti_te_ratio(s)
        t_ion = ratio * te  # the EFIT's own prior; its partition is refused as circular
        t_record = f"ti_te_ratio={ratio:g}; sigma=0.5; status=assumed; source=electron_efit prior" + lineage_text
    ods[f"{cpb}.ion.0.label"] = "H+"
    ods[f"{cpb}.ion.0.z_ion"] = 1.0
    ods[f"{cpb}.ion.0.element.0.z_n"] = 1.0
    ods[f"{cpb}.ion.0.element.0.a"] = 1.008
    ods[f"{cpb}.ion.0.element.0.atoms_n"] = 1
    ods[f"{cpb}.ion.0.temperature"] = t_ion
    ods[f"{cpb}.ion.0.temperature_fit.parameters"] = t_record

    # --- the composition: the real writer -----------------------------------
    radial = _radial_composition(comp, rho, te, ne, t)
    ods = populate_radial_impurity_profiles(ods, radial, time=t)
    defined = comp["valid_charge"]
    _check("main-ion density", ods[f"{cpb}.ion.0.density"], comp["nH"], defined)
    charged = sum(np.asarray(ods[f"{cpb}.ion.{k}.state.{q}.density"], float)
                  for k in (1, 2) for q in range(len(ods[f"{cpb}.ion.{k}.state"])))
    _check("charged impurity density", charged, comp["nimp"], defined)
    _check("Z_eff", ods[f"{cpb}.zeff"], comp["zeff"], defined)

    # --- t_i_average and pressure_ion_total (hand-built: the stage writer needs a multi-slice equilibrium)
    ods[f"{cpb}.t_i_average"] = t_ion
    ods[f"{cpb}.t_i_average_fit.parameters"] = t_record
    if lineage == "magnetics":
        n_ion = np.asarray(ods[f"{cpb}.ion.0.density"], float) + charged
        ods[f"{cpb}.pressure_ion_total"] = E * n_ion * t_ion
    ods["core_profiles.code.name"] = "workflow/kinetic_state/archive_to_ods.py (#1837)"
    if not records:
        for path in [p for p in ods.flat() if p.endswith("_fit.parameters")]:
            del ods[path]

    # --- thomson_scattering (hand-built; archived channels) ------------------
    ods["thomson_scattering.ids_properties.homogeneous_time"] = 1
    ods["thomson_scattering.time"] = np.array([t])
    for c in range(ts_r.size):
        ch = f"thomson_scattering.channel.{c}"
        ods[f"{ch}.name"] = f"TS{c}"
        ods[f"{ch}.position.r"] = float(ts_r[c])
        ods[f"{ch}.position.z"] = float(ts_z[c])
        ods[f"{ch}.position.phi"] = 0.0
        ods[f"{ch}.n_e.data"] = np.array([ne_ts[c]])
        ods[f"{ch}.n_e.data_error_upper"] = np.array([sne_ts[c]])
        ods[f"{ch}.t_e.data"] = np.array([te_ts[c]])
        ods[f"{ch}.t_e.data_error_upper"] = np.array([ste_ts[c]])

    return KineticStateCase(
        ods=ods, shot=shot, time=t, lineage=lineage, rho=rho, p_eq=peq, rho_eq=rho_eq,
        p_e=pe, p_i=pi, p_kin=pkin, n_e=ne, n_h=comp["nH"], n_imp=comp["nimp"], t_e=te, t_i=ti,
        zeff=comp["zeff"], valid_charge=comp["valid_charge"], valid_ti=valid_ti,
        ts_rho=ts_rho, ts_psi_norm=ts_psi, ts_r=ts_r, ts_p_e=p_ts, ts_sigma_p_e=sp_ts,
        ts_n_e=ne_ts, ts_t_e=te_ts, target_zeff=comp["target_zeff"],
        notes={"zeff_mean": comp["zeff_mean"], "refusal": refusal, "occurrence": occurrence},
    )


def kinetic_state_ods(shot: int = 40326, lineage: str = "magnetics", **options: Any):
    """Just the ODS of :func:`kinetic_state_case`."""
    return kinetic_state_case(shot, lineage, **options).ods


# ---------------------------------------------------------------------------
# The Romero-first chain: a SYNTHETIC variant on one archived single-slice case
# ---------------------------------------------------------------------------

#: The scalar Romero Z_eff the synthetic chain is given.  **Synthetic**: no
#: Romero flux balance was run for it; it stands in for the stage-4 product.
SYNTHETIC_ZEFF_ROMERO = 2.0
#: Its 1 sigma, as synthetic as the value.
SYNTHETIC_ZEFF_ROMERO_SIGMA = 0.2

#: The T_i record of the chain's assumption, in the composition grammar.
TI_EQUALS_TE_RECORD = "origin=assumed; method=ti_equals_te; assumption=T_i=T_e"


def romero_first_case(shot: int = 40326, *, zeff_romero: float = SYNTHETIC_ZEFF_ROMERO,
                      zeff_romero_sigma: float = SYNTHETIC_ZEFF_ROMERO_SIGMA) -> KineticStateCase:
    """The maintainer's Romero-first chain on one archived case, as a plot-ready ODS.

    Inputs are the archive's **magnetic equilibrium and Thomson** only, with
    the archived electron fits standing in for the stage-2 Thomson fits:

    1. data -- the Thomson channels and ``p_eq``, as :func:`kinetic_state_case`;
    2. ``n_e``, ``T_e`` fits -- ``coordinate=rho_tor_norm; method=external``;
    3. ``T_i = T_e`` -- ``t_i_average`` and ``ion[0].temperature`` equal to
       ``T_e``, record :data:`TI_EQUALS_TE_RECORD` (``origin=assumed``);
    4. Romero ``Z_eff`` -- ``zeff_romero`` in
       ``global_quantities.z_eff_resistive``, and ``resistive_target=``,
       ``resistive_sigma=`` (``zeff_romero_sigma``), ``resistive_method=romero;
       resistive_source=synthetic`` on every record stating the
       ``normalization=`` (main ion, C, O, ``zeff``).  **Synthetic**;
    5. ``n_I,eff`` -- C and O with ``n_C = n_O``, the amplitude closing the
       ``n_e``-weighted volume mean of ``Z_eff(rho)`` at ``zeff_romero`` (the
       archive's own normalization, with the Romero value as its target), through
       the real writer :func:`vaft.process.impurity.populate_radial_impurity_profiles`
       as an ``inferred`` composition (``method=romero_partition;
       normalization=resistive_closure``) with ``assumption=n_C=n_O`` appended
       to the impurity and ``zeff`` records.  The charge-state fractions are
       the archive's OpenADAS replay (the ADF11 tables are not downloaded in
       tests) -- an input beyond equilibrium + Thomson, stated here;
    6. ``n_i`` -- quasineutrality, written by the same writer;
    7. ``Z_eff(rho)`` -- the writer's ``zeff``;
    8. closure -- ``electrons.pressure = e n_e T_e``, ``ion[k].pressure = e
       n_k T_i`` (impurities by their charged density) and
       ``pressure_ion_total`` their sum, hand-built (no writer yet).

    ``p_e + p_i`` is **not** ``p_eq`` here: nothing partitions the equilibrium
    pressure, so the closure is a test, not an identity.
    """
    from omas import ODS  # noqa: F401  (the ODS comes from kinetic_state_case)

    from vaft.process.impurity import populate_radial_impurity_profiles

    zeff_romero = float(zeff_romero)
    if not np.isfinite(zeff_romero) or zeff_romero <= 1.0:
        raise ValueError(f"zeff_romero must be a finite Z_eff > 1; got {zeff_romero!r}")
    base_case = kinetic_state_case(shot, "magnetics", records=True)
    ods = base_case.ods
    s = load_archive(shot)
    t = base_case.time
    rho, ne, te = base_case.rho, base_case.n_e, base_case.t_e
    volume = np.asarray(s["magnetic_equilibrium"]["volume_m3"], float)

    # --- 5: n_C = n_O, the amplitude closing <Z_eff> at the Romero value -------
    unit = json.loads(json.dumps(s))
    unit["impurity_model"]["scale"] = 1.0
    shape = _composition(unit, ne, rho, volume)            # Z_eff = 1 + scale * excess
    excess = shape["zeff"] - 1.0
    edges = np.r_[0, .5 * (rho[:-1] + rho[1:]), rho[-1]]
    dV = np.clip(np.diff(np.interp(edges, rho, volume)), 0, None)
    valid = shape["valid_charge"]
    scale = (zeff_romero - 1.0) / (np.nansum((ne * dV * excess)[valid]) / np.nansum((ne * dV)[valid]))
    model = json.loads(json.dumps(s))
    model["impurity_model"].update(scale=float(scale), target_zeff=zeff_romero, normalization="resistive_closure")
    comp = _composition(model, ne, rho, volume)
    if not np.isclose(comp["zeff_mean"], zeff_romero, rtol=1e-10):
        raise ValueError(f"the closure gives <Z_eff> = {comp['zeff_mean']!r}, not {zeff_romero!r}")

    cpb = "core_profiles.profiles_1d.0"
    # Strip the archive's composition, T_i and pressures: this chain writes its own.
    for path in [p for p in ods.flat() if p.startswith(f"{cpb}.ion.") or p.startswith(f"{cpb}.t_i_average")
                 or p.startswith(f"{cpb}.zeff") or p.startswith(f"{cpb}.pressure_ion_total")]:
        del ods[path]
    ods[f"{cpb}.ion.0.label"] = "H+"
    ods[f"{cpb}.ion.0.z_ion"] = 1.0
    ods[f"{cpb}.ion.0.element.0.z_n"] = 1.0
    ods[f"{cpb}.ion.0.element.0.a"] = 1.008
    ods[f"{cpb}.ion.0.element.0.atoms_n"] = 1

    # --- 3: T_i = T_e (assumed) --------------------------------------------------
    lineage = "; equilibrium_lineage=magnetics; equilibrium_occurrence=0"
    ods[f"{cpb}.ion.0.temperature"] = te
    ods[f"{cpb}.ion.0.temperature_fit.parameters"] = TI_EQUALS_TE_RECORD + lineage
    ods[f"{cpb}.t_i_average"] = te
    ods[f"{cpb}.t_i_average_fit.parameters"] = TI_EQUALS_TE_RECORD + lineage

    # --- 5-7: the composition through the real writer ---------------------------
    radial = _radial_composition(comp, rho, te, ne, t, kind="inferred",
                                 normalization={"method": "resistive_closure", "resistive_target": zeff_romero})
    ods = populate_radial_impurity_profiles(ods, radial, time=t, method="romero_partition")
    defined = comp["valid_charge"]
    _check("main-ion density", ods[f"{cpb}.ion.0.density"], comp["nH"], defined)
    charged = {k: sum(np.asarray(ods[f"{cpb}.ion.{k}.state.{q}.density"], float)
                      for q in range(len(ods[f"{cpb}.ion.{k}.state"]))) for k in (1, 2)}
    _check("charged impurity density", charged[1] + charged[2], comp["nimp"], defined)
    _check("Z_eff", ods[f"{cpb}.zeff"], comp["zeff"], defined)
    # The closure's own value on every record that states the normalization
    # (the writer emits only normalization=resistive_closure yet).
    closure = (f"; resistive_target={zeff_romero:g}; resistive_sigma={zeff_romero_sigma:g}; "
               "resistive_method=romero; resistive_source=synthetic")
    for path in (f"{cpb}.ion.0.density_fit.parameters", f"{cpb}.ion.1.density_fit.parameters",
                 f"{cpb}.ion.2.density_fit.parameters", f"{cpb}.zeff_fit.parameters"):
        assumption = "" if path.endswith("ion.0.density_fit.parameters") else "; assumption=n_C=n_O"
        ods[path] = f"{ods[path]}{assumption}{closure}"

    # --- 4: the Romero scalar (SYNTHETIC) ----------------------------------------
    ods["core_profiles.global_quantities.z_eff_resistive"] = np.array([zeff_romero])

    # --- 8: the pressures (hand-built: no writer yet) -----------------------------
    n_h = np.asarray(ods[f"{cpb}.ion.0.density"], float)
    ods[f"{cpb}.ion.0.pressure"] = E * n_h * te
    for k in (1, 2):
        ods[f"{cpb}.ion.{k}.pressure"] = E * charged[k] * te
    pi = E * (n_h + charged[1] + charged[2]) * te
    ods[f"{cpb}.pressure_ion_total"] = pi
    ods["core_profiles.code.name"] = ("workflow/kinetic_state/archive_to_ods.py romero_first_case (#1957): "
                                      "SYNTHETIC Romero Z_eff, T_i = T_e, n_C = n_O")
    pe = E * ne * te
    return KineticStateCase(
        ods=ods, shot=shot, time=t, lineage="magnetics", rho=rho, p_eq=base_case.p_eq, rho_eq=base_case.rho_eq,
        p_e=pe, p_i=pi, p_kin=pe + pi, n_e=ne, n_h=comp["nH"], n_imp=comp["nimp"], t_e=te, t_i=te,
        zeff=comp["zeff"], valid_charge=defined, valid_ti=np.ones(rho.size, bool),
        ts_rho=base_case.ts_rho, ts_psi_norm=base_case.ts_psi_norm, ts_r=base_case.ts_r,
        ts_p_e=base_case.ts_p_e, ts_sigma_p_e=base_case.ts_sigma_p_e, ts_n_e=base_case.ts_n_e,
        ts_t_e=base_case.ts_t_e, target_zeff=zeff_romero,
        notes={"zeff_mean": comp["zeff_mean"], "scale": float(scale), "p_imp": E * (charged[1] + charged[2]) * te,
               "synthetic": ("z_eff_resistive (no Romero balance was run)",
                             "n_C = n_O composition closed at that value",
                             "T_i = T_e")},
    )

