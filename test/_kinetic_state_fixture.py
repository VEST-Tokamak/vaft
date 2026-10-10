"""Offline kinetic-state ODS fixtures from the #1839 archive snapshots (issue #1837).

The ``kinetic_overview_state`` plot reads stored IMAS paths only.  The two
archived cases (``workflow/kinetic_state/archive_data/vest_40326_322ms.json``
and ``vest_39915_316ms.json``) are not ODS files: they hold the selected
equilibrium and core-profile arrays, the mapped Thomson channels and the
transient C/O charge states, and the archive notebooks derive the ion
composition, ``T_i`` and ``Z_eff`` inline.  This converter does that
derivation -- **exactly as the notebooks do it**, here in the test tree and
never in the plot -- and stores the results into the paths the plot reads:

* ``equilibrium.time_slice[0]``: ``profiles_1d.{psi, rho_tor_norm, pressure,
  volume, q}``, ``global_quantities.{psi_axis, psi_boundary, magnetic_axis}``
  and a ``profiles_2d[0]`` psi map;
* ``core_profiles.profiles_1d[0]``: ``grid.{rho_tor_norm, psi}``,
  ``electrons.{density, density_thermal, temperature, pressure}``, one
  ``ion[k]`` per species (H+, C, O; charged densities), ``t_i_average``,
  ``pressure_ion_total`` and ``zeff``;
* ``core_profiles.code.parameters``: the JSON ``kinetic_state`` record
  proposed on #1837 (lineage, occurrence, evidence roles, ``t_i_valid`` on the
  core grid, composition preset and target);
* ``thomson_scattering``: one time sample, every channel's ``position.{r, z}``
  and ``n_e``/``t_e`` with ``data_error_upper``.

**Thomson positions and the psi map are synthetic, and say so.**  The archive
keeps each channel's *mapped* coordinate, not the 2-D equilibrium it was mapped
through.  The plot maps every channel from its native ``(R, Z)`` through the
selected equilibrium (``vaft.process.profile.equilibrium_mapping_points``), so
the fixture has to provide a map.  It builds the smallest honest one: the
channels sit at the archived VEST Thomson radii (``R_m``, ``Z = 0``) -- 39915
stores none, and is given the 40326 radii, the same system -- and
``psi_N(R, Z) = s(R)^2 + (Z / b)^2``, where ``s(R)`` is the monotone
(PCHIP) signed square root of the archived channel ``psi_N`` through those
radii.  Every channel therefore maps back to its archived ``psi_N`` exactly
(it sits on a grid node), and ``q`` is set to ``d(rho_tor^2)/d psi_N`` of the
archived ``rho_tor_norm`` table so the equilibrium's own ``rho_tor_norm(psi_N)``
reproduces that table.  The resulting channel ``rho_tor_norm`` agrees with the
archived mapping within the trapezoid error of that integral (tested), which
is what makes the comparison against the notebook numbers meaningful.  The map
is not a Grad-Shafranov solution and away from the midplane it is invented;
only its midplane and its 1-D flux tables are used.

For 39915 the archive maps Thomson through the magnetic EFIT only.  Its
electron-kinetic fixture places the channels at the *magnetic* ``psi_N``
(the kinetic mapping is not archived), so its Thomson ``rho_tor_norm`` there is
approximate; that case exists to test the refused ``T_i``, not the mapping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ARCHIVE = Path(__file__).resolve().parents[1] / "workflow" / "kinetic_state" / "archive_data"
CASES = {40326: "vest_40326_322ms.json", 39915: "vest_39915_316ms.json"}
LINEAGES = ("magnetics", "electron_kinetic")

#: Elementary charge [C], as the notebooks write it.
E = 1.602176634e-19

#: The VEST Thomson radii of the 40326 archive [m]; 39915 stores no positions.
_VEST_TS_R = (0.475, 0.425, 0.37, 0.31, 0.255)

#: Vertical scale of the synthetic psi map [m] (only the midplane is used).
_Z_SCALE = 0.6


@dataclass
class KineticStateCase:
    """The fixture ODS and the notebook-derived arrays it was built from."""

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
            "zeff": zeff, "valid_charge": valid_charge, "zeff_mean": zeff_mean,
            "target_zeff": float(model["target_zeff"])}


def _partition(peq, ne, te, comp, lineage):
    """T_i by pressure partition, point by point, as the notebooks call it."""
    from vaft.validation.kinetic_state import infer_ti_pressure_partition

    rho_size = ne.size
    ti = np.full(rho_size, np.nan)
    if lineage != "magnetics":
        refused = infer_ti_pressure_partition(
            peq[0], ne[0], te[0],
            composition={"ion_density_per_electron": float(comp["nion"][0] / ne[0]),
                         "composition_source": "transient C/O"},
            sigma_p_eq=np.nan, sigma_n_e=np.nan, sigma_t_e=np.nan, equilibrium_lineage=lineage)
        assert not refused["eligible"] and "circular" in refused["reason"]
        return ti, np.zeros(rho_size, bool), refused["reason"]
    for j in np.flatnonzero(comp["valid_charge"]):
        result = infer_ti_pressure_partition(
            peq[j], ne[j], te[j],
            composition={"ion_density_per_electron": comp["nion"][j] / ne[j],
                         "composition_source": "transient C/O, equal elemental densities, mean Zeff=2"},
            sigma_p_eq=np.nan, sigma_n_e=np.nan, sigma_t_e=np.nan, equilibrium_lineage="magnetics")
        assert result["eligible"]
        ti[j] = float(np.asarray(result["t_i"]))
    valid_ti = comp["valid_charge"] & np.isfinite(ti) & (comp["nion"] > 0)
    return ti, valid_ti, ""


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


def _roles(lineage: str) -> dict:
    independent = lineage == "magnetics"
    return {
        "equilibrium.pressure": "reconstruction",
        "electrons.density": "fit",
        "electrons.temperature": "fit",
        "electrons.pressure": "derived",
        "ion.density": "assumed",
        "zeff": "assumed",
        "composition": "assumed",
        "t_i_average": "inferred" if independent else "invalid",
        "pressure_ion_total": "inferred" if independent else "invalid",
        "p_kin": "closure identity" if independent else "invalid",
        "thomson_scattering.n_e": "fit input",
        "thomson_scattering.t_e": "fit input",
        "thomson_scattering.p_e": "independent validation" if independent else "fit input",
    }


def kinetic_state_case(shot: int = 40326, lineage: str = "magnetics", *, occurrence: int | None = None,
                       roles: bool = True, kinetic_state: bool = True) -> KineticStateCase:
    """One archived case as a plot-ready ODS plus the notebook arrays.

    ``lineage`` selects the equilibrium the ODS carries (``magnetics`` or
    ``electron_kinetic``); ``occurrence`` is recorded in ``kinetic_state`` (default
    0 for magnetics, 1 for electron-kinetic).  ``roles=False`` leaves the roles
    out of the record and ``kinetic_state=False`` writes no record at all.
    """
    from omas import ODS

    if lineage not in LINEAGES:
        raise ValueError(f"lineage must be one of {LINEAGES}")
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

    comp = _composition(s, ne, rho, volume)
    ti, valid_ti, refusal = _partition(peq, ne, te, comp, lineage)
    pe = E * ne * te
    pi = np.where(valid_ti, E * comp["nion"] * ti, np.nan)
    pkin = pe + pi

    # Thomson: the archived channels, at the archived (or 40326) radii.
    # 40326 keeps one mapping per equilibrium; 39915 one (magnetic) mapping.
    mapped = s["mapped_thomson"]
    ts = mapped[lineage] if isinstance(mapped.get(lineage), dict) else mapped
    ts_r = np.asarray(ts.get("R_m", _VEST_TS_R), float)
    ts_z = np.asarray(ts.get("Z_m", np.zeros(ts_r.size)), float)
    ts_rho = np.asarray(ts["rho_tor_norm"], float)
    if "psi_norm" in ts:
        ts_psi = np.asarray(ts["psi_norm"], float)
    else:  # 39915: invert the magnetic table the channels were mapped through
        psi_n_mag = (np.asarray(eq_mag["psi_Wb"], float) - eq_mag["psi_Wb"][0]) / (eq_mag["psi_Wb"][-1] - eq_mag["psi_Wb"][0])
        ts_psi = np.interp(ts_rho, np.asarray(eq_mag["rho_tor_norm"], float), psi_n_mag)
    ne_ts = np.asarray(ts["electron_density_m3"], float)
    te_ts = np.asarray(ts["electron_temperature_eV"], float)
    sne_ts = np.asarray(ts["electron_density_error_m3"], float)
    ste_ts = np.asarray(ts["electron_temperature_error_eV"], float)
    p_ts = E * ne_ts * te_ts
    sp_ts = E * np.hypot(te_ts * sne_ts, ne_ts * ste_ts)

    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = shot
    ods["dataset_description.data_entry.machine"] = "VEST"

    # --- equilibrium ---------------------------------------------------------
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

    # --- core_profiles -------------------------------------------------------
    cpb = "core_profiles.profiles_1d.0"
    ods["core_profiles.ids_properties.homogeneous_time"] = 1
    ods["core_profiles.time"] = np.array([t])
    ods[f"{cpb}.time"] = t
    ods[f"{cpb}.grid.rho_tor_norm"] = rho
    ods[f"{cpb}.grid.psi"] = psi_cp
    ods[f"{cpb}.electrons.density"] = ne
    ods[f"{cpb}.electrons.density_thermal"] = ne
    ods[f"{cpb}.electrons.temperature"] = te
    ods[f"{cpb}.electrons.pressure"] = pe
    for k, (label, z_n, a, density) in enumerate((
        ("H+", 1.0, 1.008, comp["nH"]),
        ("C", 6.0, 12.011, comp["nC_ion"]),
        ("O", 8.0, 15.999, comp["nO_ion"]),
    )):
        ods[f"{cpb}.ion.{k}.label"] = label
        ods[f"{cpb}.ion.{k}.element.0.z_n"] = z_n
        ods[f"{cpb}.ion.{k}.element.0.a"] = a
        ods[f"{cpb}.ion.{k}.density"] = np.where(comp["valid_charge"], density, np.nan)
    if lineage == "magnetics":
        ods[f"{cpb}.t_i_average"] = ti
        ods[f"{cpb}.pressure_ion_total"] = pi
    else:
        # The electron-kinetic EFIT's own prior, T_i = T_e: stored so the test
        # can prove the plot hides it when t_i_valid says it is not a result.
        ods[f"{cpb}.t_i_average"] = te.copy()
        ods[f"{cpb}.pressure_ion_total"] = E * comp["nion"] * te
    ods[f"{cpb}.zeff"] = np.where(comp["valid_charge"], comp["zeff"], np.nan)
    ods["core_profiles.code.name"] = "kinetic_state archive fixture (#1837)"
    if kinetic_state:
        record: dict[str, Any] = {
            "schema": 1,
            "equilibrium": {"lineage": lineage,
                            "occurrence": (0 if lineage == "magnetics" else 1) if occurrence is None else occurrence},
            "t_i_valid": [bool(v) for v in valid_ti],
            "composition": {"preset": "VEST transient C/O, n_C = n_O, <Z_eff>_ne dV",
                            "target_zeff": comp["target_zeff"]},
        }
        if refusal:
            record["t_i_refused"] = refusal
        if roles:
            record["roles"] = _roles(lineage)
        ods["core_profiles.code.parameters"] = json.dumps({"kinetic_state": record})

    # --- thomson_scattering --------------------------------------------------
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
        notes={"zeff_mean": comp["zeff_mean"], "refusal": refusal},
    )


def kinetic_state_ods(shot: int = 40326, lineage: str = "magnetics", **options: Any):
    """Just the ODS of :func:`kinetic_state_case`."""
    return kinetic_state_case(shot, lineage, **options).ods
