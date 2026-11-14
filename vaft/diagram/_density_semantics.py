"""Electron density in four representations: measured, stored, flux-mapped, midplane-cut (#2253).

``electron_density_semantics`` draws one time-matched sample -- by default
the packaged VEST kinetic slice, shot 48224 at 300 ms -- as

(a) ``n_e(rho_tor_norm)``, (b) ``n_e(psi_N)`` and (c) ``n_e(R, Z = 0)`` on the
left, and (d) the flux-surface-mapped ``n_e(R, Z)`` spanning all three rows on
the right. The four views are four *representations* of one quantity, and the
figure keeps their status apart:

* **measured**: Thomson scattering ``n_e +/- sigma_n``, ``sigma_n`` being the
  channel's ``data_error_upper``: a measurement uncertainty, not a confidence
  band of any fitted profile;
* **stored state**: ``core_profiles.profiles_1d[*].electrons.density`` on its
  own grid, as written; no profile is re-fitted here;
* **equilibrium-derived**: the Thomson positions in flux coordinates, the 2-D
  map (density constant on each flux surface: not a 2-D measurement), and the
  ``Z = 0`` cut of that map.

Above the panels a keyword strip names the responsibility boundaries the panels
pass through: scientific identity and IMAS data representation feed a semantic
adapter, which builds the typed view models (:class:`vaft.plot.models.Profile1D`,
:class:`vaft.plot.models.Field2D`); a scientific display policy
(:func:`vaft.plot.display.resolve_display`) and a graphical presentation act on
them separately. The scene takes every data value, error bar and validity flag from
those view models and the display policy, never from an IMAS path; the geometry it
overlays (boundary, wall, magnetic axis, channel positions in (d)) comes from the
same model. The axis and colour ranges are fixed for VEST and warn when a value
would be clamped. ``metadata["architecture_edges"]`` records the boundaries.

Every slice is chosen by physical time, never by index. The common radial
coordinate is ``psi_N``: the core profile is placed through its own
``grid.psi``, the channels through
:func:`vaft.process.profile.equilibrium_mapping_thomson_scattering`, and both go
to ``rho_tor_norm`` through the matched equilibrium's own
``profiles_1d.rho_tor_norm(psi)`` table -- the table ``core_profiles`` was
written on -- which is refused when it is the legacy ``sqrt(psi_N)`` proxy.
The 2-D state is :func:`vaft.process.equilibrium.psi_to_rz` masked by
:func:`vaft.process.equilibrium.plasma_cell_weights`; panel (c) is that same
array interpolated to ``Z = 0``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene

#: the packaged kinetic sample the canonical figure is drawn from
SAMPLE_SHOT = 48224
SAMPLE_TIME = 0.300
SAMPLE_PATH = Path("vaft") / "data" / "kineticEfit" / "ods_48224_300ms.json"

#: channel status at the chosen time, in the order a reader is told about them
USABLE = "usable"
NO_SAMPLE = "no_sample"
REJECTED = "rejected"
OUTSIDE = "outside_lcfs"
STATUSES = (USABLE, NO_SAMPLE, REJECTED, OUTSIDE)


@dataclass(frozen=True, eq=False)
class ThomsonPoints:
    """Thomson channels at one time, with their status and equilibrium-mapped positions.

    Every array has one entry per channel. ``sigma`` is ``data_error_upper``
    (NaN when the channel gives none) and ``sigma_lower`` ``data_error_lower``,
    which is ``sigma`` when the channel stores none; ``status`` is one of :data:`STATUSES`;
    ``psi_norm`` and ``rho_tor_norm`` are NaN outside the last closed flux
    surface; ``outboard`` is ``R > R_axis``, the side a flux coordinate forgets.
    """

    name: Tuple[str, ...]
    r: np.ndarray
    z: np.ndarray
    n_e: np.ndarray
    sigma: np.ndarray
    status: Tuple[str, ...]
    psi_norm: np.ndarray
    rho_tor_norm: np.ndarray
    outboard: np.ndarray
    on_midplane: np.ndarray
    time: float
    sigma_lower: np.ndarray = None

    @property
    def usable(self) -> np.ndarray:
        return np.array([s == USABLE for s in self.status], dtype=bool)


@dataclass(frozen=True, eq=False)
class DensitySemantics:
    """What the four panels draw, in physical units (m, m^-3, normalized flux)."""

    shot: Optional[int]
    machine: str
    time: float
    thomson: ThomsonPoints
    #: the stored core profile: its psi_N, rho_tor_norm and n_e
    profile_psi_norm: np.ndarray
    profile_rho_tor_norm: np.ndarray
    profile_n_e: np.ndarray
    #: the equilibrium's rho_tor_norm(psi_N) table that places both
    table_psi_norm: np.ndarray
    table_rho_tor_norm: np.ndarray
    #: the 2-D derived state on the equilibrium grid, indexed (R, Z); NaN outside the plasma
    grid_r: np.ndarray
    grid_z: np.ndarray
    map_n_e: np.ndarray
    #: the Z = 0 cut of ``map_n_e`` at the grid's R nodes
    midplane_r: np.ndarray
    midplane_n_e: np.ndarray
    lcfs_r: np.ndarray
    lcfs_z: np.ndarray
    axis_r: float
    axis_z: float
    metadata: Dict[str, object] = field(default_factory=dict)
    #: the machine's limiter outline (``wall.description_2d.0.limiter.unit.0``); empty when absent
    wall_r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wall_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: the typed scientific view models the panels are drawn from: ``"rho"``,
    #: ``"psi"`` and ``"mid"`` are :class:`vaft.plot.models.Profile1D`, ``"map"`` a
    #: :class:`vaft.plot.models.Field2D`; every plotted value comes from here
    views: Dict[str, object] = field(default_factory=dict)
    #: the scientific display policy, :func:`vaft.plot.display.resolve_display`
    display: object = None


# --------------------------------------------------------------------------
# selection by physical time
# --------------------------------------------------------------------------

def _array(ods, path):
    from vaft.ods_access import path_value

    value = path_value(ods, path)
    return None if value is None else np.asarray(value, dtype=float).reshape(-1)


def _nearest(times, time: float, tolerance: float, what: str) -> int:
    """Index of the sample of ``times`` at ``time``, or a refusal when none is within ``tolerance``."""
    times = np.asarray(times, dtype=float).reshape(-1)
    if not times.size or not np.isfinite(times).any():
        raise ValueError(f"{what} carries no time base; cannot match it to t = {time:g} s")
    index = int(np.nanargmin(np.abs(times - time)))
    if abs(times[index] - time) > tolerance:
        raise ValueError(f"{what} has no sample within {tolerance:g} s of t = {time:g} s "
                         f"(nearest {times[index]:g} s)")
    return index


def _validity(ods, prefix: str, index: int) -> int:
    """The channel's validity at sample ``index``: ``validity_timed`` there, else ``validity``, else 0."""
    timed = _array(ods, prefix + ".validity_timed")
    if timed is not None and index < timed.size and np.isfinite(timed[index]):
        return int(timed[index])
    whole = _array(ods, prefix + ".validity")
    if whole is not None and whole.size and np.isfinite(whole[0]):
        return int(whole[0])
    return 0


def thomson_points(ods, time: float, *, time_tolerance: float = 5e-4,
                   midplane_tolerance: float = 1e-3, equilibrium_time: Optional[float] = None) -> ThomsonPoints:
    """Thomson ``n_e +/- sigma_n`` of every channel at ``time``, with its status and flux position.

    Parameters
    ----------
    ods : ODS
        Carries ``thomson_scattering`` and the matched ``equilibrium`` [-].
    time : float
        The sample time; the Thomson sample nearest it is used [s].
    time_tolerance : float
        Largest accepted distance to that sample [s].
    midplane_tolerance : float
        Largest ``|Z|`` of a channel drawn at its own ``R`` in the midplane
        panel [m].
    equilibrium_time : float, optional
        The time of the equilibrium slice the channels are mapped through, so
        that they share the slice every other panel uses; ``None`` takes the
        slice nearest the Thomson sample [s].

    Returns
    -------
    ThomsonPoints
        A channel is ``rejected`` when its IMAS validity is negative, has
        ``no_sample`` when the value is not finite, is ``outside_lcfs`` when
        the equilibrium places it outside the plasma, and ``usable``
        otherwise [-].
    """
    from vaft.ods_access import path_value
    from vaft.process.profile import equilibrium_mapping_thomson_scattering

    index = _nearest(_array(ods, "thomson_scattering.time"), time, time_tolerance, "thomson_scattering")
    sample_time = float(_array(ods, "thomson_scattering.time")[index])
    count = len(ods["thomson_scattering.channel"])
    names, r, z, n_e, sigma, lower, valid = [], [], [], [], [], [], []
    for channel in range(count):
        prefix = f"thomson_scattering.channel.{channel}"
        names.append(str(path_value(ods, prefix + ".name") or f"channel {channel}"))
        r.append(float(ods[prefix + ".position.r"]))
        z.append(float(ods[prefix + ".position.z"]))
        data = _array(ods, prefix + ".n_e.data")
        error = _array(ods, prefix + ".n_e.data_error_upper")
        n_e.append(float(data[index]) if data is not None and index < data.size else math.nan)
        sigma.append(float(error[index]) if error is not None and index < error.size else math.nan)
        below = _array(ods, prefix + ".n_e.data_error_lower")
        lower.append(float(below[index]) if below is not None and index < below.size else sigma[-1])
        valid.append(_validity(ods, prefix + ".n_e", index))
    mapped = equilibrium_mapping_thomson_scattering(
        ods, ods, time=sample_time if equilibrium_time is None else equilibrium_time)
    psi_norm = np.asarray(mapped.psi_norm, dtype=float)
    status = []
    for k in range(count):
        if valid[k] < 0:
            status.append(REJECTED)
        elif not np.isfinite(n_e[k]):
            status.append(NO_SAMPLE)
        elif not np.isfinite(psi_norm[k]):
            status.append(OUTSIDE)
        else:
            status.append(USABLE)
    r, z = np.asarray(r), np.asarray(z)
    return ThomsonPoints(
        name=tuple(names), r=r, z=z, n_e=np.asarray(n_e), sigma=np.asarray(sigma), status=tuple(status),
        psi_norm=psi_norm, rho_tor_norm=np.full(count, np.nan), outboard=np.zeros(count, dtype=bool),
        on_midplane=np.abs(z) <= midplane_tolerance, time=sample_time, sigma_lower=np.asarray(lower),
    )


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

def _rho_table(ods, slice_index: int, psi_axis: float, psi_boundary: float):
    """The equilibrium's own ``rho_tor_norm(psi_N)``, refused when it is the ``sqrt(psi_N)`` proxy."""
    from vaft.data._derived import is_rho_pol_proxy

    base = f"equilibrium.time_slice.{slice_index}.profiles_1d"
    psi = _array(ods, base + ".psi")
    rho = _array(ods, base + ".rho_tor_norm")
    if psi is None or rho is None or psi.size != rho.size:
        raise ValueError(f"{base} carries no rho_tor_norm(psi) table")
    psi_norm = (psi - psi_axis) / (psi_boundary - psi_axis)
    if is_rho_pol_proxy(rho, psi_norm):
        raise ValueError(f"{base}.rho_tor_norm is sqrt(psi_N), the legacy rho_pol proxy (#276); "
                         "this figure does not substitute it for the toroidal-flux coordinate")
    order = np.argsort(psi_norm)
    return psi_norm[order], rho[order]


_WALL = "wall.description_2d.0.limiter.unit.0.outline"


def _sample_source():
    """Path and git blob SHA of the packaged sample, or ``(None, None)`` outside a checkout."""
    root = Path(__file__).resolve().parents[2]
    path = root / SAMPLE_PATH
    if not path.is_file():
        return None, None
    data = path.read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    return SAMPLE_PATH.as_posix(), blob


def density_semantics(ods=None, *, time: float = SAMPLE_TIME, time_tolerance: float = 5e-4,
                      midplane_tolerance: float = 1e-3, refine: int = 3) -> DensitySemantics:
    """The four representations of ``n_e`` at ``time``, from one ODS.

    Parameters
    ----------
    ods : ODS, optional
        Carries ``thomson_scattering``, ``core_profiles`` and ``equilibrium``;
        ``None`` loads the packaged sample ``vaft.omas.sample_ods(48224)`` [-].
    time : float
        The physical time every slice is matched to [s].
    time_tolerance : float
        Largest accepted distance of each IDS's slice from ``time`` [s].
    midplane_tolerance : float
        Largest ``|Z|`` of a Thomson channel shown at its own ``R`` in the
        midplane cut; a channel further off is omitted there and listed in
        ``metadata["off_midplane_channels"]`` [m].
    refine : int
        Map cells per equilibrium grid spacing: ``psi`` is read bilinearly
        between the equilibrium's nodes (as the channel mapper reads it), and
        every node is kept [-].

    Returns
    -------
    DensitySemantics
        In m, m^-3 and normalized flux; ``metadata`` names the slices, the
        coordinate table and the assumptions [-].
    """
    from vaft.ods_access import path_value
    from vaft.process.equilibrium import plasma_cell_weights, psi_to_rz

    source_path = source_blob = None
    if ods is None:
        from vaft.omas import sample_ods

        ods = sample_ods(SAMPLE_SHOT)
        source_path, source_blob = _sample_source()

    profile_index = _nearest(_array(ods, "core_profiles.time"), time, time_tolerance, "core_profiles")
    eq_index = _nearest(_array(ods, "equilibrium.time"), time, time_tolerance, "equilibrium")
    eq_time = float(_array(ods, "equilibrium.time")[eq_index])
    thomson = thomson_points(ods, time, time_tolerance=time_tolerance, midplane_tolerance=midplane_tolerance,
                             equilibrium_time=eq_time)
    eq = f"equilibrium.time_slice.{eq_index}"
    psi_axis = float(ods[eq + ".global_quantities.psi_axis"])
    psi_boundary = float(ods[eq + ".global_quantities.psi_boundary"])
    table_psi, table_rho = _rho_table(ods, eq_index, psi_axis, psi_boundary)

    # the stored profile, placed by its own grid.psi when it has one
    cp = f"core_profiles.profiles_1d.{profile_index}"
    n_e = _array(ods, cp + ".electrons.density")
    grid_psi = _array(ods, cp + ".grid.psi")
    grid_rho = _array(ods, cp + ".grid.rho_tor_norm")
    if n_e is None:
        raise ValueError(f"{cp}.electrons.density is missing")
    profile_psi = None
    if grid_psi is not None and grid_psi.size == n_e.size:
        candidate = (grid_psi - psi_axis) / (psi_boundary - psi_axis)
        # a grid.psi in another convention (Wb/rad against Wb, #278) normalizes off [0, 1]: not usable here
        if np.nanmin(candidate) >= -0.05 and np.nanmax(candidate) <= 1.05:
            profile_psi, placed_by = candidate, "core_profiles grid.psi"
    if profile_psi is not None:
        pass
    elif grid_rho is not None and grid_rho.size == n_e.size:
        profile_psi = np.interp(grid_rho, table_rho, table_psi)
        placed_by = "core_profiles grid.rho_tor_norm through the equilibrium table"
    else:
        raise ValueError(f"{cp}.grid gives neither a psi consistent with {eq} nor rho_tor_norm "
                         "for electrons.density")
    profile_rho = np.interp(profile_psi, table_psi, table_rho)

    # the channels in the same table; the side of the axis a flux coordinate forgets
    axis_r = float(ods[eq + ".global_quantities.magnetic_axis.r"])
    axis_z = float(ods[eq + ".global_quantities.magnetic_axis.z"])
    finite = np.isfinite(thomson.psi_norm)
    rho = np.full(thomson.r.shape, np.nan)
    rho[finite] = np.interp(thomson.psi_norm[finite], table_psi, table_rho)
    thomson = replace(thomson, rho_tor_norm=rho, outboard=thomson.r > axis_r)

    # the 2-D derived state: psi_to_rz on the equilibrium grid, masked by the boundary outline
    grid_r = _array(ods, eq + ".profiles_2d.0.grid.dim1")
    grid_z = _array(ods, eq + ".profiles_2d.0.grid.dim2")
    psi_rz = np.asarray(ods[eq + ".profiles_2d.0.psi"], dtype=float)
    if psi_rz.shape != (grid_r.size, grid_z.size):
        raise ValueError(f"{eq}.profiles_2d.0.psi is {psi_rz.shape}, not (R, Z) = ({grid_r.size}, {grid_z.size})")
    lcfs_r = _array(ods, eq + ".boundary.outline.r")
    lcfs_z = _array(ods, eq + ".boundary.outline.z")
    if lcfs_r is not None and lcfs_r.size >= 3:
        # the outline decides what is plasma, for the channels as for the map: psi_N < 1 also holds
        # where psi turns over near the coils
        from matplotlib.path import Path as _Outline

        inside = _Outline(np.column_stack([lcfs_r, lcfs_z])).contains_points(np.column_stack([thomson.r, thomson.z]))
        status = tuple(OUTSIDE if st == USABLE and not ok else st for st, ok in zip(thomson.status, inside))
        thomson = replace(thomson, status=status)
    if refine > 1:
        # psi bilinear between equilibrium nodes, as the channel mapper reads it; every node is kept
        from scipy.interpolate import RegularGridInterpolator

        fine_r = np.linspace(grid_r[0], grid_r[-1], refine * (grid_r.size - 1) + 1)
        fine_z = np.linspace(grid_z[0], grid_z[-1], refine * (grid_z.size - 1) + 1)
        mesh = np.stack(np.meshgrid(fine_r, fine_z, indexing="ij"), axis=-1)
        psi_rz = RegularGridInterpolator((grid_r, grid_z), psi_rz)(mesh)
        grid_r, grid_z = fine_r, fine_z
    # "edge": a node whose psi sits at or a hair below the reported psi_axis (psi_N = 0 to rounding)
    # keeps the axis value instead of a zero hole; the boundary outline, not psi_N, decides what is plasma
    map_n_e, psi_norm_rz = psi_to_rz(profile_psi, n_e, psi_rz, psi_axis, psi_boundary, fill_outside="edge")
    plasma = plasma_cell_weights(grid_r, grid_z, psi_norm_rz, lcfs_r, lcfs_z) >= 0.5
    map_n_e = np.where(plasma, map_n_e, np.nan)

    # panel (c) is that array at Z = 0: the row itself when the grid has one, else linear between the two
    # rows that bracket it (a row at Z = 0 is taken alone, since 0 * NaN would blank its edge cells)
    row = np.flatnonzero(np.isclose(grid_z, 0.0, rtol=0.0, atol=1e-12))
    if row.size:
        midplane = map_n_e[:, int(row[0])].copy()
    else:
        upper = int(np.searchsorted(grid_z, 0.0))
        if upper == 0 or upper == grid_z.size:
            raise ValueError("Z = 0 lies outside the equilibrium grid")
        lower = upper - 1
        weight = (0.0 - grid_z[lower]) / (grid_z[upper] - grid_z[lower])
        midplane = (1.0 - weight) * map_n_e[:, lower] + weight * map_n_e[:, upper]

    ds = "dataset_description.data_entry"
    shot = _array(ods, ds + ".pulse")
    machine = path_value(ods, ds + ".machine") or ""
    from vaft.data._derived import rho_tor_profile

    q_derived = None
    q = _array(ods, eq + ".profiles_1d.q")
    psi_1d = _array(ods, eq + ".profiles_1d.psi")
    if q is not None and psi_1d is not None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # an on-axis q outlier is reported, not hidden
            derived = rho_tor_profile(q, psi_1d)
        if derived is not None:
            q_psi = (psi_1d - psi_axis) / (psi_boundary - psi_axis)
            q_derived = float(np.nanmax(np.abs(np.interp(q_psi, table_psi, table_rho) - derived.rho_tor_norm)))
    metadata = {
        "shot": None if shot is None else int(shot[0]),
        "machine": str(machine),
        "time_s": float(time),
        "thomson_time_s": thomson.time,
        "core_profiles_time_s": float(_array(ods, "core_profiles.time")[profile_index]),
        "equilibrium_time_s": float(_array(ods, "equilibrium.time")[eq_index]),
        "core_profiles_slice": profile_index,
        "equilibrium_slice": eq_index,
        "equilibrium_code": str(path_value(ods, "equilibrium.ids_properties.comment") or "").strip(),
        "measurement": "thomson_scattering.channel[*].n_e.data +/- data_error_upper (measurement sigma)",
        "stored_state": f"{cp}.electrons.density, placed by {placed_by}",
        "coordinate_table": f"{eq}.profiles_1d.rho_tor_norm(psi), not sqrt(psi_N)",
        "q_derived_rho_tor_norm_max_difference": q_derived,
        "mapping": (f"psi_to_rz of the stored profile on equilibrium psi(R,Z), bilinear at {refine}x the grid; "
                    "plasma = boundary-outline cells"),
        "physical_assumption": "n_e constant on flux surfaces; the 2-D map is derived, not measured",
        "midplane_policy": f"channels with |Z| <= {midplane_tolerance:g} m drawn at native R; others omitted",
        "off_midplane_channels": [thomson.name[k] for k in range(len(thomson.name))
                                  if thomson.status[k] == USABLE and not thomson.on_midplane[k]],
        "flagged_channels": {thomson.name[k]: thomson.status[k] for k in range(len(thomson.name))
                             if thomson.status[k] != USABLE},
        "not_claimed": "core_profiles density is not asserted to be a fit to these Thomson points",
        "source_path": source_path,
        "source_blob_sha": source_blob,
    }
    metadata.update(_fit_provenance(ods, profile_index, thomson))
    model = DensitySemantics(
        shot=metadata["shot"], machine=metadata["machine"], time=float(time), thomson=thomson,
        profile_psi_norm=profile_psi, profile_rho_tor_norm=profile_rho, profile_n_e=n_e,
        table_psi_norm=table_psi, table_rho_tor_norm=table_rho,
        grid_r=grid_r, grid_z=grid_z, map_n_e=map_n_e, midplane_r=grid_r.copy(), midplane_n_e=midplane,
        lcfs_r=lcfs_r, lcfs_z=lcfs_z, axis_r=axis_r, axis_z=axis_z, metadata=metadata,
        wall_r=_array(ods, _WALL + ".r") if _array(ods, _WALL + ".r") is not None else np.zeros(0),
        wall_z=_array(ods, _WALL + ".z") if _array(ods, _WALL + ".z") is not None else np.zeros(0),
        display=_display_policy(),
    )
    return replace(model, views=_views(model))


def _display_policy():
    """The density display policy: SI m^-3 shown in 10^18 m^-3, as ``vaft.plot`` resolves it."""
    from vaft.plot.display import resolve_display

    return resolve_display("m^-3", unit="10^18 m^-3", subject="electron_density")


def _thomson_series(T: ThomsonPoints, x, keep, scale) -> tuple:
    """The Thomson trace at abscissa ``x``: ``yerr`` = sigma, ``valid_mask`` = usable."""
    from vaft.plot.models import Series

    yerr = np.vstack([T.sigma_lower[keep], T.sigma[keep]]) * scale
    if np.array_equal(yerr[0], yerr[1], equal_nan=True):
        yerr = yerr[1]  # symmetric: data_error_lower absent or equal
    return (Series(x=np.asarray(x, float)[keep], y=T.n_e[keep] * scale, yerr=yerr,
                   valid_mask=T.usable[keep], label="Thomson", channel="thomson_scattering"),)


def _views(m: DensitySemantics) -> Dict[str, object]:
    """The typed view models: three ``Profile1D`` and one ``Field2D``, sharing one display policy.

    As the ``vaft.plot`` recipes build them, values are in the display unit and
    ``display`` records the resolution (SI times ``display.scale``).

    (a) and (b) are one stored profile under two flux coordinates, a
    reparameterization through the equilibrium's table; (c) and (d) need the
    equilibrium's ``psi(R, Z)``, a mapping into physical space.
    """
    from vaft.plot.models import Field2D, Profile1D, Series

    T = m.thomson
    every = np.ones(T.r.shape, dtype=bool)
    stored = "core_profiles"
    k = m.display.scale
    common = dict(y_label="Electron Density", y_unit=m.display.unit, display=m.display)
    views = {
        "rho": Profile1D(series=(Series(x=m.profile_rho_tor_norm, y=m.profile_n_e * k, label=stored),)
                         + _thomson_series(T, T.rho_tor_norm, every, k),
                         coordinate_label="rho_tor_norm", x_limits=(0.0, 1.0),
                         metadata={"coordinate": "toroidal flux, from the equilibrium's rho_tor_norm(psi_N)"}, **common),
        "psi": Profile1D(series=(Series(x=m.profile_psi_norm, y=m.profile_n_e * k, label=stored),)
                         + _thomson_series(T, T.psi_norm, every, k),
                         coordinate_label="psi_norm", x_limits=(0.0, 1.0),
                         metadata={"coordinate": "normalized poloidal flux"}, **common),
        "mid": Profile1D(series=(Series(x=m.midplane_r, y=m.midplane_n_e * k, label="Z = 0 cut of the mapped field"),)
                         + _thomson_series(T, T.r, T.on_midplane, k),
                         coordinate_label="R [m]", x_limits=_R_RANGE,
                         metadata={"coordinate": "major radius at Z = 0, channels at their native R",
                                   "magnetic_axis_r": m.axis_r}, **common),
        # the DD stores (R, Z); Field2D is (Z, R)
        "map": Field2D(r=m.grid_r, z=m.grid_z, values=m.map_n_e.T * k,
                       value_label=f"Electron Density [{m.display.unit}]",
                       display=m.display, metadata={"derived": "flux-surface-constant mapping, not a measurement"}),
    }
    return views


def _fit_provenance(ods, profile_index: int, thomson: ThomsonPoints) -> Dict[str, object]:
    """How far ``electrons.density_fit`` documents the stored profile as a fit to these channels."""
    base = f"core_profiles.profiles_1d.{profile_index}.electrons.density_fit"
    measured = _array(ods, base + ".measured")
    positions = _array(ods, base + ".rho_tor_norm")
    if measured is None:
        return {"density_fit": "absent"}
    usable = thomson.usable
    matched, offsets = 0, []
    for k in np.flatnonzero(usable):
        hit = np.flatnonzero(np.isclose(measured, thomson.n_e[k], rtol=1e-9, atol=0.0))
        if hit.size:
            matched += 1
            if positions is not None and positions.size == measured.size:
                offsets.append(abs(float(positions[hit[0]]) - float(thomson.rho_tor_norm[k])))
    return {
        "density_fit_measured_matches": f"{matched} of {int(usable.sum())} usable Thomson values",
        "density_fit_rho_tor_norm_max_offset": max(offsets) if offsets else None,
    }


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------

#: left-column panels: (key, centre height [cm], heading, description)
_ROWS = (
    ("rho", 2.90, r"\textbf{(a)} $n_e(\rho_{\mathrm{tor},N})$", r"toroidal-flux coordinate"),
    ("psi", 0.00, r"\textbf{(b)} $n_e(\psi_N)$", r"poloidal-flux coordinate"),
    ("mid", -2.90, r"\textbf{(c)} $n_e(R,\,Z{=}0)$", r"midplane cut of (d)"),
)
_LEFT = (-8.80, -0.25)        # left column box [cm]
_RIGHT = (0.25, 8.80)         # right column box [cm]
_ROW_HALF = 1.375             # half height of a left box [cm]
_PLOT_X = (-7.45, -1.75)      # 1-D plotting box [cm]
_PLOT_DY = (-0.78, 0.78)      # 1-D plotting box, about the row centre [cm]
_Y_RANGE = (0.0, 14.0)        # 1-D density axis [10^18 m^-3]
_Y_TICKS = (0.0, 4.0, 8.0, 12.0)
_R_RANGE = (0.05, 0.75)       # major radius shown in (c) and (d) [m]
_MAP_R_RANGE = (0.05, 0.82)   # major radius shown in (d): the wall's full radial extent [m]
_Z_RANGE = (-0.66, 0.66)      # height shown in (d), past the limiter steps at |Z| = 0.6 m [m]
_MAP_ORIGIN = (1.75, -3.55)   # (d) lower-left corner [cm]
_MAP_SCALE = 5.45             # (d) cm per metre, the same in R and Z
_COLOR_TOP = 12.0             # colour-scale top [10^18 m^-3]
_COLOR_BINS = 24              # quantized colour levels, the colour bar's too

_PROFILE = "islandblue,line width=1.25pt"
_MAPPED = "absequil,densely dashed,line width=1.25pt"
_ERROR = "black!70,line width=0.5pt"
_POINT = "draw=black!85,fill=white,line width=0.55pt"
_WALL_STYLE = "black!55,line width=1.3pt"
_LCFS_STYLE = "driftred,line width=0.9pt"
_MARKER_R = 0.07              # data-point radius [cm]


def _q(values):
    """Data values rounded before they become coordinates or colour levels.

    Grid abscissae such as ``k/128`` land on 4-decimal ties in centimetres, and a
    cell value on a colour-level edge flips with the last bit; rounding here
    keeps the TikZ source -- and so the committed SVG's hash -- independent of
    platform floating-point noise.
    """
    return np.round(np.asarray(values, dtype=float), 9)


def _text(x, y, text, role, size="scriptsize", anchor="west", color=None):
    style = f"concept plain,font=\\{size},inner sep=0.7pt"
    if color:
        style += f",text={color}"
    return Label((float(x), float(y)), text, style, anchor, role)


def _rect(x0, y0, x1, y1, style, role):
    return Polyline.of([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], style, role, True)


def _dot(x, y, style, role):
    t = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]
    return Polyline.of(np.round(np.column_stack([x + _MARKER_R * np.cos(t), y + _MARKER_R * np.sin(t)]), 4),
                       style, role, True)


def _rgb(value):
    """``{rgb,255:...}`` of the quantized viridis level holding ``value`` [10^18 m^-3]."""
    from matplotlib import colormaps

    level = min(_COLOR_BINS - 1, max(0, int(math.floor(value / _COLOR_TOP * _COLOR_BINS))))
    r, g, b, _ = colormaps["viridis"]((level + 0.5) / _COLOR_BINS)
    return level, f"{{rgb,255:red,{round(255 * r)};green,{round(255 * g)};blue,{round(255 * b)}}}"


class _Axis:
    """A 1-D panel's data -> cm map."""

    def __init__(self, centre, x_range):
        self.x0, self.x1 = _PLOT_X
        self.y0, self.y1 = centre + _PLOT_DY[0], centre + _PLOT_DY[1]
        self.xr = x_range

    def x(self, value):
        return self.x0 + (value - self.xr[0]) / (self.xr[1] - self.xr[0]) * (self.x1 - self.x0)

    def y(self, value):
        value = min(max(value, _Y_RANGE[0]), _Y_RANGE[1])
        return self.y0 + (value - _Y_RANGE[0]) / (_Y_RANGE[1] - _Y_RANGE[0]) * (self.y1 - self.y0)


def _measurement(axis, x, n, below, above, role):
    """One Thomson point at abscissa ``x``: value ``n`` - ``below`` / + ``above`` [display unit]."""
    items = []
    x, n, below, above = float(_q(x)), float(_q(n)), float(_q(below)), float(_q(above))
    cx = axis.x(x)
    if np.isfinite(above):
        lo, hi = axis.y(n - (below if np.isfinite(below) else above)), axis.y(n + above)
        items += [Polyline.of([(cx, lo), (cx, hi)], _ERROR, f"{role}:error"),
                  Polyline.of([(cx - 0.05, lo), (cx + 0.05, lo)], _ERROR, f"{role}:cap"),
                  Polyline.of([(cx - 0.05, hi), (cx + 0.05, hi)], _ERROR, f"{role}:cap")]
    items.append(_dot(cx, axis.y(n), _POINT, role))
    return items


def _unit_tex(unit: str) -> str:
    """``10^18 m^-3`` typeset as ``$[10^{18}\\,\\mathrm{m}^{-3}]$``: the display policy's unit."""
    parts = []
    for token in unit.split():
        base, _, power = token.partition("^")
        base = base if base.isdigit() else f"\\mathrm{{{base}}}"
        parts.append(f"{base}^{{{power}}}" if power else base)
    return "$[" + "\\,".join(parts) + "]$"


def _frame_1d(axis, key, ticks, tick_text, x_label):
    items = [Polyline.of([(axis.x0, axis.y1), (axis.x0, axis.y0), (axis.x1, axis.y0)],
                         "black!70,line width=0.6pt", f"axes:{key}")]
    for v in _Y_TICKS:
        yy = axis.y(v)
        items += [Polyline.of([(axis.x0 - 0.07, yy), (axis.x0, yy)], "black!55,line width=0.45pt", f"ytick:{key}"),
                  _text(axis.x0 - 0.12, yy, f"${v:g}$", f"yticklabel:{key}", "scriptsize", "east", "black!75")]
    for v, text in zip(ticks, tick_text):
        xx = axis.x(v)
        items += [Polyline.of([(xx, axis.y0), (xx, axis.y0 - 0.06)], "black!55,line width=0.45pt", f"xtick:{key}"),
                  _text(xx, axis.y0 - 0.1, text, f"xticklabel:{key}", "scriptsize", "north", "black!75")]
    items.append(_text(axis.x1 + 0.12, axis.y0, x_label, f"xlabel:{key}", "footnotesize", "west"))
    # no rotated y title: with its unit it was longer than the axis and reached the panel title; the
    # quantity and unit stand on the title line instead (``view:<key>:unit``)
    return items


def _curve(axis, x, n, style, role):
    x, n = _q(x), _q(n)
    keep = np.isfinite(x) & np.isfinite(n) & (x >= axis.xr[0]) & (x <= axis.xr[1])
    pts = [(axis.x(a), axis.y(b)) for a, b in zip(x[keep], n[keep])]
    return [Polyline.of(np.round(pts, 4), style, role)] if len(pts) >= 2 else []


def _map_xy(r, z):
    return (_MAP_ORIGIN[0] + (r - _MAP_R_RANGE[0]) * _MAP_SCALE, _MAP_ORIGIN[1] + (z - _Z_RANGE[0]) * _MAP_SCALE)


def _clip_polyline(r, z, r_range, z_range) -> List[List[Tuple[float, float]]]:
    """The runs of the closed outline ``(r, z)`` inside the box, each segment clipped (Liang-Barsky)."""
    r, z = _q(r), _q(z)
    if r.size < 2:
        return []
    runs, run = [], []
    for k in range(r.size - 1):
        (ra, za), (rb, zb) = (r[k], z[k]), (r[k + 1], z[k + 1])
        t0, t1, dr, dz = 0.0, 1.0, rb - ra, zb - za
        for p, q in ((-dr, ra - r_range[0]), (dr, r_range[1] - ra), (-dz, za - z_range[0]), (dz, z_range[1] - za)):
            if p == 0.0:
                if q < 0.0:
                    t0, t1 = 1.0, 0.0
            elif p < 0.0:
                t0 = max(t0, q / p)
            else:
                t1 = min(t1, q / p)
        if t0 > t1:
            if run:
                runs.append(run)
            run = []
            continue
        start = (float(ra + t0 * dr), float(za + t0 * dz))
        end = (float(ra + t1 * dr), float(za + t1 * dz))
        if not run or run[-1] != start:
            if run:
                runs.append(run)
            run = [start]
        run.append(end)
    if run:
        runs.append(run)
    return [run for run in runs if len(run) >= 2]


def _map_cells(field):
    """A ``Field2D`` (values on ``(Z, R)``, display unit) as filled cells, one run per row of equal colour level."""
    items = []
    r, z, f = _q(field.r), _q(field.z), _q(field.values).T
    dr = np.diff(r)
    dz = np.diff(z)
    r_edges = np.concatenate([[r[0] - dr[0] / 2], r[:-1] + dr / 2, [r[-1] + dr[-1] / 2]])
    z_edges = np.concatenate([[z[0] - dz[0] / 2], z[:-1] + dz / 2, [z[-1] + dz[-1] / 2]])
    for j in range(z.size):
        if not (_Z_RANGE[0] <= z[j] <= _Z_RANGE[1]):
            continue
        f[(r < _MAP_R_RANGE[0]) | (r > _MAP_R_RANGE[1]), j] = np.nan
        i = 0
        while i < r.size:
            if not np.isfinite(f[i, j]):
                i += 1
                continue
            level, color = _rgb(f[i, j])
            k = i
            while k + 1 < r.size and np.isfinite(f[k + 1, j]) and _rgb(f[k + 1, j])[0] == level:
                k += 1
            x0, y0 = _map_xy(r_edges[i], z_edges[j])
            x1, y1 = _map_xy(r_edges[k + 1], z_edges[j + 1])
            # a hair of overlap hides the anti-aliasing seams between neighbouring cells
            items.append(_rect(round(x0, 4), round(y0, 4), round(x1 + 0.004, 4), round(y1 + 0.004, 4),
                               "draw=none,fill=" + color, "map:cell"))
            i = k + 1
    return items


def _series(view, label):
    return next(s for s in view.series if s.label == label)


def _draw_thomson(axis, view, key):
    """The Thomson trace of a ``Profile1D``: usable samples as points with bars; the rest are not drawn."""
    items = []
    trace = _series(view, "Thomson")
    err = np.asarray(trace.yerr)
    below, above = (err[0], err[1]) if err.ndim == 2 else (err, err)
    for x, y, lo, hi, ok in zip(trace.x, trace.y, below, above, trace.valid_mask):
        if ok and not (axis.xr[0] <= x <= axis.xr[1]):
            continue  # outside the panel's abscissa: never pinned to its edge
        if ok:
            items += _measurement(axis, x, y, lo, hi, f"ts:{key}")
    return items


#: the responsibility boundaries the top strip draws, as (from, to, kind): conceptual, not a runtime order
ARCHITECTURE_EDGES = (
    ("identity", "adapter", "input"),
    ("data", "adapter", "input"),
    ("adapter", "model", "extraction"),
    ("model", "display", "policy"),
    ("model", "presentation", "policy"),
    ("model", "panels", "representation"),
)

_ARCH_TOP = (5.9, 7.0)        # upper strip row [cm]
_ARCH_BOTTOM = (4.7, 5.8)     # lower strip row [cm]
#: column widths fit the widest bold heading with the same inset on both sides; 0.6 cm gaps
_ARCH_COLUMNS = {"a": (-8.80, -4.20), "b": (-3.60, -0.40), "c": (0.20, 4.10), "d": (4.70, 8.80)}


def _arch_box(column, rows, key, lines, style="concept leaf"):
    (x0, x1), (y0, y1) = _ARCH_COLUMNS[column], rows
    items = [_rect(x0, y0, x1, y1, style, f"arch:{key}")]
    y = y1 - 0.33  # first baseline: lines are placed on their baselines, so descenders do not shift them
    for k, line in enumerate(lines):
        size = "footnotesize" if k == 0 else "scriptsize"
        items.append(_text(x0 + 0.14, y, line, f"arch:{key}:{'head' if k == 0 else 'line'}", size,
                           anchor="base west", color=None if k == 0 else "black!78"))
        y -= 0.33 if k == 0 else 0.28
    return items


def _architecture() -> List:
    """Identity and data feed the adapter, which builds the views; two separate policies act on them.

    Keywords only: the design argument, the API names and the development
    state belong to the caption and the Methods text, not to the figure.
    """
    full = (_ARCH_BOTTOM[0], _ARCH_TOP[1])
    items = []
    items += _arch_box("a", _ARCH_TOP, "identity", (
        r"\textbf{Scientific identity}",
        r"electron density $n_e$"), "concept strong")
    items += _arch_box("a", _ARCH_BOTTOM, "data", (
        r"\textbf{IMAS data representation}",
        r"Thomson scattering",
        r"core profiles $\cdot$ equilibrium"))
    items += _arch_box("b", full, "adapter", (
        r"\textbf{Semantic adapter}",
        r"time alignment",
        r"coordinate mapping",
        r"uncertainty $\cdot$ validity"))
    items += _arch_box("c", full, "model", (
        r"\textbf{Scientific view model}",
        r"1-D profiles $\cdot$ midplane cut",
        r"2-D field",
        r"coordinates $\cdot$ units",
        r"uncertainty $\cdot$ validity"))
    items += _arch_box("d", _ARCH_TOP, "display", (
        r"\textbf{Scientific display policy}",
        r"unit $\cdot$ scaling $\cdot$ notation"))
    items += _arch_box("d", _ARCH_BOTTOM, "presentation", (
        r"\textbf{Graphical presentation}",
        r"layout $\cdot$ style $\cdot$ backend"))
    mid_top, mid_bottom = sum(_ARCH_TOP) / 2, sum(_ARCH_BOTTOM) / 2
    a1, b0 = _ARCH_COLUMNS["a"][1], _ARCH_COLUMNS["b"][0]
    b1, c0 = _ARCH_COLUMNS["b"][1], _ARCH_COLUMNS["c"][0]
    c1, d0 = _ARCH_COLUMNS["c"][1], _ARCH_COLUMNS["d"][0]
    items += [Arrow((a1, mid_top), (b0, mid_top), "connector", "edge:identity-adapter"),
              Arrow((a1, mid_bottom), (b0, mid_bottom), "connector", "edge:data-adapter"),
              Arrow((b1, sum(full) / 2), (c0, sum(full) / 2), "connector strong", "edge:adapter-model"),
              Arrow((c1, mid_top), (d0, mid_top), "connector", "edge:model-display"),
              Arrow((c1, mid_bottom), (d0, mid_bottom), "connector", "edge:model-presentation")]
    cx = sum(_ARCH_COLUMNS["c"]) / 2
    items.append(Arrow((cx, full[0]), (cx, 2.90 + _ROW_HALF), "connector strong", "edge:model-panels"))
    return items


def _warn_out_of_range(views) -> None:
    """The axis and colour ranges are fixed for VEST; say so when a value would be clamped to an edge."""
    import warnings

    top = 0.0
    for key in ("rho", "psi", "mid"):
        for series in views[key].series:
            y = np.asarray(series.y, float)
            if series.yerr is not None:
                err = np.asarray(series.yerr, float)
                y = y + (err[1] if err.ndim == 2 else err)
            if series.valid_mask is not None:
                y = y[series.valid_mask]
            if y.size and np.isfinite(y).any():
                top = max(top, float(np.nanmax(y)))
    if top > _Y_RANGE[1]:
        warnings.warn(f"n_e reaches {top:.3g} in display units, above the 1-D axis top {_Y_RANGE[1]:g}; "
                      "those points and bars are clamped to the edge", RuntimeWarning, stacklevel=3)
    field = np.asarray(views["map"].values, float)
    if np.isfinite(field).any() and np.nanmax(field) > _COLOR_TOP:
        warnings.warn(f"the map reaches {np.nanmax(field):.3g}, above the colour top {_COLOR_TOP:g}; "
                      "it saturates there", RuntimeWarning, stacklevel=3)


def _scene(m: DensitySemantics) -> Scene:
    """The figure, drawn from ``m.views`` and ``m.display``; ``m`` adds only geometry and provenance text."""
    views = m.views  # every value is already in the display unit
    _warn_out_of_range(views)
    T = m.thomson
    items: List = list(_architecture())

    # (a)-(c): three Profile1D views
    for key, centre, head, desc in _ROWS:
        view = views[key]
        items.append(_rect(_LEFT[0], centre - _ROW_HALF, _LEFT[1], centre + _ROW_HALF, "concept leaf", f"view:{key}"))
        axis = _Axis(centre, view.x_limits)
        axis.unit = _unit_tex(view.y_unit)
        items.append(_text(_LEFT[0] + 0.15, centre + _ROW_HALF - 0.33, head, f"view:{key}:title", "footnotesize",
                           "base west"))
        items.append(_text(_LEFT[1] - 0.15, centre + _ROW_HALF - 0.33, r"$n_e\;$" + axis.unit, f"view:{key}:unit",
                           "footnotesize", "base east", "black!75"))
        if key in ("rho", "psi"):
            items += _frame_1d(axis, key, (0.0, 0.5, 1.0), ("$0$", "$0.5$", "$1$"),
                               r"$\rho_{\mathrm{tor},N}$" if key == "rho" else r"$\psi_N$")
            stored = _series(view, "core_profiles")
            items += _curve(axis, stored.x, stored.y, _PROFILE, f"curve:{key}")
        else:
            ticks = (0.1, 0.3, 0.5, 0.7)
            items += _frame_1d(axis, key, ticks, [f"${v:g}$" for v in ticks], r"$R\;[\mathrm{m}]$")
            ax = axis.x(view.metadata["magnetic_axis_r"])
            items.append(Polyline.of([(ax, axis.y0), (ax, axis.y1)], "black!40,densely dotted,line width=0.5pt",
                                     "mid:axis"))
            items.append(_text(ax - 0.05, axis.y0 + 0.03, r"$R_{\mathrm{axis}}$", "mid:axis:label", "footnotesize",
                               "south east", "black!60"))
            cut = _series(view, "Z = 0 cut of the mapped field")
            items += _curve(axis, cut.x, cut.y, _MAPPED, "curve:mid")
        items += _draw_thomson(axis, view, key)

    # (d): the Field2D view, equal R and Z scale
    items.append(_rect(_RIGHT[0], -2.90 - _ROW_HALF, _RIGHT[1], 2.90 + _ROW_HALF, "concept leaf", "view:map"))
    items.append(_text(_RIGHT[0] + 0.15, 2.90 + _ROW_HALF - 0.33, r"\textbf{(d)} $n_e(R,Z)$", "view:map:title",
                       "footnotesize", "base west"))
    items += _map_cells(views["map"])
    lcfs = [_map_xy(a, b) for a, b in zip(m.lcfs_r, m.lcfs_z)]
    items.append(Polyline.of(np.round(lcfs, 4), _LCFS_STYLE, "map:lcfs", closed=True))
    for run in _clip_polyline(m.wall_r, m.wall_z, _MAP_R_RANGE, _Z_RANGE):
        items.append(Polyline.of(np.round([_map_xy(a, b) for a, b in run], 4), _WALL_STYLE, "map:wall"))
    x0, y0 = _map_xy(_MAP_R_RANGE[0], _Z_RANGE[0])
    x1, y1 = _map_xy(_MAP_R_RANGE[1], _Z_RANGE[1])
    ym = _map_xy(0.0, 0.0)[1]
    items.append(Polyline.of([(x0, ym), (x1, ym)], "black!55,densely dashed,line width=0.6pt", "map:midplane"))
    axis_xy = _map_xy(m.axis_r, m.axis_z)
    ax_x, ax_y, h = round(axis_xy[0], 4), round(axis_xy[1], 4), 0.1
    items.append(Polyline.of([(ax_x - h, ax_y), (ax_x + h, ax_y)], "white,line width=0.8pt", "map:axis"))
    items.append(Polyline.of([(ax_x, ax_y - h), (ax_x, ax_y + h)], "white,line width=0.8pt", "map:axis"))
    usable = T.usable
    for k in np.flatnonzero(usable):
        cx, cy = _map_xy(T.r[k], T.z[k])
        items.append(_dot(round(cx, 4), round(cy, 4), _POINT, "map:ts"))
    items.append(_rect(x0, y0, x1, y1, "black!60,line width=0.6pt", "map:axes"))
    for v in (0.2, 0.4, 0.6, 0.8):
        xx = _map_xy(v, 0.0)[0]
        items += [Polyline.of([(xx, y0), (xx, y0 - 0.06)], "black!55,line width=0.45pt", "map:xtick"),
                  _text(xx, y0 - 0.1, f"${v:g}$", "map:xticklabel", "scriptsize", "north", "black!75")]
    for v in (-0.6, -0.3, 0.0, 0.3, 0.6):
        yy = _map_xy(0.0, v)[1]
        items += [Polyline.of([(x0, yy), (x0 - 0.06, yy)], "black!55,line width=0.45pt", "map:ytick"),
                  _text(x0 - 0.1, yy, f"${v:g}$", "map:yticklabel", "scriptsize", "east", "black!75")]
    items.append(_text(0.5 * (x0 + x1), y0 - 0.42, r"$R\;[\mathrm{m}]$", "map:xlabel", "footnotesize", "north"))
    items.append(Label((x0 - 0.75, 0.5 * (y0 + y1)), r"$Z\;[\mathrm{m}]$",
                       "concept plain,font=\\footnotesize,inner sep=0.7pt,rotate=90", "south", "map:ylabel"))
    # colour bar: the same quantized levels, the same unit as the 1-D axes
    bx, bottom, top = x1 + 0.55, y0, y1
    for level in range(_COLOR_BINS):
        _, color = _rgb((level + 0.5) / _COLOR_BINS * _COLOR_TOP)
        lo = bottom + (top - bottom) * level / _COLOR_BINS
        hi = bottom + (top - bottom) * (level + 1) / _COLOR_BINS
        items.append(_rect(bx, round(lo, 4), bx + 0.25, round(hi + 0.004, 4), "draw=none,fill=" + color,
                           "map:colorbar"))
    items.append(_rect(bx, bottom, bx + 0.25, top, "black!60,line width=0.45pt", "map:colorbar:frame"))
    for v in (0.0, 4.0, 8.0, 12.0):
        yy = bottom + (top - bottom) * v / _COLOR_TOP
        items.append(_text(bx + 0.32, yy, f"${v:g}$", "map:colorbar:tick", "scriptsize", color="black!75"))
    items.append(Label((bx + 0.95, 0.5 * (bottom + top)), r"$n_e\;$" + _unit_tex(m.display.unit),
                       "concept plain,font=\\footnotesize,inner sep=0.7pt,rotate=90", "north", "map:colorbar:label"))

    # legend: measured, stored and equilibrium-derived kept apart; the rest is the caption's
    ly, ly2 = -4.62, -4.98
    items.append(Polyline.of([(-8.6, ly - 0.15), (-8.6, ly + 0.15)], _ERROR, "legend:ts:error"))
    items.append(_dot(-8.6, ly, _POINT, "legend:ts"))
    items.append(_text(-8.35, ly, r"Thomson $n_e \pm \sigma_n$", "legend:measured", "footnotesize"))
    items.append(Polyline.of([(-5.2, ly), (-4.6, ly)], _PROFILE, "legend:profile"))
    items.append(_text(-4.45, ly, r"stored density profile", "legend:stored", "footnotesize"))
    items.append(Polyline.of([(-5.2, ly2), (-4.6, ly2)], _MAPPED, "legend:mapped"))
    items.append(_text(-4.45, ly2, r"equilibrium-mapped density", "legend:derived", "footnotesize"))
    items.append(Polyline.of([(0.45, ly), (1.05, ly)], _WALL_STYLE, "legend:wall"))
    items.append(_text(1.2, ly, r"VEST wall (limiter)", "legend:wall:text", "footnotesize"))
    items.append(Polyline.of([(0.45, ly2), (1.05, ly2)], _LCFS_STYLE, "legend:lcfs"))
    items.append(_text(1.2, ly2, r"plasma boundary (LCFS)", "legend:lcfs:text", "footnotesize"))
    return Scene(tuple(items))


def electron_density_semantics(ods=None, *, time: float = SAMPLE_TIME, time_tolerance: float = 5e-4,
                               midplane_tolerance: float = 1e-3, refine: int = 3) -> Diagram:
    r"""Electron density as measured, stored and flux-mapped, at one time-matched sample.

    Panels (a) $n_e(\rho_{\mathrm{tor},N})$, (b) $n_e(\psi_N)$ and (c)
    $n_e(R, Z=0)$ carry the Thomson points with their vertical measurement
    error bars; (a) and (b) the stored ``core_profiles`` density, (c) the
    ``Z = 0`` cut of (d), the flux-surface-mapped $n_e(R, Z)$.

    Parameters
    ----------
    ods : ODS, optional
        ``None`` (default) draws the packaged VEST sample, shot 48224 at
        300 ms (``vaft/data/kineticEfit/ods_48224_300ms.json``) [-].
    time : float
        The physical time every IDS slice is matched to [s].
    time_tolerance : float
        Largest accepted distance of a slice from ``time`` [s].
    midplane_tolerance : float
        Largest ``|Z|`` of a channel drawn at its own ``R`` in (c); one
        further off is omitted there, not relabelled [m].
    refine : int
        Map cells per equilibrium grid spacing, as :func:`density_semantics` [-].

    Returns
    -------
    Diagram
        ``Diagram.model`` is the :class:`DensitySemantics` the panels draw:
        Thomson points and status, the stored profile in both flux
        coordinates, the masked 2-D state, its ``Z = 0`` cut and the
        provenance in ``metadata``.
    """
    model = density_semantics(ods, time=time, time_tolerance=time_tolerance, midplane_tolerance=midplane_tolerance,
                              refine=refine)
    model.metadata["architecture_edges"] = ARCHITECTURE_EDGES
    return Diagram("electron_density_semantics", _scene(model), model=model)
