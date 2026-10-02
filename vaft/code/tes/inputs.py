"""Build TES input from an ODS.

``prepare_tes_inputs`` reads machine geometry (wall limiter), the external-coil
set (``pf_active`` and, optionally, ``pf_passive`` eddy loops), and global plasma
targets from an ODS, then writes the strict C-format file that ``rtes`` consumes
directly. The intermediate Fortran namelist used by the legacy ``pytes.py`` is
skipped; an optional human-readable namelist can still be emitted for debugging
via ``TESConfig.emit_namelist``.

The C-format layout mirrors what ``rtes`` (``read_input.cpp``) expects. That
parser is purely positional: the leading keyword on each line is read into a
throwaway buffer, so only token order/count matters. Section banners are skipped
with ``fgets`` except for ``"Magnetic Diagnostics"`` which is located by
substring search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import TESConfig, TESInputs


# ----------------------------------------------------------------------------- #
#  ODS readers
# ----------------------------------------------------------------------------- #
def _infer_shot(ods: Any, config: TESConfig) -> int:
    if config.shot is not None:
        return int(config.shot)
    if ods is not None:
        for path in (
            "dataset_description.data_entry.pulse",
            "summary.global_quantities.pulse",
        ):
            try:
                return int(ods[path])
            except Exception:
                pass
    raise ValueError("TES shot number is required in TESConfig.shot or ODS metadata")


def _resolve_time(ods: Any, config: TESConfig) -> float:
    """Resolve the case time [s] from an explicit slice index or ``time``.

    When ``time_index`` is given it selects a slice of the chosen constraint
    source (the ``equilibrium`` time array, or the ``magnetics`` Ip time array)
    and the corresponding time is returned; otherwise ``time`` is used directly.
    """
    if config.time_index is not None:
        idx = int(config.time_index)
        if config.constraint_source == "magnetics":
            t = np.asarray(ods["magnetics.ip.0.time"], dtype=float)[idx]
        else:
            t = np.asarray(ods["equilibrium.time"], dtype=float)[idx]
        return float(t)
    if config.time is not None:
        return float(config.time)
    raise ValueError("TESConfig.time (seconds) or TESConfig.time_index is required")


def _grid_box(config: TESConfig) -> tuple[float, float, float, float]:
    """The grid rectangle shrunk by ``limiter_grid_margin`` cells per side."""
    dr = (config.rmax - config.rmin) / max(config.nr - 1, 1)
    dz = (config.zmax - config.zmin) / max(config.nz - 1, 1)
    m = config.limiter_grid_margin
    return (config.rmin + m * dr, config.rmax - m * dr, config.zmin + m * dz, config.zmax - m * dz)


def _open_polygon(r: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop a repeated closing vertex: TES closes the limiter itself."""
    if r.size > 1 and r[0] == r[-1] and z[0] == z[-1]:
        return r[:-1], z[:-1]
    return r, z


def _clip_limiter_to_grid(r: np.ndarray, z: np.ndarray, config: TESConfig) -> tuple[np.ndarray, np.ndarray]:
    """Clip the limiter polygon to the computational grid.

    ``get_psi()`` in TES calls ``exit()`` for any coordinate outside
    [RMIN, RMAX] x [ZMIN, ZMAX]; a margin of a few cells keeps interpolation
    stencils in-bounds. The polygon is clipped against that box
    (Sutherland-Hodgman), so a wall leaving the grid is replaced by the box
    edge. Dropping the outside vertices instead would join the remaining ones
    with chords that cut across the vessel.
    """
    r, z = _open_polygon(np.asarray(r, dtype=float), np.asarray(z, dtype=float))
    r0, r1, z0, z1 = _grid_box(config)
    points = list(zip(r.tolist(), z.tolist()))
    # (inside test, intersection with the clip line) per box side
    sides = (
        (lambda p: p[0] >= r0, lambda a, b: (r0, a[1] + (b[1] - a[1]) * (r0 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[0] <= r1, lambda a, b: (r1, a[1] + (b[1] - a[1]) * (r1 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[1] >= z0, lambda a, b: (a[0] + (b[0] - a[0]) * (z0 - a[1]) / (b[1] - a[1]), z0)),
        (lambda p: p[1] <= z1, lambda a, b: (a[0] + (b[0] - a[0]) * (z1 - a[1]) / (b[1] - a[1]), z1)),
    )
    for inside, cross in sides:
        clipped: list[tuple[float, float]] = []
        for i, current in enumerate(points):
            previous = points[i - 1]
            if inside(current):
                if not inside(previous):
                    clipped.append(cross(previous, current))
                clipped.append(current)
            elif inside(previous):
                clipped.append(cross(previous, current))
        points = clipped
        if not points:
            break
    # consecutive duplicates appear where a vertex lies on the box
    unique = [pt for i, pt in enumerate(points) if i == 0 or pt != points[i - 1]]
    if len(unique) > 1 and unique[0] == unique[-1]:
        unique.pop()
    if len(unique) < 3:
        raise ValueError(
            "Limiter has fewer than 3 points inside the computational grid; "
            "check the wall outline or supply TESConfig.limiter explicitly."
        )
    out = np.asarray(unique, dtype=float)
    return out[:, 0], out[:, 1]


def _densify_limiter(
    r: np.ndarray,
    z: np.ndarray,
    spacing: float,
    box: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split every edge of the closed polygon into pieces no longer than ``spacing``.

    Edges lying along a side of ``box`` (the grid clip) are left whole: they
    are the cut, not wall, and resampling them would hand TES dozens of
    candidate limiting points across the open chamber neck.
    """
    if spacing <= 0.0:
        raise ValueError(f"TESConfig.limiter_spacing must be positive, got {spacing!r}")
    r, z = _open_polygon(np.asarray(r, dtype=float), np.asarray(z, dtype=float))
    r_next, z_next = np.roll(r, -1), np.roll(z, -1)
    out_r: list[np.ndarray] = []
    out_z: list[np.ndarray] = []
    def on_box(ra, za, rb, zb):
        if box is None:
            return False
        r0, r1, z0, z1 = box
        return any(abs(a - edge) < 1e-12 and abs(b - edge) < 1e-12
                   for a, b, edge in ((ra, rb, r0), (ra, rb, r1), (za, zb, z0), (za, zb, z1)))

    for ra, za, rb, zb in zip(r, z, r_next, z_next):
        n = 1 if on_box(ra, za, rb, zb) else max(1, int(np.ceil(np.hypot(rb - ra, zb - za) / spacing)))
        t = np.arange(n) / n
        out_r.append(ra + t * (rb - ra))
        out_z.append(za + t * (zb - za))
    return np.concatenate(out_r), np.concatenate(out_z)


def _limiter_from_ods(ods: Any, config: TESConfig) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the limiter polygon from config or the ODS wall IDS."""
    if config.limiter is not None:
        r, z = config.limiter
        return np.asarray(r, dtype=float), np.asarray(z, dtype=float)
    try:
        r = np.asarray(ods["wall.description_2d.0.limiter.unit.0.outline.r"], dtype=float)
        z = np.asarray(ods["wall.description_2d.0.limiter.unit.0.outline.z"], dtype=float)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            "No limiter found: ODS lacks wall.description_2d.0.limiter.unit.0.outline "
            "and TESConfig.limiter was not supplied."
        ) from exc
    r, z = _clip_limiter_to_grid(r, z, config)
    if config.limiter_spacing is not None:
        r, z = _densify_limiter(r, z, float(config.limiter_spacing), box=_grid_box(config))
    return r, z


def limited_iso_points(ods: Any, time: float, *, time_index: int | None = None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Iso-flux points for a plasma resting on the inboard limiter.

    Returns ``(isor, isoz)``: the inboard limiter face at the axis height (the
    target limiting point TES picks as nearest the wall), followed by the
    outboard-midplane, top and bottom points of the reference equilibrium's
    LCFS (the slice nearest ``time``; an explicit ``time_index`` wins). Use it
    with ``TESConfig.limited_shape_control`` to hold the reference shape while
    another target, such as Ip, is scanned. TES measures "nearest the wall" to
    the limiter's listed points, which the default ``limiter_spacing``
    resamples finely enough to include this face.
    """
    if "equilibrium.time" not in ods:
        raise ValueError("limited_iso_points needs a reference equilibrium IDS")
    idx = time_index
    if idx is None:
        idx = int(np.argmin(np.abs(np.asarray(ods["equilibrium.time"], dtype=float) - time)))
    base = f"equilibrium.time_slice.{int(idx)}"
    # `in` first: reading a missing ODS path creates it.
    if f"{base}.boundary.outline.r" not in ods:
        raise ValueError(f"reference equilibrium slice {idx} has no boundary outline")
    r = np.asarray(ods[f"{base}.boundary.outline.r"], dtype=float)
    z = np.asarray(ods[f"{base}.boundary.outline.z"], dtype=float)
    z_axis = (
        float(ods[f"{base}.global_quantities.magnetic_axis.z"])
        if f"{base}.global_quantities.magnetic_axis.z" in ods else 0.0
    )
    wall = "wall.description_2d.0.limiter.unit.0.outline"
    if f"{wall}.r" not in ods:
        raise ValueError("limited_iso_points needs the limiter outline in ods['wall']")
    wall_r = np.asarray(ods[f"{wall}.r"], dtype=float)
    wall_z = np.asarray(ods[f"{wall}.z"], dtype=float)
    # inboard face: the innermost crossing of the wall polygon with Z = z_axis
    crossings = []
    for ra, za, rb, zb in zip(wall_r, wall_z, np.roll(wall_r, -1), np.roll(wall_z, -1)):
        if (za - z_axis) * (zb - z_axis) <= 0.0 and za != zb:
            crossings.append(ra + (rb - ra) * (z_axis - za) / (zb - za))
    if not crossings:
        raise ValueError("the wall polygon does not cross the magnetic-axis height")
    r_in = float(min(crossings))
    outboard = np.argmax(np.where(np.abs(z - z_axis) < 0.25 * float(np.ptp(z)), r, -np.inf))
    top, bottom = int(np.argmax(z)), int(np.argmin(z))
    isor = (r_in, float(r[outboard]), float(r[top]), float(r[bottom]))
    isoz = (z_axis, float(z[outboard]), float(z[top]), float(z[bottom]))
    return isor, isoz


def _ip0_from_magnetics(ods: Any, time: float) -> float:
    mg = ods["magnetics"]
    return float(np.interp(time, mg["ip.0.time"], mg["ip.0.data"])) / 1000.0


def _bt0_from_tf(ods: Any, time: float) -> float:
    tf = ods["tf"]
    btor = np.asarray(tf["b_field_tor_vacuum_r.data"], dtype=float) / tf["r0"]
    return float(np.interp(time, tf["time"], btor))


def _resolve_targets(ods: Any, config: TESConfig, time: float) -> tuple[float, float, float]:
    """Resolve (IP0 [kA], BT0 [T], BETAP) honouring ``constraint_source``.

    - ``"equilibrium"``: Ip and betap come from a single equilibrium slice — the
      explicit ``time_index`` when given, else the slice nearest ``time``. Both
      scalars are therefore read at the *same* physical time.
    - ``"magnetics"``: Ip comes from magnetics only; the equilibrium IDS is never
      read, so ``betap`` must be supplied via the config.

    Any of ``ip0_kA`` / ``bt0`` / ``betap`` set explicitly on the config always
    take precedence.
    """
    source = config.constraint_source
    if source not in ("equilibrium", "magnetics"):
        raise ValueError(
            f"TESConfig.constraint_source must be 'equilibrium' or 'magnetics', got {source!r}"
        )

    # --- BT0 (same in both modes) ---
    bt0 = float(config.bt0) if config.bt0 is not None else _bt0_from_tf(ods, time)

    if source == "magnetics":
        # Ip from magnetics; never touch the equilibrium IDS.
        ip0_kA = float(config.ip0_kA) if config.ip0_kA is not None else _ip0_from_magnetics(ods, time)
        if config.betap is None:
            raise ValueError(
                "constraint_source='magnetics' does not read the equilibrium IDS, "
                "so TESConfig.betap must be set explicitly (beta is chosen by the user)."
            )
        return ip0_kA, bt0, float(config.betap)

    # --- equilibrium mode ---
    eq = ods["equilibrium"]
    idx = config.time_index
    if idx is None:
        # select the equilibrium slice nearest the requested time so that Ip and
        # betap are read at the same physical instant
        eqtime = np.asarray(eq["time"], dtype=float)
        idx = int(np.argmin(np.abs(eqtime - time)))
    idx = int(idx)

    # Ip: explicit > equilibrium slice global_quantities.ip > magnetics fallback
    if config.ip0_kA is not None:
        ip0_kA = float(config.ip0_kA)
    else:
        try:
            ip0_kA = float(eq[f"time_slice.{idx}.global_quantities.ip"]) / 1000.0
        except Exception:
            ip0_kA = _ip0_from_magnetics(ods, time)

    # betap: explicit > equilibrium slice beta_pol
    if config.betap is not None:
        betap = float(config.betap)
    else:
        # `in` first: reading a missing ODS path creates it (and returns an ODS).
        path = f"time_slice.{idx}.global_quantities.beta_pol"
        if path not in eq:
            raise ValueError(
                f"equilibrium slice {idx} carries no global_quantities.beta_pol; "
                "set TESConfig.betap explicitly."
            )
        betap = float(eq[path])

    return ip0_kA, bt0, betap


def _legacy_pf_rows(ods: Any, time: float) -> list[tuple]:
    """pf_active rows in the legacy ``VEST_tes`` layout (``coil_model="legacy"``).

    Ported faithfully from the legacy ``VEST_tes.generate_tes_input_ods``:
    PF1 and PF2 are each discretised into 5 upper + 5 lower sub-coils, the
    remaining 8 PF coils into 1 upper + 1 lower each (36 sub-coils, grouped
    1..18 with two elements per group).
    """
    PF = ods["pf_active"]
    pf_time = PF["time"]
    rows: list[tuple] = []

    def _interp_current(coil_idx: int) -> float:
        return float(np.interp(time, pf_time, PF[f"coil.{coil_idx}.current.data"]))

    # --- PF1 (coil 0): 5 upper + 5 lower ---
    nbelt = len(PF["coil.0.element"])
    e0_r = PF["coil.0.element.0.geometry.rectangle.r"]
    e0_z = PF["coil.0.element.0.geometry.rectangle.z"]
    e0_dr = PF["coil.0.element.0.geometry.rectangle.width"]
    e0_dz = PF["coil.0.element.0.geometry.rectangle.height"]
    nbturn_total = sum(PF[f"coil.0.element.{i}.turns_with_sign"] for i in range(nbelt))
    PF1R = e0_r
    PF1DR = e0_dr
    PF1Z = e0_z + e0_dz / 2.0
    PF1DZ = PF1Z / 5.0
    PF1CUR = _interp_current(0)
    nbturn1 = nbturn_total / 10.0

    # --- PF2 (coil 1): 5 upper + 5 lower ---
    nbelt = len(PF["coil.1.element"])
    e_r = [PF[f"coil.1.element.{i}.geometry.rectangle.r"] for i in range(nbelt)]
    e_z = [PF[f"coil.1.element.{i}.geometry.rectangle.z"] for i in range(nbelt)]
    e_dr = [PF[f"coil.1.element.{i}.geometry.rectangle.width"] for i in range(nbelt)]
    e_dz = [PF[f"coil.1.element.{i}.geometry.rectangle.height"] for i in range(nbelt)]
    nbturn_total = sum(PF[f"coil.1.element.{i}.turns_with_sign"] for i in range(nbelt))
    PF2R = sum(e_r) / nbelt
    PF2DR = e_dr[0]
    high2Z = max(abs(np.array(e_z))) + e_dz[0] / 2.0
    low2Z = min(abs(np.array(e_z))) - e_dz[0] / 2.0
    PF2Z = high2Z - low2Z
    PF2DZ = PF2Z / 5.0
    PF2CUR = _interp_current(1)
    nbturn2 = nbturn_total / 10.0

    # --- remaining 8 PF coils (2..9): 1 upper + 1 lower each ---
    PFR, PFZ, PFDR, PFDZ, PFturn, PFCUR = [], [], [], [], [], []
    for j in range(8):
        cidx = 2 + j
        nbelt = len(PF[f"coil.{cidx}.element"])
        r = [PF[f"coil.{cidx}.element.{i}.geometry.rectangle.r"] for i in range(nbelt)]
        z = [PF[f"coil.{cidx}.element.{i}.geometry.rectangle.z"] for i in range(nbelt)]
        dr = [PF[f"coil.{cidx}.element.{i}.geometry.rectangle.width"] for i in range(nbelt)]
        dz = [PF[f"coil.{cidx}.element.{i}.geometry.rectangle.height"] for i in range(nbelt)]
        nbturn_total = sum(PF[f"coil.{cidx}.element.{i}.turns_with_sign"] for i in range(nbelt))
        PFR.append(sum(r) / nbelt)
        PFDR.append((max(r) - min(r)) + sum(dr) / nbelt)
        PFZ.append(sum(abs(np.array(z))) / nbelt)
        PFDZ.append((max(z) - min(abs(np.array(z)))) + sum(dz) / nbelt)
        PFCUR.append(_interp_current(cidx))
        PFturn.append(nbturn_total / 2.0)

    # Per-coil currents [A] are written in kA, the rtes convention
    # (read_input.cpp multiplies by 1000).
    def _i_kA(cur_A: float) -> float:
        return round(cur_A, 6) / 1000.0

    # Upper half: groups 1..18
    grp = 1
    for i in range(5):
        rows.append((PF1R, PF1DZ / 2 + i * PF1DZ, PF1DR, PF1DZ, int(nbturn1), _i_kA(PF1CUR), grp, 1.00)); grp += 1
    for i in range(5):
        rows.append((PF2R, low2Z + PF2DZ / 2.0 + i * PF2DZ, 2 * PF2DR, PF2DZ, int(nbturn2), _i_kA(PF2CUR), grp, 1.00)); grp += 1
    for i in range(8):
        rows.append((PFR[i], PFZ[i], PFDR[i], PFDZ[i], int(PFturn[i]), _i_kA(PFCUR[i]), grp, 1.00)); grp += 1

    # Lower half: groups 1..18 again (mirror in Z)
    grp = 1
    for i in range(5):
        rows.append((PF1R, -(PF1DZ / 2 + i * PF1DZ), PF1DR, PF1DZ, int(nbturn1), _i_kA(PF1CUR), grp, 1.00)); grp += 1
    for i in range(5):
        rows.append((PF2R, -(low2Z + PF2DZ / 2.0 + i * PF2DZ), 2 * PF2DR, PF2DZ, int(nbturn2), _i_kA(PF2CUR), grp, 1.00)); grp += 1
    for i in range(8):
        rows.append((PFR[i], -PFZ[i], PFDR[i], PFDZ[i], int(PFturn[i]), _i_kA(PFCUR[i]), grp, 1.00)); grp += 1

    return rows


def _coil_index_by_name(PF: Any) -> dict[str, int]:
    names = {}
    for cidx in range(len(PF["coil"])):
        # `in` first: reading a missing ODS path creates it.
        name = PF[f"coil.{cidx}.name"] if f"coil.{cidx}.name" in PF else f"PF{cidx + 1}"
        names[str(name).upper()] = cidx
    return names


def _element_pf_rows(ods: Any, time: float, lumped: Sequence[str] = ()) -> list[tuple]:
    """pf_active rows as one filament per element (``coil_model="elements"``).

    TES models a row outside its grid as a single filament at (R, Z), so each
    pf_active element becomes its own row. The turns are folded into the
    current (N_turn = 1, I = element ampere-turns), which keeps fractional
    ``turns_with_sign`` exact in TES's integer turn column. Every row gets its
    own coil group.

    Coils named in ``lumped`` (the shape-control coils) are written first
    instead: one row per up/down half at the half's mean position, with both
    halves in one group (ids 1..k in ``lumped`` order) and the coil's
    per-turn current, so TES's coil fit re-scales the whole coil and reports
    its current per turn (ampere-turns with N = 1 when a half's turns are not
    whole). Coils in series (VEST PF9/PF10) are still separate groups here. TES caps a group at 10 rows; the VEST shaping coils
    sit at R >= 0.71 m, far enough from the plasma for one filament per half.
    """
    PF = ods["pf_active"]
    pf_time = np.asarray(PF["time"], dtype=float)
    by_name = _coil_index_by_name(PF)
    unknown = [name for name in lumped if str(name).upper() not in by_name]
    if unknown:
        raise ValueError(
            f"shape_coils {unknown} are not pf_active coils; available: {sorted(by_name)}"
        )
    lumped_idx = [by_name[str(name).upper()] for name in lumped]

    def current(cidx: int) -> float:
        return float(np.interp(time, pf_time, np.asarray(PF[f"coil.{cidx}.current.data"], dtype=float)))

    rows: list[tuple] = []
    grp = 1
    for cidx in lumped_idx:
        elements = [PF[f"coil.{cidx}.element.{i}"] for i in range(len(PF[f"coil.{cidx}.element"]))]
        for upper in (True, False):
            half = [e for e in elements if (float(e["geometry.rectangle.z"]) >= 0.0) == upper]
            if not half:
                continue
            r = np.array([float(e["geometry.rectangle.r"]) for e in half])
            z = np.array([float(e["geometry.rectangle.z"]) for e in half])
            w = np.array([float(e["geometry.rectangle.width"]) for e in half])
            h = np.array([float(e["geometry.rectangle.height"]) for e in half])
            turns = float(sum(float(e["turns_with_sign"]) for e in half))
            if turns == 0.0:
                raise ValueError(f"shape coil {lumped[lumped_idx.index(cidx)]} has a half with zero net turns")
            if turns.is_integer():
                n_turn, i_kA = int(turns), round(current(cidx), 6) / 1000.0
            else:
                n_turn, i_kA = 1, round(current(cidx) * turns, 6) / 1000.0
            rows.append((
                float(r.mean()), float(z.mean()),
                float((r + w / 2).max() - (r - w / 2).min()),
                float((z + h / 2).max() - (z - h / 2).min()),
                n_turn, i_kA, grp, 1.00,
            ))
        grp += 1
    for cidx in range(len(PF["coil"])):
        if cidx in lumped_idx:
            continue
        coil_current = current(cidx)
        for i in range(len(PF[f"coil.{cidx}.element"])):
            element = PF[f"coil.{cidx}.element.{i}"]
            turns = float(element["turns_with_sign"])
            rows.append((
                float(element["geometry.rectangle.r"]),
                float(element["geometry.rectangle.z"]),
                float(element["geometry.rectangle.width"]),
                float(element["geometry.rectangle.height"]),
                1,
                round(coil_current * turns, 6) / 1000.0,
                grp,
                1.00,
            ))
            grp += 1
    return rows


def _merge_in_grid_rows(rows: list[tuple], config: TESConfig) -> list[tuple]:
    """Fold rows that TES would deposit into the same grid cell into one row.

    TES models a row whose centre lies inside the computational grid as a
    current density on ONE grid node: ``JPHI = I*N / (w*h)``, ASSIGNED (not
    added) in ``update_jphi.cpp``. Two rows rounding to the same node therefore
    keep only the last one, while the boundary flux (``update_psiv.cpp``)
    still sums them all. The 5 mm-pitch inner-wall filaments (W4, W11) put up
    to three rows in a 12.5 mm cell. Each occupied node becomes one row, sized
    one cell and carrying the summed ampere-turns, so the deposited and the
    boundary currents agree. The row sits at the mean position of the rows it
    replaces: that rounds to the same node, while a row placed exactly ON a
    grid node puts a filament singularity on the grid and TES's
    post-processing then fails ("(r,z) is out of range"). Rows whose group
    shape control re-fits are kept apart so their group is not altered.
    """
    dr = (config.rmax - config.rmin) / max(config.nr - 1, 1)
    dz = (config.zmax - config.zmin) / max(config.nz - 1, 1)
    refit = set(_shape_groups(config)) if config.fix_shape else set()
    merged: dict[tuple, list] = {}
    order: list[tuple | None] = []      # a merged-node key, or None for an out-of-grid row
    outside: list[tuple] = []
    for row in rows:
        r, z = row[0], row[1]
        if not (config.rmin < r < config.rmax and config.zmin < z < config.zmax):
            order.append(None)
            outside.append(row)
            continue
        # TES's own node choice (initialize.cpp / update_jphi.cpp), 0-based
        ll = int((r - config.rmin) / dr)
        jj = int((z - config.zmin) / dz)
        if r - (config.rmin + ll * dr) > dr / 2.0:
            ll += 1
        if z - (config.zmin + jj * dz) > dz / 2.0:
            jj += 1
        key = (ll, jj, row[6] if row[6] in refit else None)
        ampere_turns_kA = row[4] * row[5]
        if key in merged:
            entry = merged[key]
            entry["rows"].append(row)
            entry["kA"] += ampere_turns_kA
        else:
            merged[key] = {"rows": [row], "kA": ampere_turns_kA}
            order.append(key)

    def folded(entry: dict) -> tuple:
        rows_here = entry["rows"]
        if len(rows_here) == 1:
            # alone on its node: keep its turns and per-turn current, which a
            # coil fit scales (TES's shape response is built from COILT)
            only = rows_here[0]
            return (only[0], only[1], dr, dz) + tuple(only[4:])
        first = rows_here[0]
        return (float(np.mean([row[0] for row in rows_here])),
                float(np.mean([row[1] for row in rows_here])),
                dr, dz, 1, entry["kA"], first[6], first[7])

    remaining = iter(outside)
    rows = [next(remaining) if key is None else folded(merged[key]) for key in order]
    # TES indexes its group table by group id in an array sized NCOIL, so the
    # ids must stay within 1..NCOIL: renumber them in order of appearance.
    # Shape-control groups come first and keep their ids.
    renumber: dict[int, int] = {}
    for row in rows:
        renumber.setdefault(row[6], len(renumber) + 1)
    if any(renumber.get(group, group) != group for group in refit):
        raise ValueError("shape-control coil groups must lead the coil table")
    return [row[:6] + (renumber[row[6]],) + tuple(row[7:]) for row in rows]


def _coils_from_ods(ods: Any, config: TESConfig, time: float) -> list[tuple]:
    """Build the TES external-coil table from pf_active (+ optional pf_passive).

    ``config.coil_model`` selects how pf_active is laid out (see
    ``TESConfig.coil_model``). When ``config.eddy`` is set, every
    ``pf_passive`` loop is appended as its own single-element group.

    Rows inside the computational grid are folded per TES grid node (see
    ``_merge_in_grid_rows``), so the current TES deposits equals the rows'
    ampere-turns.

    Each row is ``(R, Z, dR, dZ, N_turn, I_kA, group_id, scale)``.
    """
    if config.coil_model == "elements":
        rows = _element_pf_rows(ods, time, lumped=config.shape_coils)
    elif config.coil_model == "legacy":
        if config.shape_coils:
            raise ValueError("shape_coils applies to coil_model='elements'; the legacy layout has fixed groups")
        rows = _legacy_pf_rows(ods, time)
    else:
        raise ValueError(
            f"TESConfig.coil_model must be 'elements' or 'legacy', got {config.coil_model!r}"
        )
    grp = max((row[6] for row in rows), default=0) + 1

    # --- optional eddy filaments from pf_passive ---
    if config.eddy:
        PFP = ods["pf_passive"]
        pfp_time = PFP["time"]
        nbloop = len(PFP["loop"])
        for i in range(nbloop):
            r = PFP[f"loop.{i}.element.0.geometry.outline.r"]
            z = PFP[f"loop.{i}.element.0.geometry.outline.z"]
            cur = PFP[f"loop.{i}.current"]
            R = sum(r) / 4
            Z = sum(z) / 4
            DR = r[1] - r[0]
            DZ = z[2] - z[1]
            CUR = float(np.interp(time, pfp_time, cur))   # [A]
            rows.append((R, Z, DR, DZ, 1, round(CUR, 6) / 1000.0, grp, 1.00)); grp += 1

    if config.fix_shape:
        groups = _shape_groups(config)
        if not groups:
            raise ValueError(
                "TES shape control (fix_shape=1) re-fits coil groups: name them with "
                "TESConfig.shape_coils (coil_model='elements') or grpid (legacy layout)."
            )
        sizes = {g: sum(1 for row in rows if row[6] == g) for g in groups}
        crowded = {g: n for g, n in sizes.items() if n > 10 or n == 0}
        if crowded:
            raise ValueError(
                f"shape-control coil groups must hold 1..10 rows (TES NGRPCOILMAX); got {crowded}"
            )
    return _merge_in_grid_rows(rows, config)


def _shape_groups(config: TESConfig) -> tuple[int, ...]:
    """Coil groups TES re-fits: explicit ``grpid``, else the lumped shape coils."""
    if config.grpid:
        return tuple(int(g) for g in config.grpid)
    return tuple(range(1, len(config.shape_coils) + 1))


def _mag_diagnostics_from_ods(ods: Any) -> dict[str, np.ndarray]:
    """Flux-loop and magnetic-probe positions for synthetic-diagnostic output."""
    MG = ods["magnetics"]
    nbprobe = len(MG["b_field_pol_probe"])
    mpr = np.array([MG[f"b_field_pol_probe.{i}.position.r"] for i in range(nbprobe)], dtype=float)
    mpz = np.array([MG[f"b_field_pol_probe.{i}.position.z"] for i in range(nbprobe)], dtype=float)
    mpt = 90.0 * np.ones(nbprobe)             # degrees; rtes converts to radians

    nbflux = len(MG["flux_loop"])
    flr = np.array([MG[f"flux_loop.{i}.position.0.r"] for i in range(nbflux)], dtype=float)
    flz = np.array([MG[f"flux_loop.{i}.position.0.z"] for i in range(nbflux)], dtype=float)
    return {"MPR": mpr, "MPZ": mpz, "MPT": mpt, "FLR": flr, "FLZ": flz}


# ----------------------------------------------------------------------------- #
#  Writers
# ----------------------------------------------------------------------------- #
def _chunked(values: Sequence[float], fmt: str, per_line: int = 5, prefix: str = "   ") -> str:
    out, line, n = [], prefix, 0
    for v in values:
        line += fmt % v
        n += 1
        if n == per_line:
            out.append(line)
            line, n = prefix, 0
    if n:
        out.append(line)
    return "\n".join(out)


def write_tes_cinput(path: Path, P: dict) -> Path:
    """Write the strict C-format input consumed directly by ``rtes``."""
    L: list[str] = []
    w = L.append

    w("== CONTROL FLAG ==")
    w("MAG_DIAGNOSTICS     %d" % P["mag_diagnostics"])
    w("MSE_DIAGNOSTICS     %d" % P["mse_diagnostics"])
    w("")
    w("== NUMERICAL CONFIGURATION ==")
    w("SHOT                %d" % P["shot"])
    w("CTIME               %d" % P["ctime"])
    w("PROF_TYPE           %d" % P["prof_type"])
    w("NR                  %d" % P["nr"])
    w("NZ                  %d" % P["nz"])
    w("RMIN                %f" % P["rmin"])
    w("RMAX                %f" % P["rmax"])
    w("ZMIN                %f" % P["zmin"])
    w("ZMAX                %f" % P["zmax"])
    w("")
    w("INIT_R0             %f" % P["init_r0"])
    w("INIT_Z0             %f" % P["init_z0"])
    w("INIT_A0             %f" % P["init_a0"])
    w("")
    w("IP0[kA]             %f" % P["ip0_kA"])
    w("MAJOR_R             %f" % P["major_r"])
    w("BT0                 %f" % P["bt0"])
    w("BETAP               %f" % P["betap"])
    w("BETAP_TYPE          %d" % P["betap_type"])
    w("ALPHA_P_A           %f" % P["alpha_p_a"])
    w("ALPHA_P_B           %f" % P["alpha_p_b"])
    w("ALPHA_F_A           %f" % P["alpha_f_a"])
    w("ALPHA_F_B           %f" % P["alpha_f_b"])
    w("")
    w("NFLUX               %d" % P["nflux"])
    w("NTHETA              %d" % P["ntheta"])
    w("")
    w("GPS                 %f" % P["gps"])
    w("RELAX_SOR           %f" % P["relax_sor"])
    w("RELAX_SHP           %f" % P["relax_shp"])
    w("TIKHONOV_FACTOR     %f" % P["tikhonov_factor"])
    w("")
    w("ERRTOL_LOOP         %le" % P["errtol_loop"])
    w("ERRTOL_SHAPE        %le" % P["errtol_shape"])
    w("")

    # --- limiter ---
    limr, limz = P["limr"], P["limz"]
    w("== Limiter ==")
    w("NLIM                %d" % len(limr))
    w("LIMR")
    w(_chunked(limr, "%9.5f "))
    w("LIMZ")
    w(_chunked(limz, "%9.5f "))
    w("")

    # --- external currents ---
    coils = P["coils"]
    w("== EXTERNAL CURRENTS [kA] ==")
    w("NCOIL               %d" % len(coils))
    for (r, z, dr, dz, nturn, ikA, grp, scale) in coils:
        w("%9.5f %9.5f %9.5f %9.5f %5d %9.5f %5d %9.5f"
          % (r, z, dr, dz, nturn, ikA, grp, scale))
    w("")

    # --- shaping control ---
    w("== SHAPING CONTROL ==")
    w("FIX_SHAPE           %d" % P["fix_shape"])
    w("")
    w("FLUX_LINKAGE        %d  %9.5f" % (P["flux_linkage"][0], P["flux_linkage"][1]))
    w("")
    w("NXPT                %d" % P["nxpt"])
    w("XPR                 " + " ".join("%-9.5f" % v for v in P["xpr"]))
    w("XPZ                 " + " ".join("%-9.5f" % v for v in P["xpz"]))
    w("ACTIVE              " + " ".join("%-9d" % v for v in P["active"]))
    w("SNOWFLAKE           " + " ".join("%-9d" % v for v in P["snowflake"]))
    w("DRSEP               %d  %9.5f" % (P["drsep"][0], P["drsep"][1]))
    w("DSEP                %d  %9.5f" % (P["dsep"][0], P["dsep"][1]))
    w("")
    w("NISO                %d" % len(P["isor"]))
    w("ISOR                " + " ".join("%-9.5f" % v for v in P["isor"]))
    w("ISOZ                " + " ".join("%-9.5f" % v for v in P["isoz"]))
    w("")
    w("NCGRP_FOR_SHAPE     %d" % len(P["grpid"]))
    w("GRPID               " + " ".join("%-4d" % v for v in P["grpid"]))
    w("")

    # --- magnetic diagnostics ---
    if P["mag_diagnostics"]:
        md = P["mag_diag"]
        w("== Magnetic Diagnostics ==")
        w("NFL                 %d" % len(md["FLR"]))
        w("FLR")
        w(_chunked(md["FLR"], "%13.6le "))
        w("FLZ")
        w(_chunked(md["FLZ"], "%13.6le "))
        w("")
        w("NMP                 %d" % len(md["MPR"]))
        w("MPR")
        w(_chunked(md["MPR"], "%13.6le "))
        w("MPZ")
        w(_chunked(md["MPZ"], "%13.6le "))
        w("MPT")
        w(_chunked(md["MPT"], "%13.6le "))
        w("")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def write_tes_namelist(path: Path, P: dict) -> Path:
    """Write a human-readable Fortran-style namelist (debugging aid only)."""
    L: list[str] = []
    w = L.append
    w("$MAIN")
    w("  MAG_DIAGNOSTICS  = %d" % P["mag_diagnostics"])
    w("  MSE_DIAGNOSTICS  = %d" % P["mse_diagnostics"])
    w("  SHOT             = %06d" % P["shot"])
    w("  CTIME            = %06d" % P["ctime"])
    w("  NR               = %d" % P["nr"])
    w("  NZ               = %d" % P["nz"])
    w("  RMIN             = %g" % P["rmin"])
    w("  RMAX             = %g" % P["rmax"])
    w("  ZMIN             = %g" % P["zmin"])
    w("  ZMAX             = %g" % P["zmax"])
    w("  INIT_R0          = %g" % P["init_r0"])
    w("  INIT_Z0          = %g" % P["init_z0"])
    w("  INIT_A0          = %g" % P["init_a0"])
    w("  NFLUX            = %d" % P["nflux"])
    w("  NTHETA           = %d" % P["ntheta"])
    w("  GPS              = %g" % P["gps"])
    w("  RELAX_SOR        = %g" % P["relax_sor"])
    w("  RELAX_SHP        = %g" % P["relax_shp"])
    w("  TIKHONOV_FACTOR  = %g" % P["tikhonov_factor"])
    w("  ERRTOL_LOOP      = %g" % P["errtol_loop"])
    w("  ERRTOL_SHAPE     = %g" % P["errtol_shape"])
    w("  LIMR             = " + ", ".join("%.4f" % v for v in P["limr"]))
    w("  LIMZ             = " + ", ".join("%.4f" % v for v in P["limz"]))
    w("/")
    w("")
    w("$EQ_CONSTRAINT")
    w("  PROF_TYPE        = %d" % P["prof_type"])
    w("  IP0_kA           = %g" % P["ip0_kA"])
    w("  MAJOR_R          = %g" % P["major_r"])
    w("  BT0              = %g" % P["bt0"])
    w("  BETAP            = %g" % P["betap"])
    w("  BETAP_TYPE       = %d" % P["betap_type"])
    w("  ALPHA_P_A        = %g" % P["alpha_p_a"])
    w("  ALPHA_P_B        = %g" % P["alpha_p_b"])
    w("  ALPHA_F_A        = %g" % P["alpha_f_a"])
    w("  ALPHA_F_B        = %g" % P["alpha_f_b"])
    w("/")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- #
#  Public entry point
# ----------------------------------------------------------------------------- #
def prepare_tes_inputs(ods: Any, config: TESConfig) -> TESInputs:
    """Build the TES C-format input from an ODS and configuration."""
    workdir = Path(config.workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    shot = _infer_shot(ods, config)
    time = _resolve_time(ods, config)
    ctime = int(round(time * 1000))

    limr, limz = _limiter_from_ods(ods, config)
    ip0_kA, bt0, betap = _resolve_targets(ods, config, time)
    coils = _coils_from_ods(ods, config, time)
    mag_diag = _mag_diagnostics_from_ods(ods) if config.mag_diagnostics else None

    P = {
        "shot": shot, "ctime": ctime,
        "mag_diagnostics": config.mag_diagnostics, "mse_diagnostics": config.mse_diagnostics,
        "prof_type": config.prof_type,
        "nr": config.nr, "nz": config.nz,
        "rmin": config.rmin, "rmax": config.rmax, "zmin": config.zmin, "zmax": config.zmax,
        "init_r0": config.init_r0, "init_z0": config.init_z0, "init_a0": config.init_a0,
        "ip0_kA": ip0_kA, "major_r": config.major_r, "bt0": bt0, "betap": betap,
        "betap_type": config.betap_type,
        "alpha_p_a": config.alpha_p_a, "alpha_p_b": config.alpha_p_b,
        "alpha_f_a": config.alpha_f_a, "alpha_f_b": config.alpha_f_b,
        "nflux": config.nflux, "ntheta": config.ntheta,
        "gps": config.gps, "relax_sor": config.relax_sor, "relax_shp": config.relax_shp,
        "tikhonov_factor": config.tikhonov_factor,
        "errtol_loop": config.errtol_loop, "errtol_shape": config.errtol_shape,
        "fix_shape": config.fix_shape, "flux_linkage": config.flux_linkage,
        "nxpt": config.nxpt, "xpr": config.xpr, "xpz": config.xpz,
        "active": config.active, "snowflake": config.snowflake,
        "drsep": config.drsep, "dsep": config.dsep,
        "isor": config.isor, "isoz": config.isoz,
        "grpid": _shape_groups(config) if config.fix_shape else tuple(config.grpid),
        "limr": limr, "limz": limz,
        "coils": coils,
        "mag_diag": mag_diag,
    }

    cinput = write_tes_cinput(workdir / f"{shot:06d}_{ctime:06d}_tes.cin", P)
    namelist = None
    if config.emit_namelist:
        namelist = write_tes_namelist(workdir / f"{shot:06d}_tes.in", P)

    return TESInputs(workdir=workdir, cinput=cinput, ods=ods, namelist=namelist, files=(cinput,))
