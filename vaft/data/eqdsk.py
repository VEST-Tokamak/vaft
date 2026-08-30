"""EFIT EQDSK/GEQDSK compatibility helpers implemented inside VAFT."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional
import re

import numpy as np


_STANDARD_KEYS = (
    "CASE",
    "NW",
    "NH",
    "RDIM",
    "ZDIM",
    "RCENTR",
    "RLEFT",
    "ZMID",
    "RMAXIS",
    "ZMAXIS",
    "SIMAG",
    "SIBRY",
    "BCENTR",
    "CURRENT",
    "FPOL",
    "PRES",
    "FFPRIM",
    "PPRIME",
    "PSIRZ",
    "QPSI",
    "NBBBS",
    "LIMITR",
    "RBBBS",
    "ZBBBS",
    "RLIM",
    "ZLIM",
)

_FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")


@dataclass
class GEQDSK:
    """Neutral VAFT representation of an EFIT GEQDSK/g-file."""

    mapping: MutableMapping[str, Any]
    source: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: EFIT's trailing ``&OUT1``/``&BASIS``/``&CHIOUT`` namelists, when the file
    #: carries them.  Group and variable names are lowercased, as f90nml reads
    #: them.
    namelists: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.mapping[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.mapping[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self.mapping

    def get(self, key: str, default: Any = None) -> Any:
        return self.mapping.get(key, default)

    def keys(self) -> Iterable[str]:
        return self.mapping.keys()

    def to_omas(
        self,
        ods: Any = None,
        time_index: int = 0,
        profile_index: int = 0,
        allow_derived_data: bool = True,
    ) -> Any:
        """Convert this GEQDSK into an OMAS ODS equilibrium subtree."""
        return to_omas(
            self,
            ods=ods,
            time_index=time_index,
            profile_index=profile_index,
            allow_derived_data=allow_derived_data,
        )

    def to_imas(self, target: Any, *, occurrence: Optional[dict] = None, imas_version: Optional[str] = None) -> Any:
        """Convert this GEQDSK through OMAS and save it to an IMAS target."""
        return to_imas(self, target, occurrence=occurrence, imas_version=imas_version)

    def write(self, path: str | Path) -> Path:
        """Write this GEQDSK to disk."""
        return write_geqdsk(self, path)


def _parse_header(line: str) -> tuple[str, int, int]:
    ints = [int(match) for match in re.findall(r"[-+]?\d+", line)]
    if len(ints) < 2:
        raise ValueError(f"Could not find GEQDSK grid dimensions in header: {line!r}")
    nw, nh = ints[-2], ints[-1]
    idx = line.rfind(str(nw))
    case = line[:idx].strip() if idx >= 0 else line.strip()
    return case or "VAFT GEQDSK", nw, nh


def _numbers_from_text(text: str) -> list[float]:
    return [float(token.replace("D", "E").replace("d", "e")) for token in _FLOAT_RE.findall(text)]


def _take(values: list[float], cursor: int, count: int) -> tuple[np.ndarray, int]:
    end = cursor + count
    if end > len(values):
        raise ValueError(f"GEQDSK ended early while reading {count} values")
    return np.asarray(values[cursor:end], dtype=float), end


def _metadata(mapping: Mapping[str, Any], parser: str, source: str | Path | None = None) -> dict[str, Any]:
    meta = {
        "standard_keys": [key for key in _STANDARD_KEYS if key in mapping],
        "parser": parser,
    }
    if source is not None:
        meta["source"] = str(source)
    return meta


def _coerce_geqdsk(geqdsk: GEQDSK | Mapping[str, Any]) -> GEQDSK:
    if isinstance(geqdsk, GEQDSK):
        return geqdsk
    if isinstance(geqdsk, Mapping):
        return GEQDSK(dict(geqdsk), metadata=_metadata(geqdsk, "mapping"))
    if hasattr(geqdsk, "keys"):
        mapping = {key: geqdsk[key] for key in geqdsk.keys()}
        return GEQDSK(mapping, source=Path(getattr(geqdsk, "filename", "")) if getattr(geqdsk, "filename", None) else None)
    raise TypeError("Expected a GEQDSK or mapping-like object")


_NAMELIST_START = re.compile(r"^\s*&(\w+)\s*$")


def _trailing_namelists(lines: list[str]) -> dict[str, dict[str, Any]]:
    """Read the Fortran namelists EFIT appends after the g-file body.

    EFIT writes ``&OUT1``/``&BASIS``/``&CHIOUT`` -- the reconstruction's own
    inputs and fit diagnostics -- after the last limiter point. They are the
    only record of those settings that travels with the g-file, so dropping
    them loses data that exists nowhere else in the equilibrium.
    """
    for index, line in enumerate(lines):
        if _NAMELIST_START.match(line):
            break
    else:
        return {}

    import f90nml

    from vaft.data.keqdsk import _plain

    try:
        return _plain(f90nml.reads("\n".join(lines[index:])))
    except Exception:
        # A malformed trailing block must not cost us the equilibrium itself.
        return {}


def read_geqdsk(path: str | Path) -> GEQDSK:
    """Read an EFIT GEQDSK/g-file using VAFT's standalone parser."""
    source = Path(path).expanduser()
    lines = source.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty GEQDSK file: {source}")

    case, nw, nh = _parse_header(lines[0])
    values = _numbers_from_text("\n".join(lines[1:]))
    cursor = 0

    scalars, cursor = _take(values, cursor, 20)
    fpol, cursor = _take(values, cursor, nw)
    pres, cursor = _take(values, cursor, nw)
    ffprim, cursor = _take(values, cursor, nw)
    pprime, cursor = _take(values, cursor, nw)
    psirz_flat, cursor = _take(values, cursor, nw * nh)
    qpsi, cursor = _take(values, cursor, nw)

    if cursor + 2 > len(values):
        raise ValueError("GEQDSK missing boundary/limiter counts")
    nbbbs = int(round(values[cursor]))
    limitr = int(round(values[cursor + 1]))
    cursor += 2

    boundary, cursor = _take(values, cursor, 2 * max(nbbbs, 0))
    limiter, cursor = _take(values, cursor, 2 * max(limitr, 0))

    mapping: dict[str, Any] = {
        "CASE": case,
        "NW": int(nw),
        "NH": int(nh),
        "RDIM": scalars[0],
        "ZDIM": scalars[1],
        "RCENTR": scalars[2],
        "RLEFT": scalars[3],
        "ZMID": scalars[4],
        "RMAXIS": scalars[5],
        "ZMAXIS": scalars[6],
        "SIMAG": scalars[7],
        "SIBRY": scalars[8],
        "BCENTR": scalars[9],
        "CURRENT": scalars[10],
        "FPOL": fpol,
        "PRES": pres,
        "FFPRIM": ffprim,
        "PPRIME": pprime,
        "PSIRZ": psirz_flat.reshape(nh, nw).T,
        "QPSI": qpsi,
        "NBBBS": int(nbbbs),
        "LIMITR": int(limitr),
        "RBBBS": boundary[0::2],
        "ZBBBS": boundary[1::2],
        "RLIM": limiter[0::2],
        "ZLIM": limiter[1::2],
    }
    return GEQDSK(
        mapping=mapping,
        source=source,
        metadata=_metadata(mapping, "vaft", source),
        namelists=_trailing_namelists(lines),
    )


def _format_floats(values: Iterable[Any]) -> list[str]:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    lines = []
    for idx in range(0, arr.size, 5):
        lines.append("".join(f"{value:16.9E}" for value in arr[idx : idx + 5]))
    return lines


def _pairs(r: Any, z: Any) -> np.ndarray:
    r_arr = np.asarray(r, dtype=float).reshape(-1)
    z_arr = np.asarray(z, dtype=float).reshape(-1)
    count = min(r_arr.size, z_arr.size)
    out = np.empty(count * 2, dtype=float)
    out[0::2] = r_arr[:count]
    out[1::2] = z_arr[:count]
    return out


def write_geqdsk(geqdsk: GEQDSK | Mapping[str, Any], path: str | Path) -> Path:
    """Write GEQDSK data to disk using standard EFIT g-file ordering."""
    item = _coerce_geqdsk(geqdsk)
    data = item.mapping
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    nw = int(data["NW"])
    nh = int(data["NH"])
    case = str(data.get("CASE", "VAFT GEQDSK"))[:48]
    psirz = np.asarray(data["PSIRZ"], dtype=float)
    if psirz.shape != (nw, nh):
        psirz = psirz.reshape(nw, nh)

    scalars = [
        data.get("RDIM", 0.0),
        data.get("ZDIM", 0.0),
        data.get("RCENTR", 0.0),
        data.get("RLEFT", 0.0),
        data.get("ZMID", 0.0),
        data.get("RMAXIS", 0.0),
        data.get("ZMAXIS", 0.0),
        data.get("SIMAG", 0.0),
        data.get("SIBRY", 0.0),
        data.get("BCENTR", 0.0),
        data.get("CURRENT", 0.0),
        data.get("SIMAG", 0.0),
        0.0,
        data.get("RMAXIS", 0.0),
        0.0,
        data.get("ZMAXIS", 0.0),
        0.0,
        data.get("SIBRY", 0.0),
        0.0,
        0.0,
    ]

    rbbbs = np.asarray(data.get("RBBBS", []), dtype=float)
    zbbbs = np.asarray(data.get("ZBBBS", []), dtype=float)
    rlim = np.asarray(data.get("RLIM", []), dtype=float)
    zlim = np.asarray(data.get("ZLIM", []), dtype=float)
    nbbbs = int(min(rbbbs.size, zbbbs.size))
    limitr = int(min(rlim.size, zlim.size))

    lines = [f"{case:<48}{0:4d}{nw:4d}{nh:4d}"]
    lines.extend(_format_floats(scalars))
    for key in ("FPOL", "PRES", "FFPRIM", "PPRIME"):
        lines.extend(_format_floats(np.asarray(data.get(key, np.zeros(nw)), dtype=float)[:nw]))
    lines.extend(_format_floats(psirz.T.reshape(-1)))
    lines.extend(_format_floats(np.asarray(data.get("QPSI", np.zeros(nw)), dtype=float)[:nw]))
    lines.append(f"{nbbbs:5d}{limitr:5d}")
    if nbbbs:
        lines.extend(_format_floats(_pairs(rbbbs[:nbbbs], zbbbs[:nbbbs])))
    if limitr:
        lines.extend(_format_floats(_pairs(rlim[:limitr], zlim[:limitr])))
    target.write_text("\n".join(lines) + "\n")
    return target


def _path_get(ods: Any, path: str, default: Any = None) -> Any:
    try:
        return ods[path]
    except Exception:
        return default


def _profile(values: Any, size: int, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=float).reshape(-1)
    if arr.size == size:
        return arr
    if arr.size == 0:
        return np.full(size, default, dtype=float)
    source = np.linspace(0.0, 1.0, arr.size)
    target = np.linspace(0.0, 1.0, size)
    return np.interp(target, source, arr)


def _scalar(value: Any, default: float = 0.0) -> float:
    try:
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return float(default)
        return float(arr.reshape(-1)[0])
    except Exception:
        return float(default)


# EFIT ``g<shot>.<time_ms>`` naming, e.g. ``g039915.00319``.  Shared with
# ``vaft.machine_mapping.equilibrium`` so both infer shot and time identically.
GFILE_NAME_PATTERN = re.compile(r"[a-zA-Z](?P<shot>\d+)\.(?P<time>\d+)?")
GFILE_HEADER_PATTERN = re.compile(
    r"#\s*(?P<shot>\d+)\s+(?P<time>\d+(?:\.\d+)?)\s*(?P<unit>ms)?\b",
    re.IGNORECASE,
)


def _header_shot_time(source: Path) -> tuple[int, Optional[float]]:
    try:
        with source.open(encoding="ascii", errors="ignore") as stream:
            header = stream.readline()
    except OSError:
        return 0, None
    match = GFILE_HEADER_PATTERN.search(header)
    if match is None:
        return 0, None
    raw_time = float(match.group("time"))
    # EFIT headers commonly use integer milliseconds with an explicit "ms".
    # VFIT headers in this repository use 10-microsecond ticks without a unit.
    scale = 1.0e-3 if match.group("unit") else (1.0e-5 if raw_time >= 1000 else 1.0e-3)
    return int(match.group("shot")), raw_time * scale


def infer_source_shot_time(source: Optional[Path]) -> tuple[int, Optional[float]]:
    """Infer ``(shot, time_in_seconds)`` from an EFIT g-file name.

    The time is ``None`` when the name carries no numeric time field, so callers
    can distinguish "unknown" from a genuine ``t = 0``.
    """
    if source is None:
        return 0, None
    path = Path(source)
    header_shot, header_time = _header_shot_time(path)
    if header_time is not None:
        return header_shot, header_time
    match = GFILE_NAME_PATTERN.match(path.name)
    if match is None:
        return 0, None
    time = match.group("time")
    return int(match.group("shot")), None if time is None else float(time) / 1000.0


def _infer_source_time(source: Optional[Path]) -> float:
    time = infer_source_shot_time(source)[1]
    return 0.0 if time is None else time


def _infer_source_shot(source: Optional[Path]) -> int:
    return infer_source_shot_time(source)[0]


def from_omas(
    ods: Any,
    time_index: int = 0,
    profile_index: int = 0,
    allow_derived_data: bool = True,
) -> GEQDSK:
    """Build a GEQDSK wrapper from an OMAS ODS equilibrium time slice."""
    _ = allow_derived_data
    ts = ods[f"equilibrium.time_slice.{time_index}"]
    prof2d = ts[f"profiles_2d.{profile_index}"]
    r = np.asarray(prof2d["grid.dim1"], dtype=float)
    z = np.asarray(prof2d["grid.dim2"], dtype=float)
    # The DD declares psi(:,:)'s coordinates as [grid.dim1, grid.dim2], i.e.
    # axis 0 = dim1 (R) and axis 1 = dim2 (Z) -- exactly what to_omas() below
    # writes (`psi.reshape(nw, nh)`, nw=len(R)). A shape-based "is this
    # secretly transposed?" heuristic is ambiguous whenever nw == nh (VEST's
    # EFIT/CHEASE grids always are: 129x129, 513x513) and was silently
    # transposing psi that to_omas() had already written correctly -- this
    # was the sole source of a "xin not in ascending order" CHEASE failure
    # on every ODS-sourced refinement. Trust the DD convention unconditionally.
    psi = np.asarray(prof2d["psi"], dtype=float)
    nw, nh = int(r.size), int(z.size)

    psi_axis = float(_path_get(ts, "global_quantities.psi_axis", np.nanmin(psi)))
    psi_boundary = float(_path_get(ts, "global_quantities.psi_boundary", np.nanmax(psi)))
    mapping: dict[str, Any] = {
        "CASE": str(_path_get(ods, "equilibrium.ids_properties.comment", "VAFT GEQDSK")),
        "NW": nw,
        "NH": nh,
        "RDIM": float(np.max(r) - np.min(r)) if nw else 0.0,
        "ZDIM": float(np.max(z) - np.min(z)) if nh else 0.0,
        "RCENTR": float(_path_get(ods, "equilibrium.vacuum_toroidal_field.r0", np.mean(r) if nw else 0.0)),
        "RLEFT": float(np.min(r)) if nw else 0.0,
        "ZMID": float((np.max(z) + np.min(z)) / 2.0) if nh else 0.0,
        "RMAXIS": float(_path_get(ts, "global_quantities.magnetic_axis.r", np.mean(r) if nw else 0.0)),
        "ZMAXIS": float(_path_get(ts, "global_quantities.magnetic_axis.z", 0.0)),
        "SIMAG": psi_axis,
        "SIBRY": psi_boundary,
        "BCENTR": _scalar(_path_get(ods, f"equilibrium.vacuum_toroidal_field.b0.{time_index}", _path_get(ods, "equilibrium.vacuum_toroidal_field.b0", 0.0))),
        "CURRENT": float(_path_get(ts, "global_quantities.ip", 0.0)),
        "FPOL": _profile(_path_get(ts, "profiles_1d.f"), nw),
        "PRES": _profile(_path_get(ts, "profiles_1d.pressure"), nw),
        "FFPRIM": _profile(_path_get(ts, "profiles_1d.f_df_dpsi"), nw),
        "PPRIME": _profile(_path_get(ts, "profiles_1d.dpressure_dpsi"), nw),
        "PSIRZ": psi.reshape(nw, nh),
        "QPSI": _profile(_path_get(ts, "profiles_1d.q"), nw),
        "RBBBS": np.asarray(_path_get(ts, "boundary.outline.r", []), dtype=float),
        "ZBBBS": np.asarray(_path_get(ts, "boundary.outline.z", []), dtype=float),
        "RLIM": np.asarray(_path_get(ods, "wall.description_2d.0.limiter.unit.0.outline.r", []), dtype=float),
        "ZLIM": np.asarray(_path_get(ods, "wall.description_2d.0.limiter.unit.0.outline.z", []), dtype=float),
    }
    mapping["NBBBS"] = int(min(mapping["RBBBS"].size, mapping["ZBBBS"].size))
    mapping["LIMITR"] = int(min(mapping["RLIM"].size, mapping["ZLIM"].size))
    return GEQDSK(mapping=mapping, metadata=_metadata(mapping, "omas"))


#: Minimum marching-squares vertex count for a traced flux surface to be
#: trusted for a volume; below this the polygon is grid resolution, not
#: plasma geometry.
_MIN_CONTOUR_POINTS = 24


def _rho_tor_profile(
    qpsi: Any, psi_1d: np.ndarray, b0: Any
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Integrate ``q`` into toroidal flux and the rho_tor coordinate.

    EFIT writes ``psi`` in Wb/rad, so ``dPhi/dpsi = 2*pi*q`` and

        Phi(psi) = 2*pi * cumulative_trapezoid(q, psi)
        rho_tor  = sqrt(|Phi| / (pi * |B0|))

    This is the same coordinate OMFIT's ``fluxSurfaces`` produces -- they agree
    to within 1.5e-3 in normalized units on VEST g-files. The ``sqrt(psi_N)``
    proxy VAFT wrote before is not: it is off by percent-level amounts, and
    every kinetic profile is mapped onto this grid.

    Returns ``(phi, rho_tor, rho_tor_norm)``, or ``None`` when the profile is
    degenerate (flat psi, zero field, no toroidal flux) and the caller should
    fall back to the proxy.
    """
    from vaft.compat import cumtrapz_compat

    qpsi = np.asarray(qpsi, dtype=float).reshape(-1)
    psi_1d = np.asarray(psi_1d, dtype=float).reshape(-1)
    b0 = abs(float(b0))
    if qpsi.size != psi_1d.size or qpsi.size < 2 or b0 == 0.0:
        return None
    if not np.all(np.isfinite(qpsi)) or not np.all(np.isfinite(psi_1d)):
        return None

    phi = 2.0 * np.pi * np.asarray(cumtrapz_compat(qpsi, x=psi_1d), dtype=float)
    rho_tor = np.sqrt(np.abs(phi) / (np.pi * b0))
    edge = float(rho_tor[-1])
    if not np.isfinite(edge) or edge <= 0.0:
        return None
    return phi, rho_tor, rho_tor / edge


def _volume_profile(
    data: Mapping[str, Any], psi_norm: np.ndarray
) -> Optional[np.ndarray]:
    """Enclosed volume on each psi_N level, by tracing the flux surfaces.

    Each level's longest traced contour is revolved with the exact
    ``V = pi * closed_integral(R^2 dZ)`` form. The axis is 0 by construction and
    the edge uses the g-file's own ``RBBBS``/``ZBBBS`` boundary rather than a
    traced contour, which is both exact and cheaper.

    Returns ``None`` when the grid or the boundary cannot support the trace, so
    the caller simply leaves ``profiles_1d.volume`` unwritten.
    """
    from vaft.formula.equilibrium import exact_volume_from_RZ_contour
    from vaft.process.equilibrium import extract_flux_surface_contours

    psi_norm = np.asarray(psi_norm, dtype=float).reshape(-1)
    if psi_norm.size < 3:
        return None
    try:
        nw, nh = int(data["NW"]), int(data["NH"])
        psi_axis, psi_boundary = float(data["SIMAG"]), float(data["SIBRY"])
        if psi_axis == psi_boundary:
            return None
        r_grid = np.linspace(0.0, float(data["RDIM"]), nw) + float(data["RLEFT"])
        z_grid = (
            np.linspace(0.0, float(data["ZDIM"]), nh)
            - float(data["ZDIM"]) / 2.0
            + float(data["ZMID"])
        )
        psi_2d = np.asarray(data["PSIRZ"], dtype=float).reshape(nw, nh)

        r_bnd = np.asarray(data.get("RBBBS", []), dtype=float).reshape(-1)
        z_bnd = np.asarray(data.get("ZBBBS", []), dtype=float).reshape(-1)
        if r_bnd.size < 3 or r_bnd.size != z_bnd.size:
            return None
        edge_volume = exact_volume_from_RZ_contour(r_bnd, z_bnd)

        # The axis level has no contour and the boundary level comes from
        # RBBBS/ZBBBS, so only the interior levels are traced.
        interior = psi_norm[1:-1]
        contours = extract_flux_surface_contours(
            psi_2d, r_grid, z_grid, psi_axis, psi_boundary, interior
        )

        volume = np.empty(psi_norm.size, dtype=float)
        volume[0] = 0.0
        volume[-1] = edge_volume
        for index, level in enumerate(interior, start=1):
            segments = contours.get(float(level)) or []
            # Marching squares returns a handful of vertices for the innermost
            # surfaces, where the flux surface is only a few cells across; the
            # volume those polygons enclose is grid artifact, not geometry.
            # Drop them and let the fill below interpolate -- V is very nearly
            # linear in psi_N near the axis, so that is the better estimate.
            segments = [seg for seg in segments if seg[0].size >= _MIN_CONTOUR_POINTS]
            if not segments:
                volume[index] = np.nan
                continue
            r_seg, z_seg = max(segments, key=lambda segment: segment[0].size)
            volume[index] = exact_volume_from_RZ_contour(r_seg, z_seg)
    except Exception:
        return None

    # Levels with no closed contour -- common right at the axis, where the
    # surface is smaller than a grid cell -- are filled from their neighbours
    # rather than left as NaN, which would poison every downstream integral.
    missing = ~np.isfinite(volume)
    if missing.all():
        return None
    if missing.any():
        good = ~missing
        volume[missing] = np.interp(psi_norm[missing], psi_norm[good], volume[good])
    if not np.all(np.diff(volume) >= 0.0):
        return None
    return volume


def to_omas(
    geqdsk: GEQDSK | Mapping[str, Any],
    ods: Any = None,
    time_index: int = 0,
    profile_index: int = 0,
    allow_derived_data: bool = True,
) -> Any:
    """Convert GEQDSK data to an OMAS ODS without OMFIT."""
    from omas import ODS

    item = _coerce_geqdsk(geqdsk)
    data = item.mapping
    if ods is None:
        ods = ODS()
    if f"equilibrium.time_slice.{time_index}" in ods:
        ods[f"equilibrium.time_slice.{time_index}"] = ODS()

    eqt = ods[f"equilibrium.time_slice.{time_index}"]
    nw, nh = int(data["NW"]), int(data["NH"])
    psi_1d = np.linspace(float(data["SIMAG"]), float(data["SIBRY"]), nw)
    psi_norm = (psi_1d - psi_1d[0]) / (psi_1d[-1] - psi_1d[0]) if nw > 1 and psi_1d[-1] != psi_1d[0] else np.zeros(nw)

    ods["dataset_description.data_entry.pulse"] = _infer_source_shot(item.source)
    ods["equilibrium.ids_properties.comment"] = str(data.get("CASE", "VAFT GEQDSK"))
    eqt["time"] = _infer_source_time(item.source)
    eqt["global_quantities.magnetic_axis.r"] = float(data["RMAXIS"])
    eqt["global_quantities.magnetic_axis.z"] = float(data["ZMAXIS"])
    eqt["global_quantities.psi_axis"] = float(data["SIMAG"])
    eqt["global_quantities.psi_boundary"] = float(data["SIBRY"])
    eqt["global_quantities.ip"] = float(data["CURRENT"])
    ods["equilibrium.vacuum_toroidal_field.r0"] = float(data["RCENTR"])
    try:
        ods.set_time_array("equilibrium.vacuum_toroidal_field.b0", time_index, float(data["BCENTR"]))
    except Exception:
        ods[f"equilibrium.vacuum_toroidal_field.b0.{time_index}"] = float(data["BCENTR"])

    eqt["profiles_1d.psi"] = psi_1d
    eqt["profiles_1d.f"] = np.asarray(data["FPOL"], dtype=float)
    eqt["profiles_1d.pressure"] = np.asarray(data["PRES"], dtype=float)
    eqt["profiles_1d.f_df_dpsi"] = np.asarray(data["FFPRIM"], dtype=float)
    eqt["profiles_1d.dpressure_dpsi"] = np.asarray(data["PPRIME"], dtype=float)
    eqt["profiles_1d.q"] = np.asarray(data["QPSI"], dtype=float)

    rho_tor_terms = _rho_tor_profile(data["QPSI"], psi_1d, data["BCENTR"])
    if rho_tor_terms is None:
        eqt["profiles_1d.rho_tor_norm"] = np.sqrt(np.clip(psi_norm, 0.0, 1.0))
    else:
        phi, rho_tor, rho_tor_norm = rho_tor_terms
        eqt["profiles_1d.phi"] = phi
        eqt["profiles_1d.rho_tor"] = rho_tor
        eqt["profiles_1d.rho_tor_norm"] = rho_tor_norm

    prof2d = eqt[f"profiles_2d.{profile_index}"]
    prof2d["grid_type.index"] = 1
    prof2d["grid.dim1"] = np.linspace(0.0, float(data["RDIM"]), nw) + float(data["RLEFT"])
    prof2d["grid.dim2"] = np.linspace(0.0, float(data["ZDIM"]), nh) - float(data["ZDIM"]) / 2.0 + float(data["ZMID"])
    prof2d["psi"] = np.asarray(data["PSIRZ"], dtype=float).reshape(nw, nh)

    if allow_derived_data and float(data["CURRENT"]) != 0.0:
        eqt["global_quantities.magnetic_axis.b_field_tor"] = float(data["BCENTR"]) * float(data["RCENTR"]) / float(data["RMAXIS"])
        eqt["global_quantities.q_axis"] = float(np.asarray(data["QPSI"])[0])
        eqt["global_quantities.q_95"] = float(np.interp(0.95, np.linspace(0.0, 1.0, nw), np.asarray(data["QPSI"], dtype=float)))
        qpsi = np.asarray(data["QPSI"], dtype=float)
        qmin_idx = int(np.argmin(np.abs(qpsi)))
        eqt["global_quantities.q_min.value"] = float(qpsi[qmin_idx])
        eqt["global_quantities.q_min.rho_tor_norm"] = float(eqt["profiles_1d.rho_tor_norm"][qmin_idx])

    if float(data["CURRENT"]) != 0.0:
        eqt["boundary.outline.r"] = np.asarray(data.get("RBBBS", []), dtype=float)
        eqt["boundary.outline.z"] = np.asarray(data.get("ZBBBS", []), dtype=float)

    if allow_derived_data:
        # Tracing every flux surface is the expensive part of this conversion,
        # so it sits behind the same flag as the other derived quantities.
        volume = _volume_profile(data, psi_norm)
        if volume is not None:
            eqt["profiles_1d.volume"] = volume
            eqt["global_quantities.volume"] = float(volume[-1])

    try:
        ods.set_time_array("equilibrium.time", time_index, eqt["time"])
    except Exception:
        ods[f"equilibrium.time.{time_index}"] = eqt["time"]

    ods["wall.description_2d.0.limiter.type.name"] = "first_wall"
    ods["wall.description_2d.0.limiter.type.index"] = 0
    ods["wall.description_2d.0.limiter.type.description"] = "first wall"
    ods["wall.description_2d.0.limiter.unit.0.outline.r"] = np.asarray(data.get("RLIM", []), dtype=float)
    ods["wall.description_2d.0.limiter.unit.0.outline.z"] = np.asarray(data.get("ZLIM", []), dtype=float)
    try:
        ods.set_time_array("wall.time", time_index, eqt["time"])
    except Exception:
        ods[f"wall.time.{time_index}"] = eqt["time"]
    eqt["constraints.ip.reconstructed"] = float(data["CURRENT"])

    namelists = getattr(item, "namelists", {})
    if namelists:
        from vaft.data.keqdsk import write_namelists_to_ods

        write_namelists_to_ods(ods, namelists, time_index=time_index)
    return ods


def from_imas(
    source: Any,
    *,
    paths: Optional[list] = None,
    time: Optional[float] = None,
    occurrence: Optional[dict] = None,
    imas_version: Optional[str] = None,
) -> GEQDSK:
    """Load equilibrium data from IMAS through OMAS and return GEQDSK."""
    from vaft.imas.omas_imas import load_omas_imas

    if isinstance(source, (str, Path)) and str(source).startswith("imas:"):
        ods = load_omas_imas(
            occurrence=occurrence or {},
            paths=paths,
            time=time,
            imas_version=imas_version,
            uri=str(source),
        )
    else:
        ods = source
    return from_omas(ods)


def to_imas(
    geqdsk: GEQDSK | Mapping[str, Any],
    target: Any,
    *,
    occurrence: Optional[dict] = None,
    imas_version: Optional[str] = None,
) -> Any:
    """Convert GEQDSK through OMAS and write to an IMAS target."""
    ods = to_omas(geqdsk)
    from vaft.imas.omas_imas import save_omas_imas

    if isinstance(target, (str, Path)) and str(target).startswith("imas:"):
        return save_omas_imas(
            ods,
            occurrence=occurrence or {},
            imas_version=imas_version,
            uri=str(target),
        )
    if isinstance(target, (str, Path)):
        return save_omas_imas(
            ods,
            occurrence=occurrence or {},
            imas_version=imas_version,
            uri="imas:hdf5?path=" + str(Path(target).expanduser()),
        )
    raise TypeError("target must be an IMAS URI or local IMAS HDF5 directory")
