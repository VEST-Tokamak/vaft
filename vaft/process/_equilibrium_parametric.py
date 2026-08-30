"""Convention-aware parametric equilibrium algorithms.

The public names in this module are re-exported by :mod:`vaft.process.equilibrium`.
No external equilibrium solver is used.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.constants import mu_0 as MU0
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from scipy.optimize import least_squares, root
from scipy.spatial import cKDTree

from vaft.data.equilibrium import (
    BoundaryRepresentation,
    Contour,
    DerivationProvenance,
    DerivedValue,
    EquilibriumConvention,
    EquilibriumData,
    Gap,
    GlobalEquilibriumDescriptors,
    MillerFitResult,
    MillerSequenceResult,
    MillerSurface,
    SolovevConstraint,
    SolovevEquilibrium,
    StationaryPoint,
    StrikePoint,
    Topology,
    ValidationIssue,
    ValidationReport,
    XPoint,
)


def _path_get(item: Any, path: str, default: Any = None) -> Any:
    try:
        return item[path]
    except Exception:
        node = item
        try:
            for part in path.split("."):
                if part.isdigit():
                    node = node[int(part)]
                elif isinstance(node, Mapping):
                    node = node[part]
                elif hasattr(node, part):
                    node = getattr(node, part)
                else:
                    node = node[part]
            return node
        except Exception:
            return default


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    return arr if arr.size else None


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(np.asarray(value, dtype=float).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    return out if np.isfinite(out) else None


def _sign(value: float | None) -> int | None:
    if value is None or not np.isfinite(value) or abs(value) < 1e-14:
        return None
    return 1 if value > 0 else -1


def _detect_convention(
    *, explicit: int | None, bt0: float | None, ip: float | None,
    q: np.ndarray | None, psi_1d: np.ndarray | None, source: str,
    clockwise_phi: bool | None = None,
) -> EquilibriumConvention:
    if explicit is not None:
        if explicit not in range(1, 19):
            raise ValueError("COCOS index must be in the range 1..18")
        return EquilibriumConvention(
            cocos=explicit, candidates=(explicit,), psi_per_radian=explicit < 10,
            clockwise_phi=clockwise_phi, ip_sign=_sign(ip), bt_sign=_sign(bt0),
            q_sign=_sign(float(np.nanmedian(q))) if q is not None else None,
            source=source,
        )
    candidates: tuple[int, ...] = ()
    if bt0 is not None and ip is not None and q is not None and psi_1d is not None:
        try:
            from omas import identify_cocos

            identified = identify_cocos(bt0, ip, q, psi_1d, clockwise_phi=clockwise_phi)
            candidates = tuple(sorted({int(value) for value in identified}))
        except Exception:
            candidates = ()
    unique = candidates[0] if len(candidates) == 1 else None
    return EquilibriumConvention(
        cocos=unique, candidates=candidates,
        psi_per_radian=(unique < 10) if unique is not None else None,
        clockwise_phi=clockwise_phi, ip_sign=_sign(ip), bt_sign=_sign(bt0),
        q_sign=_sign(float(np.nanmedian(q))) if q is not None and q.size else None,
        source=source,
    )


def _from_geqdsk(source: Any, convention: int | None) -> EquilibriumData:
    data = source.mapping if hasattr(source, "mapping") else source
    nw, nh = int(data["NW"]), int(data["NH"])
    r = np.linspace(float(data["RLEFT"]), float(data["RLEFT"]) + float(data["RDIM"]), nw)
    z = np.linspace(
        float(data["ZMID"]) - float(data["ZDIM"]) / 2,
        float(data["ZMID"]) + float(data["ZDIM"]) / 2,
        nh,
    )
    psi = np.asarray(data["PSIRZ"], dtype=float).reshape(nw, nh)
    psi_axis, psi_boundary = float(data["SIMAG"]), float(data["SIBRY"])
    psi_1d = np.linspace(psi_axis, psi_boundary, nw)
    q = _array(data.get("QPSI"))
    case = str(data.get("CASE", ""))
    match = re.search(r"COCOS\s*[=_-]?\s*(\d{1,2})", case, re.IGNORECASE)
    explicit = convention if convention is not None else (int(match.group(1)) if match else None)
    bt0, ip = _scalar(data.get("BCENTR")), _scalar(data.get("CURRENT"))
    conv = _detect_convention(
        explicit=explicit, bt0=bt0, ip=ip, q=q, psi_1d=psi_1d,
        source="argument" if convention is not None else ("GEQDSK header" if match else "GEQDSK signs"),
    )
    rb, zb = _array(data.get("RBBBS")), _array(data.get("ZBBBS"))
    rl, zl = _array(data.get("RLIM")), _array(data.get("ZLIM"))
    metadata = dict(getattr(source, "metadata", {}))
    metadata.update({"source_type": "geqdsk", "case": case})
    return EquilibriumData(
        r=r, z=z, psi=psi, psi_axis=psi_axis, psi_boundary=psi_boundary,
        magnetic_axis=(float(data["RMAXIS"]), float(data["ZMAXIS"])),
        lcfs=Contour(rb, zb, True) if rb is not None and zb is not None else None,
        limiter=Contour(rl, zl, True) if rl is not None and zl is not None else None,
        psi_1d=psi_1d, pressure=_array(data.get("PRES")), f=_array(data.get("FPOL")),
        q=q, ip=ip, bt0=bt0, r0=_scalar(data.get("RCENTR")), convention=conv,
        metadata=metadata,
    )


def _from_ods(source: Any, time_index: int, profile_index: int, convention: int | None) -> EquilibriumData:
    ts = _path_get(source, f"equilibrium.time_slice.{time_index}")
    if ts is None:
        raise ValueError(f"equilibrium.time_slice.{time_index} is unavailable")
    prof2d = _path_get(ts, f"profiles_2d.{profile_index}")
    r = _array(_path_get(prof2d, "grid.dim1")) if prof2d is not None else None
    z = _array(_path_get(prof2d, "grid.dim2")) if prof2d is not None else None
    psi = None if prof2d is None else _path_get(prof2d, "psi")
    if psi is not None:
        psi = np.asarray(psi, dtype=float)
    psi_1d = _array(_path_get(ts, "profiles_1d.psi"))
    q = _array(_path_get(ts, "profiles_1d.q"))
    bt0 = _scalar(_path_get(source, f"equilibrium.vacuum_toroidal_field.b0.{time_index}"))
    if bt0 is None:
        bt0 = _scalar(_path_get(source, "equilibrium.vacuum_toroidal_field.b0"))
    ip = _scalar(_path_get(ts, "global_quantities.ip"))
    explicit_meta = _scalar(_path_get(source, "equilibrium.ids_properties.cocos"))
    explicit = convention if convention is not None else (int(explicit_meta) if explicit_meta else None)
    conv = _detect_convention(
        explicit=explicit, bt0=bt0, ip=ip, q=q, psi_1d=psi_1d,
        source="argument" if convention is not None else ("ODS metadata" if explicit else "ODS signs"),
    )
    rb, zb = _array(_path_get(ts, "boundary.outline.r")), _array(_path_get(ts, "boundary.outline.z"))
    rl = _array(_path_get(source, "wall.description_2d.0.limiter.unit.0.outline.r"))
    zl = _array(_path_get(source, "wall.description_2d.0.limiter.unit.0.outline.z"))
    axis_r = _scalar(_path_get(ts, "global_quantities.magnetic_axis.r"))
    axis_z = _scalar(_path_get(ts, "global_quantities.magnetic_axis.z"))
    return EquilibriumData(
        r=r, z=z, psi=psi,
        psi_axis=_scalar(_path_get(ts, "global_quantities.psi_axis")),
        psi_boundary=_scalar(_path_get(ts, "global_quantities.psi_boundary")),
        magnetic_axis=(axis_r, axis_z) if axis_r is not None and axis_z is not None else None,
        lcfs=Contour(rb, zb, True) if rb is not None and zb is not None else None,
        limiter=Contour(rl, zl, True) if rl is not None and zl is not None else None,
        psi_1d=psi_1d, pressure=_array(_path_get(ts, "profiles_1d.pressure")),
        f=_array(_path_get(ts, "profiles_1d.f")), q=q, ip=ip, bt0=bt0,
        r0=_scalar(_path_get(source, "equilibrium.vacuum_toroidal_field.r0")),
        time=_scalar(_path_get(ts, "time")), convention=conv,
        metadata={"source_type": "omas", "time_index": time_index, "profile_index": profile_index},
    )


def as_equilibrium(
    source: Any, *, time_index: int = 0, profile_index: int = 0,
    convention: int | None = None,
) -> EquilibriumData:
    """Adapt a GEQDSK, ODS, IMAS handle, or native model to ``EquilibriumData``."""
    if isinstance(source, EquilibriumData):
        return convert_cocos(source, convention) if convention is not None else source
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        from vaft.data.eqdsk import read_geqdsk

        source = read_geqdsk(source)
    if hasattr(source, "to_omas") and not hasattr(source, "mapping"):
        source = source.to_omas()
    # A native equilibrium IDS is already the authoritative root object but,
    # unlike ODS, it does not contain an outer ``equilibrium`` mapping.  Wrap
    # that root and let the same path adapter traverse native attributes/AoS.
    metadata = getattr(source, "metadata", None)
    if getattr(metadata, "name", None) == "equilibrium":
        source = {"equilibrium": source}
    mapping = source.mapping if hasattr(source, "mapping") else source
    try:
        keys = set(mapping.keys())
    except Exception:
        keys = set()
    if {"NW", "NH", "PSIRZ"}.issubset(keys):
        return _from_geqdsk(source, convention)
    return _from_ods(source, time_index, profile_index, convention)


def validate_equilibrium(equilibrium: EquilibriumData, *, required_for: str = "general") -> ValidationReport:
    issues: list[ValidationIssue] = []
    if equilibrium.psi is not None and equilibrium.r is not None and equilibrium.z is not None:
        expected = (equilibrium.r.size, equilibrium.z.size)
        if equilibrium.psi.shape != expected:
            issues.append(ValidationIssue("error", "psi_shape", "psi", f"psi shape {equilibrium.psi.shape} must be {expected}"))
    elif required_for in {"global", "miller", "edge", "solovev"}:
        issues.append(ValidationIssue("error", "missing_grid", "psi", "R, Z, and 2-D psi are required"))
    if equilibrium.psi_axis is None or equilibrium.psi_boundary is None:
        issues.append(ValidationIssue("error", "missing_flux_bounds", "psi_axis", "psi_axis and psi_boundary are required"))
    elif equilibrium.psi_axis == equilibrium.psi_boundary:
        issues.append(ValidationIssue("error", "degenerate_flux", "psi_boundary", "psi_boundary must differ from psi_axis"))
    if equilibrium.lcfs is None or equilibrium.lcfs.r.size < 3:
        issues.append(ValidationIssue("error" if required_for in {"global", "miller", "edge"} else "warning", "missing_lcfs", "lcfs", "a closed LCFS with at least three points is required"))
    if equilibrium.convention.ambiguous:
        candidates = equilibrium.convention.candidates or ("none",)
        issues.append(ValidationIssue("warning", "ambiguous_cocos", "convention", f"COCOS is ambiguous ({candidates}); pass convention= explicitly before conversion"))
    return ValidationReport(tuple(issues))


def convert_cocos(equilibrium: EquilibriumData, target_cocos: int) -> EquilibriumData:
    """Return a copy converted between explicitly known COCOS conventions."""
    if target_cocos not in range(1, 19):
        raise ValueError("target_cocos must be in the range 1..18")
    source_cocos = equilibrium.convention.cocos
    if source_cocos is None and len(equilibrium.convention.candidates) == 1:
        source_cocos = equilibrium.convention.candidates[0]
    if source_cocos is None:
        raise ValueError("source COCOS is ambiguous; provide an explicit convention when adapting the equilibrium")
    if source_cocos == target_cocos:
        return equilibrium
    from omas import cocos_transform

    factors = cocos_transform(source_cocos, target_cocos)
    scale = lambda value, key: None if value is None else np.asarray(value) * factors[key]
    axis = equilibrium.magnetic_axis
    convention = EquilibriumConvention(
        cocos=target_cocos, candidates=(target_cocos,), psi_per_radian=target_cocos < 10,
        clockwise_phi=equilibrium.convention.clockwise_phi,
        ip_sign=_sign(None if equilibrium.ip is None else equilibrium.ip * factors["IP"]),
        bt_sign=_sign(None if equilibrium.bt0 is None else equilibrium.bt0 * factors["BT"]),
        q_sign=_sign(None if equilibrium.q is None else float(np.nanmedian(equilibrium.q * factors["Q"]))),
        source=f"converted from COCOS {source_cocos}",
    )
    return replace(
        equilibrium, psi=scale(equilibrium.psi, "PSI"),
        psi_axis=None if equilibrium.psi_axis is None else float(equilibrium.psi_axis * factors["PSI"]),
        psi_boundary=None if equilibrium.psi_boundary is None else float(equilibrium.psi_boundary * factors["PSI"]),
        psi_1d=scale(equilibrium.psi_1d, "PSI"), f=scale(equilibrium.f, "F"),
        q=scale(equilibrium.q, "Q"),
        ip=None if equilibrium.ip is None else float(equilibrium.ip * factors["IP"]),
        bt0=None if equilibrium.bt0 is None else float(equilibrium.bt0 * factors["BT"]),
        magnetic_axis=axis, convention=convention,
    )


def _provenance(eq: EquilibriumData, method: str, fields: Sequence[str] = (), **kwargs: Any) -> DerivationProvenance:
    return DerivationProvenance(
        method=method, source_type=str(eq.metadata.get("source_type", "native")),
        source_fields=tuple(fields), source_time=eq.time, convention=eq.convention, **kwargs,
    )


def _derived(eq: EquilibriumData, value: Any, unit: str, definition: str, method: str, fields: Sequence[str] = (), quality: Mapping[str, Any] | None = None) -> DerivedValue:
    return DerivedValue(value, unit, definition, _provenance(eq, method, fields), quality or {})


def _unavailable(eq: EquilibriumData, unit: str, definition: str, reason: str, fields: Sequence[str] = ()) -> DerivedValue:
    return DerivedValue(None, unit, definition, _provenance(eq, "unavailable", fields), reason=reason)


def _closed_points(contour: Contour) -> tuple[np.ndarray, np.ndarray]:
    r, z = contour.r, contour.z
    if r.size and (r[0] != r[-1] or z[0] != z[-1]):
        r, z = np.r_[r, r[0]], np.r_[z, z[0]]
    return r, z


def _polygon_geometry(contour: Contour) -> dict[str, float]:
    r, z = _closed_points(contour)
    cross = r[:-1] * z[1:] - r[1:] * z[:-1]
    signed_area = 0.5 * np.sum(cross)
    if abs(signed_area) < 1e-15:
        raise ValueError("LCFS polygon area is zero")
    cr = np.sum((r[:-1] + r[1:]) * cross) / (6 * signed_area)
    cz = np.sum((z[:-1] + z[1:]) * cross) / (6 * signed_area)
    dl = np.hypot(np.diff(r), np.diff(z))
    surface = np.sum(2 * np.pi * 0.5 * (r[:-1] + r[1:]) * dl)
    volume = 2 * np.pi * abs(signed_area) * cr
    return {"area": abs(signed_area), "r": float(cr), "z": float(cz), "surface": float(surface), "volume": float(volume)}


def derive_radial_coordinates(equilibrium: Any) -> Mapping[str, DerivedValue]:
    eq = as_equilibrium(equilibrium)
    if eq.psi_1d is None or eq.psi_axis is None or eq.psi_boundary is None or eq.psi_axis == eq.psi_boundary:
        reason = "psi_1d and distinct axis/boundary flux values are required"
        return {name: _unavailable(eq, "1", name, reason) for name in ("psi_n", "rho_pol_n", "rho_tor_n")}
    psi_n = (eq.psi_1d - eq.psi_axis) / (eq.psi_boundary - eq.psi_axis)
    result = {
        "psi_n": _derived(eq, psi_n, "1", "(psi-psi_axis)/(psi_boundary-psi_axis)", "direct normalization", ("psi_1d", "psi_axis", "psi_boundary")),
        "rho_pol_n": _derived(eq, np.sqrt(np.clip(psi_n, 0, None)), "1", "sqrt(psi_n)", "direct normalization", ("psi_1d",)),
    }
    if eq.q is None or eq.q.size != psi_n.size or not np.all(np.isfinite(eq.q)):
        result["rho_tor_n"] = _unavailable(eq, "1", "sqrt(Phi/Phi_boundary), Phi=int(q dpsi)", "finite q on the psi_1d grid is required", ("q", "psi_1d"))
        return result
    order = np.argsort(psi_n)
    x, q = psi_n[order], eq.q[order]
    if np.any(np.diff(x) <= 0):
        result["rho_tor_n"] = _unavailable(eq, "1", "sqrt(Phi/Phi_boundary), Phi=int(q dpsi)", "psi_n must be strictly monotonic", ("q", "psi_1d"))
        return result
    cumulative = np.r_[0.0, np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(x))]
    if cumulative[-1] == 0 or np.any(cumulative / cumulative[-1] < -1e-12) or np.any(np.diff(cumulative / cumulative[-1]) < -1e-10):
        result["rho_tor_n"] = _unavailable(eq, "1", "sqrt(Phi/Phi_boundary), Phi=int(q dpsi)", "q dpsi does not produce a monotonic toroidal-flux coordinate", ("q", "psi_1d"))
        return result
    rho_sorted = np.sqrt(np.clip(cumulative / cumulative[-1], 0, 1))
    rho = np.empty_like(rho_sorted); rho[order] = rho_sorted
    result["rho_tor_n"] = _derived(eq, rho, "1", "sqrt(int_axis^psi q dpsi / int_axis^boundary q dpsi)", "trapezoidal integration", ("q", "psi_1d"))
    return result


def _psi_per_radian_factor(eq: EquilibriumData) -> float:
    """Multiplicative factor turning the stored psi into Wb/rad.

    COCOS 1-8 store psi per radian (B_pol = grad(psi)/R directly); COCOS 11-18
    and IMAS store the full poloidal flux in weber, so field construction needs
    the extra 1/(2*pi). An ambiguous convention keeps the historical per-radian
    assumption (factor 1) rather than guessing.
    """
    per_radian = eq.convention.psi_per_radian
    if per_radian is None and eq.convention.cocos is not None:
        per_radian = eq.convention.cocos < 10
    return 1.0 if per_radian in (None, True) else 1.0 / (2.0 * np.pi)


def _grid_fields(eq: EquilibriumData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if eq.r is None or eq.z is None or eq.psi is None or eq.psi.shape != (eq.r.size, eq.z.size):
        raise ValueError("a correctly shaped R/Z/psi grid is required")
    factor = _psi_per_radian_factor(eq)
    dpsi_dr, dpsi_dz = np.gradient(eq.psi * factor, eq.r, eq.z, edge_order=2)
    rm, zm = np.meshgrid(eq.r, eq.z, indexing="ij")
    br = -dpsi_dz / rm
    bz = dpsi_dr / rm
    return rm, zm, br, bz, np.hypot(br, bz)


def derive_global_descriptors(
    equilibrium: Any, *, rational_q: Sequence[float] = (1.0, 1.5, 2.0, 3.0),
) -> GlobalEquilibriumDescriptors:
    eq = as_equilibrium(equilibrium)
    validation = validate_equilibrium(eq, required_for="global")
    values: dict[str, DerivedValue] = {}
    direct = {
        "ip": (eq.ip, "A", "source plasma current", ("ip",)),
        "bt0": (eq.bt0, "T", "source vacuum toroidal field at r0", ("bt0", "r0")),
    }
    for name, (value, unit, definition, fields) in direct.items():
        values[name] = _derived(eq, value, unit, definition, "source", fields) if value is not None else _unavailable(eq, unit, definition, f"{fields[0]} is unavailable", fields)
    geometry = None
    if eq.lcfs is not None and eq.lcfs.r.size >= 3:
        try:
            geometry = _polygon_geometry(eq.lcfs)
            rin, rout = float(np.min(eq.lcfs.r)), float(np.max(eq.lcfs.r))
            zbot, ztop = float(np.min(eq.lcfs.z)), float(np.max(eq.lcfs.z))
            minor = 0.5 * (rout - rin)
            # Shape descriptors use the conventional geometric centre
            # (R_out+R_in)/2, which is what IMAS boundary.geometric_axis and
            # boundary.triangularity report.  The area centroid is a different
            # quantity: it is what Pappus's theorem needs for the volume, and it
            # can sit centimetres away at low aspect ratio.
            rgeo = 0.5 * (rout + rin)
            zgeo = 0.5 * (ztop + zbot)
            zmin_i, zmax_i = int(np.argmin(eq.lcfs.z)), int(np.argmax(eq.lcfs.z))
            kappa = (ztop - zbot) / (2 * minor)
            tri_upper = (rgeo - float(eq.lcfs.r[zmax_i])) / minor
            tri_lower = (rgeo - float(eq.lcfs.r[zmin_i])) / minor
            geo_values = {
                "major_radius": (rgeo, "m", "(R_out+R_in)/2 at the LCFS"),
                "minor_radius": (minor, "m", "(R_out-R_in)/2 at the LCFS"),
                "inverse_aspect_ratio": (minor / rgeo, "1", "minor_radius/major_radius"),
                "elongation": (kappa, "1", "(Z_max-Z_min)/(2a)"),
                "triangularity_upper": (tri_upper, "1", "(R_geo-R_at_Zmax)/a"),
                "triangularity_lower": (tri_lower, "1", "(R_geo-R_at_Zmin)/a"),
                "volume": (geometry["volume"], "m^3", "2*pi*LCFS_area*area_centroid_R"),
                "surface_area": (geometry["surface"], "m^2", "integral_LCFS 2*pi*R dl"),
                "geometric_center_r": (rgeo, "m", "(R_out+R_in)/2 at the LCFS"),
                "geometric_center_z": (zgeo, "m", "(Z_max+Z_min)/2 at the LCFS"),
                "area_centroid_r": (geometry["r"], "m", "LCFS area centroid R (the volume's Pappus radius)"),
                "area_centroid_z": (geometry["z"], "m", "LCFS area centroid Z"),
                "cross_section_area": (geometry["area"], "m^2", "LCFS poloidal cross-section area"),
            }
            for name, (value, unit, definition) in geo_values.items():
                values[name] = _derived(eq, value, unit, definition, "closed-polygon geometry", ("lcfs",))
            if eq.magnetic_axis is not None:
                values["magnetic_axis_r"] = _derived(eq, eq.magnetic_axis[0], "m", "source magnetic-axis R", "source", ("magnetic_axis",))
                values["magnetic_axis_z"] = _derived(eq, eq.magnetic_axis[1], "m", "source magnetic-axis Z", "source", ("magnetic_axis",))
                values["shafranov_shift"] = _derived(eq, eq.magnetic_axis[0] - rgeo, "m", "R_axis-R_geo", "difference", ("magnetic_axis", "lcfs"))
        except ValueError as exc:
            geometry = None
            validation = ValidationReport(validation.issues + (ValidationIssue("error", "invalid_lcfs", "lcfs", str(exc)),))
    for name, unit, definition in (
        ("major_radius", "m", "(R_out+R_in)/2"), ("minor_radius", "m", "(R_out-R_in)/2"),
        ("inverse_aspect_ratio", "1", "a/R"), ("elongation", "1", "LCFS elongation"),
        ("triangularity_upper", "1", "upper LCFS triangularity"), ("triangularity_lower", "1", "lower LCFS triangularity"),
        ("volume", "m^3", "axisymmetric LCFS volume"), ("surface_area", "m^2", "axisymmetric LCFS surface area"),
        ("geometric_center_r", "m", "(R_out+R_in)/2"), ("geometric_center_z", "m", "(Z_max+Z_min)/2"),
        ("area_centroid_r", "m", "LCFS area centroid R"), ("area_centroid_z", "m", "LCFS area centroid Z"),
        ("cross_section_area", "m^2", "LCFS poloidal cross-section area"),
        ("magnetic_axis_r", "m", "magnetic-axis R"), ("magnetic_axis_z", "m", "magnetic-axis Z"),
        ("shafranov_shift", "m", "R_axis-R_geo"),
    ):
        values.setdefault(name, _unavailable(eq, unit, definition, "a valid LCFS and magnetic axis are required", ("lcfs", "magnetic_axis")))

    pressure_integral = None
    average_pressure = None
    bpa = None
    virial: dict[str, float] = {}
    if geometry is not None and eq.pressure is not None and eq.psi_1d is not None:
        try:
            rm, zm, br, bz, _ = _grid_fields(eq)
            psi_n_grid = (eq.psi - eq.psi_axis) / (eq.psi_boundary - eq.psi_axis)
            psi_n_1d = (eq.psi_1d - eq.psi_axis) / (eq.psi_boundary - eq.psi_axis)
            order = np.argsort(psi_n_1d)
            p_grid = np.interp(psi_n_grid, psi_n_1d[order], eq.pressure[order], left=eq.pressure[order][0], right=eq.pressure[order][-1])
            mask = MplPath(eq.lcfs.points).contains_points(np.column_stack((rm.ravel(), zm.ravel()))).reshape(rm.shape)
            dr = np.gradient(eq.r)[:, None]; dz = np.gradient(eq.z)[None, :]
            dv = 2 * np.pi * rm * dr * dz * mask
            pressure_integral = float(np.nansum(p_grid * dv))
            volume_grid = float(np.nansum(dv))
            average_pressure = pressure_integral / volume_grid
            values["pressure_integral"] = _derived(eq, pressure_integral, "J", "integral_plasma p dV", "grid quadrature", ("pressure", "psi", "lcfs"), {"grid_volume": volume_grid})
            values["thermal_energy"] = _derived(eq, 1.5 * pressure_integral, "J", "(3/2)*integral_plasma p dV", "grid quadrature", ("pressure", "psi", "lcfs"))
            from vaft.process.equilibrium import calculate_average_boundary_poloidal_field, efit_virial_volume_integrals, poloidal_field_at_boundary, shafranov_integrals
            rb, zb = _closed_points(eq.lcfs)
            bp_boundary, _, _ = poloidal_field_at_boundary(
                eq.r, eq.z, eq.psi * _psi_per_radian_factor(eq), rb, zb
            )
            bpa = float(calculate_average_boundary_poloidal_field(rb, zb, bp_boundary))
            s1, s2, s3, alpha = shafranov_integrals(rb, zb, bp_boundary, rm, zm, br, bz, R_0=geometry["r"], Z_0=geometry["z"], B_ref=bpa, volume=geometry["volume"])
            virial.update(s1=float(s1), s2=float(s2), s3=float(s3), alpha=float(alpha))
            if eq.f is not None and eq.f.size == eq.psi_1d.size and eq.bt0 is not None and eq.r0 is not None:
                f_grid = np.interp(psi_n_grid, psi_n_1d[order], eq.f[order])
                details = efit_virial_volume_integrals(rm, zm, rb, zb, br, bz, p_tot_grid=p_grid, B_phi_grid=f_grid/rm, B_phi_vac_grid=eq.bt0*eq.r0/rm, F_grid=f_grid, F_boundary=float(eq.f[order][-1]))
                virial["rt"] = details["rt"]
                from vaft.process.equilibrium import computed_diamagnetism_from_phi
                virial["mui"] = computed_diamagnetism_from_phi(details["phi_dia_comp"], eq.bt0, eq.r0, details["volume"], bpa)
                from vaft.formula.equilibrium import virial_li_from_S_alpha_rt
                virial["li"] = virial_li_from_S_alpha_rt(s1, s2, s3, alpha, details["rt"] / geometry["r"])
        except Exception as exc:
            validation = ValidationReport(validation.issues + (ValidationIssue("warning", "pressure_integration", "pressure", str(exc)),))
    for name, unit, definition in (("pressure_integral", "J", "integral p dV"), ("thermal_energy", "J", "3/2 integral p dV")):
        values.setdefault(name, _unavailable(eq, unit, definition, "pressure, psi profile/grid, and LCFS are required"))
    if average_pressure is not None and eq.bt0 not in (None, 0):
        beta_t = 2 * MU0 * average_pressure / eq.bt0**2
        values["beta_t"] = _derived(eq, beta_t, "1", "2*mu0*<p>/Bt0^2", "volume and pressure integration", ("pressure", "bt0"))
        if geometry and eq.ip not in (None, 0):
            beta_n = 100 * beta_t * values["minor_radius"].value * abs(eq.bt0) / (abs(eq.ip) / 1e6)
            values["beta_n"] = _derived(eq, beta_n, "% m T / MA", "100*beta_t*a*abs(Bt0)/abs(Ip_MA)", "derived", ("beta_t", "minor_radius", "bt0", "ip"))
    if average_pressure is not None and bpa not in (None, 0):
        values["beta_p_boundary_average"] = _derived(eq, 2 * MU0 * average_pressure / bpa**2, "1", "2*mu0*<p>/<Bp>_boundary^2", "boundary-field average", ("pressure", "psi", "lcfs"))
    for name, definition in (("beta_t", "2*mu0*<p>/Bt0^2"), ("beta_n", "100*beta_t*a*abs(Bt0)/abs(Ip_MA)"), ("beta_p_boundary_average", "2*mu0*<p>/<Bp>_boundary^2")):
        values.setdefault(name, _unavailable(eq, "1" if name != "beta_n" else "% m T / MA", definition, "required pressure, geometry, current, or magnetic field is unavailable"))
    for name in ("s1", "s2", "s3", "alpha"):
        values[name] = _derived(eq, virial[name], "1", f"Shafranov boundary integral {name}", "shafranov_integrals", ("psi", "lcfs")) if name in virial else _unavailable(eq, "1", f"Shafranov {name}", "valid grid, LCFS, and boundary poloidal field are required")
    values["li_virial"] = _derived(eq, virial["li"], "1", "[S1/2+S2/2*(1-RT/R0)-S3]/(alpha-1)", "Lao virial closure", ("psi", "f", "pressure", "lcfs")) if "li" in virial else _unavailable(eq, "1", "Lao virial internal inductance", "F, pressure, field grid, and a well-conditioned alpha are required")

    radial = derive_radial_coordinates(eq)
    rational: dict[float, tuple[DerivedValue, ...]] = {}
    if eq.q is not None and "psi_n" in radial and radial["psi_n"].available and eq.q.size == np.asarray(radial["psi_n"].value).size:
        x = np.asarray(radial["psi_n"].value)
        values["q0"] = _derived(eq, float(eq.q[np.argmin(x)]), "1", "q at psi_n=0", "profile endpoint", ("q",))
        values["q95"] = _derived(eq, float(np.interp(0.95, x, eq.q)), "1", "q interpolated at psi_n=0.95", "linear interpolation", ("q", "psi_1d"))
        values["q_edge"] = _derived(eq, float(eq.q[np.argmax(x)]), "1", "q at psi_n=1", "profile endpoint", ("q",))
        rho = np.sqrt(np.clip(x, 0, None))
        shear = np.divide(rho * np.gradient(eq.q, rho, edge_order=1), eq.q, out=np.full_like(eq.q, np.nan), where=eq.q != 0)
        values["magnetic_shear"] = _derived(eq, shear, "1", "(rho_pol_n/q)*dq/drho_pol_n", "finite differences", ("q", "psi_1d"))
        for target in rational_q:
            crossings = []
            crossing_locations: list[float] = []
            for i in np.where((eq.q[:-1] - target) * (eq.q[1:] - target) <= 0)[0]:
                if eq.q[i + 1] == eq.q[i]:
                    xn = float(x[i])
                else:
                    xn = float(x[i] + (target - eq.q[i]) * (x[i + 1] - x[i]) / (eq.q[i + 1] - eq.q[i]))
                if not any(abs(xn-previous) <= 1e-10 for previous in crossing_locations):
                    crossing_locations.append(xn)
                    crossings.append(_derived(eq, xn, "1", f"psi_n where q={target:g}", "piecewise-linear root", ("q", "psi_1d")))
            rational[float(target)] = tuple(crossings)
    else:
        for name in ("q0", "q95", "q_edge", "magnetic_shear"):
            values[name] = _unavailable(eq, "1", name, "q and psi_1d profiles with equal length are required")
        rational = {float(value): () for value in rational_q}
    return GlobalEquilibriumDescriptors(values, radial, rational, validation)


def evaluate_miller(surface: MillerSurface, theta: Any) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    if surface.r <= 0 or surface.kappa <= 0 or abs(surface.delta) >= 1:
        raise ValueError("Miller geometry requires r>0, kappa>0, and abs(delta)<1")
    angle = theta + np.arcsin(surface.delta) * np.sin(theta)
    return surface.r0 + surface.r * np.cos(angle), surface.z0 + surface.kappa * surface.r * np.sin(theta)


def _resample_contour(contour: Contour, count: int = 256) -> Contour:
    r, z = _closed_points(contour)
    length = np.r_[0.0, np.cumsum(np.hypot(np.diff(r), np.diff(z)))]
    if length[-1] <= 0:
        raise ValueError("contour has zero arc length")
    target = np.linspace(0, length[-1], count, endpoint=False)
    return Contour(np.interp(target, length, r), np.interp(target, length, z), True)


def fit_miller_surface(
    contour: Contour | tuple[Any, Any], *, radial_value: float | None = None,
    radial_coordinate: str = "psi_n", max_normalized_rms: float = 0.02,
    near_xpoint: bool = False,
) -> MillerFitResult:
    if not isinstance(contour, Contour):
        contour = Contour(contour[0], contour[1], True)
    observed = _resample_contour(contour)
    rmin, rmax = float(np.min(observed.r)), float(np.max(observed.r))
    zmin, zmax = float(np.min(observed.z)), float(np.max(observed.z))
    minor = 0.5 * (rmax - rmin)
    if minor <= 0:
        raise ValueError("contour must have nonzero radial extent")
    initial = np.array([0.5 * (rmin + rmax), 0.5 * (zmin + zmax), minor, (zmax-zmin)/(2*minor), 0.0])
    theta = np.linspace(0, 2*np.pi, observed.r.size, endpoint=False)
    observed_points = observed.points

    def model(params: np.ndarray) -> np.ndarray:
        s = MillerSurface(params[2], params[0], params[1], params[3], params[4])
        rr, zz = evaluate_miller(s, theta)
        return np.column_stack((rr, zz))

    def nearest_model_points(params: np.ndarray) -> np.ndarray:
        dense_theta = np.linspace(0, 2*np.pi, 8*observed.r.size, endpoint=False)
        surface = MillerSurface(params[2], params[0], params[1], params[3], params[4])
        dense_points = np.column_stack(evaluate_miller(surface, dense_theta))
        nearest = cKDTree(dense_points).query(observed_points)[1]
        local_theta = dense_theta[nearest]
        asin_delta = np.arcsin(params[4])
        for _ in range(5):
            angle = local_theta + asin_delta*np.sin(local_theta)
            angle_prime = 1 + asin_delta*np.cos(local_theta)
            angle_second = -asin_delta*np.sin(local_theta)
            points = np.column_stack((params[0]+params[2]*np.cos(angle), params[1]+params[3]*params[2]*np.sin(local_theta)))
            first = np.column_stack((-params[2]*np.sin(angle)*angle_prime, params[3]*params[2]*np.cos(local_theta)))
            second = np.column_stack((-params[2]*(np.cos(angle)*angle_prime**2+np.sin(angle)*angle_second), -params[3]*params[2]*np.sin(local_theta)))
            delta_points = points-observed_points
            numerator = np.sum(delta_points*first, axis=1)
            denominator = np.sum(first*first+delta_points*second, axis=1)
            step = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=np.abs(denominator)>1e-14)
            local_theta = np.mod(local_theta-step, 2*np.pi)
        return np.column_stack(evaluate_miller(surface, local_theta))

    def residual(params: np.ndarray) -> np.ndarray:
        return (nearest_model_points(params)-observed_points).reshape(-1)

    span_r = max(rmax-rmin, 1e-9); span_z = max(zmax-zmin, 1e-9)
    result = least_squares(
        residual, initial,
        bounds=([rmin-span_r, zmin-span_z, 1e-9, 0.05, -0.999], [rmax+span_r, zmax+span_z, 2*span_r, 10.0, 0.999]),
        xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=1000,
    )
    nearest_points = nearest_model_points(result.x)
    fitted = MillerSurface(float(result.x[2]), float(result.x[0]), float(result.x[1]), float(result.x[3]), float(result.x[4]), radial_value, radial_coordinate)
    predicted = Contour(nearest_points[:, 0], nearest_points[:, 1], True)
    d1 = np.linalg.norm(nearest_points-observed_points, axis=1)
    d2 = cKDTree(observed_points).query(predicted.points)[0]
    distances = np.r_[d1, d2]
    rms = float(np.sqrt(np.mean(distances**2))); nrms = rms / fitted.r
    hausdorff = float(max(np.max(d1), np.max(d2))); maximum = float(np.max(distances))
    reason = None
    if radial_value is not None and radial_coordinate == "psi_n" and radial_value >= 0.995:
        reason = "surface is at psi_n>=0.995, where X-point/separatrix geometry is not locally Miller-like"
    elif near_xpoint:
        reason = "surface lies within 0.05 minor radii of an X-point"
    elif nrms > max_normalized_rms:
        reason = f"normalized RMS {nrms:.4g} exceeds {max_normalized_rms:.4g}"
    elif not result.success:
        reason = result.message
    provenance = DerivationProvenance(
        "bounded symmetric-contour least squares", radial_coordinate=radial_coordinate,
        fit_range=(radial_value, radial_value) if radial_value is not None else None,
        tolerances={"max_normalized_rms": max_normalized_rms},
    )
    return MillerFitResult(fitted, observed, predicted, rms, nrms, maximum, hausdorff, bool(result.success), reason is None, reason, provenance)


def _contour_at_level(eq: EquilibriumData, level: float) -> Contour | None:
    from vaft.process.equilibrium import extract_flux_surface_contours

    raw = extract_flux_surface_contours(eq.psi, eq.r, eq.z, eq.psi_axis, eq.psi_boundary, [level]).get(float(level), [])
    if not raw:
        return None
    axis = eq.magnetic_axis
    candidates = []
    for r, z in raw:
        contour = Contour(r, z, bool(np.hypot(r[0]-r[-1], z[0]-z[-1]) < 3*max(np.mean(np.diff(eq.r)), np.mean(np.diff(eq.z)))))
        contains = bool(axis and MplPath(contour.points).contains_point(axis))
        candidates.append((contains, contour.r.size, contour))
    return max(candidates, key=lambda value: (value[0], value[1]))[2]


def fit_miller_sequence(
    equilibrium: Any, levels: Sequence[float], *, radial_coordinate: str = "psi_n",
    max_normalized_rms: float = 0.02,
) -> MillerSequenceResult:
    eq = as_equilibrium(equilibrium)
    if radial_coordinate != "psi_n":
        radial = derive_radial_coordinates(eq).get(radial_coordinate)
        if radial is None or not radial.available:
            raise ValueError(f"{radial_coordinate} is unavailable")
        psi_n_grid = np.asarray(derive_radial_coordinates(eq)["psi_n"].value)
        coord_grid = np.asarray(radial.value)
        psi_levels = np.interp(levels, coord_grid, psi_n_grid)
    else:
        psi_levels = np.asarray(levels, dtype=float)
    boundary = derive_boundary_representation(eq)
    fits: list[MillerFitResult] = []
    for requested, psi_level in zip(levels, psi_levels):
        contour = _contour_at_level(eq, float(psi_level))
        if contour is None or not contour.closed:
            continue
        # Only a boundary-relevant X-point breaks the local Miller symmetry;
        # rejected saddles elsewhere in the domain must not veto a good fit.
        near = any(
            np.min(np.hypot(contour.r-x.r, contour.z-x.z)) <= 0.05 * (np.ptp(contour.r)/2)
            for x in boundary.x_points if x.active
        )
        fits.append(fit_miller_surface(contour, radial_value=float(requested), radial_coordinate=radial_coordinate, max_normalized_rms=max_normalized_rms, near_xpoint=near))
    valid = [item for item in fits if item.accepted]
    reason = None
    if len(valid) < 4:
        reason = "at least four accepted surfaces are required for radial derivatives"
    else:
        radii = np.array([item.surface.r for item in valid])
        order = np.argsort(radii); radii = radii[order]
        if np.any(np.diff(radii) <= 0):
            reason = "fitted minor radii must be strictly increasing"
        else:
            def derivative(values: Sequence[float]) -> np.ndarray:
                return UnivariateSpline(radii, np.asarray(values)[order], k=min(3, len(radii)-1), s=0).derivative()(radii)
            dr0 = derivative([item.surface.r0 for item in valid])
            dk = derivative([item.surface.kappa for item in valid])
            dd = derivative([item.surface.delta for item in valid])
            q_values = None; shear = None; alpha_values = None
            if eq.q is not None and eq.psi_1d is not None and eq.psi_axis is not None and eq.psi_boundary is not None:
                psi_prof = (eq.psi_1d-eq.psi_axis)/(eq.psi_boundary-eq.psi_axis)
                # Apply `order` so q/p (and hence shear/alpha) stay aligned with
                # the radius-sorted arrays fed to the splines below.
                valid_levels = np.array([item.surface.radial_value for item in valid])[order]
                psi_valid = valid_levels if radial_coordinate == "psi_n" else np.interp(valid_levels, np.asarray(derive_radial_coordinates(eq)[radial_coordinate].value), psi_prof)
                q_values = np.interp(psi_valid, psi_prof, eq.q)
                dqdr = UnivariateSpline(radii, q_values, k=min(3, len(radii)-1), s=0).derivative()(radii)
                shear = radii / q_values * dqdr
                if eq.pressure is not None and eq.bt0 not in (None, 0):
                    p_values = np.interp(psi_valid, psi_prof, eq.pressure)
                    dpdr = UnivariateSpline(radii, p_values, k=min(3, len(radii)-1), s=0).derivative()(radii)
                    r0_values = np.array([item.surface.r0 for item in valid])[order]
                    alpha_values = -2*MU0*q_values**2*r0_values*dpdr/eq.bt0**2
            updates = {}
            for position, original_index in enumerate(order):
                item = valid[original_index]
                surface = replace(item.surface, d_r0_dr=float(dr0[position]), d_kappa_dr=float(dk[position]), d_delta_dr=float(dd[position]), q=None if q_values is None else float(q_values[position]), magnetic_shear=None if shear is None else float(shear[position]), alpha=None if alpha_values is None else float(alpha_values[position]))
                updates[id(item)] = replace(item, surface=surface)
            fits = [updates.get(id(item), item) for item in fits]
    provenance = _provenance(eq, "flux-contour extraction plus Miller least squares", ("psi", "lcfs"), radial_coordinate=radial_coordinate, fit_range=(float(min(levels)), float(max(levels))))
    return MillerSequenceResult(tuple(fits), reason, provenance)


def _solovev_components(model: SolovevEquilibrium, r: Any, z: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float); z = np.asarray(z, dtype=float)
    if np.any(r <= 0):
        raise ValueError("Solovev evaluation requires R>0")
    log = np.log(r/model.rref)
    basis = np.array([np.ones_like(r), r**2, r**2*log-z**2, r**4-4*r**2*z**2, r**6-12*r**4*z**2+8*r**2*z**4])
    dr = np.array([np.zeros_like(r), 2*r, r*(2*log+1), 4*r**3-8*r*z**2, 6*r**5-48*r**3*z**2+16*r*z**4])
    dz = np.array([np.zeros_like(r), np.zeros_like(r), -2*z, -8*r**2*z, -24*r**4*z+32*r**2*z**3])
    particular = -MU0*model.pprime*r**4/8 - model.ffprime*r**2*log/2
    particular_dr = -MU0*model.pprime*r**3/2 - model.ffprime*r*(2*log+1)/2
    psi = particular + np.tensordot(model.coefficients, basis, axes=(0, 0))
    dpsi_dr = particular_dr + np.tensordot(model.coefficients, dr, axes=(0, 0))
    dpsi_dz = np.tensordot(model.coefficients, dz, axes=(0, 0))
    return psi, dpsi_dr, dpsi_dz, basis


def evaluate_solovev(model: SolovevEquilibrium, r: Any, z: Any) -> Mapping[str, np.ndarray]:
    psi, dpsi_dr, dpsi_dz, _ = _solovev_components(model, r, z)
    rr = np.asarray(r, dtype=float)
    delta = psi-model.psi_boundary
    pressure = model.pressure_boundary + model.pprime*delta
    f_squared = model.f_boundary**2 + 2*model.ffprime*delta
    f = model.f_sign*np.sqrt(np.clip(f_squared, 0, None))
    return {
        "psi": psi, "dpsi_dr": dpsi_dr, "dpsi_dz": dpsi_dz,
        "b_r": -dpsi_dz/rr, "b_z": dpsi_dr/rr, "b_phi": f/rr,
        "pressure": pressure, "f": f,
        "j_phi": rr*model.pprime + model.ffprime/(MU0*rr),
        "grad_shafranov_source": -MU0*rr**2*model.pprime-model.ffprime,
    }


def solve_solovev_constraints(
    constraints: Sequence[SolovevConstraint], *, pprime: float, ffprime: float,
    rref: float, psi_boundary: float = 0.0, pressure_boundary: float = 0.0,
    f_boundary: float = 1.0, f_sign: int = 1,
) -> SolovevEquilibrium:
    if len(constraints) < 5:
        raise ValueError("at least five independent Solovev constraints are required")
    zero = SolovevEquilibrium(np.zeros(5), pprime, ffprime, rref, psi_boundary, pressure_boundary, f_boundary, f_sign)
    rows, rhs = [], []
    for constraint in constraints:
        psi, d_r, d_z, basis = _solovev_components(zero, constraint.r, constraint.z)
        if constraint.kind == "psi":
            row, particular = basis, psi
        elif constraint.kind == "dpsi_dr":
            eps = max(1e-6*rref, 1e-8)
            basis_plus = _solovev_components(zero, constraint.r+eps, constraint.z)[3]
            basis_minus = _solovev_components(zero, constraint.r-eps, constraint.z)[3]
            row, particular = (basis_plus-basis_minus)/(2*eps), d_r
        elif constraint.kind == "dpsi_dz":
            eps = max(1e-6*rref, 1e-8)
            basis_plus = _solovev_components(zero, constraint.r, constraint.z+eps)[3]
            basis_minus = _solovev_components(zero, constraint.r, constraint.z-eps)[3]
            row, particular = (basis_plus-basis_minus)/(2*eps), d_z
        else:
            raise ValueError("Solovev constraint kind must be psi, dpsi_dr, or dpsi_dz")
        rows.append(np.asarray(row, dtype=float).reshape(5)); rhs.append(float(constraint.value-np.asarray(particular)))
    matrix = np.asarray(rows); vector = np.asarray(rhs)
    coefficients, residuals, rank, _ = np.linalg.lstsq(matrix, vector, rcond=None)
    if rank < 5:
        raise ValueError(f"Solovev constraints are rank deficient ({rank}/5)")
    residual_norm = float(np.linalg.norm(matrix@coefficients-vector))
    return SolovevEquilibrium(coefficients, pprime, ffprime, rref, psi_boundary, pressure_boundary, f_boundary, f_sign, int(rank), residual_norm, {"constraint_count": len(constraints)})


def _locate_solovev_axis(
    model: SolovevEquilibrium, r: np.ndarray, z: np.ndarray, psi: np.ndarray
) -> tuple[float, float]:
    """Find the O-point of an analytic Solovev field on the given grid.

    The constant-source particular solution grows with R, so the raw grid
    argmax of ``|psi - psi_boundary|`` can land on a domain corner instead of
    the magnetic axis. Locate true stationary points (grad psi = 0 with a
    definite Hessian, i.e. extrema rather than saddles) by multi-start root
    finding on the analytic field, then keep the one around which the
    psi_boundary contour actually closes.
    """
    spline = RectBivariateSpline(r, z, psi, kx=min(3, r.size - 1), ky=min(3, z.size - 1))

    def gradient(point: np.ndarray) -> np.ndarray:
        return np.array(
            [spline.ev(point[0], point[1], dx=1, dy=0), spline.ev(point[0], point[1], dx=0, dy=1)],
            dtype=float,
        )

    scale = max(float(np.max(np.abs(psi - model.psi_boundary))), 1e-30)
    candidates: list[tuple[float, float, float]] = []
    for rr in np.linspace(r[1], r[-2], min(9, r.size - 2)):
        for zz in np.linspace(z[1], z[-2], min(9, z.size - 2)):
            solved = root(gradient, (rr, zz))
            if not solved.success:
                continue
            pr, pz = map(float, solved.x)
            if not (r[0] < pr < r[-1] and z[0] < pz < z[-1]):
                continue
            if np.linalg.norm(gradient(solved.x)) > 1e-8 * scale:
                continue
            drr = float(spline.ev(pr, pz, dx=2, dy=0))
            dzz = float(spline.ev(pr, pz, dx=0, dy=2))
            drz = float(spline.ev(pr, pz, dx=1, dy=1))
            if drr * dzz - drz**2 <= 0:
                continue  # saddle (X-point), not an O-point
            if any(np.hypot(pr - a, pz - b) < 1e-6 * (r[-1] - r[0]) for a, b, _ in candidates):
                continue
            candidates.append((pr, pz, abs(float(spline.ev(pr, pz)) - model.psi_boundary)))
    for pr, pz, _depth in sorted(candidates, key=lambda item: -item[2]):
        temp = EquilibriumData(
            r=r, z=z, psi=psi, psi_axis=float(spline.ev(pr, pz)),
            psi_boundary=model.psi_boundary, magnetic_axis=(pr, pz),
        )
        if _closed_boundary_contour(temp, (pr, pz))[0] is not None:
            return (pr, pz)
    # Fallback: interior grid extremum (still excludes the domain edge).
    interior = np.abs(psi - model.psi_boundary).copy()
    interior[0, :] = interior[-1, :] = -np.inf
    interior[:, 0] = interior[:, -1] = -np.inf
    index = np.unravel_index(np.argmax(interior), psi.shape)
    return (float(r[index[0]]), float(z[index[1]]))


def _closed_boundary_contour(
    temp: EquilibriumData,
    axis: tuple[float, float],
    levels: tuple[float, ...] = (1.0, 0.9995, 0.999, 0.995, 0.99),
) -> tuple[Contour | None, float | None]:
    """Return a CLOSED psi_boundary contour enclosing ``axis``, or (None, None).

    ``_contour_at_level`` deliberately falls back to the longest open segment
    (useful for separatrix legs); the Solovev export must not accept that -- an
    open "LCFS" silently corrupts Ip, pressure integrals, and every shape
    descriptor. Step slightly inward from psi_n=1 to tolerate a boundary
    surface that grazes the grid edge.
    """
    for level in levels:
        contour = _contour_at_level(temp, level)
        if (
            contour is not None
            and contour.closed
            and MplPath(contour.points).contains_point(axis)
        ):
            return contour, level
    return None, None


def solovev_to_equilibrium(
    model: SolovevEquilibrium, r: Any, z: Any, *, magnetic_axis: tuple[float, float] | None = None,
    limiter: Contour | None = None, convention: int = 11,
) -> EquilibriumData:
    """Export an analytic Solovev model as a gridded :class:`EquilibriumData`.

    ``evaluate_solovev`` works with psi in Wb/rad (B_pol = grad(psi)/R). The
    exported flux quantities are scaled to honor the declared ``convention``:
    a full-weber COCOS (11-18) multiplies psi by 2*pi so descriptor
    derivation, which divides full-weber psi gradients by 2*pi, recovers the
    analytic fields exactly.

    Raises ``ValueError`` when ``psi_boundary`` does not form a CLOSED contour
    enclosing the magnetic axis on this grid (checked explicitly, stepping to
    psi_n=0.99 before giving up): an open boundary would silently corrupt Ip,
    the pressure integrals, and every shape descriptor.
    """
    r = np.asarray(r, dtype=float).reshape(-1); z = np.asarray(z, dtype=float).reshape(-1)
    rm, zm = np.meshgrid(r, z, indexing="ij")
    values = evaluate_solovev(model, rm, zm); psi = values["psi"]
    if magnetic_axis is None:
        magnetic_axis = _locate_solovev_axis(model, r, z, psi)
    psi_axis = float(evaluate_solovev(model, *magnetic_axis)["psi"])
    temp = EquilibriumData(r=r, z=z, psi=psi, psi_axis=psi_axis, psi_boundary=model.psi_boundary, magnetic_axis=magnetic_axis)
    lcfs, lcfs_level = _closed_boundary_contour(temp, magnetic_axis)
    if lcfs is None:
        raise ValueError(
            "psi_boundary does not form a closed contour containing the magnetic axis on "
            "this grid; enlarge the grid, adjust the model, or check the surface topology"
        )
    psi_1d = np.linspace(psi_axis, model.psi_boundary, max(65, min(r.size, z.size)))
    pressure = model.pressure_boundary + model.pprime*(psi_1d-model.psi_boundary)
    f = model.f_sign*np.sqrt(np.clip(model.f_boundary**2+2*model.ffprime*(psi_1d-model.psi_boundary), 0, None))
    mask = MplPath(lcfs.points).contains_points(np.column_stack((rm.ravel(), zm.ravel()))).reshape(rm.shape)
    ip = float(np.sum(values["j_phi"]*np.gradient(r)[:, None]*np.gradient(z)[None, :]*mask))
    if convention not in range(1, 19):
        raise ValueError("convention must be a COCOS index in the range 1..18")
    psi_factor = 2.0*np.pi if convention >= 11 else 1.0
    conv = _detect_convention(explicit=convention, bt0=float(model.f_boundary/model.rref), ip=ip, q=None, psi_1d=psi_1d*psi_factor, source="analytic Solovev")
    return EquilibriumData(
        r, z, psi*psi_factor, psi_axis*psi_factor, model.psi_boundary*psi_factor, magnetic_axis, lcfs, limiter,
        psi_1d*psi_factor, pressure, f, None, ip, float(model.f_boundary/model.rref), model.rref,
        None, conv, {"source_type": "solovev", "model": model, "lcfs_psi_n": lcfs_level,
                     "topology_assumptions": "axisymmetric limited or upper/lower-null"},
    )


def _grid_spacing(eq: EquilibriumData) -> float:
    """Coarsest cell size of the R/Z grid."""
    return float(max(np.max(np.abs(np.diff(eq.r))), np.max(np.abs(np.diff(eq.z)))))


def _flux_resolution(eq: EquilibriumData) -> float | None:
    """The psi_n step a single grid cell resolves near the plasma boundary.

    Two flux labels closer together than the psi change across one cell are not
    distinguishable, so this sets how far inside the boundary the confined
    region has to be probed for a contour that is actually resolved.  It is
    built from the grid spacing and the field's own gradient, so it adapts to
    any machine and carries no hard-coded geometry.  The sharper, per-saddle
    window for the flux-identity test is derived separately in
    :func:`_boundary_relevant_saddles`, because psi is stationary at a saddle
    and is therefore known there far more precisely than one cell.
    """
    if eq.psi_axis is None or eq.psi_boundary is None or eq.psi_axis == eq.psi_boundary:
        return None
    dpsi_dr, dpsi_dz = np.gradient(eq.psi, eq.r, eq.z, edge_order=2)
    magnitude = np.hypot(dpsi_dr, dpsi_dz)
    sample = magnitude.ravel()
    if eq.lcfs is not None and eq.lcfs.r.size >= 3:
        spline = RectBivariateSpline(eq.r, eq.z, magnitude, kx=min(3, eq.r.size-1), ky=min(3, eq.z.size-1))
        on_boundary = np.abs(np.asarray(spline.ev(eq.lcfs.r, eq.lcfs.z), dtype=float))
        on_boundary = on_boundary[np.isfinite(on_boundary)]
        if on_boundary.size:
            sample = on_boundary
    sample = sample[np.isfinite(sample)]
    if not sample.size:
        return None
    step = float(np.median(sample)) * _grid_spacing(eq) / abs(eq.psi_boundary - eq.psi_axis)
    return float(np.clip(step, 1e-6, 0.25))


def _stationary_points(eq: EquilibriumData) -> tuple[StationaryPoint, ...]:
    """Every point where grad(psi) vanishes, split into O-points and saddles.

    Seeds come from local minima of |grad psi| on the grid itself, so the search
    adapts to the resolution instead of sampling a fixed lattice, and the
    convergence test is expressed as a fraction of the field's own gradient
    scale rather than an absolute number of tesla-metres.
    """
    if eq.r is None or eq.z is None or eq.psi is None or eq.psi.shape != (eq.r.size, eq.z.size):
        return ()
    if eq.r.size < 4 or eq.z.size < 4:
        return ()
    spline = RectBivariateSpline(eq.r, eq.z, eq.psi, kx=min(3, eq.r.size-1), ky=min(3, eq.z.size-1))
    dpsi_dr, dpsi_dz = np.gradient(eq.psi, eq.r, eq.z, edge_order=2)
    magnitude = np.hypot(dpsi_dr, dpsi_dz)
    inner = magnitude[1:-1, 1:-1]
    seeds = (
        (inner <= magnitude[:-2, 1:-1]) & (inner <= magnitude[2:, 1:-1])
        & (inner <= magnitude[1:-1, :-2]) & (inner <= magnitude[1:-1, 2:])
    )
    spacing = _grid_spacing(eq)
    if eq.psi_axis is not None and eq.psi_boundary is not None and eq.psi_axis != eq.psi_boundary:
        scale = abs(eq.psi_boundary - eq.psi_axis)
    else:
        scale = float(np.ptp(eq.psi)) or 1.0
    residual_tolerance = 1e-6 * scale / spacing

    def gradient(point: np.ndarray) -> np.ndarray:
        return np.array([spline.ev(point[0], point[1], dx=1, dy=0), spline.ev(point[0], point[1], dx=0, dy=1)], dtype=float)

    found: list[tuple[float, float, float]] = []
    for index_r, index_z in np.argwhere(seeds):
        solved = root(gradient, (eq.r[index_r + 1], eq.z[index_z + 1]))
        if not solved.success:
            continue
        pr, pz = float(solved.x[0]), float(solved.x[1])
        if not (eq.r[0] < pr < eq.r[-1] and eq.z[0] < pz < eq.z[-1]):
            continue
        if float(np.linalg.norm(gradient(np.array([pr, pz])))) > residual_tolerance:
            continue
        if any(np.hypot(pr - a, pz - b) < 0.5 * spacing for a, b, _, _ in found):
            continue
        drr = float(spline.ev(pr, pz, dx=2, dy=0))
        dzz = float(spline.ev(pr, pz, dx=0, dy=2))
        drz = float(spline.ev(pr, pz, dx=1, dy=1))
        half_trace = 0.5 * (drr + dzz)
        offset = float(np.hypot(0.5 * (drr - dzz), drz))
        curvature = min(abs(half_trace + offset), abs(half_trace - offset))
        found.append((pr, pz, drr * dzz - drz**2, curvature))
    points: list[StationaryPoint] = []
    for pr, pz, determinant, curvature in found:
        if determinant == 0:
            continue  # degenerate: neither an extremum nor a saddle
        psi = float(spline.ev(pr, pz))
        if eq.psi_axis is not None and eq.psi_boundary is not None and eq.psi_axis != eq.psi_boundary:
            psi_n = (psi - eq.psi_axis) / (eq.psi_boundary - eq.psi_axis)
        else:
            psi_n = float("nan")
        points.append(StationaryPoint(pr, pz, psi, float(psi_n), "o" if determinant > 0 else "x", determinant, curvature))
    return tuple(points)


def _confined_contour(eq: EquilibriumData, level: float) -> tuple[Contour | None, str | None]:
    """The closed flux surface at ``psi_n = level`` that encloses the axis.

    Returns ``(None, reason)`` when the confined region is not resolvable at
    that level, which is the signal that a topology cannot be decided.
    """
    contour = _contour_at_level(eq, level)
    if contour is None:
        return None, f"no flux surface could be extracted at psi_n={level:.4g}"
    if not contour.closed:
        return None, f"the flux surface at psi_n={level:.4g} is clipped by the grid boundary"
    if eq.magnetic_axis is not None and not MplPath(contour.points).contains_point(eq.magnetic_axis):
        return None, f"the flux surface at psi_n={level:.4g} does not enclose the magnetic axis"
    return contour, None


def _boundary_relevant_saddles(
    eq: EquilibriumData, saddles: Sequence[StationaryPoint], tolerance: float | None,
    resolution: float, minor_radius: float,
) -> tuple[list[StationaryPoint], str | None]:
    """Select the saddles that actually sit on the plasma boundary.

    A saddle is a physical X-point when its flux matches the boundary flux to
    within what the grid resolves *and* the confined region's level set is
    consistent with a separatrix through it: the closed surface just inside the
    boundary flux must reach the saddle.  A saddle that passes the flux test but
    whose confined region cannot be resolved leaves the topology indeterminate
    rather than silently diverted or limited.
    """
    spacing = _grid_spacing(eq)
    scale = abs(eq.psi_boundary - eq.psi_axis)

    def flux_window(point: StationaryPoint) -> float:
        """How precisely this saddle's flux label is known.

        psi is stationary at a saddle, so a position error does not perturb it
        to first order: displacing by one grid cell changes psi only by about
        ``curvature*h^2/2``.  That makes the identity test far sharper than the
        grid's generic flux resolution, and it is set by the saddle's own local
        topology rather than by any global assumption.
        """
        if tolerance is not None:
            return tolerance
        return max(0.5 * point.curvature * spacing**2 / scale, 1e-9)

    candidates = [
        point for point in saddles
        if np.isfinite(point.psi_n) and abs(point.psi_n - 1.0) <= flux_window(point)
    ]
    if not candidates:
        return [], None
    offset = max(2.0 * resolution, 1e-4)
    level = 1.0 - offset
    contour, why = _confined_contour(eq, level)
    if contour is None:
        return [], f"a saddle lies on the boundary flux but {why}"
    # psi is quadratic about a saddle, so probing the confined region a flux
    # offset inside the boundary moves its level set away from a genuine X-point
    # by about sqrt(2*dpsi/curvature).  Comparing the measured separation with
    # that intrinsic scale keeps the test free of any absolute length.
    delta_psi = offset * scale
    active = []
    for point in candidates:
        separation = float(np.min(np.hypot(contour.r - point.r, contour.z - point.z)))
        expected = np.sqrt(2.0 * delta_psi / point.curvature) if point.curvature > 0 else np.inf
        # A saddle so flat that its level set would retreat across a large part
        # of the plasma cannot be bounding that plasma, so the curvature scale is
        # capped by the plasma's own size.  Both bounds are relative: one to the
        # local flux curvature, one to the minor radius.
        reach = min(max(3.0 * expected, 4.0 * spacing), 0.5 * minor_radius)
        if separation <= reach:
            active.append(point)
    return active, None


def _wall_contact_distance(eq: EquilibriumData) -> float | None:
    """Closest approach between the LCFS and the limiter/first wall."""
    if eq.limiter is None or eq.limiter.r.size < 2 or eq.lcfs is None or eq.lcfs.r.size < 2:
        return None
    try:
        wall = _resample_contour(eq.limiter, max(512, 8 * eq.limiter.r.size))
    except ValueError:
        return None
    return float(np.min(cKDTree(wall.points).query(eq.lcfs.points)[0]))


def _outboard_radius_at_z(contour: Contour, z0: float) -> float | None:
    """Outboard-midplane radius: the largest R where the contour crosses z=z0.

    Distinct from ``max(contour.r)`` for up-down asymmetric or shifted
    surfaces, whose maximum-R point does not sit on the midplane.
    """
    points = contour.points
    if points.shape[0] < 2:
        return None
    if contour.closed and np.any(points[0] != points[-1]):
        points = np.vstack([points, points[0]])
    crossings: list[float] = []
    for (r1, z1), (r2, z2) in zip(points[:-1], points[1:]):
        if (z1 - z0) * (z2 - z0) > 0 or z1 == z2:
            continue
        fraction = (z0 - z1) / (z2 - z1)
        if 0.0 <= fraction <= 1.0:
            crossings.append(float(r1 + fraction * (r2 - r1)))
    return max(crossings) if crossings else None


def _ray_intersections(contour: Contour, origin: tuple[float, float], angle: float) -> list[tuple[float, tuple[float, float]]]:
    points = contour.points
    if points.shape[0] < 2:
        return []
    points = np.vstack((points, points[0]))
    direction = np.array([np.cos(angle), np.sin(angle)]); origin_array = np.asarray(origin)
    result = []
    cross2 = lambda a,b: a[0]*b[1]-a[1]*b[0]
    for p, q in zip(points[:-1], points[1:]):
        segment = q-p; denominator = cross2(direction, segment)
        if abs(denominator) < 1e-14:
            continue
        delta = p-origin_array
        t = cross2(delta, segment)/denominator; u = cross2(delta, direction)/denominator
        if t >= 0 and -1e-12 <= u <= 1+1e-12:
            result.append((float(t), tuple(origin_array+t*direction)))
    return sorted(result)


def _fourier_boundary(contour: Contour, modes: int) -> tuple[dict[str, np.ndarray], float]:
    sampled = _resample_contour(contour, max(256, 8*modes))
    start = int(np.argmax(sampled.r)); r = np.roll(sampled.r, -start); z = np.roll(sampled.z, -start)
    # Ensure a deterministic CCW traversal.
    area = 0.5*np.sum(r*np.roll(z,-1)-np.roll(r,-1)*z)
    if area < 0:
        r = np.r_[r[0], r[:0:-1]]; z = np.r_[z[0], z[:0:-1]]
    theta = np.linspace(0, 2*np.pi, r.size, endpoint=False)
    matrix = [np.ones_like(theta)]
    for n in range(1, modes+1): matrix.extend((np.cos(n*theta), np.sin(n*theta)))
    design = np.column_stack(matrix)
    cr = np.linalg.lstsq(design, r, rcond=None)[0]; cz = np.linalg.lstsq(design, z, rcond=None)[0]
    reconstructed = np.column_stack((design@cr, design@cz))
    error = float(np.sqrt(np.mean(np.sum((reconstructed-np.column_stack((r,z)))**2, axis=1))))
    def split(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.r_[c[0], c[1::2]]; b = np.r_[0.0, c[2::2]]
        return a, b
    ar, br = split(cr); az, bz = split(cz)
    return {"r_cos": ar, "r_sin": br, "z_cos": az, "z_sin": bz}, error


def _segment_intersections(first: Contour, second: Contour) -> list[tuple[float, float, int]]:
    a = np.vstack((first.points, first.points[0])); b = np.vstack((second.points, second.points[0])); out=[]
    cross2=lambda x,y:x[0]*y[1]-x[1]*y[0]
    for p1,p2 in zip(a[:-1],a[1:]):
        d1=p2-p1
        for index,(q1,q2) in enumerate(zip(b[:-1],b[1:])):
            d2=q2-q1; den=cross2(d1,d2)
            if abs(den)<1e-14: continue
            delta=q1-p1; t=cross2(delta,d2)/den; u=cross2(delta,d1)/den
            if -1e-10<=t<=1+1e-10 and -1e-10<=u<=1+1e-10:
                point=p1+t*d1
                if not any(np.hypot(point[0]-x,point[1]-y)<1e-6 for x,y,_ in out): out.append((float(point[0]),float(point[1]),index))
    return out


def derive_boundary_representation(
    equilibrium: Any, *, gap_angles: Mapping[str, float] | None = None,
    flux_tolerance: float | None = None, fourier_modes: int = 16,
) -> BoundaryRepresentation:
    """Classify the plasma boundary and derive its geometric diagnostics.

    Topology is decided from the flux map alone.  Stationary points of psi are
    located and split into O-points and saddles by the sign of the Hessian
    determinant; a saddle counts as a physical X-point only when it is relevant
    to the boundary, meaning its flux agrees with the boundary flux to within
    what the grid resolves and the confined region's level set reaches it.  At
    least one such X-point makes the equilibrium diverted; none, together with
    an LCFS in contact with the wall, makes it limited.  Anything that cannot be
    settled -- a grid-clipped confined region, no wall to test against, or an
    LCFS bounded by neither -- returns ``Topology.AMBIGUOUS`` with a reason.

    ``flux_tolerance`` overrides the psi_n window of the flux test with a single
    value for every saddle.  Leave it at ``None`` -- the numerically justified
    choice -- and each saddle gets its own window from its Hessian curvature and
    the grid spacing, since psi is stationary there and a one-cell position
    error perturbs its flux only at second order.
    """
    eq = as_equilibrium(equilibrium)
    unavailable = lambda definition, reason: _unavailable(eq, "m", definition, reason)
    have_grid = eq.psi is not None and eq.r is not None and eq.z is not None and eq.r.size > 1 and eq.z.size > 1
    resolution = _flux_resolution(eq) if have_grid else None
    spacing = _grid_spacing(eq) if have_grid else None
    provenance = _provenance(
        eq, "stationary-point classification and contour geometry", ("psi", "lcfs", "limiter"),
        tolerances={
            # NaN means "derived per saddle from its own curvature" rather than
            # taken from a single caller-supplied window.
            "xpoint_flux_psi_n": float(flux_tolerance) if flux_tolerance is not None else float("nan"),
            "grid_flux_resolution_psi_n": float(resolution) if resolution is not None else float("nan"),
            "grid_spacing_m": float(spacing) if spacing is not None else float("nan"),
        },
    )
    if eq.lcfs is None:
        return BoundaryRepresentation(None, eq.limiter, (), Topology.AMBIGUOUS, unavailable("outboard upper-minus-lower separatrix offset", "LCFS is unavailable"), (), (), {}, unavailable("Fourier LCFS RMS", "LCFS is unavailable"), provenance, "LCFS is unavailable")

    center_geo = _polygon_geometry(eq.lcfs); center = (center_geo["r"], center_geo["z"])
    minor_radius = 0.5 * float(np.max(eq.lcfs.r) - np.min(eq.lcfs.r))
    stationary = _stationary_points(eq)
    saddles = [point for point in stationary if point.kind == "x"]

    reason: str | None = None
    if resolution is None:
        topology, active = Topology.AMBIGUOUS, []
        reason = ("an R/Z psi grid with distinct psi_axis and psi_boundary is required to test a "
                  "saddle against the boundary flux")
    else:
        active, indeterminate = _boundary_relevant_saddles(eq, saddles, flux_tolerance, resolution, minor_radius)
        if indeterminate is not None:
            topology = Topology.AMBIGUOUS
            reason = f"{indeterminate}; the boundary cannot be classified at this resolution"
        elif active:
            if eq.magnetic_axis is None:
                topology = Topology.DIVERTED
                reason = "the magnetic axis is unavailable, so X-points cannot be attributed to an upper or lower branch"
            else:
                upper = [point for point in active if point.z >= eq.magnetic_axis[1]]
                lower = [point for point in active if point.z < eq.magnetic_axis[1]]
                topology = (
                    Topology.DOUBLE_NULL if upper and lower
                    else Topology.UPPER_SINGLE_NULL if upper
                    else Topology.LOWER_SINGLE_NULL
                )
        else:
            contact = _wall_contact_distance(eq)
            touching = max(3.0 * spacing, 0.02 * minor_radius) if spacing is not None else 0.02 * minor_radius
            if contact is None:
                topology = Topology.AMBIGUOUS
                reason = ("no saddle is relevant to the boundary flux, and no limiter or first-wall "
                          "contour is available to confirm a limited boundary")
            elif contact <= touching:
                topology = Topology.LIMITED
            else:
                topology = Topology.AMBIGUOUS
                reason = (f"no saddle is relevant to the boundary flux, yet the LCFS stands {contact:.4g} m "
                          f"clear of the wall (contact needs {touching:.4g} m), so the boundary is set by neither")

    active_keys = {(point.r, point.z) for point in active}
    xpoints = tuple(sorted(
        (XPoint(point.r, point.z, point.psi, point.psi_n, (point.r, point.z) in active_keys, point.hessian_determinant)
         for point in saddles),
        key=lambda point: (abs(point.psi_n - 1), point.z),
    ))

    contact_distance = _wall_contact_distance(eq)
    wall_contact = (
        _derived(eq, contact_distance, "m", "min distance from the LCFS to the limiter/first wall", "nearest-point search", ("lcfs", "limiter"))
        if contact_distance is not None
        else unavailable("min distance from the LCFS to the limiter/first wall", "a limiter/first-wall contour is required")
    )

    # dRsep separates two X-point flux surfaces, so it is only meaningful once
    # the configuration is known to be diverted.
    d_r_sep = unavailable("R_out(psi_X,upper)-R_out(psi_X,lower)", "a diverted topology with both an upper and a lower X-point is required")
    if topology.is_diverted and eq.psi_axis is not None and eq.psi_boundary is not None:
        r_axis, z_mid = eq.magnetic_axis if eq.magnetic_axis else center
        ordered = sorted(saddles, key=lambda point: abs(point.psi_n - 1))
        all_upper = [point for point in ordered if point.z >= z_mid]
        all_lower = [point for point in ordered if point.z < z_mid]
        if all_upper and all_lower:
            r_up = _outboard_midplane_radius(eq, all_upper[0].psi, r_axis, z_mid)
            r_low = _outboard_midplane_radius(eq, all_lower[0].psi, r_axis, z_mid)
            if r_up is None or r_low is None:
                d_r_sep = unavailable("R_out(psi_X,upper)-R_out(psi_X,lower)", "an X-point flux level is not reached on the outboard midplane")
            else:
                d_r_sep = _derived(eq, float(r_up-r_low), "m", "R_out(psi_X,upper)-R_out(psi_X,lower) at Z=Z_axis", "outboard-midplane flux inversion", ("psi", "magnetic_axis"))
    gaps: list[Gap] = []
    angles = dict(gap_angles or {"outboard": 0.0, "top": np.pi/2, "inboard": np.pi, "bottom": 3*np.pi/2})
    center_geo = _polygon_geometry(eq.lcfs); center=(center_geo["r"],center_geo["z"])
    for name, angle in angles.items():
        plasma_hits = _ray_intersections(eq.lcfs, center, angle)
        wall_hits = _ray_intersections(eq.limiter, center, angle) if eq.limiter is not None else []
        if plasma_hits and wall_hits:
            plasma = plasma_hits[-1]; farther = [hit for hit in wall_hits if hit[0] >= plasma[0]-1e-10]
            if farther:
                wall=min(farther); distance=max(0.0,wall[0]-plasma[0])
                gaps.append(Gap(name,float(angle),_derived(eq,distance,"m","ray distance from LCFS to first wall","ray/segment intersection",("lcfs","limiter")),plasma[1],wall[1])); continue
        gaps.append(Gap(name,float(angle),unavailable("ray distance from LCFS to first wall","LCFS/wall ray intersection is unavailable")))
    coefficients, error = _fourier_boundary(eq.lcfs, int(fourier_modes))
    fourier_error = _derived(eq,error,"m",f"RMS reconstruction error for {fourier_modes}-mode periodic Fourier boundary","arc-length least squares",("lcfs",),{"modes":fourier_modes})
    strikes: list[StrikePoint] = []
    if eq.limiter is not None and active and eq.r is not None and eq.z is not None and eq.psi is not None:
        try:
            rm,zm,br,bz,bp=_grid_fields(eq); brs=RectBivariateSpline(eq.r,eq.z,br); bzs=RectBivariateSpline(eq.r,eq.z,bz)
            fs=RectBivariateSpline(eq.r,eq.z,eq.psi)
            upstream=(float(np.max(eq.lcfs.r)), center[1]); bp_up=float(np.hypot(brs.ev(*upstream),bzs.ev(*upstream)))
            for point in active:
                separatrix=_contour_at_level(eq,point.psi_n)
                if separatrix is None: continue
                for rr,zz,wall_index in _segment_intersections(separatrix,eq.limiter):
                    bp_target=float(np.hypot(brs.ev(rr,zz),bzs.ev(rr,zz)))
                    expansion=(upstream[0]*bp_up)/(rr*bp_target) if rr*bp_target else None
                    wall_points=np.vstack((eq.limiter.points,eq.limiter.points[0])); tangent=wall_points[wall_index+1]-wall_points[wall_index]; tangent=tangent/np.linalg.norm(tangent); normal=np.array([-tangent[1],tangent[0]])
                    bpol=np.array([float(brs.ev(rr,zz)),float(bzs.ev(rr,zz))]); bphi=0.0
                    if eq.f is not None and eq.psi_1d is not None:
                        bphi=float(np.interp(float(fs.ev(rr,zz)),np.sort(eq.psi_1d),eq.f[np.argsort(eq.psi_1d)])/rr)
                    incidence=float(np.arcsin(np.clip(abs(np.dot(bpol,normal))/np.sqrt(np.dot(bpol,bpol)+bphi**2),0,1)))
                    strikes.append(StrikePoint(rr,zz,"upper" if point.z>=center[1] else "lower",_derived(eq,expansion,"1","(R_up Bp_up)/(R_target Bp_target)","local field ratio",("psi","f")) if expansion is not None else _unavailable(eq,"1","poloidal flux expansion","target Bp is zero"),_derived(eq,incidence,"rad","asin(abs(B_pol dot wall_normal)/abs(B))","wall intersection geometry",("psi","f","limiter"))))
        except Exception:
            strikes=[]
    # A topology reason already explains an unresolved classification; only add a
    # diagnostic here when the topology itself came out fine.
    if reason is None:
        if eq.limiter is None:
            reason = "strike points and gaps require a limiter/first-wall contour"
        elif active and not strikes:
            reason = "no separatrix/wall intersections were found for the boundary-relevant X-points"
    return BoundaryRepresentation(
        eq.lcfs, eq.limiter, xpoints, topology, d_r_sep, tuple(gaps), tuple(strikes),
        coefficients, fourier_error, provenance, reason, stationary, wall_contact,
    )


__all__ = [
    "as_equilibrium", "convert_cocos", "derive_boundary_representation",
    "derive_global_descriptors", "derive_radial_coordinates", "evaluate_miller",
    "evaluate_solovev", "fit_miller_sequence", "fit_miller_surface",
    "solovev_to_equilibrium", "solve_solovev_constraints", "validate_equilibrium",
]
