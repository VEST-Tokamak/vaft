"""Edge safety-factor estimates from an ODS, with or without an equilibrium (#1583).

Most VEST discharges have no converged equilibrium, so their q95 is unknown.
:func:`edge_q_estimate` evaluates the analytic edge-q proxies of
:mod:`vaft.formula.equilibrium` on what the ODS has:

* ``source="equilibrium"``: every time slice's boundary shape (the IMAS
  scalars, or the outline when they are absent), ``global_quantities.ip`` and
  the vacuum field. This is mainly a check of the estimate against the
  equilibrium's own ``global_quantities.q_95``, which is returned beside it.
* ``source="magnetics"``: the measured ``magnetics.ip`` trace and the
  ``tf`` field, with a shape that is either the caller's or the machine's
  stated default (``vest.yaml:edge_q_estimate.default_shape``). The returned
  ``shape_source`` and ``provenance`` say which; there is no silent default.

The scaling and START configuration come from the machine description
(:func:`vaft.machine_mapping.edge_q_estimate.vest_edge_q_estimate_policy`)
unless the caller names them. Each proxy keeps its own name: the estimate is
q95 from a scaling, Menard's q* and Freidberg's q* are different quantities,
and none of them is q_a.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from vaft.ods_access import path_count, path_value

__all__ = [
    "EdgeQEstimate",
    "edge_q_estimate",
]

_SOURCES = ("auto", "equilibrium", "magnetics")
_SCALING_LABELS = {"start": "START", "iter": "ITER"}


@dataclass(frozen=True)
class EdgeQEstimate:
    """Edge-q proxies on one time base, with where every input came from.

    Arrays share ``time`` [s]. ``plasma_current`` is in A, ``toroidal_field``
    is the vacuum field at the geometric centre ``shape["major_radius"]`` in T,
    ``normalized_current`` is I_N = I_p[MA]/(a B_T) in MA/(m T); the safety
    factors are dimensionless. Samples below ``current_threshold`` (A) are NaN.
    ``equilibrium_q95`` (|q_95|, on ``equilibrium_time``) is ``None`` when the
    ODS has no equilibrium q95.
    """

    time: np.ndarray
    estimated_q95: np.ndarray
    q_star_cylindrical: np.ndarray
    q_star_kink: np.ndarray
    normalized_current: np.ndarray
    plasma_current: np.ndarray
    toroidal_field: np.ndarray
    shape: Mapping[str, np.ndarray]
    scaling: str
    configuration: str
    source: str
    shape_source: str
    current_source: str
    field_source: str
    current_threshold: float
    equilibrium_time: Optional[np.ndarray] = None
    equilibrium_q95: Optional[np.ndarray] = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        """``"q95 (START estimate)"`` or ``"q95 (ITER estimate)"``; never bare q95."""
        return f"q95 ({_SCALING_LABELS[self.scaling]} estimate)"

    @property
    def provenance(self) -> str:
        """One line naming the scaling and the source of every input."""
        config = f", {self.configuration}" if self.scaling == "start" else ""
        parts = [
            f"{self.label}: scaling={self.scaling}{config}",
            f"I_p from {self.current_source}",
            f"B_T from {self.field_source}",
            f"shape from {self.shape_source}",
        ]
        return "; ".join(parts + list(self.notes))


def _as_array(value: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=float))


def _caller_shape(shape: Mapping[str, Any]) -> dict[str, float]:
    from vaft.machine_mapping.edge_q_estimate import SHAPE_KEYS

    missing = [key for key in SHAPE_KEYS if key not in shape]
    if missing:
        raise ValueError(f"shape must give {', '.join(SHAPE_KEYS)}; missing {', '.join(missing)}")
    return {key: float(shape[key]) for key in SHAPE_KEYS}


def _slice_shape(ods: Any, i: int) -> Optional[dict[str, float]]:
    """The boundary shape of slice ``i``: IMAS scalars, else its outline, else ``None``."""
    base = f"equilibrium.time_slice.{i}.boundary"
    scalars = {
        "minor_radius": path_value(ods, f"{base}.minor_radius"),
        "major_radius": path_value(ods, f"{base}.geometric_axis.r"),
        "elongation": path_value(ods, f"{base}.elongation"),
        "triangularity": path_value(ods, f"{base}.triangularity"),
    }
    if all(value is not None and np.isfinite(float(value)) for value in scalars.values()):
        return {key: float(value) for key, value in scalars.items()}
    r = path_value(ods, f"{base}.outline.r")
    z = path_value(ods, f"{base}.outline.z")
    if r is None or z is None or np.size(r) < 4:
        return None
    from vaft.process.equilibrium import contour_shape_parameters

    try:
        params = contour_shape_parameters(np.asarray(r, dtype=float), np.asarray(z, dtype=float))
    except ValueError:
        return None
    return {
        "minor_radius": 0.5 * (params["r_outboard"] - params["r_inboard"]),
        "major_radius": 0.5 * (params["r_outboard"] + params["r_inboard"]),
        "elongation": params["elongation"],
        "triangularity": 0.5 * (params["triangularity_upper"] + params["triangularity_lower"]),
    }


def _equilibrium_q95(ods: Any) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    n = path_count(ods, "equilibrium.time_slice")
    if n == 0:
        return None, None
    time = path_value(ods, "equilibrium.time")
    values = [path_value(ods, f"equilibrium.time_slice.{i}.global_quantities.q_95") for i in range(n)]
    if time is None or all(value is None for value in values):
        return None, None
    q = np.array([np.nan if value is None else abs(float(value)) for value in values])
    return _as_array(time)[:n], q


def _has_equilibrium_shape(ods: Any) -> bool:
    n = path_count(ods, "equilibrium.time_slice")
    return n > 0 and any(_slice_shape(ods, i) is not None for i in range(n))


def _from_equilibrium(ods: Any, shape: Optional[Mapping[str, Any]]):
    n = path_count(ods, "equilibrium.time_slice")
    if n == 0:
        raise ValueError("source='equilibrium' needs equilibrium.time_slice")
    time = _as_array(path_value(ods, "equilibrium.time"))[:n]
    if shape is not None:
        fixed = _caller_shape(shape)
        slices = [fixed] * n
        shape_source = "the caller's shape"
    else:
        slices = [_slice_shape(ods, i) for i in range(n)]
        shape_source = "equilibrium.time_slice[:].boundary"
    nan_shape = dict.fromkeys(("minor_radius", "major_radius", "elongation", "triangularity"), np.nan)
    shapes = {key: np.array([(s or nan_shape)[key] for s in slices]) for key in nan_shape}
    ip = np.array([np.nan if (v := path_value(ods, f"equilibrium.time_slice.{i}.global_quantities.ip")) is None
                   else float(v) for i in range(n)])
    r0 = path_value(ods, "equilibrium.vacuum_toroidal_field.r0")
    b0 = path_value(ods, "equilibrium.vacuum_toroidal_field.b0")
    if r0 is None or b0 is None:
        raise ValueError("source='equilibrium' needs equilibrium.vacuum_toroidal_field.r0 and b0")
    rb = float(r0) * _as_array(b0)[:n]
    field = rb / shapes["major_radius"]
    return (time, ip, field, shapes, shape_source, "equilibrium.time_slice[:].global_quantities.ip",
            "equilibrium.vacuum_toroidal_field (R0 B0 / R_geo)")


def _from_magnetics(ods: Any, shape: Optional[Mapping[str, Any]], policy: Any):
    ip = path_value(ods, "magnetics.ip.0.data")
    if ip is None:
        raise ValueError("source='magnetics' needs magnetics.ip.0.data")
    ip_time = path_value(ods, "magnetics.ip.0.time")
    if ip_time is None:
        ip_time = path_value(ods, "magnetics.time")
    if ip_time is None:
        raise ValueError("source='magnetics' needs magnetics.ip.0.time or magnetics.time")
    time, current = _as_array(ip_time), _as_array(ip)
    if time.size != current.size:
        raise ValueError(f"magnetics.ip.0 has {current.size} samples on a {time.size}-sample time base")

    if shape is not None:
        fixed = _caller_shape(shape)
        shape_source = "the caller's shape"
    else:
        fixed = dict(policy.default_shape)
        shape_source = (f"the default shape of {policy.source}.default_shape "
                        f"(a={fixed['minor_radius']:.3f} m, R_geo={fixed['major_radius']:.3f} m, "
                        f"kappa={fixed['elongation']:.2f}, delta={fixed['triangularity']:.2f})")
    shapes = {key: np.full(time.size, value) for key, value in fixed.items()}

    tf_time = path_value(ods, "tf.time")
    rb = path_value(ods, "tf.b_field_tor_vacuum_r.data")
    if tf_time is not None and rb is not None and np.size(tf_time) == np.size(rb):
        tf_t, rb_arr = _as_array(tf_time), _as_array(rb)
        rb_at = np.interp(time, tf_t, rb_arr, left=np.nan, right=np.nan)
        field_source = "tf.b_field_tor_vacuum_r (R B / R_geo)"
    else:
        r0 = path_value(ods, "equilibrium.vacuum_toroidal_field.r0")
        b0 = path_value(ods, "equilibrium.vacuum_toroidal_field.b0")
        eq_time = path_value(ods, "equilibrium.time")
        if r0 is None or b0 is None or eq_time is None:
            raise ValueError("source='magnetics' needs tf.b_field_tor_vacuum_r (or equilibrium.vacuum_toroidal_field)")
        rb_at = np.interp(time, _as_array(eq_time), float(r0) * _as_array(b0), left=np.nan, right=np.nan)
        field_source = "equilibrium.vacuum_toroidal_field (R0 B0 / R_geo)"
    field = rb_at / shapes["major_radius"]
    return time, current, field, shapes, shape_source, "magnetics.ip.0", field_source


def edge_q_estimate(
    ods: Any,
    *,
    source: str = "auto",
    shape: Optional[Mapping[str, Any]] = None,
    scaling: Optional[str] = None,
    configuration: Optional[str] = None,
    current_threshold: Optional[float] = None,
    info_file: Optional[str] = None,
) -> EdgeQEstimate:
    """Estimated q95, Menard q*, Freidberg q* and I_N of one discharge.

    Parameters
    ----------
    ods : ODS
        One discharge.
    source : {"auto", "equilibrium", "magnetics"}
        Where shape and current come from. ``"auto"`` takes the equilibrium when
        a time slice has a boundary, else the magnetics.
    shape : mapping, optional
        ``minor_radius`` [m], ``major_radius`` (geometric centre) [m],
        ``elongation`` and ``triangularity``, overriding the equilibrium's or
        the machine default. All four or none.
    scaling, configuration : str, optional
        :func:`vaft.formula.equilibrium.estimated_q95` options; default from
        ``vest.yaml:edge_q_estimate``.
    current_threshold : float, optional
        |I_p| [A] below which every proxy is NaN (no plasma to estimate); default
        ``vest.yaml:equilibrium_regime.phase.vacuum_current_amperes``.
    info_file : str, optional
        Alternative machine description.

    Returns
    -------
    EdgeQEstimate
        The proxies and the provenance of every input.
    """
    from vaft.formula.equilibrium import (
        estimated_q95,
        normalized_plasma_current,
        q_star_cylindrical,
        q_star_kink,
    )
    from vaft.machine_mapping.edge_q_estimate import vest_edge_q_estimate_policy

    if source not in _SOURCES:
        raise ValueError(f"source must be one of {list(_SOURCES)}, not {source!r}")
    policy = vest_edge_q_estimate_policy(info_file=info_file)
    scaling = policy.scaling if scaling is None else scaling
    if configuration is None:
        configuration = policy.configuration if scaling == "start" else "limiter"
    if current_threshold is None:
        from vaft.machine_mapping.equilibrium_regime import vest_equilibrium_regime_policy

        current_threshold = vest_equilibrium_regime_policy(info_file=info_file).vacuum_current_amperes
    current_threshold = float(current_threshold)

    resolved = source
    if source == "auto":
        resolved = "equilibrium" if _has_equilibrium_shape(ods) else "magnetics"
    if resolved == "equilibrium":
        parts = _from_equilibrium(ods, shape)
    else:
        parts = _from_magnetics(ods, shape, policy)
    time, current, field, shapes, shape_source, current_source, field_source = parts

    current = np.where(np.abs(current) >= current_threshold, current, np.nan)
    a, R, kappa, delta = (shapes[k] for k in ("minor_radius", "major_radius", "elongation", "triangularity"))
    with np.errstate(invalid="ignore", divide="ignore"):
        q95 = _as_array(estimated_q95(a, R, field, kappa, delta, current,
                                      scaling=scaling, configuration=configuration))
        q_cyl = _as_array(q_star_cylindrical(a, R, field, kappa, current))
        q_kink = _as_array(q_star_kink(a, R, field, kappa, current))
        i_n = _as_array(np.abs(normalized_plasma_current(current, R, a, np.abs(field))))

    notes = []
    if resolved == "magnetics" and shape is None:
        notes.append(f"default shape status: {policy.status['default_shape']}")
    eq_time, eq_q95 = _equilibrium_q95(ods)
    return EdgeQEstimate(
        time=time,
        estimated_q95=q95,
        q_star_cylindrical=q_cyl,
        q_star_kink=q_kink,
        normalized_current=i_n,
        plasma_current=current,
        toroidal_field=np.abs(field),
        shape=shapes,
        scaling=scaling,
        configuration=configuration,
        source=resolved,
        shape_source=shape_source,
        current_source=current_source,
        field_source=field_source,
        current_threshold=current_threshold,
        equilibrium_time=eq_time,
        equilibrium_q95=eq_q95,
        notes=tuple(notes),
    )
