"""Typed, NumPy-backed view models consumed by the canonical renderers.

Canonical renderers in :mod:`vaft.plot` accept these models and nothing else.
They never see an OMAS ``ODS``/``ODC``, a native IMAS ``IDS``, a ``DBEntry``, a
shot number, a code result, or a file path -- converting those into a view model
is the job of the adapter layers (``vaft.omas.plot_*`` and friends).

Every model is a frozen dataclass whose ``__post_init__`` coerces sequences to
read-only :class:`numpy.ndarray` instances and validates shapes, so a renderer
can assume well-formed arrays without defensive checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "Field2D",
    "GeometryLayer",
    "GeometryLayers",
    "LineSeries",
    "Panels",
    "Profile1D",
    "Series",
    "Spectrogram",
    "ViewModel",
    "as_model_array",
]


# Names of the data-object families that must never reach a renderer.  Matching
# by class name keeps this check free of hard imports on optional dependencies.
_REJECTED_TYPE_NAMES = frozenset(
    {
        "ODS",
        "ODC",
        "CodeParameters",
        "DBEntry",
        "IDSToplevel",
        "IDSStructure",
        "IDSStructArray",
    }
)


def _reject_data_objects(value: Any, *, where: str) -> None:
    """Raise an actionable ``TypeError`` when a data object is passed as data."""
    for klass in type(value).__mro__:
        if klass.__name__ in _REJECTED_TYPE_NAMES:
            raise TypeError(
                f"{where} received a {type(value).__name__} object. Canonical "
                "vaft.plot renderers accept view models only; build one with the "
                "matching adapter (for example vaft.omas.plot_* for ODS/ODC "
                "inputs) instead of passing the data object directly."
            )


def as_model_array(value: Any, *, where: str, dtype: Any = float) -> np.ndarray:
    """Return ``value`` as a read-only ``ndarray``, rejecting data objects.

    When ``value`` is already an ``ndarray`` of the right dtype, a read-only
    *view* is taken rather than freezing the caller's own array: constructing a
    view model must not make the array the caller passed in immutable.
    """
    _reject_data_objects(value, where=where)
    array = np.asarray(value, dtype=dtype)
    if array is value or array.base is not None or np.shares_memory(array, value):
        array = array.view()
    array.flags.writeable = False
    return array


def _frozen_style(style: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(style or {}))


@dataclass(frozen=True)
class ViewModel:
    """Marker base class shared by every renderer input model."""


@dataclass(frozen=True)
class Series(ViewModel):
    """One labeled x-y trace, optionally with symmetric or asymmetric error bars."""

    x: np.ndarray
    y: np.ndarray
    label: str = ""
    yerr: np.ndarray | None = None
    style: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x = as_model_array(self.x, where="Series.x")
        y = as_model_array(self.y, where="Series.y")
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(
                f"Series.x and Series.y must be 1D; got shapes {x.shape} and {y.shape}"
            )
        if x.size != y.size:
            raise ValueError(
                f"Series.x and Series.y must have equal length; got {x.size} and {y.size}"
            )
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        if self.yerr is not None:
            yerr = as_model_array(self.yerr, where="Series.yerr")
            if yerr.shape not in {y.shape, (2, y.size)}:
                raise ValueError(
                    "Series.yerr must match y's shape or be (2, len(y)); "
                    f"got {yerr.shape} for y of length {y.size}"
                )
            object.__setattr__(self, "yerr", yerr)
        object.__setattr__(self, "style", _frozen_style(self.style))
        object.__setattr__(self, "label", str(self.label))


def _as_series_tuple(series: Iterable[Series] | Series, *, where: str) -> tuple[Series, ...]:
    if isinstance(series, Series):
        return (series,)
    _reject_data_objects(series, where=where)
    items = tuple(series)
    for item in items:
        if not isinstance(item, Series):
            raise TypeError(f"{where} entries must be Series; got {type(item).__name__}")
    return items


@dataclass(frozen=True)
class LineSeries(ViewModel):
    """A set of traces sharing one pair of axes: time histories and generic x-y."""

    series: tuple[Series, ...]
    x_label: str = ""
    y_label: str = ""
    x_unit: str = ""
    y_unit: str = ""
    title: str = ""
    x_limits: tuple[float, float] | None = None
    y_limits: tuple[float, float] | None = None
    log_y: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "series", _as_series_tuple(self.series, where="LineSeries.series")
        )
        for name in ("x_limits", "y_limits"):
            limits = getattr(self, name)
            if limits is not None:
                low, high = (float(limits[0]), float(limits[1]))
                object.__setattr__(self, name, (low, high))


@dataclass(frozen=True)
class Profile1D(ViewModel):
    """One or more 1D profiles sharing a radial coordinate label."""

    series: tuple[Series, ...]
    coordinate_label: str = ""
    y_label: str = ""
    y_unit: str = ""
    title: str = ""
    x_limits: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "series", _as_series_tuple(self.series, where="Profile1D.series")
        )
        if self.x_limits is not None:
            object.__setattr__(
                self, "x_limits", (float(self.x_limits[0]), float(self.x_limits[1]))
            )


@dataclass(frozen=True)
class Field2D(ViewModel):
    """A scalar field sampled on an ``(r, z)`` grid, e.g. a poloidal flux map."""

    r: np.ndarray
    z: np.ndarray
    values: np.ndarray
    value_label: str = ""
    x_label: str = "R [m]"
    y_label: str = "Z [m]"
    title: str = ""
    contour_levels: int | Sequence[float] | None = None
    filled: bool = True
    aspect_equal: bool = True
    overlays: tuple["GeometryLayer", ...] = ()

    def __post_init__(self) -> None:
        r = as_model_array(self.r, where="Field2D.r")
        z = as_model_array(self.z, where="Field2D.z")
        values = as_model_array(self.values, where="Field2D.values")
        if r.ndim != 1 or z.ndim != 1:
            raise ValueError("Field2D.r and Field2D.z must be 1D grid axes")
        if values.shape != (z.size, r.size):
            raise ValueError(
                "Field2D.values must have shape (len(z), len(r)); got "
                f"{values.shape} for len(z)={z.size} and len(r)={r.size}"
            )
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "values", values)
        if self.contour_levels is not None and not isinstance(self.contour_levels, int):
            object.__setattr__(
                self,
                "contour_levels",
                as_model_array(self.contour_levels, where="Field2D.contour_levels"),
            )
        object.__setattr__(self, "overlays", tuple(self.overlays))


@dataclass(frozen=True)
class GeometryLayer(ViewModel):
    """One geometric element drawn in a machine view.

    ``kind`` selects how the coordinates are drawn: ``polyline`` connects the
    points, ``polygon`` additionally closes the outline, and ``points`` draws
    markers only.  Coordinates are named ``r``/``z`` for poloidal views and reused
    as ``x``/``y`` for top views.
    """

    r: np.ndarray
    z: np.ndarray
    kind: str = "polyline"
    label: str = ""
    style: Mapping[str, Any] = field(default_factory=dict)

    KINDS = ("polyline", "polygon", "points")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ValueError(
                f"GeometryLayer.kind must be one of {self.KINDS}; got {self.kind!r}"
            )
        r = as_model_array(self.r, where="GeometryLayer.r")
        z = as_model_array(self.z, where="GeometryLayer.z")
        if r.ndim != 1 or z.ndim != 1 or r.size != z.size:
            raise ValueError(
                "GeometryLayer.r and GeometryLayer.z must be 1D and equal length; "
                f"got shapes {r.shape} and {z.shape}"
            )
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "style", _frozen_style(self.style))
        object.__setattr__(self, "label", str(self.label))


@dataclass(frozen=True)
class GeometryLayers(ViewModel):
    """A stack of geometry layers drawn into one equal-aspect view."""

    layers: tuple[GeometryLayer, ...]
    x_label: str = "R [m]"
    y_label: str = "Z [m]"
    title: str = ""
    aspect_equal: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.layers, GeometryLayer):
            object.__setattr__(self, "layers", (self.layers,))
            return
        _reject_data_objects(self.layers, where="GeometryLayers.layers")
        layers = tuple(self.layers)
        for layer in layers:
            if not isinstance(layer, GeometryLayer):
                raise TypeError(
                    f"GeometryLayers.layers entries must be GeometryLayer; "
                    f"got {type(layer).__name__}"
                )
        object.__setattr__(self, "layers", layers)


@dataclass(frozen=True)
class Spectrogram(ViewModel):
    """Time-frequency magnitude map on a ``(frequency, time)`` grid."""

    time: np.ndarray
    frequency: np.ndarray
    magnitude: np.ndarray
    x_label: str = "Time [s]"
    y_label: str = "Frequency [Hz]"
    value_label: str = "Magnitude"
    title: str = ""
    max_frequency: float | None = None
    cmap: str = "hot_r"

    def __post_init__(self) -> None:
        time = as_model_array(self.time, where="Spectrogram.time")
        frequency = as_model_array(self.frequency, where="Spectrogram.frequency")
        magnitude = as_model_array(self.magnitude, where="Spectrogram.magnitude")
        if time.ndim != 1 or frequency.ndim != 1:
            raise ValueError("Spectrogram.time and Spectrogram.frequency must be 1D")
        if magnitude.shape != (frequency.size, time.size):
            raise ValueError(
                "Spectrogram.magnitude must have shape (len(frequency), len(time)); "
                f"got {magnitude.shape} for len(frequency)={frequency.size} and "
                f"len(time)={time.size}"
            )
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "magnitude", magnitude)
        if self.max_frequency is not None:
            object.__setattr__(self, "max_frequency", float(self.max_frequency))

    @classmethod
    def from_result(cls, result: Any, **overrides: Any) -> "Spectrogram":
        """Build from any object exposing ``time``/``frequency``/``magnitude``.

        This accepts the result of
        :func:`vaft.process.magnetics.compute_mirnov_spectrogram` directly.
        """
        return cls(
            time=result.time,
            frequency=result.frequency,
            magnitude=result.magnitude,
            **overrides,
        )


@dataclass(frozen=True)
class Panels(ViewModel):
    """A grid of view models rendered into one figure.

    Multi-axis renderers stay inside the single renderer contract by taking a
    ``Panels`` model and returning ``(Figure, ndarray[Axes])``.
    """

    models: tuple[ViewModel, ...]
    nrows: int | None = None
    ncols: int = 1
    share_x: bool = True
    share_y: bool = False
    suptitle: str = ""
    squeeze: bool = False

    def __post_init__(self) -> None:
        _reject_data_objects(self.models, where="Panels.models")
        models = tuple(self.models)
        if not models:
            raise ValueError("Panels.models must contain at least one view model")
        for model in models:
            if not isinstance(model, ViewModel) or isinstance(model, Panels):
                raise TypeError(
                    "Panels.models entries must be non-nested view models; got "
                    f"{type(model).__name__}"
                )
        object.__setattr__(self, "models", models)
        ncols = max(1, int(self.ncols))
        nrows = self.nrows
        nrows = -(-len(models) // ncols) if nrows is None else max(1, int(nrows))
        if nrows * ncols < len(models):
            raise ValueError(
                f"Panels grid {nrows}x{ncols} cannot hold {len(models)} models"
            )
        object.__setattr__(self, "nrows", nrows)
        object.__setattr__(self, "ncols", ncols)
