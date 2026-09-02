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

from .display import DisplaySpec

__all__ = [
    "Field2D",
    "Geometry3DLayer",
    "Geometry3DLayers",
    "GeometryLayer",
    "GeometryLayers",
    "Image2D",
    "ImageSequence",
    "LineSeries",
    "Panels",
    "PowerSpectrum",
    "Profile1D",
    "ReferenceSlope",
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
    #: IMAS channel validity code for the whole trace: ``0`` (or ``None``)
    #: means valid, any negative value means the data provider or an automatic
    #: check flagged this channel invalid.  Plotting renders that flag; it
    #: never computes it.
    validity: int | None = None
    #: Per-sample validity: ``True`` where the sample is usable.  ``None`` means
    #: every sample is valid, which is not the same as a channel whose
    #: ``validity`` flag is negative.
    valid_mask: np.ndarray | None = None
    #: Structured identity, kept apart so a legend can say each thing once:
    #: ``entry`` is the shot or collection key the trace came from, ``channel``
    #: the canonical channel label (``[5] (9.1 cm, 4.0 cm)``), ``position`` the
    #: stored (R, Z) in metres.  ``label`` remains the composed fallback for
    #: hand-built models.
    entry: str = ""
    channel: str = ""
    position: tuple[float, float] | None = None
    #: What this trace is relative to its channel: ``""`` for the measurement
    #: itself, ``"reconstruction"`` for an equilibrium's prediction of it,
    #: ``"constraint"`` for the value the solver was given (issue #261).  A
    #: role shares its channel's panel and is named beside it in the legend.
    role: str = ""

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
        if self.valid_mask is not None:
            mask = np.asarray(self.valid_mask, dtype=bool)
            if mask.shape != y.shape:
                raise ValueError(
                    "Series.valid_mask must match y's shape; "
                    f"got {mask.shape} for y of length {y.size}"
                )
            mask.setflags(write=False)
            object.__setattr__(self, "valid_mask", mask)
        if self.validity is not None:
            object.__setattr__(self, "validity", int(self.validity))
        if self.position is not None:
            r, z = self.position
            object.__setattr__(self, "position", (float(r), float(z)))
        object.__setattr__(self, "entry", str(self.entry or ""))
        object.__setattr__(self, "channel", str(self.channel or ""))
        object.__setattr__(self, "role", str(self.role or ""))
        object.__setattr__(self, "style", _frozen_style(self.style))
        object.__setattr__(self, "label", str(self.label))

    @property
    def is_invalid_channel(self) -> bool:
        """Whether the whole trace is flagged invalid (IMAS: a negative code)."""
        return self.validity is not None and self.validity < 0


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
    #: Resolved display policy (unit/scale/notation) the series were built
    #: with; ``None`` for models assembled outside the display layer.
    display: "DisplaySpec | None" = None

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
    #: Resolved display policy (unit/scale/notation); see :class:`LineSeries`.
    display: "DisplaySpec | None" = None

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
class Geometry3DLayer(ViewModel):
    """One polyline or point cloud drawn in machine Cartesian coordinates."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    kind: str = "polyline"
    label: str = ""
    style: Mapping[str, Any] = field(default_factory=dict)

    KINDS = ("polyline", "points")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ValueError(
                f"Geometry3DLayer.kind must be one of {self.KINDS}; got {self.kind!r}"
            )
        x = as_model_array(self.x, where="Geometry3DLayer.x")
        y = as_model_array(self.y, where="Geometry3DLayer.y")
        z = as_model_array(self.z, where="Geometry3DLayer.z")
        if not (x.ndim == y.ndim == z.ndim == 1 and x.size == y.size == z.size):
            raise ValueError(
                "Geometry3DLayer.x/y/z must be 1D and equal length; got shapes "
                f"{x.shape}, {y.shape} and {z.shape}"
            )
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "style", _frozen_style(self.style))
        object.__setattr__(self, "label", str(self.label))


@dataclass(frozen=True)
class Geometry3DLayers(ViewModel):
    """A stack of 3D geometry layers drawn into one machine-coordinate view."""

    layers: tuple[Geometry3DLayer, ...]
    x_label: str = "x [m]"
    y_label: str = "y [m]"
    z_label: str = "z [m]"
    title: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.layers, Geometry3DLayer):
            object.__setattr__(self, "layers", (self.layers,))
            return
        _reject_data_objects(self.layers, where="Geometry3DLayers.layers")
        layers = tuple(self.layers)
        for layer in layers:
            if not isinstance(layer, Geometry3DLayer):
                raise TypeError(
                    f"Geometry3DLayers.layers entries must be Geometry3DLayer; "
                    f"got {type(layer).__name__}"
                )
        object.__setattr__(self, "layers", layers)


@dataclass(frozen=True)
class Image2D(ViewModel):
    """A raster image in pixel space, e.g. one camera frame, with optional overlays.

    Unlike :class:`Field2D` (a scalar field on a physical ``(r, z)`` grid,
    rendered as contours), ``values`` is drawn directly with ``imshow`` --
    the right tool for a dense pixel raster, where a per-pixel contour
    computation would be both far too slow and would destroy the image's own
    detail. ``overlays`` reuses :class:`GeometryLayer`, with its ``r``/``z``
    fields holding pixel *column*/*row* coordinates instead of physical
    machine coordinates -- the same drawing code applies unchanged, since
    :func:`~vaft.plot.renderers.geometry.draw_geometry_layer` never assumes a
    particular coordinate system.
    """

    values: np.ndarray
    value_label: str = ""
    x_label: str = "Column"
    y_label: str = "Row"
    title: str = ""
    cmap: str = "gray"
    vmin: float | None = None
    vmax: float | None = None
    origin: str = "upper"
    aspect_equal: bool = True
    overlays: tuple[GeometryLayer, ...] = ()

    def __post_init__(self) -> None:
        values = as_model_array(self.values, where="Image2D.values")
        if values.ndim != 2:
            raise ValueError(f"Image2D.values must be 2D; got shape {values.shape}")
        object.__setattr__(self, "values", values)
        if self.origin not in ("upper", "lower"):
            raise ValueError(f"Image2D.origin must be 'upper' or 'lower'; got {self.origin!r}")
        if isinstance(self.overlays, GeometryLayer):
            object.__setattr__(self, "overlays", (self.overlays,))
        else:
            _reject_data_objects(self.overlays, where="Image2D.overlays")
            overlays = tuple(self.overlays)
            for layer in overlays:
                if not isinstance(layer, GeometryLayer):
                    raise TypeError(
                        f"Image2D.overlays entries must be GeometryLayer; got {type(layer).__name__}"
                    )
            object.__setattr__(self, "overlays", overlays)
        if self.vmin is not None:
            object.__setattr__(self, "vmin", float(self.vmin))
        if self.vmax is not None:
            object.__setattr__(self, "vmax", float(self.vmax))


@dataclass(frozen=True)
class ImageSequence(ViewModel):
    """A sequence of raster frames sharing one color scale, e.g. an animation.

    ``vmin``/``vmax`` default to the sequence's own overall min/max so every
    frame is drawn on the same intensity scale, keeping frames directly
    comparable -- explicit values override that.
    """

    frames: tuple[np.ndarray, ...]
    time: np.ndarray
    value_label: str = ""
    x_label: str = "Column"
    y_label: str = "Row"
    title: str = ""
    cmap: str = "gray"
    vmin: float | None = None
    vmax: float | None = None
    origin: str = "upper"
    aspect_equal: bool = True

    def __post_init__(self) -> None:
        _reject_data_objects(self.frames, where="ImageSequence.frames")
        frames = tuple(
            as_model_array(frame, where="ImageSequence.frames[i]") for frame in self.frames
        )
        if not frames:
            raise ValueError("ImageSequence.frames must contain at least one frame")
        shape = frames[0].shape
        for frame in frames:
            if frame.ndim != 2:
                raise ValueError(f"ImageSequence frames must be 2D; got shape {frame.shape}")
            if frame.shape != shape:
                raise ValueError("ImageSequence frames must all share the same shape")
        object.__setattr__(self, "frames", frames)

        time = as_model_array(self.time, where="ImageSequence.time")
        if time.ndim != 1 or time.size != len(frames):
            raise ValueError(
                f"ImageSequence.time must be 1D with length {len(frames)}; got shape {time.shape}"
            )
        object.__setattr__(self, "time", time)

        if self.origin not in ("upper", "lower"):
            raise ValueError(f"ImageSequence.origin must be 'upper' or 'lower'; got {self.origin!r}")
        vmin = self.vmin if self.vmin is not None else min(float(f.min()) for f in frames)
        vmax = self.vmax if self.vmax is not None else max(float(f.max()) for f in frames)
        object.__setattr__(self, "vmin", float(vmin))
        object.__setattr__(self, "vmax", float(vmax))


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
class ReferenceSlope:
    """One caller-supplied power-law guide line for a log-log spectrum plot.

    ``slope`` is any number the caller wants drawn.  VAFT ships no slope
    constants and attaches no meaning to any value: whether a guide represents a
    turbulence cascade, an instrument roll-off, or nothing at all is the
    caller's assertion, made in ``label``.

    When ``anchor`` is ``None`` the guide is positioned to pass through the
    measured PSD at the geometric-mean frequency of the drawn range -- a
    deterministic, purely numerical choice.  Pass ``anchor=(frequency, psd)`` to
    place it explicitly, for instance to offset a guide off the data for
    legibility.
    """

    slope: float
    label: str = ""
    anchor: tuple[float, float] | None = None
    style: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "slope", float(self.slope))
        object.__setattr__(self, "label", str(self.label))
        if self.anchor is not None:
            frequency, psd = (float(self.anchor[0]), float(self.anchor[1]))
            if frequency <= 0 or psd <= 0:
                raise ValueError(
                    "ReferenceSlope.anchor must be positive in both coordinates to "
                    f"place a line on log-log axes; got ({frequency}, {psd})"
                )
            object.__setattr__(self, "anchor", (frequency, psd))
        object.__setattr__(self, "style", _frozen_style(self.style))


@dataclass(frozen=True)
class PowerSpectrum(ViewModel):
    """A power spectral density, with optional fitted segments and guide lines.

    ``fits`` are drawn segments the caller has already computed (typically from
    :func:`vaft.process.fluctuation.fit_power_law_spectrum`); ``reference_slopes``
    are comparison guides; ``marker_frequencies`` are labeled vertical lines, for
    a characteristic frequency the caller wants to mark.  All three default to
    empty -- nothing is drawn that the caller did not ask for, and no value is
    given a physical name here.
    """

    frequency: np.ndarray
    psd: np.ndarray
    fits: tuple[Series, ...] = ()
    reference_slopes: tuple[ReferenceSlope, ...] = ()
    marker_frequencies: tuple[tuple[float, str], ...] = ()
    label: str = ""
    x_label: str = "Frequency [Hz]"
    y_label: str = "PSD"
    title: str = ""
    log_x: bool = True
    log_y: bool = True
    x_limits: tuple[float, float] | None = None
    y_limits: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        frequency = as_model_array(self.frequency, where="PowerSpectrum.frequency")
        psd = as_model_array(self.psd, where="PowerSpectrum.psd")
        if frequency.ndim != 1 or psd.ndim != 1:
            raise ValueError(
                "PowerSpectrum.frequency and PowerSpectrum.psd must be 1D; "
                f"got shapes {frequency.shape} and {psd.shape}"
            )
        if frequency.size != psd.size:
            raise ValueError(
                "PowerSpectrum.frequency and PowerSpectrum.psd must have equal "
                f"length; got {frequency.size} and {psd.size}"
            )
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "psd", psd)
        object.__setattr__(self, "fits", _as_series_tuple(self.fits, where="PowerSpectrum.fits"))

        if isinstance(self.reference_slopes, ReferenceSlope):
            slopes: tuple[ReferenceSlope, ...] = (self.reference_slopes,)
        else:
            _reject_data_objects(self.reference_slopes, where="PowerSpectrum.reference_slopes")
            slopes = tuple(
                item if isinstance(item, ReferenceSlope) else ReferenceSlope(slope=item)
                for item in self.reference_slopes
            )
        object.__setattr__(self, "reference_slopes", slopes)

        markers = []
        for entry in self.marker_frequencies:
            if isinstance(entry, (int, float)):
                markers.append((float(entry), ""))
            else:
                markers.append((float(entry[0]), str(entry[1])))
        object.__setattr__(self, "marker_frequencies", tuple(markers))

        for name in ("x_limits", "y_limits"):
            limits = getattr(self, name)
            if limits is not None:
                object.__setattr__(self, name, (float(limits[0]), float(limits[1])))
        object.__setattr__(self, "label", str(self.label))

    @classmethod
    def from_result(cls, result: Any, **overrides: Any) -> "PowerSpectrum":
        """Build from any object exposing ``frequency``/``psd``.

        This accepts a :class:`vaft.process.fluctuation.FluctuationSpectrum`
        directly.  Fitted segments are *not* generated here: pass them through
        ``fits=`` if you want them drawn, so the model never invents a curve the
        caller did not compute.
        """
        overrides.setdefault("y_label", f"PSD [{result.units}]" if getattr(result, "units", "") else "PSD")
        return cls(frequency=result.frequency, psd=result.psd, **overrides)


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
    #: Grid slots that hold a note instead of a model, as ``(slot, text)``.  A
    #: composite with a fixed member list keeps its shape on every input by
    #: rendering an unavailable member as a labelled empty panel rather than
    #: dropping it (issue #260).
    placeholders: tuple[tuple[int, str], ...] = ()
    #: Per-model renderer defaults, one mapping per model, applied beneath the
    #: caller's own keyword arguments -- how an overview asks its members for
    #: ``validity="mask"`` without forcing that on every individual plot.
    member_styles: tuple[Mapping[str, Any], ...] | None = None

    def __post_init__(self) -> None:
        _reject_data_objects(self.models, where="Panels.models")
        models = tuple(self.models)
        placeholders = tuple((int(slot), str(text)) for slot, text in self.placeholders)
        object.__setattr__(self, "placeholders", placeholders)
        if self.member_styles is not None:
            styles = tuple(_frozen_style(s) for s in self.member_styles)
            if len(styles) != len(models):
                raise ValueError(
                    f"Panels.member_styles has {len(styles)} entries for {len(models)} models"
                )
            object.__setattr__(self, "member_styles", styles)
        if not models and not placeholders:
            raise ValueError("Panels.models must contain at least one view model")
        for model in models:
            if not isinstance(model, ViewModel) or isinstance(model, Panels):
                raise TypeError(
                    "Panels.models entries must be non-nested view models; got "
                    f"{type(model).__name__}"
                )
        object.__setattr__(self, "models", models)
        ncols = max(1, int(self.ncols))
        occupied = len(models) + len(placeholders)
        nrows = self.nrows
        nrows = -(-occupied // ncols) if nrows is None else max(1, int(nrows))
        if nrows * ncols < occupied:
            raise ValueError(
                f"Panels grid {nrows}x{ncols} cannot hold {occupied} panels"
            )
        if any(slot >= nrows * ncols for slot, _ in placeholders):
            raise ValueError("Panels.placeholders names a slot outside the grid")
        object.__setattr__(self, "nrows", nrows)
        object.__setattr__(self, "ncols", ncols)
