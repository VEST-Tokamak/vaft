"""The view models as :class:`xarray.Dataset` objects (umbrella #434).

Each ``ViewModel.to_xarray()`` delegates here.  The rule is *lossless and
plain*: every array of the model comes back as a variable or coordinate,
every scalar fact as an attribute, and the attributes are restricted to what
netCDF can store (strings and numbers; a mapping or a list is JSON text), so a
dataset can be written to disk and read back by anything.  A symmetric
``yerr`` is stored as its two bounds with ``yerr_symmetric`` set, so the
model's own shape is recoverable.

Traces of unequal length -- the channels of one diagnostic, or the same
quantity from two shots -- are stacked on a ``series`` dimension and padded
with ``NaN`` along ``sample``; the ``length`` coordinate holds each trace's
true sample count, so ``ds.y[k, :ds.length[k]]`` is exactly the model's array
even where the data itself carries ``NaN``.  Nothing here imports Matplotlib;
``xarray`` is imported on first use so ``import vaft.plot`` stays light.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

__all__ = ["dataset_attrs"]


def _xr():
    import xarray as xr

    return xr


# ---------------------------------------------------------------------------
# attributes
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """``value`` as something a netCDF attribute holds: str, int, float, or JSON."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, Mapping):
        return json.dumps({str(k): _jsonable(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return json.dumps([_jsonable(v) for v in value])
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def dataset_attrs(model: Any, *fields: str, **extra: Any) -> dict[str, Any]:
    """The attributes every dataset carries, plus the named model fields.

    ``model`` is the class name; a ``display`` field becomes
    ``display_unit``/``display_scale``/``display_notation``/``display_quantity``;
    ``extra`` is the caller's passthrough (``dd_paths``, ``plot_name``).
    """
    from vaft import __version__

    attrs: dict[str, Any] = {"model": type(model).__name__, "vaft_version": str(__version__)}
    for name in fields:
        attrs[name] = _plain(getattr(model, name))
    display = getattr(model, "display", None)
    attrs["display_quantity"] = _plain(getattr(display, "quantity", ""))
    attrs["display_unit"] = _plain(getattr(display, "unit", ""))
    attrs["display_scale"] = _plain(getattr(display, "scale", 1.0) if display is not None else "")
    attrs["display_notation"] = _plain(getattr(display, "notation", ""))
    for key, value in extra.items():
        attrs[key] = _plain(value)
    return attrs


# ---------------------------------------------------------------------------
# padding
# ---------------------------------------------------------------------------


def _padded(
    arrays: Sequence[np.ndarray], *, dtype: Any = float, fill: Any = np.nan, width: int | None = None,
) -> np.ndarray:
    """``arrays`` stacked on a leading axis, padded to the longest (or ``width``) with ``fill``."""
    if width is None:
        width = max((a.shape[-1] for a in arrays), default=0)
    lead = arrays[0].shape[:-1] if arrays else ()
    out = np.full((len(arrays), *lead, width), fill, dtype=dtype)
    for k, array in enumerate(arrays):
        out[k, ..., : array.shape[-1]] = array
    return out


# ---------------------------------------------------------------------------
# series models
# ---------------------------------------------------------------------------


def _series_dataset(series: Sequence[Any]) -> "xr.Dataset":
    """The ``(series, sample)`` layout shared by :class:`LineSeries` and :class:`Profile1D`."""
    xr = _xr()
    xs, ys, errs, masks, symmetric = [], [], [], [], []
    for trace in series:
        xs.append(np.asarray(trace.x, dtype=float))
        ys.append(np.asarray(trace.y, dtype=float))
        yerr = trace.yerr
        if yerr is not None:
            yerr = np.asarray(yerr, dtype=float)
            symmetric.append(yerr.ndim == 1)
            errs.append(np.vstack([yerr, yerr]) if yerr.ndim == 1 else yerr)
        else:
            symmetric.append(False)
            errs.append(None)
        masks.append(None if trace.valid_mask is None else np.asarray(trace.valid_mask, dtype=bool))
    lengths = np.array([y.size for y in ys], dtype=int)
    width = int(lengths.max()) if lengths.size else 0
    data_vars: dict[str, Any] = {
        "x": (("series", "sample"), _padded(xs)),
        "y": (("series", "sample"), _padded(ys)),
    }
    if any(e is not None for e in errs):
        stacked = [e if e is not None else np.full((2, 0), np.nan) for e in errs]
        data_vars["yerr"] = (("series", "bound", "sample"), _padded(stacked, width=width))
    if any(m is not None for m in masks):
        stacked = [m if m is not None else np.zeros(0, dtype=bool) for m in masks]
        data_vars["valid_mask"] = (("series", "sample"), _padded(stacked, dtype=bool, fill=False, width=width))
    coords: dict[str, Any] = {
        "sample": np.arange(width),
        "label": ("series", np.array([t.label for t in series], dtype=object)),
        "entry": ("series", np.array([t.entry for t in series], dtype=object)),
        "channel": ("series", np.array([t.channel for t in series], dtype=object)),
        "role": ("series", np.array([t.role for t in series], dtype=object)),
        "index": ("series", np.array([-1 if t.index is None else t.index for t in series], dtype=int)),
        "validity": ("series", np.array([np.nan if t.validity is None else t.validity for t in series], dtype=float)),
        "position_r": ("series", np.array([np.nan if t.position is None else t.position[0] for t in series], dtype=float)),
        "position_z": ("series", np.array([np.nan if t.position is None else t.position[1] for t in series], dtype=float)),
        "length": ("series", lengths),
        "has_yerr": ("series", np.array([e is not None for e in errs], dtype=bool)),
        "yerr_symmetric": ("series", np.array(symmetric, dtype=bool)),
        "has_valid_mask": ("series", np.array([m is not None for m in masks], dtype=bool)),
        "series_style": ("series", np.array([_plain(dict(t.style)) for t in series], dtype=object)),
    }
    if "yerr" in data_vars:
        coords["bound"] = np.array(["lower", "upper"], dtype=object)
    return xr.Dataset(data_vars, coords=coords)


def line_series_dataset(model: Any, **extra: Any) -> "xr.Dataset":
    ds = _series_dataset(model.series)
    ds.attrs.update(dataset_attrs(
        model, "x_label", "y_label", "x_unit", "y_unit", "title", "x_limits", "y_limits", "log_y", **extra,
    ))
    return ds


def profile_dataset(model: Any, **extra: Any) -> "xr.Dataset":
    ds = _series_dataset(model.series)
    ds.attrs.update(dataset_attrs(model, "coordinate_label", "y_label", "y_unit", "title", "x_limits", **extra))
    ds.attrs["reference_lines"] = _plain(
        [{"x": line.x, "label": line.label, "style": dict(line.style)} for line in model.reference_lines]
    )
    return ds


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _layer_variables(layers: Sequence[Any], axes: Sequence[str], *, prefix: str = "") -> tuple[dict, dict]:
    """Padded ``(layer, point)`` arrays for each axis name, plus the layer coordinates."""
    dim = f"{prefix}layer"
    point = f"{prefix}point"
    data_vars = {
        f"{prefix}{axis}": ((dim, point), _padded([np.asarray(getattr(layer, axis), dtype=float) for layer in layers]))
        for axis in axes
    }
    coords = {
        f"{prefix}kind": (dim, np.array([layer.kind for layer in layers], dtype=object)),
        f"{prefix}label": (dim, np.array([layer.label for layer in layers], dtype=object)),
        f"{prefix}entry": (dim, np.array([getattr(layer, "entry", "") for layer in layers], dtype=object)),
        f"{prefix}role": (dim, np.array([getattr(layer, "role", "") for layer in layers], dtype=object)),
        f"{prefix}length": (dim, np.array([np.asarray(getattr(layer, axes[0])).size for layer in layers], dtype=int)),
        f"{prefix}layer_style": (dim, np.array([_plain(dict(layer.style)) for layer in layers], dtype=object)),
    }
    return data_vars, coords


def geometry_layers_dataset(model: Any, **extra: Any) -> "xr.Dataset":
    xr = _xr()
    data_vars, coords = _layer_variables(model.layers, ("r", "z"))
    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs.update(dataset_attrs(model, "x_label", "y_label", "title", "aspect_equal", "legend", **extra))
    return ds


def geometry_3d_layers_dataset(model: Any, **extra: Any) -> "xr.Dataset":
    xr = _xr()
    data_vars, coords = _layer_variables(model.layers, ("x", "y", "z"))
    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs.update(dataset_attrs(model, "x_label", "y_label", "z_label", "title", **extra))
    return ds


def _with_overlays(ds: "xr.Dataset", overlays: Sequence[Any]) -> "xr.Dataset":
    if not overlays:
        return ds
    data_vars, coords = _layer_variables(overlays, ("r", "z"), prefix="overlay_")
    return ds.assign_coords(coords).assign(data_vars)


# ---------------------------------------------------------------------------
# grids, images, spectra, text
# ---------------------------------------------------------------------------


def field_dataset(model: Any, **extra: Any) -> "xr.Dataset":
    xr = _xr()
    data_vars: dict[str, Any] = {"values": (("z", "r"), np.asarray(model.values, dtype=float))}
    if model.region is not None:
        data_vars["region"] = (("z", "r"), np.asarray(model.region, dtype=bool))
    ds = xr.Dataset(data_vars, coords={"r": np.asarray(model.r, dtype=float), "z": np.asarray(model.z, dtype=float)})
    ds = _with_overlays(ds, model.overlays)
    ds.attrs.update(dataset_attrs(
        model, "value_label", "x_label", "y_label", "title", "contour_levels", "filled",
        "aspect_equal", "secondary_levels", "colorbar", "extend", **extra,
    ))
    return ds


def image_dataset(model: Any, **extra: Any) -> "xr.Dataset":
    xr = _xr()
    ds = xr.Dataset({"values": (("row", "column"), np.asarray(model.values, dtype=float))})
    ds = _with_overlays(ds, model.overlays)
    ds.attrs.update(dataset_attrs(
        model, "value_label", "x_label", "y_label", "title", "cmap", "vmin", "vmax", "origin", "aspect_equal", **extra,
    ))
    return ds


def image_sequence_dataset(model: Any, **extra: Any) -> "xr.Dataset":
    xr = _xr()
    frames = np.stack([np.asarray(frame, dtype=float) for frame in model.frames])
    ds = xr.Dataset(
        {"frames": (("time", "row", "column"), frames)},
        coords={"time": np.asarray(model.time, dtype=float)},
    )
    ds.attrs.update(dataset_attrs(
        model, "value_label", "x_label", "y_label", "title", "cmap", "vmin", "vmax", "origin", "aspect_equal", **extra,
    ))
    return ds


def spectrogram_dataset(model: Any, **extra: Any) -> "xr.Dataset":
    xr = _xr()
    ds = xr.Dataset(
        {"magnitude": (("frequency", "time"), np.asarray(model.magnitude, dtype=float))},
        coords={
            "time": np.asarray(model.time, dtype=float),
            "frequency": np.asarray(model.frequency, dtype=float),
        },
    )
    ds.attrs.update(dataset_attrs(model, "x_label", "y_label", "value_label", "title", "max_frequency", "cmap", **extra))
    return ds


def power_spectrum_dataset(model: Any, **extra: Any) -> "xr.Dataset":
    xr = _xr()
    ds = xr.Dataset(
        {"psd": (("frequency",), np.asarray(model.psd, dtype=float))},
        coords={"frequency": np.asarray(model.frequency, dtype=float)},
    )
    if model.fits:
        fits = model.fits
        ds = ds.assign_coords({
            "fit_label": ("fit", np.array([f.label for f in fits], dtype=object)),
            "fit_length": ("fit", np.array([np.asarray(f.y).size for f in fits], dtype=int)),
            "fit_style": ("fit", np.array([_plain(dict(f.style)) for f in fits], dtype=object)),
        }).assign({
            "fit_frequency": (("fit", "fit_sample"), _padded([np.asarray(f.x, dtype=float) for f in fits])),
            "fit_psd": (("fit", "fit_sample"), _padded([np.asarray(f.y, dtype=float) for f in fits])),
        })
    ds.attrs.update(dataset_attrs(
        model, "label", "x_label", "y_label", "title", "log_x", "log_y", "x_limits", "y_limits", **extra,
    ))
    ds.attrs["reference_slopes"] = _plain([
        {"slope": s.slope, "label": s.label, "anchor": s.anchor, "style": dict(s.style)}
        for s in model.reference_slopes
    ])
    ds.attrs["marker_frequencies"] = _plain([list(marker) for marker in model.marker_frequencies])
    return ds


def text_panel_dataset(model: Any, **extra: Any) -> "xr.Dataset":
    xr = _xr()
    ds = xr.Dataset(coords={"line": np.array(model.lines, dtype=object)})
    ds.attrs.update(dataset_attrs(model, "title", **extra))
    return ds


def panels_datatree(model: Any, **extra: Any) -> Any:
    """A :class:`xarray.DataTree` with one child dataset per panel, in panel order."""
    xr = _xr()
    if not hasattr(xr, "DataTree"):  # pragma: no cover - depends on the installed xarray
        raise ImportError("Panels.to_xarray() needs xarray >= 2024.10 for xarray.DataTree")
    children = {
        f"{position:02d}_{type(member).__name__}": member.to_xarray()
        for position, member in enumerate(model.models)
    }
    attrs = dataset_attrs(
        model, "nrows", "ncols", "share_x", "share_y", "suptitle", "squeeze", "spans", **extra,
    )
    attrs["member_styles"] = _plain(
        [] if model.member_styles is None else [dict(style) for style in model.member_styles]
    )
    root = xr.Dataset(attrs=attrs)
    return xr.DataTree.from_dict({"/": root, **{f"/{name}": ds for name, ds in children.items()}})
