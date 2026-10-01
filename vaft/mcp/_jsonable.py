"""Bounded, JSON-safe conversion of VAFT result objects for the MCP adapter (#1423).

Discovery objects (catalog specs, plot capabilities, boundaries) and extracted
view models are frozen dataclasses holding tuples, read-only mappings and
NumPy arrays.  :class:`Bounded` turns any of them into plain ``dict``/``list``/
scalar JSON, dropping callables, replacing non-finite floats by ``None`` and
summarising every array by its shape, dtype, finite range and a strided
preview of at most ``max_points`` values.  Every place where it shortened
something is recorded in :attr:`Bounded.truncated`, so a tool result says
what it left out instead of silently looking complete.

The strided preview is a transport bound, not a resampling: the values are
``array[::stride]`` exactly, without filtering, and ``min``/``max`` are taken
over the whole array.  Standard library and NumPy only.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

__all__ = ["Bounded", "array_summary", "preview_strides"]


def preview_strides(shape: tuple[int, ...], max_points: int) -> tuple[int, ...]:
    """Per-axis strides keeping a strided preview of ``shape`` at or under ``max_points``."""
    if not shape:
        return ()
    budget = max(1, int(max_points))
    per_axis = max(1, int(math.floor(budget ** (1.0 / len(shape)))))
    return tuple(max(1, math.ceil(n / per_axis)) for n in shape)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _plain_values(array) -> Any:
    """``array.tolist()`` with NaN/inf mapped to ``None``."""
    values = array.tolist()

    def clean(item):
        if isinstance(item, list):
            return [clean(i) for i in item]
        if isinstance(item, float):
            return _finite_or_none(item)
        if isinstance(item, complex):
            return [_finite_or_none(item.real), _finite_or_none(item.imag)]
        if isinstance(item, (bytes, bytearray)):
            return item.decode("utf-8", "replace")
        if item is None or isinstance(item, (bool, int, str)):
            return item
        return str(item)

    return clean(values)


def array_summary(array, max_points: int) -> dict[str, Any]:
    """Shape, dtype, finite range and a strided preview of one array."""
    import numpy as np

    array = np.asarray(array)
    strides = preview_strides(array.shape, max_points)
    preview = array[tuple(slice(None, None, s) for s in strides)] if array.ndim else array
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
    }
    if array.dtype.kind in "biuf" and array.size:
        numeric = array.astype(float)
        finite = np.isfinite(numeric)
        summary["finite_count"] = int(finite.sum())
        summary["min"] = float(numeric[finite].min()) if finite.any() else None
        summary["max"] = float(numeric[finite].max()) if finite.any() else None
    summary["stride"] = list(strides)
    summary["preview_shape"] = list(np.shape(preview))
    summary["truncated"] = any(s > 1 for s in strides)
    summary["values"] = _plain_values(preview)
    return summary


class Bounded:
    """Convert one result to JSON-safe data, bounded in sequence length and points.

    Parameters
    ----------
    max_items : int
        Longest list kept from any tuple/list/set; the rest is dropped and the
        path recorded in :attr:`truncated` [-].
    max_points : int
        Largest strided preview kept from any array [-].
    drop_keys : collection of str
        Dictionary/dataclass keys left out wherever they occur [-].
    """

    def __init__(self, *, max_items: int = 200, max_points: int = 200, drop_keys=()) -> None:
        self.max_items = max(1, int(max_items))
        self.max_points = max(1, int(max_points))
        self.drop_keys = frozenset(drop_keys)
        self.truncated: list[str] = []

    def __call__(self, value: Any, path: str = "$") -> Any:
        return self._convert(value, path)

    def _convert(self, value: Any, path: str) -> Any:
        import numpy as np

        if value is None or isinstance(value, (bool, str)):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, float):
            return _finite_or_none(value)
        if isinstance(value, enum.Enum):
            return self._convert(value.value, path)
        if isinstance(value, np.ndarray):
            summary = array_summary(value, self.max_points)
            if summary["truncated"]:
                self.truncated.append(path)
            return summary
        if isinstance(value, np.generic):
            return self._convert(value.item(), path)
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", "replace")
        if isinstance(value, PurePath):
            return str(value)
        if isinstance(value, Mapping):
            return {
                str(key): self._convert(item, f"{path}.{key}")
                for key, item in value.items()
                if str(key) not in self.drop_keys and not _is_code(item)
            }
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: self._convert(getattr(value, field.name), f"{path}.{field.name}")
                for field in dataclasses.fields(value)
                if field.name not in self.drop_keys and not _is_code(getattr(value, field.name))
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            items = list(value)
            if isinstance(value, (set, frozenset)):
                items = sorted(items, key=str)
            if len(items) > self.max_items:
                self.truncated.append(f"{path} ({len(items)} items, kept {self.max_items})")
                items = items[: self.max_items]
            return [self._convert(item, f"{path}[{index}]") for index, item in enumerate(items)]
        if _is_code(value):
            return None
        return str(value)


def _is_code(value: Any) -> bool:
    """Functions, classes and modules carry behaviour, not data: never serialised."""
    import types

    return isinstance(value, (type, types.ModuleType)) or (
        callable(value) and not dataclasses.is_dataclass(value)
    )
