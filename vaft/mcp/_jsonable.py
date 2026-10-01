"""Bounded, JSON-safe conversion of VAFT result objects for the MCP adapter (#1423).

Discovery objects (catalog specs, plot capabilities, boundaries) and extracted
view models are frozen dataclasses holding tuples, read-only mappings and
NumPy arrays.  :class:`Bounded` turns any of them into plain ``dict``/``list``/
scalar JSON: callables are dropped, non-finite floats become ``None``, lists,
mappings and strings are capped, and every array is summarised by its shape,
dtype and finite range plus a strided preview.

``max_points`` is a budget for the *whole* result: it is shared out across
the arrays it contains, so a 300-array overview gets a few points per array
(or statistics only) where a single profile gets them all.  :func:`bounded_json`
additionally holds the serialised result under a byte cap, falling back to
statistics-only arrays and then to shorter lists.  Every place something was
shortened is recorded and reported by :meth:`Bounded.report`.

The strided preview is a transport bound, not a resampling: the values are
``array[::stride]`` exactly, without filtering, and ``min``/``max`` are taken
over the whole array.  Standard library and NumPy only.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

__all__ = ["Bounded", "array_summary", "bounded_json", "count_arrays", "preview_strides"]

#: How many truncation paths a result lists before it only counts them.
REPORTED_PATHS = 20


def preview_strides(shape: tuple[int, ...], max_points: int) -> tuple[int, ...]:
    """Per-axis strides keeping a strided preview of ``shape`` at or under ``max_points``."""
    if not shape:
        return ()
    budget = max(1, int(max_points))
    per_axis = max(1, int(math.floor(budget ** (1.0 / len(shape)) + 1e-9)))
    return tuple(max(1, math.ceil(n / per_axis)) for n in shape)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _plain_values(array) -> Any:
    """``array.tolist()`` with NaN/inf mapped to ``None``."""

    def clean(item):
        if isinstance(item, list):
            return [clean(i) for i in item]
        if isinstance(item, float):
            return _finite_or_none(item)
        if isinstance(item, complex):
            return [_finite_or_none(item.real), _finite_or_none(item.imag)]
        if isinstance(item, (bytes, bytearray)):
            return item.decode("utf-8", "replace")[:200]
        if item is None or isinstance(item, (bool, int)):
            return item
        return str(item)[:200]

    return clean(array.tolist())


def array_summary(array, max_points: int) -> dict[str, Any]:
    """Shape, dtype, finite range and a strided preview of one array.

    ``max_points`` 0 gives the statistics without any values.
    """
    import numpy as np

    array = np.asarray(array)
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
    if max_points <= 0 and array.size > 1:
        summary["stride"] = None
        summary["truncated"] = True
        return summary
    strides = preview_strides(array.shape, max(1, max_points))
    preview = array[tuple(slice(None, None, s) for s in strides)] if array.ndim else array
    summary["stride"] = list(strides)
    summary["preview_shape"] = list(np.shape(preview))
    summary["truncated"] = any(s > 1 for s in strides)
    summary["values"] = _plain_values(preview)
    return summary


def _is_code(value: Any) -> bool:
    """Functions, classes and modules carry behaviour, not data: never serialised."""
    import types

    return isinstance(value, (type, types.ModuleType)) or (
        callable(value) and not dataclasses.is_dataclass(value)
    )


def _children(value: Any):
    if isinstance(value, Mapping):
        return list(value.values())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [getattr(value, f.name) for f in dataclasses.fields(value)]
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return []


def count_arrays(value: Any, max_items: int | None = None, _depth: int = 0) -> int:
    """How many arrays of more than one element ``value`` holds, within ``max_items`` per list."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return int(value.size > 1)
    if _depth > 50 or isinstance(value, (str, bytes)):
        return 0
    children = _children(value)
    if max_items is not None and isinstance(value, (list, tuple, set, frozenset)):
        children = children[:max_items]
    return sum(count_arrays(child, max_items, _depth + 1) for child in children)


class Bounded:
    """Convert one result to JSON-safe data, bounded in size.

    Parameters
    ----------
    max_items : int
        Longest list kept from any tuple/list/set [-].
    max_points : int
        Points per array preview; see :meth:`for_budget` for a whole-result budget [-].
    max_keys : int
        Most keys kept from any mapping [-].
    max_string : int
        Longest string kept, in characters [-].
    drop_keys : collection of str
        Dictionary/dataclass keys left out wherever they occur [-].
    """

    def __init__(self, *, max_items: int = 200, max_points: int = 200, max_keys: int = 500,
                 max_string: int = 20_000, drop_keys=()) -> None:
        self.max_items = max(1, int(max_items))
        self.max_points = max(0, int(max_points))
        self.max_keys = max(1, int(max_keys))
        self.max_string = max(1, int(max_string))
        self.drop_keys = frozenset(drop_keys)
        self.truncated: list[str] = []

    @classmethod
    def for_budget(cls, value: Any, budget: int, **kwargs: Any) -> "Bounded":
        """A converter sharing ``budget`` points across every array in ``value``."""
        max_items = kwargs.get("max_items", 200)
        arrays = max(1, count_arrays(value, max_items))
        return cls(max_points=int(budget) // arrays, **kwargs)

    def report(self) -> dict[str, Any]:
        """What was shortened: the count and the first :data:`REPORTED_PATHS` places."""
        return {"count": len(self.truncated), "paths": self.truncated[:REPORTED_PATHS]}

    def __call__(self, value: Any, path: str = "$") -> Any:
        return self._convert(value, path)

    def _string(self, text: str, path: str) -> str:
        if len(text) > self.max_string:
            self.truncated.append(f"{path} ({len(text)} chars, kept {self.max_string})")
            return text[: self.max_string]
        return text

    def _pairs(self, pairs, path: str) -> dict[str, Any]:
        pairs = [(str(k), v) for k, v in pairs if str(k) not in self.drop_keys and not _is_code(v)]
        if len(pairs) > self.max_keys:
            self.truncated.append(f"{path} ({len(pairs)} keys, kept {self.max_keys})")
            pairs = pairs[: self.max_keys]
        out: dict[str, Any] = {}
        for key, item in pairs:
            name, suffix = key, 1
            while name in out:  # 1 and "1" both stringify to "1"
                suffix += 1
                name = f"{key}#{suffix}"
            out[name] = self._convert(item, f"{path}.{key}")
        return out

    def _convert(self, value: Any, path: str) -> Any:
        import numpy as np

        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str):
            return self._string(value, path)
        if isinstance(value, int):
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
            return self._string(value.decode("utf-8", "replace"), path)
        if isinstance(value, PurePath):
            return str(value)
        if isinstance(value, Mapping):
            return self._pairs(value.items(), path)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return self._pairs(((f.name, getattr(value, f.name)) for f in dataclasses.fields(value)), path)
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
        return self._string(str(value), path)


def _size(data: Any) -> int:
    return len(json.dumps(data, allow_nan=False))


def bounded_json(value: Any, *, budget: int, max_bytes: int, max_items: int = 50,
                 **kwargs: Any) -> tuple[Any, Bounded]:
    """``value`` converted with a whole-result point ``budget`` and held under ``max_bytes``.

    First the budget is shared across the arrays.  If the JSON is still larger
    than ``max_bytes``, arrays are reduced to their statistics; if even that is
    too large, lists (and mappings, down to ten keys) are cut shorter until it
    fits or a list is down to one item.
    """
    converter = Bounded.for_budget(value, budget, max_items=max_items, **kwargs)
    data = converter(value)
    if _size(data) <= max_bytes:
        return data, converter
    items = max_items
    while True:
        converter = Bounded(max_points=0, max_items=items, **{"max_keys": max(10, items), **kwargs})
        data = converter(value)
        if _size(data) <= max_bytes or items == 1:
            converter.truncated.insert(0, f"$ (over {max_bytes} bytes: arrays reduced to statistics)")
            return data, converter
        items = max(1, items // 2)
