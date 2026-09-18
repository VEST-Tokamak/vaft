"""Renderer-neutral drawing primitives.

A diagram builder turns physics into a :class:`Scene`: a flat list of
polylines, markers, arrows and labels in centimetres, each tagged with a
``style`` (how it looks, resolved by the renderer's template) and a ``role``
(what it means, which is what the topology tests query). Nothing here knows
about TikZ, so a second renderer consumes the same scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Tuple, Union

import numpy as np

Point = Tuple[float, float]


def _points(xy) -> Tuple[Point, ...]:
    arr = np.asarray(xy, dtype=float).reshape(-1, 2)
    return tuple((float(x), float(y)) for x, y in arr)


@dataclass(frozen=True)
class Polyline:
    """An open or closed polyline; dense sampling stands in for curves."""

    points: Tuple[Point, ...]
    style: str
    role: str = ""
    closed: bool = False

    @classmethod
    def of(cls, xy, style: str, role: str = "", closed: bool = False) -> "Polyline":
        return cls(_points(xy), style, role, closed)


@dataclass(frozen=True)
class Marker:
    """A point feature: ``kind`` is ``"o"`` (filled dot) or ``"x"`` (cross)."""

    at: Point
    kind: str
    style: str
    role: str = ""


@dataclass(frozen=True)
class Arrow:
    """A straight arrow; ``both`` puts heads on both ends."""

    start: Point
    end: Point
    style: str
    role: str = ""
    both: bool = False


@dataclass(frozen=True)
class Label:
    """Text set by the renderer; ``text`` may contain inline LaTeX math."""

    at: Point
    text: str
    style: str = "label"
    anchor: str = "center"
    role: str = ""


Item = Union[Polyline, Marker, Arrow, Label]


def _map(item: Item, f) -> Item:
    if isinstance(item, Polyline):
        return replace(item, points=tuple(f(p) for p in item.points))
    if isinstance(item, Marker):
        return replace(item, at=f(item.at))
    if isinstance(item, Arrow):
        return replace(item, start=f(item.start), end=f(item.end))
    return replace(item, at=f(item.at))


@dataclass(frozen=True)
class Scene:
    """An ordered drawing: later items paint over earlier ones."""

    items: Tuple[Item, ...] = field(default_factory=tuple)

    def __add__(self, other: "Scene") -> "Scene":
        return Scene(self.items + other.items)

    def role(self, role: str) -> Tuple[Item, ...]:
        """Every item carrying ``role``, in drawing order."""
        return tuple(item for item in self.items if item.role == role)

    def transformed(self, scale: float = 1.0, offset: Point = (0.0, 0.0)) -> "Scene":
        """The same scene scaled about the origin, then shifted."""
        dx, dy = offset

        def f(p):
            return (scale * p[0] + dx, scale * p[1] + dy)

        return Scene(tuple(_map(item, f) for item in self.items))
