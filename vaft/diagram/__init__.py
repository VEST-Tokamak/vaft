"""Reproducible scientific schematics (issue #890).

``vaft.diagram`` explains concepts -- topology, coordinates, workflows --
where :mod:`vaft.plot` shows data and results. The boundary:

``vaft.formula``
    owns the physics: every equation a diagram depends on is a formula
    function, called here rather than restated.
``vaft.diagram``
    owns explanatory geometry: sampling, projection, camera, annotation.
``vaft.plot``
    owns data and numerical results.

Diagrams: ``magnetic_island`` (poloidal, top and 3-D projections of one
island model) and the stability / operational-space charts
``peeling_ballooning`` (schematic), ``s_alpha_ballooning``, ``hugill`` and
``troyon``.

A builder returns a :class:`Diagram`, which holds the TikZ source at once
and renders it to SVG -- the canonical artifact -- on first request (inline
in Jupyter through ``_repr_svg_``). Rendering needs ``latex`` and
``dvisvgm``; importing this package and building a diagram do not.

The committed reference SVGs are regenerated and checked with::

    python -m vaft.diagram.build          # render what changed
    python -m vaft.diagram.build --check  # verify, needs no TeX
"""

from importlib import import_module

__all__ = [
    "magnetic_island",
    "peeling_ballooning",
    "s_alpha_ballooning",
    "hugill",
    "troyon",
    "Diagram",
    "DiagramToolchainError",
]

_LOCATIONS = {
    "magnetic_island": "._magnetic_island",
    "peeling_ballooning": "._stability_space",
    "s_alpha_ballooning": "._stability_space",
    "hugill": "._stability_space",
    "troyon": "._stability_space",
    "Diagram": "._render",
    "DiagramToolchainError": "._render",
}


def __getattr__(name: str):
    if name in _LOCATIONS:
        value = getattr(import_module(_LOCATIONS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
