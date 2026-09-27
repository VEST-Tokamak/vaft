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
``troyon``; single-particle motion: ``exb_drift``, ``curvature_drift``,
``magnetization_current`` and ``toroidal_drift``; tearing physics upstream
of the island: ``rational_surface``, ``delta_prime`` and
``tearing_layer_matching``; 3-D perturbation harmonics:
``normal_field_component``, ``complex_harmonic``, ``toroidal_harmonic_phase``,
``harmonic_real_space_projection`` and ``complex_field_superposition``;
the classification ``collision_processes``.

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
    "exb_drift",
    "curvature_drift",
    "magnetization_current",
    "toroidal_drift",
    "rational_surface",
    "delta_prime",
    "tearing_layer_matching",
    "normal_field_component",
    "complex_harmonic",
    "toroidal_harmonic_phase",
    "harmonic_real_space_projection",
    "complex_field_superposition",
    "collision_processes",
    "Diagram",
    "DiagramToolchainError",
]

_LOCATIONS = {
    "magnetic_island": "._magnetic_island",
    "peeling_ballooning": "._stability_space",
    "s_alpha_ballooning": "._stability_space",
    "hugill": "._stability_space",
    "troyon": "._stability_space",
    "exb_drift": "._particle_motion",
    "curvature_drift": "._particle_motion",
    "magnetization_current": "._particle_motion",
    "toroidal_drift": "._particle_motion",
    "rational_surface": "._tearing",
    "delta_prime": "._tearing",
    "tearing_layer_matching": "._tearing",
    "normal_field_component": "._harmonic",
    "complex_harmonic": "._harmonic",
    "toroidal_harmonic_phase": "._harmonic",
    "harmonic_real_space_projection": "._harmonic",
    "complex_field_superposition": "._harmonic",
    "collision_processes": "._collision",
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
