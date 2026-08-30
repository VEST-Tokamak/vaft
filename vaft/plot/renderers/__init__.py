"""Canonical low-level renderers, grouped by view model family.

Importing this package registers every canonical renderer in
:mod:`vaft.plot.registry`.  Renderers are re-exported from :mod:`vaft.plot`, so
application code imports them from there rather than reaching into these modules.
"""

from . import fields, geometry, images, lines, panels, profiles, spectra, spectrograms
from .fields import render_field_2d
from .geometry import draw_geometry_layer, render_geometry_layers
from .images import render_image_2d, render_image_sequence
from .lines import render_line_series
from .panels import render_panels
from .profiles import render_profile_1d
from .spectra import render_power_spectrum
from .spectrograms import render_spectrogram

__all__ = [
    "draw_geometry_layer",
    "fields",
    "geometry",
    "images",
    "lines",
    "panels",
    "profiles",
    "render_field_2d",
    "render_geometry_layers",
    "render_image_2d",
    "render_image_sequence",
    "render_line_series",
    "render_panels",
    "render_power_spectrum",
    "render_profile_1d",
    "render_spectrogram",
    "spectra",
    "spectrograms",
]
