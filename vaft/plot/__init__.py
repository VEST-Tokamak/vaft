"""Low-level plotting for VAFT.

``vaft.plot`` owns rendering and nothing else.  Its canonical renderers are named

.. code-block:: text

    vaft.plot.<domain>_<view>_<quantity>

for example :func:`magnetics_time_ip`, :func:`equilibrium_profile_pressure`, and
:func:`soft_x_rays_spectrogram`.

``<domain>`` is the IDS root the plot belongs to -- ``magnetics``,
``equilibrium``, ``pf_active``, ``pf_passive``, ``tf``, ``wall``,
``soft_x_rays``, ``core_profiles``, ``thomson_scattering``, ``charge_exchange``,
``barometry``, ``spectrometer_uv``, ``camera_visible`` -- plus two composed
domains, ``machine`` for cross-IDS machine views and ``summary`` for cross-IDS
summary panels.  ``<view>`` is one of ``time``, ``profile``, ``field``,
``geometry``, ``spectrogram``, ``overview``, ``image``, ``animation``.
``<quantity>`` may be dropped when the domain and view are already
unambiguous, as in ``soft_x_rays_spectrogram``.

There is no redundant ``plot_`` prefix here; adapter layers and object methods
use ``plot_<canonical-stem>``, so ``vaft.plot.magnetics_time_ip`` is rendered
from an ODS by ``vaft.omas.plot_magnetics_time_ip``.

The radial coordinate is *not* part of a profile renderer's name.  One
:func:`equilibrium_profile_q` serves every coordinate; pick one with the
adapter's ``coordinate=`` argument (``rho_tor_norm``, ``psi_norm``, ``r_major``,
``r_minor``).

The renderer contract
---------------------

Every canonical renderer has this shape::

    def magnetics_time_ip(model: LineSeries, *, ax=None, show=False, **style
                          ) -> tuple[Figure, Axes]:

* ``ax=None`` -- when you pass axes they are authoritative: nothing new is
  created and the returned figure is theirs.  Multi-panel renderers take a
  sequence of axes whose length matches the panel count.
* ``show=False`` -- rendering never displays implicitly.  Pass ``show=True`` to
  call ``plt.show()``.
* The return value is ``(Figure, Axes)``, or ``(Figure, ndarray[Axes])`` for
  multi-panel renderers -- with one exception: ``<domain>_animation_<quantity>``
  renderers return ``(Figure, Axes, FuncAnimation)``, since none of the other
  view kinds models a time animation.

Renderers take a typed view model from :mod:`vaft.plot.models` plus styling and
layout options, and nothing else.  None of them interprets an OMAS
``ODS``/``ODC``, a native IMAS ``IDS``, a ``DBEntry``, a shot number, a code
result, or a data file -- passing one raises a ``TypeError`` naming the adapter
to use instead.

View models
-----------

:mod:`vaft.plot.models` holds frozen, NumPy-backed dataclasses that validate on
construction, so renderers never need defensive checks.

* :class:`~vaft.plot.models.Series` -- the building block: one labeled ``x``/``y``
  trace with optional error bars and per-trace style.
* :class:`~vaft.plot.models.LineSeries` -- traces sharing one pair of axes, used
  by ``<domain>_time_<quantity>``.
* :class:`~vaft.plot.models.Profile1D` -- 1D profiles against a named radial
  coordinate, used by ``<domain>_profile_<quantity>``.
* :class:`~vaft.plot.models.Field2D` -- a scalar field on an ``(r, z)`` grid plus
  geometry overlays, used by ``<domain>_field_<quantity>``.
* :class:`~vaft.plot.models.GeometryLayers` -- polylines, polygons and point sets
  in a machine view, used by ``<domain>_geometry_<quantity>``.
* :class:`~vaft.plot.models.Image2D` -- a raster image in pixel space (e.g. a
  camera frame) with optional pixel-space overlays, drawn with ``imshow``
  rather than a contour -- used by ``<domain>_image_<quantity>``.
* :class:`~vaft.plot.models.ImageSequence` -- a sequence of raster frames on a
  shared color scale, used by ``<domain>_animation_<quantity>``.
* :class:`~vaft.plot.models.Spectrogram` -- a ``(frequency, time)`` magnitude map.
* :class:`~vaft.plot.models.Panels` -- a grid of the above rendered into one
  figure, used by ``<domain>_overview`` and the composite renderers.

``Spectrogram.from_result()`` wraps anything exposing ``time``/``frequency``/
``magnitude``, including the result of ``vaft.process.mirnov_spectrogram``.

The shared bodies that do the drawing are public for ad-hoc plots with no
canonical name: :func:`render_line_series`, :func:`render_profile_1d`,
:func:`render_field_2d`, :func:`render_geometry_layers`,
:func:`render_image_2d`, :func:`render_image_sequence`,
:func:`render_spectrogram` and :func:`render_panels`.

Discovery
---------

::

    for row in vaft.plot.available_plots(domain="magnetics"):
        print(row["name"], row["model"], row["required_paths"])

Each row carries the canonical ``name``, its ``domain``/``view``/``quantity``,
the ``model`` the renderer consumes, the ``ids`` roots and ``required_paths`` an
adapter must supply, and a short ``description``.  :func:`available_plots`,
``__all__`` and ``dir()`` report the same canonical set because all three derive
from :mod:`vaft.plot.registry`.  The ``ids`` and ``required_paths`` fields are
what will let a database adapter fetch only the data a plot needs once selective
loading (issue #51) lands.

Adding a renderer means adding a ``@renderer(...)``-decorated function; the
decorator registers it and returns it unchanged, so the name stays a real
module-level ``def`` that documentation tools and type checkers can see.

Rendering from data
-------------------

:mod:`vaft.omas` exposes one ``plot_<canonical-stem>`` per canonical plot,
accepting a single ``ODS``, an ``ODC``, or a list of either::

    fig, ax = vaft.omas.plot_magnetics_time_ip(ods)
    fig, ax = vaft.omas.plot_magnetics_time_ip([ods_a, ods_b], label="shot")

Ordering follows the caller's ordering -- ODC key order, or list order -- and
labels default to the data-entry pulse number, falling back to the key.  Pass
``label="key"``, ``label="run"``, or an explicit sequence to override.
``vaft.omas.available_plots(obj)`` filters the registry down to the plots whose
required data the object actually holds.

Importing ``vaft`` never mutates OMAS.  Opt in to ODS methods explicitly with
``vaft.omas.enable_plot_methods()``; registration is idempotent, and any name
that would replace one of OMAS's own ``plot_*`` methods raises ``RuntimeError``
unless ``overwrite=True`` is passed.

Migration
---------

Names from before the redesign still resolve here and emit a
``DeprecationWarning`` naming their replacement.  Print the full per-symbol
table, including the removal release, with::

    python -c "import vaft.plot; print(vaft.plot.migration_table())"
"""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any

from . import models, registry, renderers, style
from ._migration import (
    DEPRECATED,
    LEGACY,
    LEGACY_MODULES,
    RELOCATED,
    REMOVAL_RELEASE,
    REMOVED,
)
from ._migration import render_markdown_table as migration_table
from .models import (
    Field2D,
    GeometryLayer,
    GeometryLayers,
    Image2D,
    ImageSequence,
    LineSeries,
    Panels,
    Profile1D,
    Series,
    Spectrogram,
    ViewModel,
)
from .registry import PlotSpec, available_plots, canonical_names, get_spec
from .renderers.fields import render_field_2d
from .renderers.geometry import render_geometry_layers
from .renderers.images import render_image_2d, render_image_sequence
from .renderers.lines import render_line_series
from .renderers.panels import render_panels
from .renderers.profiles import render_profile_1d
from .renderers.spectrograms import render_spectrogram
from .style import save_figure

# Canonical renderers are re-exported explicitly rather than bound in a loop, so
# documentation tools, IDEs and type checkers see every ``vaft.plot.<name>``.
# ``test_plot_registry`` asserts this block stays in step with the registry.
from .renderers.fields import (
    core_profiles_field_electron_density,
    core_profiles_field_electron_temperature,
    equilibrium_field_psi,
    equilibrium_field_psi_vacuum,
)
from .renderers.geometry import (
    charge_exchange_geometry_poloidal,
    equilibrium_geometry_boundary,
    equilibrium_geometry_topview,
    machine_geometry_poloidal,
    machine_geometry_topview,
    magnetics_geometry_poloidal,
    pf_active_geometry_poloidal,
    pf_passive_geometry_poloidal,
    soft_x_rays_geometry_lines_of_sight,
    thomson_scattering_geometry_poloidal,
    wall_geometry_poloidal,
)
from .renderers.images import (
    camera_visible_animation_frames,
    camera_visible_image_efit_overlay,
    camera_visible_image_field_line,
    camera_visible_image_frame,
)
from .renderers.lines import (
    barometry_time_pressure,
    charge_exchange_time_ion_temperature,
    charge_exchange_time_velocity_tor,
    core_profiles_time_electron_density,
    core_profiles_time_electron_temperature,
    equilibrium_time_beta_n,
    equilibrium_time_beta_pol,
    equilibrium_time_beta_tor,
    equilibrium_time_diamagnetic_flux,
    equilibrium_time_li,
    equilibrium_time_major_radius,
    equilibrium_time_plasma_current,
    equilibrium_time_q0,
    equilibrium_time_q95,
    equilibrium_time_qa,
    equilibrium_time_w_mag,
    equilibrium_time_w_mhd,
    equilibrium_time_w_tot,
    interferometer_time_n_e_line,
    magnetics_time_b_field_pol_probe_field,
    magnetics_time_diamagnetic_flux,
    magnetics_time_flux_loop_flux,
    magnetics_time_flux_loop_voltage,
    magnetics_time_impa_field,
    magnetics_time_impa_voltage,
    magnetics_time_ip,
    magnetics_time_mirnov_voltage,
    pf_active_time_current,
    pf_active_time_current_turns,
    soft_x_rays_time_power,
    spectrometer_uv_time_intensity,
    tf_time_b_field_tor,
    tf_time_b_field_tor_vacuum_r,
    tf_time_coil_current,
    thomson_scattering_time_electron_density,
    thomson_scattering_time_electron_temperature,
)
from .renderers.panels import (
    core_profiles_time_volume_averaged,
    electromagnetics_time_current,
    equilibrium_overview,
    equilibrium_time_virial,
    interferometer_overview,
    magnetics_overview,
    magnetics_overview_impa,
    soft_x_rays_overview,
    spectrometer_uv_time_impurity,
    summary_time_beta,
    summary_time_energy,
    summary_time_power_balance,
    summary_time_voltage_consumption,
)
from .renderers.profiles import (
    charge_exchange_profile_ion_temperature,
    charge_exchange_profile_velocity_tor,
    core_profiles_profile_electron_density,
    core_profiles_profile_electron_temperature,
    core_profiles_profile_ion_temperature,
    core_profiles_profile_pressure,
    equilibrium_profile_f,
    equilibrium_profile_ffprime,
    equilibrium_profile_j_tor,
    equilibrium_profile_pprime,
    equilibrium_profile_pressure,
    equilibrium_profile_q,
    magnetics_profile_impa_tf,
    thomson_scattering_profile_electron_density,
    thomson_scattering_profile_electron_temperature,
)
from .renderers.spectrograms import (
    interferometer_spectrogram,
    magnetics_spectrogram_mirnov,
    soft_x_rays_spectrogram,
)

# Public surface that is not a canonical renderer.
_SUPPORT_EXPORTS = (
    "Field2D",
    "GeometryLayer",
    "GeometryLayers",
    "Image2D",
    "ImageSequence",
    "LineSeries",
    "Panels",
    "PlotSpec",
    "Profile1D",
    "Series",
    "Spectrogram",
    "ViewModel",
    "available_plots",
    "canonical_names",
    "get_spec",
    "migration_table",
    "render_field_2d",
    "render_geometry_layers",
    "render_image_2d",
    "render_image_sequence",
    "render_line_series",
    "render_panels",
    "render_profile_1d",
    "render_spectrogram",
    "save_figure",
)

__all__ = sorted(_SUPPORT_EXPORTS + registry.canonical_names())

#: Legacy submodules that stay importable as ``vaft.plot.<name>``.
_LEGACY_SUBMODULES = frozenset(LEGACY_MODULES.values()) | {"utils"}


def _resolve_legacy(name: str) -> Any:
    module = import_module(f".{LEGACY_MODULES[name]}", __name__)
    return getattr(module, name)


def _resolve_relocated(name: str) -> Any:
    target = RELOCATED[name]
    module_path, _, attribute = target.rpartition(".")
    return getattr(import_module(module_path), attribute)


def __getattr__(name: str) -> Any:
    if name in REMOVED:
        raise AttributeError(
            f"vaft.plot.{name} was an internal helper and is no longer exported "
            f"({REMOVED[name]}). See vaft.plot.migration_table()."
        )
    if name in RELOCATED:
        warnings.warn(
            f"vaft.plot.{name} moved to {RELOCATED[name]}; import it from there. "
            f"The vaft.plot alias is removed in {REMOVAL_RELEASE}.",
            DeprecationWarning,
            stacklevel=2,
        )
        value = _resolve_relocated(name)
    elif name in DEPRECATED:
        warnings.warn(
            f"vaft.plot.{name} is deprecated; use vaft.plot.{DEPRECATED[name]} "
            f"with a view model, or vaft.omas.plot_{DEPRECATED[name]} to render "
            f"directly from an ODS. Removed in {REMOVAL_RELEASE}.",
            DeprecationWarning,
            stacklevel=2,
        )
        value = _resolve_legacy(name)
    elif name in LEGACY:
        value = _resolve_legacy(name)
    elif name in _LEGACY_SUBMODULES:
        # Submodule access such as ``vaft.plot.time`` keeps working.
        value = import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(LEGACY))
