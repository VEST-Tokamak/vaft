"""Low-level plotting for VAFT.

``vaft.plot`` owns rendering and nothing else.  Its canonical renderers are named

.. code-block:: text

    vaft.plot.<domain>_<view>_<quantity>

for example :func:`plasma_current_time`, :func:`equilibrium_profile_pressure`, and
:func:`soft_x_rays_spectrogram`.

``<domain>`` is the IDS root the plot belongs to -- ``magnetics``,
``equilibrium``, ``pf_active``, ``pf_passive``, ``tf``, ``wall``,
``soft_x_rays``, ``core_profiles``, ``thomson_scattering``, ``charge_exchange``,
``barometry``, ``spectrometer_uv``, ``camera_visible`` -- plus two composed
domains, ``machine`` for cross-IDS machine views and ``summary`` for cross-IDS
summary panels.  ``<view>`` is one of ``time``, ``profile``, ``field``,
``geometry``, ``spectrum``, ``spectrogram``, ``overview``, ``image``,
``animation``.  ``<quantity>`` may be dropped when the domain and view are
already unambiguous, as in ``soft_x_rays_spectrogram``.

There is no redundant ``plot_`` prefix here; adapter layers and object methods
use ``plot_<canonical-stem>``, so ``vaft.plot.plasma_current_time`` is rendered
from an ODS by ``vaft.omas.plot_plasma_current_time``.

The radial coordinate is *not* part of a profile renderer's name.  One
:func:`equilibrium_profile_q` serves every coordinate; pick one with the
adapter's ``coordinate=`` argument (``rho_tor_norm``, ``psi_norm``, ``r_major``,
``r_minor``).

The renderer contract
---------------------

Every canonical renderer has this shape::

    def plasma_current_time(model: LineSeries, *, ax=None, show=False, **style
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
* :class:`~vaft.plot.models.PowerSpectrum` -- a power spectral density on log-log
  axes, with optional fitted segments and caller-supplied reference-slope guides.
* :class:`~vaft.plot.models.Spectrogram` -- a ``(frequency, time)`` magnitude map.
* :class:`~vaft.plot.models.Panels` -- a grid of the above rendered into one
  figure, used by ``<domain>_overview`` and the composite renderers.

``Spectrogram.from_result()`` wraps anything exposing ``time``/``frequency``/
``magnitude``, including the result of ``vaft.process.mirnov_spectrogram``.

The shared bodies that do the drawing are public for ad-hoc plots with no
canonical name: :func:`render_line_series`, :func:`render_profile_1d`,
:func:`render_field_2d`, :func:`render_geometry_layers`,
:func:`render_image_2d`, :func:`render_image_sequence`,
:func:`render_power_spectrum`, :func:`render_spectrogram` and
:func:`render_panels`.

Discovery
---------

::

    print(vaft.plot.available_plots(query="ip"))      # a subject / view / quantity tree
    for row in vaft.plot.available_plots(domain="magnetics"):
        print(row["name"], row["model"], row["required_paths"])

:func:`available_plots` returns a :class:`~vaft.plot.discovery.PlotCatalog`: it
prints as a tree organised by the taxonomy's ``subject / view / [quantity]``
identity, and iterates as :class:`~vaft.plot.discovery.PlotCapability` records
that still answer the flat-row keys -- the canonical ``name``, its
``domain``/``view``/``quantity``, the ``model`` the renderer consumes, the
``ids`` roots and ``required_paths`` an adapter must supply, and a short
``description`` (issue #262).  ``query=`` resolves strictly through the alias
registry.  :func:`available_plots`, ``__all__`` and ``dir()`` report the same
canonical set because all three derive from :mod:`vaft.plot.registry`.  The
``ids`` and ``required_paths`` fields are what will let a database adapter
fetch only the data a plot needs once selective loading (issue #51) lands.

Adding a renderer means adding a ``@renderer(...)``-decorated function; the
decorator registers it and returns it unchanged, so the name stays a real
module-level ``def`` that documentation tools and type checkers can see.

Rendering from data
-------------------

:mod:`vaft.omas` exposes one ``plot_<canonical-stem>`` per canonical plot,
accepting a single ``ODS``, an ``ODC``, or a list of either::

    fig, ax = vaft.omas.plot_plasma_current_time(ods)
    fig, ax = vaft.omas.plot_plasma_current_time([ods_a, ods_b], label="shot")

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
    RENAMED,
    RENAMED_REMOVAL_RELEASE,
)
from ._migration import render_markdown_table as migration_table
from .models import (
    Field2D,
    Geometry3DLayer,
    Geometry3DLayers,
    GeometryLayer,
    GeometryLayers,
    Image2D,
    ImageSequence,
    LineSeries,
    Panels,
    PowerSpectrum,
    Profile1D,
    ReferenceSlope,
    Series,
    Spectrogram,
    TextPanel,
    ViewModel,
)
from .discovery import PlotCapability, PlotCatalog
from .navigation import SliceNavigator
from .registry import PlotSpec, available_plots, canonical_names, get_spec
from .renderers.fields import render_field_2d
from .renderers.geometry import render_geometry_3d_layers, render_geometry_layers
from .renderers.images import render_image_2d, render_image_sequence
from .renderers.lines import render_line_series
from .renderers.panels import render_panels
from .renderers.profiles import render_profile_1d
from .renderers.spectra import render_power_spectrum
from .renderers.spectrograms import render_spectrogram
from .style import save_figure

# Canonical renderers are re-exported explicitly rather than bound in a loop, so
# documentation tools, IDEs and type checkers see every ``vaft.plot.<name>``.
# ``test_plot_registry`` asserts this block stays in step with the registry.
from .renderers.fields import (
    electron_density_field,
    electron_temperature_field,
    equilibrium_field_psi,
    equilibrium_field_psi_vacuum,
)
from .renderers.geometry import (
    charge_exchange_geometry_poloidal,
    coil_3d_geometry3d,
    coil_3d_geometry_topview,
    equilibrium_geometry_boundary,
    equilibrium_geometry_topview,
    machine_geometry_poloidal,
    machine_geometry_topview,
    magnetics_geometry_poloidal,
    pf_coil_geometry_poloidal,
    passive_structure_geometry_poloidal,
    soft_x_rays_geometry_lines_of_sight,
    thomson_scattering_geometry_poloidal,
    wall_geometry_poloidal,
)
from .renderers.images import (
    camera_visible_animation_frames,
    camera_visible_image,
    camera_visible_image_efit_overlay,
    camera_visible_image_field_line,
    camera_visible_image_frame,
)
from .renderers.lines import (
    barometry_time_pressure,
    charge_exchange_time_ion_temperature,
    charge_exchange_time_velocity_tor,
    electron_density_time,
    electron_temperature_time,
    equilibrium_time_beta_n,
    equilibrium_time_beta_p,
    equilibrium_time_beta_t,
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
    b_field_probe_time_field,
    diamagnetic_flux_time,
    flux_loop_time_flux,
    flux_loop_time_voltage,
    impa_time_field,
    impa_time_voltage,
    plasma_current_time,
    mirnov_time_voltage,
    mhd_linear_time_energy_perturbed,
    pf_coil_time_current,
    pf_coil_time_current_turns,
    soft_x_rays_time_power,
    spectrometer_uv_time_intensity,
    tf_coil_time_b_t,
    tf_coil_time_b_t_vacuum_r,
    tf_coil_time_current,
    thomson_scattering_time_electron_density,
    thomson_scattering_time_electron_temperature,
)
from .renderers.panels import (
    chease_overview_profile_validity,
    chease_overview_refinement_summary,
    core_profiles_time_volume_averaged,
    current_overview,
    diagnostics_overview,
    equilibrium_overview,
    equilibrium_overview_constraint_coverage,
    equilibrium_overview_constraints,
    equilibrium_overview_convergence,
    equilibrium_overview_fit_quality,
    equilibrium_overview_histories,
    equilibrium_overview_profiles,
    equilibrium_overview_residuals,
    equilibrium_overview_verification,
    equilibrium_time_virial,
    interferometer_overview,
    magnetics_overview,
    impa_overview,
    magnetics_overview_plasma_residual,
    magnetics_overview_vacuum,
    limiter_current_time,
    soft_x_rays_overview,
    spectrometer_uv_time_impurity,
    equilibrium_time_beta,
    summary_time_energy,
    summary_time_power_balance,
    summary_time_voltage_consumption,
)
from .renderers.profiles import (
    charge_exchange_profile_ion_temperature,
    charge_exchange_profile_velocity_tor,
    electron_density_profile,
    electron_temperature_profile,
    ion_temperature_profile,
    thermal_pressure_profile,
    equilibrium_profile_f,
    equilibrium_profile_ffprime,
    equilibrium_profile_j_tor,
    equilibrium_profile_pprime,
    equilibrium_profile_pressure,
    equilibrium_profile_q,
    impa_profile_field,
    thomson_scattering_profile_electron_density,
    thomson_scattering_profile_electron_temperature,
)
from .renderers.spectra import (
    interferometer_spectrum,
    mirnov_spectrum,
    soft_x_rays_spectrum,
)
from .renderers.spectrograms import (
    interferometer_spectrogram,
    mirnov_spectrogram,
    soft_x_rays_spectrogram,
)
from .parameter_history import plot_parameter_history

# Public surface that is not a canonical renderer.
_SUPPORT_EXPORTS = (
    "Field2D",
    "Geometry3DLayer",
    "Geometry3DLayers",
    "GeometryLayer",
    "GeometryLayers",
    "Image2D",
    "ImageSequence",
    "LineSeries",
    "Panels",
    "PlotCapability",
    "PlotCatalog",
    "PlotSpec",
    "PowerSpectrum",
    "Profile1D",
    "ReferenceSlope",
    "Series",
    "SliceNavigator",
    "Spectrogram",
    "TextPanel",
    "ViewModel",
    "available_plots",
    "canonical_names",
    "get_spec",
    "migration_table",
    "render_field_2d",
    "render_geometry_3d_layers",
    "render_geometry_layers",
    "render_image_2d",
    "render_image_sequence",
    "render_line_series",
    "render_panels",
    "render_power_spectrum",
    "render_profile_1d",
    "render_spectrogram",
    "save_figure",
    "plot_parameter_history",
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
    if name in RENAMED:
        replacement = RENAMED[name]
        warnings.warn(
            f"vaft.plot.{name} was renamed to vaft.plot.{replacement} in the "
            f"subject-taxonomy redesign (issue #251); use the new name, or "
            f"vaft.omas.plot_{replacement} to render directly from an ODS. "
            f"The old name is removed in {RENAMED_REMOVAL_RELEASE}.",
            DeprecationWarning,
            stacklevel=2,
        )
        value = globals()[replacement]
        globals()[name] = value
        return value
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
