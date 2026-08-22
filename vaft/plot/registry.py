"""Central registry of canonical VAFT plot renderers.

Every canonical renderer registers a :class:`PlotSpec` describing its name, the
view model it consumes, the IDS roots and exact data paths an adapter must
supply, and a short description.  The registry is the single source of truth
shared by:

* ``vaft.plot.__all__`` and :func:`vaft.plot.available_plots`;
* the adapter layers, which look up ``required_paths`` to decide which plots an
  object can produce and (once selective loading exists) which IDS to fetch;
* the contract tests, which iterate every spec.

Renderers are registered with the :func:`renderer` decorator, which returns the
undecorated function so each canonical name remains a real module-level ``def``
visible to documentation tools and static analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping

__all__ = [
    "PlotSpec",
    "VIEWS",
    "available_plots",
    "canonical_names",
    "get_spec",
    "register",
    "renderer",
    "specs",
]

Status = Literal["canonical", "legacy"]

#: The ``<view>`` component of ``<domain>_<view>_<quantity>``.
VIEWS = ("time", "profile", "field", "geometry", "spectrogram", "overview")


@dataclass(frozen=True)
class PlotSpec:
    """Metadata describing one canonical renderer."""

    name: str
    model: type
    renderer: Callable[..., Any]
    domain: str
    view: str
    description: str
    quantity: str = ""
    #: Top-level IDS names the adapter must read.  Drives selective loading.
    ids: tuple[str, ...] = ()
    #: Exact data paths required to build the view model.  ``{i}`` marks a
    #: repeated index that the adapter expands.
    required_paths: tuple[str, ...] = ()
    #: Paths that enrich the plot but are not required.
    optional_paths: tuple[str, ...] = ()
    status: Status = "canonical"
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def stem(self) -> str:
        """The canonical stem used to derive adapter names (``plot_<stem>``)."""
        return self.name


_REGISTRY: dict[str, PlotSpec] = {}


def register(spec: PlotSpec) -> PlotSpec:
    """Add ``spec`` to the registry, refusing to silently replace an entry."""
    existing = _REGISTRY.get(spec.name)
    if existing is not None and existing.renderer is not spec.renderer:
        raise ValueError(
            f"plot {spec.name!r} is already registered by "
            f"{existing.renderer.__module__}.{existing.renderer.__qualname__}"
        )
    if spec.view not in VIEWS:
        raise ValueError(f"unknown view {spec.view!r}; expected one of {VIEWS}")
    _REGISTRY[spec.name] = spec
    return spec


def renderer(
    *,
    domain: str,
    view: str,
    model: type,
    description: str,
    quantity: str = "",
    ids: Iterable[str] = (),
    required_paths: Iterable[str] = (),
    optional_paths: Iterable[str] = (),
    status: Status = "canonical",
    name: str | None = None,
    **options: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register the decorated function and return it unchanged."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        register(
            PlotSpec(
                name=name or function.__name__,
                model=model,
                renderer=function,
                domain=domain,
                view=view,
                quantity=quantity,
                description=description,
                ids=tuple(ids),
                required_paths=tuple(required_paths),
                optional_paths=tuple(optional_paths),
                status=status,
                options=dict(options),
            )
        )
        return function

    return decorate


def get_spec(name: str) -> PlotSpec:
    """Return the spec registered under ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no plot named {name!r}; use vaft.plot.available_plots() to list them"
        ) from None


def specs(
    *,
    domain: str | None = None,
    view: str | None = None,
    model: type | None = None,
    status: Status | None = "canonical",
) -> tuple[PlotSpec, ...]:
    """Return registered specs, filtered and ordered by canonical name."""
    selected = [
        spec
        for spec in _REGISTRY.values()
        if (domain is None or spec.domain == domain)
        and (view is None or spec.view == view)
        and (model is None or spec.model is model)
        and (status is None or spec.status == status)
    ]
    return tuple(sorted(selected, key=lambda spec: spec.name))


def canonical_names() -> tuple[str, ...]:
    """Return every canonical renderer name, sorted."""
    return tuple(spec.name for spec in specs())


def available_plots(
    *,
    domain: str | None = None,
    view: str | None = None,
    model: type | None = None,
    status: Status | None = "canonical",
) -> tuple[dict[str, Any], ...]:
    """Describe the registered plots as plain dictionaries.

    Each row carries the canonical ``name``, the ``model`` the renderer consumes,
    the ``ids`` and ``required_paths`` an adapter must provide, and a short
    ``description``.
    """
    return tuple(
        {
            "name": spec.name,
            "domain": spec.domain,
            "view": spec.view,
            "quantity": spec.quantity,
            "model": spec.model.__name__,
            "ids": spec.ids,
            "required_paths": spec.required_paths,
            "description": spec.description,
            "status": spec.status,
        }
        for spec in specs(domain=domain, view=view, model=model, status=status)
    )
