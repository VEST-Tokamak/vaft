"""Processing namespace for VAFT's scientific and numerical kernels.

Submodules are imported on demand.  ``from vaft.process.signal_processing
import smooth`` costs scipy and nothing else; it used to cost the entire
subtree -- omas, Matplotlib, ipywidgets, uncertainties, pandas, statsmodels --
because importing any submodule first executes this file, and this file used to
star-import all fifteen of them (issue #249).

``from vaft.process import *`` still works.  Reading ``__all__`` is what pulls
the whole subtree in: the module-level ``__getattr__`` below is consulted for
``__all__`` itself (PEP 562), so the star import triggers the full load at the
moment it actually needs it, and nothing before then does.

What the star import *exposes* is narrower than it was.  Six submodules
declared no ``__all__``, so their star-import surface was everything they
imported as well as everything they defined, and ``profile`` opened with
``from omas import *``.  That put 164 names into this namespace by accident --
``load_omas_json``, ``machine_mappings``, ``mdstree``, ``IntSlider``,
``raw_db``, ``np`` -- against 172 that belong here.  Every submodule now
declares ``__all__``.  ``test/data/process_export_inventory.json`` records the
old surface and ``test/test_process_lazy_namespace.py`` checks that each
dropped name is still reachable from the module that really provides it.

The discovery layer -- ``describe``, ``search``, ``list_processes``,
``categories`` and the ``catalog`` submodule they live in (issue #252) -- is
resolved the same way, on first access, and is never part of ``__all__``: the
star import does not touch it, and neither does importing a processing
submodule.
"""

from __future__ import annotations

from importlib import import_module

_SUBMODULES = {
    "atomic": ".atomic",
    "camera_geometry": ".camera_geometry",
    "cocos": ".cocos",
    "electromagnetics": ".electromagnetics",
    "equilibrium": ".equilibrium",
    "fluctuation": ".fluctuation",
    "impa": ".impa",
    "langmuir": ".langmuir",
    "magnetics": ".magnetics",
    "numerical": ".numerical",
    "onset": ".onset",
    "profile": ".profile",
    "signal_processing": ".signal_processing",
    "soft_x_rays": ".soft_x_rays",
    "statistical_analysis": ".statistical_analysis",
}

#: The order this package star-imported its submodules in when it loaded them
#: eagerly, so a name that two of them export still resolves to the same object
#: the old last-import-wins chain produced.
#:
#: Unlike :mod:`vaft.formula`, where eighteen names genuinely collide and the
#: order is load-bearing, nothing here is ambiguous: the eight names two
#: submodules share are the same objects seen twice -- ``np`` and the ``typing``
#: aliases, and ``define_baseline``/``subtract_baseline``, which ``magnetics``
#: re-exported only because it used to import them back out of this package.
#: :func:`_resolve` therefore asserts identity rather than picking a winner, so
#: a future genuine collision fails loudly instead of resolving by luck.
_IMPORT_ORDER = (
    "profile",
    "equilibrium",
    "camera_geometry",
    "signal_processing",
    "fluctuation",
    "soft_x_rays",
    "electromagnetics",
    "numerical",
    "magnetics",
    "statistical_analysis",
    "atomic",
    "langmuir",
    "impa",
    # Added after the lazy loader; nothing it exports collides with a sibling.
    "onset",
)

#: Reached as attributes but never star-imported, then or now: ``cocos`` is a
#: submodule callers import by name, and ``_equilibrium_parametric`` is private
#: and re-exported through ``equilibrium``.
_ATTRIBUTE_ONLY = ("cocos",)

#: Names served by ``.catalog`` on first access.  Deliberately not in
#: ``_SUBMODULES``: joining that map would put ``catalog`` into ``__all__`` and
#: make the star import load it, which is exactly what #252 forbids.
_CATALOG_NAMES = frozenset(
    {"catalog", "ProcessSpec", "describe", "search", "list_processes", "categories"}
)

_MODULES: dict[str, object] = {}


def _submodule(key: str):
    """Import one submodule, caching it."""
    module = _MODULES.get(key)
    if module is None:
        module = _MODULES[key] = import_module(_SUBMODULES[key], __name__)
    return module


def _exported(module) -> frozenset[str]:
    """The names ``from <module> import *`` would bind.

    Every process submodule now declares ``__all__``; the fallback is kept so a
    newly added one without it still resolves rather than silently exporting
    nothing.
    """
    declared = getattr(module, "__all__", None)
    if declared is None:
        declared = [name for name in vars(module) if not name.startswith("_")]
    return frozenset(declared)


def _resolve(name: str):
    """Find ``name`` among the submodules, refusing a genuine collision.

    Returns the bound object, or raises :class:`AttributeError` if no submodule
    exports it.  Two submodules exporting the *same* object is fine and
    happens; two exporting different objects under one name would make
    ``vaft.process.<name>`` depend on import order, which this package does not
    do and should not start doing quietly.
    """
    found: tuple[str, object] | None = None
    for key in _IMPORT_ORDER:
        try:
            module = _submodule(key)
        except Exception:
            # Not swallowed in the sense that matters: a submodule that cannot
            # import still breaks `from vaft.process import *`, because that
            # goes through _public_names().  It is skipped only while hunting
            # for one specific name, so an unrelated broken optional dependency
            # cannot hide a name that a healthy submodule provides.
            continue
        if name not in _exported(module):
            continue
        value = getattr(module, name)
        if found is None:
            found = (key, value)
        elif found[1] is not value:
            raise AttributeError(
                f"{name!r} is exported by both vaft.process.{found[0]} and "
                f"vaft.process.{key} as different objects; import order would "
                f"decide which one vaft.process.{name} means. Reach the one you "
                f"want through its own submodule."
            )
    if found is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return found[1]


def _public_names() -> list[str]:
    """Every name the eager star-imports left in this namespace."""
    collected = set(_SUBMODULES)
    for key in _IMPORT_ORDER:
        collected.update(_exported(_submodule(key)))
    return sorted(name for name in collected if not name.startswith("_"))


def __getattr__(name: str):
    if name in _SUBMODULES:
        value = _submodule(name)
        globals()[name] = value
        return value

    if name in _CATALOG_NAMES:
        module = import_module(".catalog", __name__)
        value = module if name == "catalog" else getattr(module, name)
        globals()[name] = value
        return value

    if name == "__all__":
        value = _public_names()
        globals()["__all__"] = value
        return value

    value = _resolve(name)
    globals()[name] = value
    return value


def __dir__():
    names = globals().get("__all__") or _public_names()
    return sorted(set(globals()) | set(names) | set(_ATTRIBUTE_ONLY) | _CATALOG_NAMES)
