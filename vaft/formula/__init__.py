"""Formula namespace for physics helpers.

Submodules are imported on demand.  ``import vaft.formula.statistics`` (or any
other single submodule) costs only that submodule, so a caller that needs one
pure kernel does not pay for scipy by way of ``.green`` and ``.equilibrium``,
nor inherit those modules' import-time failure surface.

``from vaft.formula import *`` still exposes exactly the same names it always
has.  Reading ``__all__`` is what pulls the whole subtree in: the module-level
``__getattr__`` below is consulted for ``__all__`` itself (PEP 562), so the star
import triggers the full load at the moment it actually needs it, and nothing
before then does.

The discovery layer -- ``describe``, ``search``, ``list_formulas``,
``categories`` and the ``catalog`` submodule they live in -- is resolved the
same way, on first access, and is never part of ``__all__``: the star import
does not touch it, and neither does importing a physics submodule.
"""

from __future__ import annotations

from importlib import import_module

_SUBMODULES = {
    "constants": ".constants",
    "utils": ".utils",
    "equilibrium": ".equilibrium",
    "stability": ".stability",
    "green": ".green",
    "atomic": ".atomic",
    "statistics": ".statistics",
    "magnetics": ".magnetics",
}

#: The order these submodules were star-imported in when this package loaded
#: them eagerly.  Eighteen names are defined by more than one submodule -- MU0,
#: gradient, trapz_integral and friends -- and under a star import the *last*
#: binding won.  Attribute resolution therefore walks this tuple in reverse, so
#: ``vaft.formula.gradient`` still resolves to ``stability``'s and not to
#: ``equilibrium``'s.  Order is load-bearing: see
#: test_formula_lazy_namespace.py, which pins every one of those names.
_IMPORT_ORDER = (
    "constants",
    "utils",
    "equilibrium",
    "stability",
    "green",
    "atomic",
    "statistics",
    "magnetics",
)

#: Names served by ``.catalog`` on first access.  Deliberately not in
#: ``_SUBMODULES``: joining that map would put ``catalog`` into ``__all__`` and
#: make the star import load it, which is exactly what issue #248 forbids.
_CATALOG_NAMES = frozenset(
    {"catalog", "FormulaSpec", "describe", "search", "list_formulas", "categories"}
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

    Honouring ``__all__`` matters here: a submodule that declares one does not
    re-export its own imports, so ``np`` and ``Union`` must not be reachable
    through it even though ``hasattr`` would say otherwise.
    """
    declared = getattr(module, "__all__", None)
    if declared is None:
        declared = [name for name in vars(module) if not name.startswith("_")]
    return frozenset(declared)


def _public_names() -> list[str]:
    """Every name the eager star-imports used to leave in this namespace."""
    collected = set(_SUBMODULES)
    for key in _IMPORT_ORDER:
        collected.update(_exported(_submodule(key)))
    return sorted(name for name in collected if not name.startswith("_"))


def __getattr__(name: str):
    if name in _SUBMODULES:
        return _submodule(name)

    if name in _CATALOG_NAMES:
        module = import_module(".catalog", __name__)
        value = module if name == "catalog" else getattr(module, name)
        globals()[name] = value
        return value

    if name == "__all__":
        # Not swallowed: a submodule that cannot import used to break the star
        # import outright, and it still should.  What changed is that plain
        # `import vaft.formula` no longer does.
        value = _public_names()
        globals()["__all__"] = value
        return value

    for key in reversed(_IMPORT_ORDER):
        try:
            module = _submodule(key)
        except Exception:
            continue
        if name in _exported(module):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    names = globals().get("__all__") or _public_names()
    return sorted(set(globals()) | set(names) | _CATALOG_NAMES)
