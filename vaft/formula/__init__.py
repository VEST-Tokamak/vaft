"""Formula namespace for physics helpers."""

from __future__ import annotations

from importlib import import_module

_SUBMODULES = {
    "constants": ".constants",
    "utils": ".utils",
    "equilibrium": ".equilibrium",
    "stability": ".stability",
    "green": ".green",
}

_SEARCH_ORDER = ("green", "equilibrium", "constants", "utils", "stability")

# Eagerly import all submodule symbols to preserve `from vaft.formula import *` behaviour.
from .constants import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
from .equilibrium import *  # noqa: F401, F403
from .stability import *  # noqa: F401, F403
from .green import *  # noqa: F401, F403

# Include both submodule accessor names and all eagerly-imported public symbols.
__all__ = sorted(
    set(list(_SUBMODULES.keys()) + [name for name in globals() if not name.startswith("_")])
)


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name], __name__)
        globals()[name] = module
        return module

    for module_key in _SEARCH_ORDER:
        try:
            module = import_module(_SUBMODULES[module_key], __name__)
        except Exception:
            continue
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
