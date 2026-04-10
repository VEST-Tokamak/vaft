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

# Snapshot names before star-imports so __all__ captures only the symbols they add.
_before = set(dir())

# Eagerly import all submodule symbols to preserve `from vaft.formula import *` behaviour.
from .constants import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
from .equilibrium import *  # noqa: F401, F403
from .stability import *  # noqa: F401, F403
from .green import *  # noqa: F401, F403

# Include both submodule accessor names and symbols added by the star imports above.
__all__ = sorted(set(list(_SUBMODULES.keys()) + list(set(dir()) - _before)))
del _before


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
