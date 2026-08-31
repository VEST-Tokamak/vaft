"""Top-level package initialization.

The package historically imported every major subpackage eagerly. That makes
`import vaft` fail whenever optional heavy dependencies such as `omas` are not
available, even if the caller only needs lightweight utilities.

We now expose the same public subpackages lazily via `__getattr__`, which keeps
the top-level import small while preserving the historical attribute-based API.
"""

from importlib import import_module

from .compat import apply_runtime_compat_patches
from .version import __version__

__all__ = [
    "__version__",
    "process",
    "formula",
    "machine_mapping",
    "plot",
    "data",
    "omas",
    "code",
    "cli",
    "database",
    "imas",
    "validation",
    "apply_runtime_compat_patches",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)


# Apply lightweight compatibility shims as early as possible so every
# subpackage observes the same NumPy/SciPy behavior.
apply_runtime_compat_patches()
