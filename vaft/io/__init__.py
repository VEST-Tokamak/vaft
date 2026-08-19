"""Local artifact and FileDB storage interfaces.

Format-specific loading remains available through :mod:`vaft.omas` and
:mod:`vaft.imas`. This namespace owns local storage layout and shared private
artifact detection; remote HSDS and SQL access remain in :mod:`vaft.database`.
"""

from importlib import import_module

__all__ = [
    "ArtifactClass",
    "FileDB",
    "FileDBConfigError",
    "FileDBDomain",
    "FileDBError",
    "FileDBPathError",
    "GPECCode",
    "LegacyAuditEntry",
    "LegacyAuditReport",
    "LegacyCollision",
    "LegacyDuplicate",
    "LegacyMissingProduct",
    "LegacyResolution",
    "OMASStage",
    "audit_legacy_filedb",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".filedb", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
