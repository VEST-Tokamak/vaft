"""File-format and data-structure compatibility helpers.

The public ``vaft.data`` namespace is for portable file/data representations
such as EFIT GEQDSK. Packaged sample files remain available through
``vaft.data.resources``.
"""

from importlib import import_module

__all__ = [
    "GEQDSK",
    "data_path",
    "from_imas",
    "from_omas",
    "read_geqdsk",
    "resources",
    "sample_geqdsk",
    "open_adas",
    "to_imas",
    "to_omas",
    "write_geqdsk",
    "VFITResult",
    "read_vfit",
]

_EXPORT_MAP = {
    "GEQDSK": (".eqdsk", "GEQDSK"),
    "from_imas": (".eqdsk", "from_imas"),
    "from_omas": (".eqdsk", "from_omas"),
    "read_geqdsk": (".eqdsk", "read_geqdsk"),
    "to_imas": (".eqdsk", "to_imas"),
    "to_omas": (".eqdsk", "to_omas"),
    "write_geqdsk": (".eqdsk", "write_geqdsk"),
    "data_path": (".resources", "data_path"),
    "sample_geqdsk": (".resources", "sample_geqdsk"),
    "VFITResult": (".vfit", "VFITResult"),
    "read_vfit": (".vfit", "read_vfit"),
}


def __getattr__(name: str):
    if name in {"resources", "open_adas"}:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
