"""Database access and storage infrastructure.

The high-level :func:`load`, :func:`open`, and :func:`save` APIs operate on
remote HSDS namespaces. Canonical local FileDB paths live in
:mod:`vaft.database.filedb`; local OMAS/IMAS artifact loading remains exposed
through :mod:`vaft.omas` and :mod:`vaft.imas`.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Literal
import warnings

__all__ = [
    "raw",
    "ods",
    "ids",
    "utils",
    "filedb",
    "load",
    "open",
    "save",
    "summary",
    "export_summary",
]


def _namespace(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a non-empty bare HSDS namespace")
    if ":" in value or "/" in value or "\\" in value:
        raise ValueError(
            f"{label} must be a bare HSDS namespace such as 'public'; "
            "the hdf5:// protocol is an internal detail."
        )
    return value


def _paths(paths: str | list[str] | None) -> list[str] | None:
    if paths is None:
        return None
    values = [paths] if isinstance(paths, str) else list(paths)
    if not values or any(not isinstance(path, str) or not path for path in values):
        raise ValueError(
            "paths must be a non-empty string or list of non-empty strings"
        )
    return values


def _ids_from_paths(paths: list[str] | None) -> list[str] | None:
    if paths is None:
        return None
    names = [path.split(".", 1)[0].split("[", 1)[0] for path in paths]
    if any(name != path for name, path in zip(names, paths)):
        raise ValueError(
            "representation='imas' accepts top-level IDS names only; use representation='omas' for leaf paths"
        )
    return list(dict.fromkeys(names))


def _root_ids(paths: list[str] | None) -> list[str] | None:
    """Return the IDS roots used to scope an OMAS request.

    OMAS paths may name a leaf (including an AOS index); only native IMAS
    dispatch requires the stricter top-level-only validation above.
    """
    if paths is None:
        return None
    return list(dict.fromkeys(path.split(".", 1)[0].split("[", 1)[0] for path in paths))


def _occurrence_map(occurrence: int | Mapping[str, int] | None) -> dict[str, int]:
    if occurrence is None:
        return {}
    if isinstance(occurrence, int):
        if occurrence < 0:
            raise ValueError("occurrence must be non-negative")
        return {"*": occurrence}
    result = {str(name): int(value) for name, value in occurrence.items()}
    if any(value < 0 for value in result.values()):
        raise ValueError("occurrence values must be non-negative")
    return result


def _discover_native_ids(source: str, shot: int, version: str) -> list[str]:
    """List stored domains that the selected IMAS DD can represent natively."""
    from .utils import _require_h5pyd

    _require_h5pyd()
    from . import utils
    import imas

    factory = imas.IDSFactory(version)
    names = sorted(
        name[:-3]
        for name in utils.h5pyd.Folder(f"/{source}/{shot}/")
        if name.endswith(".h5") and name not in {"master.h5", "dataset_description.h5"}
    )
    return [name for name in names if factory.exists(name)]


def _infer_remote_imas_version(
    source: str, shot: int, ids: list[str] | None, requested: str | None
) -> str:
    if requested is not None:
        return requested
    from .utils import _require_h5pyd

    _require_h5pyd()
    from . import utils
    from ..imas.omas_imas import IMAS_DD_VERSION_CONVERSION

    candidates = ["dataset_description", *(ids or [])]
    for name in candidates:
        try:
            with utils.h5pyd.File(f"hdf5://{source}/{shot}/{name}.h5", "r") as handle:
                root = handle[name] if name in handle else handle
                for dataset in (
                    "imas_version",
                    "ids_properties&version_put&data_dictionary",
                ):
                    if dataset in root:
                        value = root[dataset][()]
                        if isinstance(value, bytes):
                            value = value.decode("utf-8", errors="replace")
                        return str(value)
        except Exception:
            continue
    warnings.warn(
        f"Could not infer IMAS DD version for HSDS source '{source}' shot {shot}; "
        f"using {IMAS_DD_VERSION_CONVERSION}",
        RuntimeWarning,
        stacklevel=3,
    )
    return IMAS_DD_VERSION_CONVERSION


def load(
    shot: int | list[int],
    source: str = "public",
    *,
    representation: Literal["omas", "imas"] = "omas",
    paths: str | list[str] | None = None,
    occurrence: int | Mapping[str, int] | None = None,
    imas_version: str | None = None,
    cache: str = "auto",
    transport: Literal["auto", "canonical", "h5image"] = "auto",
):
    """Materialize an OMAS ODS or native IMAS IDS from remote HSDS storage."""
    source = _namespace(source, "source")
    selected_paths = _paths(paths)
    occurrences = _occurrence_map(occurrence)
    if representation == "omas":
        from .ods import load_ods

        mapped = {name: value for name, value in occurrences.items() if name != "*"}
        if "*" in occurrences:
            roots = _root_ids(selected_paths)
            mapped = (
                {name: occurrences["*"] for name in roots}
                if roots is not None
                else {"*": occurrences["*"]}
            )
        version = _infer_remote_imas_version(
            source,
            int(shot[0] if isinstance(shot, list) else shot),
            _root_ids(selected_paths),
            imas_version,
        )
        return load_ods(
            shot,
            directory=source,
            occurrence=mapped,
            paths=selected_paths,
            imas_version=version,
            cache=cache,
            transport=transport,
        )
    if representation != "imas":
        raise ValueError("representation must be 'omas' or 'imas'")
    if isinstance(shot, list):
        raise ValueError("representation='imas' loads one shot at a time")
    names = _ids_from_paths(selected_paths)
    version = _infer_remote_imas_version(source, int(shot), names, imas_version)
    if names is None:
        names = _discover_native_ids(source, int(shot), version)
        if not names:
            raise FileNotFoundError(
                f"No native IMAS IDS are stored for shot {shot} in '{source}'"
            )
    from .ids import load as load_ids

    return load_ids(
        int(shot),
        names[0] if len(names) == 1 else names,
        directory=source,
        occurrence=occurrences,
        dd_version=version,
        cache=cache,
        transport=transport,
    )


def open(
    shot: int,
    *,
    source: str = "public",
    representation: Literal["omas", "imas"] = "omas",
    paths: str | list[str] | None = None,
    occurrence: int | Mapping[str, int] | None = None,
    imas_version: str | None = None,
):
    """Open a read-only lazy OMAS ODS or native IMAS IDS adapter over HSDS."""
    if representation not in {"omas", "imas"}:
        raise ValueError("representation must be 'omas' or 'imas'")
    source = _namespace(source, "source")
    selected_paths = _paths(paths)
    occurrences = _occurrence_map(occurrence)
    if any(value != 0 for value in occurrences.values()):
        raise ValueError("lazy HSDS access currently supports occurrence 0 only")
    if representation == "imas":
        names = _ids_from_paths(selected_paths)
        stored_version = _infer_remote_imas_version(source, int(shot), names, None)
        if imas_version is not None and imas_version != stored_version:
            raise ValueError(
                "HSDS native lazy IMAS does not perform DD conversion; "
                f"requested {imas_version}, stored {stored_version}. Use database.load() for eager conversion."
            )
        from .lazy_imas import open_imas

        return open_imas(
            int(shot), directory=source, ids=names, imas_version=stored_version
        )
    from .lazy_ods import open_ods

    return open_ods(
        int(shot),
        directory=source,
        ids=_root_ids(selected_paths),
        imas_version=_infer_remote_imas_version(
            source, int(shot), _root_ids(selected_paths), imas_version
        ),
    )


def save(
    data,
    shot: int,
    *,
    target: str = "public",
    representation: Literal["omas", "imas"] | None = None,
    occurrence: int | Mapping[str, int] | None = None,
    imas_version: str | None = None,
    derived_cache: Literal["auto", "none", "imas-images", "omas", "both"] = "auto",
):
    """Write an OMAS ODS or native IDS to remote HSDS storage."""
    target = _namespace(target, "target")
    is_ids = _is_imas_ids(data)
    inferred = "imas" if is_ids else "omas"
    if representation is not None and representation != inferred:
        raise TypeError(
            f"representation={representation!r} does not match the supplied object"
        )
    if inferred == "imas":
        from .ids import save as save_ids

        return save_ids(
            data,
            shot,
            directory=target,
            dd_version=imas_version,
            derived_cache=derived_cache,
        )
    from .ods import save_ods

    mapped = _occurrence_map(occurrence)
    if "*" in mapped:
        raise ValueError("OMAS occurrence must be a mapping keyed by IDS name")
    return save_ods(
        data,
        shot,
        directory=target,
        occurrence=mapped,
        imas_version=imas_version,
        derived_cache=derived_cache,
    )


def _is_imas_ids(obj) -> bool:
    try:
        from imas.ids_toplevel import IDSToplevel
    except Exception:
        return False
    return isinstance(obj, IDSToplevel)


def summary(shot_range=None, *, preset="equilibrium_global", source="public"):
    """Return a canonical preset summary for a range or all available shots."""
    from ._summary import summary as _summary

    return _summary(shot_range, preset=preset, source=source)


def export_summary(df, path, *, mode="replace", key_columns=None, replace_groups=None):
    """Serialize or upsert an already-generated summary DataFrame."""
    from ._summary import export_summary as _export_summary

    return _export_summary(
        df,
        path,
        mode=mode,
        key_columns=key_columns,
        replace_groups=replace_groups,
    )


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    # Preserve utility access patterns used by shipped notebooks and workflows.
    # High-level remote I/O still belongs to load/open/save above; these
    # compatibility attributes are deprecated at their implementation sites.
    for module_name in ("ods", "ids", "raw", "utils"):
        try:
            module = import_module(f".{module_name}", __name__)
        except Exception:
            continue
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
