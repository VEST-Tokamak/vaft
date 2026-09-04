"""Database access and storage infrastructure.

The high-level :func:`load`, :func:`open`, and :func:`save` APIs operate on
named HSDS sources -- one namespace per analysis lineage, so an EFIT baseline
and its CHEASE refinement of the same shot never overwrite each other. The
catalog, the ``main`` default and the read-only rule for the legacy ``public``
namespace all live in :mod:`vaft.database.sources`; ``directory=`` and
``target=`` remain accepted as deprecated aliases of ``source=``.

Sources are read one at a time and never merged: :func:`load` on ``main``
returns ``main``.  Analysis that wants an optional diagnostic alongside the
baseline -- IMPA, whose sparse source is described in :mod:`vaft.database.sources`
-- asks for both through :func:`vaft.database.compose`, which keeps the source of
every channel visible.

Canonical local FileDB paths live in :mod:`vaft.database.filedb`; local
OMAS/IMAS artifact loading remains exposed through :mod:`vaft.omas` and
:mod:`vaft.imas`.
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
    "sources",
    "replication",
    "composition",
    "compose",
    "production_qa",
    "plotting",
    "load",
    "open",
    "save",
    "summary",
    "export_summary",
    "get_summary_preset",
]


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
        for name in utils.h5pyd.Folder(f"/{source}/{shot}/", mode="r")
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
    source: str | None = None,
    *,
    directory: str | None = None,
    representation: Literal["omas", "imas"] = "omas",
    paths: str | list[str] | None = None,
    occurrence: int | Mapping[str, int] | None = None,
    imas_version: str | None = None,
    cache: str = "auto",
    transport: Literal["auto", "canonical", "h5image"] = "auto",
):
    """Materialize an OMAS ODS or native IMAS IDS from remote HSDS storage.

    ``source`` names the HSDS namespace and defaults to ``main``; pass
    ``"public"`` to read the legacy reference. ``directory`` is a deprecated
    alias for ``source``.
    """
    from .sources import resolve

    source = resolve(source, directory=directory)
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
            source=source,
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
        source=source,
        occurrence=occurrences,
        dd_version=version,
        cache=cache,
        transport=transport,
    )


def open(
    shot: int,
    *,
    source: str | None = None,
    directory: str | None = None,
    representation: Literal["omas", "imas"] = "omas",
    paths: str | list[str] | None = None,
    occurrence: int | Mapping[str, int] | None = None,
    imas_version: str | None = None,
):
    """Open a read-only lazy OMAS ODS or native IMAS IDS adapter over HSDS.

    ``source`` names the HSDS namespace and defaults to ``main``; ``directory``
    is a deprecated alias for it.
    """
    if representation not in {"omas", "imas"}:
        raise ValueError("representation must be 'omas' or 'imas'")
    from .sources import resolve

    source = resolve(source, directory=directory)
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
            int(shot), source=source, ids=names, imas_version=stored_version
        )
    from .lazy_ods import open_ods

    return open_ods(
        int(shot),
        source=source,
        ids=_root_ids(selected_paths),
        imas_version=_infer_remote_imas_version(
            source, int(shot), _root_ids(selected_paths), imas_version
        ),
    )


def save(
    data,
    shot: int,
    *,
    source: str | None = None,
    target: str | None = None,
    directory: str | None = None,
    representation: Literal["omas", "imas"] | None = None,
    occurrence: int | Mapping[str, int] | None = None,
    imas_version: str | None = None,
    derived_cache: Literal["auto", "none", "imas-images", "omas", "both"] = "auto",
):
    """Write an OMAS ODS or native IDS to remote HSDS storage.

    ``source`` names the HSDS namespace to publish into and defaults to
    ``main``. The legacy ``public`` source is read-only and is refused here, so
    a lineage cannot be overwritten by falling back to it. ``target`` and
    ``directory`` are deprecated aliases for ``source``.
    """
    from .sources import resolve

    source = resolve(source, directory=directory, target=target, writable=True)
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
            source=source,
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
        source=source,
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


def summary(shot_range=None, *, preset="equilibrium_global", source=None, directory=None):
    """Return a canonical preset summary for a range or all available shots.

    ``source`` names the HSDS namespace and defaults to ``main``; ``directory``
    is a deprecated alias for it.
    """
    from ._summary import summary as _summary
    from .sources import resolve

    # Resolve here so a deprecated alias is reported against the caller's frame
    # rather than this wrapper.
    return _summary(shot_range, preset=preset, source=resolve(source, directory=directory))


def get_summary_preset(name):
    """Return the :class:`SummaryPreset` describing one canonical summary sheet.

    The preset is the single source of truth for a sheet's column names, key
    columns and sort order, so consumers validate against it instead of
    hard-coding a schema.
    """
    from ._summary import get_summary_preset as _get_summary_preset

    return _get_summary_preset(name)


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
    # Composing two sources is an explicit request, never something `load` does
    # on its own (issue #305); the helper is exposed here so asking for it is as
    # easy as asking for one source, not so that anything does it implicitly.
    if name == "compose":
        value = getattr(import_module(".composition", __name__), "compose")
        globals()[name] = value
        return value
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    # Plot adapters (issue #63): resolved lazily so importing vaft.database
    # pulls in neither the plotting stack nor Matplotlib.
    if name.startswith("plot_") or name == "available_plots":
        plotting = import_module(".plotting", __name__)
        try:
            value = getattr(plotting, name)
        except AttributeError:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
        globals()[name] = value
        return value
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
    # Listing the adapters needs the registry, which brings the plotting
    # stack in; a plain import never does -- only dir() pays for completion.
    names = set(globals()) | set(__all__)
    try:
        names |= set(dir(import_module(".plotting", __name__)))
    except Exception:  # pragma: no cover - the plotting stack is optional at dir() time
        pass
    return sorted(n for n in names if not n.startswith("_"))
