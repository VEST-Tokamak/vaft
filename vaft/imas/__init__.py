"""Local IMAS importer and writer, and the native-IMAS plotting adapters.

Low-level OMAS--IMAS conversion machinery intentionally stays private in
``omas_imas``.  The public API detects local source formats automatically.

Plotting adapters (``plot_<canonical-stem>``, ``available_plots``,
``normalize_entries``, ``IDSEntry``) live in ``.plotting`` / ``.access`` and
are resolved lazily, so importing ``vaft.imas`` does not pull in Matplotlib
(issue #63).
"""

from pathlib import Path

from ..database._local import IMASHandle
from .omas_imas import IMAS_DD_VERSION_CONVERSION

__all__ = ["IMASHandle", "load", "save", "to_equilibrium", "IMAS_DD_VERSION_CONVERSION"]

_PLOTTING_NAMES = frozenset({"available_plots", "normalize_entries", "render_plot", "plotting"})
_ACCESS_NAMES = frozenset({"IDSEntry"})
_PLOTTING_EXPORTS: frozenset | None = None


def _plotting_exports() -> frozenset:
    global _PLOTTING_EXPORTS
    if _PLOTTING_EXPORTS is None:
        from importlib import import_module

        _PLOTTING_EXPORTS = frozenset(import_module(".plotting", __name__).__all__)
    return _PLOTTING_EXPORTS


def __getattr__(name):
    # import_module rather than ``from . import``: the latter probes the
    # package attribute first, which would re-enter this hook.
    from importlib import import_module

    if name in _ACCESS_NAMES:
        value = getattr(import_module(".access", __name__), name)
    elif name in _PLOTTING_NAMES or name.startswith("plot_"):
        plotting = import_module(".plotting", __name__)
        if name == "plotting":
            value = plotting
        elif name == "render_plot":
            value = plotting.render
        elif name in _plotting_exports() or name in _PLOTTING_NAMES:
            value = getattr(plotting, name)
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | _ACCESS_NAMES | _PLOTTING_NAMES | _plotting_exports())


def load(source, *, imas_version=None):
    """Open any supported local artifact as a native IMAS context manager."""
    from ..database._local import open_imas

    return open_imas(source, imas_version=imas_version)


def _occurrence_for(occurrence, ids_name: str) -> int:
    if occurrence is None:
        return 0
    if isinstance(occurrence, int):
        return occurrence
    return int(occurrence.get(ids_name, occurrence.get("*", 0)))


def save(data, target, *, imas_version=None, occurrence=None):
    """Write OMAS, native IDS, or an IMAS handle to local IMAS HDF5/NetCDF."""
    import imas
    from ..database._local import IMASHandle
    from .omas_imas import save_omas_imas

    target_path = Path(target).expanduser()
    version = imas_version or IMAS_DD_VERSION_CONVERSION
    if isinstance(data, IMASHandle):
        data = data.to_omas()

    if isinstance(data, imas.ids_toplevel.IDSToplevel):
        if target_path.suffix.lower() == ".nc":
            uri = str(target_path)
            mode = "x" if not target_path.exists() else "w"
        else:
            target_path.mkdir(parents=True, exist_ok=True)
            uri = "imas:hdf5?path=" + str(target_path)
            mode = "x" if not (target_path / "master.h5").exists() else "w"
        with imas.DBEntry(uri, mode, dd_version=version) as entry:
            ids_name = data.metadata.name
            entry.put(data, _occurrence_for(occurrence, ids_name))
        return target_path

    if target_path.suffix.lower() == ".nc":
        # Convert through a temporary HDF5 entry, then copy each native IDS into
        # IMAS-Python's netCDF backend.
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="vaft-imas-save-") as temporary:
            root = Path(temporary)
            save_omas_imas(
                data,
                occurrence=occurrence or {},
                imas_version=version,
                new=True,
                verbose=False,
                uri="imas:hdf5?path=" + str(root),
            )
            with imas.DBEntry("imas:hdf5?path=" + str(root), "r", dd_version=version) as source_entry:
                mode = "x" if not target_path.exists() else "w"
                with imas.DBEntry(str(target_path), mode, dd_version=version) as target_entry:
                    # Image filenames encode non-zero occurrences (for example,
                    # ``equilibrium_2.h5``), so derive native IDS names from
                    # the ODS roots rather than parsing staging filenames.
                    for name in sorted(data.keys()):
                        try:
                            occurrence_value = _occurrence_for(occurrence, name)
                            target_entry.put(
                                source_entry.get(name, occurrence_value),
                                occurrence_value,
                            )
                        except Exception as exc:
                            # Newer public DDs intentionally omit legacy
                            # dataset_description; do not make a valid IDS
                            # export fail because that image is unavailable.
                            if name != "dataset_description":
                                raise RuntimeError(f"Could not export IDS {name!r} to {target_path}") from exc
        return target_path

    target_path.mkdir(parents=True, exist_ok=True)
    save_omas_imas(
        data,
        occurrence=occurrence or {},
        imas_version=version,
        new=not (target_path / "master.h5").exists(),
        verbose=False,
        uri="imas:hdf5?path=" + str(target_path),
    )
    return target_path


def to_equilibrium(source, *, time_index=0, profile_index=0, convention=None):
    """Adapt an IMAS handle (or converted native IDS) for scientific algorithms."""
    from vaft.process.equilibrium import as_equilibrium

    return as_equilibrium(
        source, time_index=time_index, profile_index=profile_index, convention=convention
    )
