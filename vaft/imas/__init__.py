"""Local IMAS importer and writer.

Low-level OMAS--IMAS conversion machinery intentionally stays private in
``omas_imas``.  The public API detects local source formats automatically.
"""

from pathlib import Path

from .._local_io import IMASHandle
from .omas_imas import IMAS_DD_VERSION_CONVERSION

__all__ = ["IMASHandle", "load", "save", "IMAS_DD_VERSION_CONVERSION"]


def load(source, *, imas_version=None):
    """Open any supported local artifact as a native IMAS context manager."""
    from .._local_io import open_imas

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
    from .._local_io import IMASHandle
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
