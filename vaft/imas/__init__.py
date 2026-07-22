"""Local IMAS importer and writer.

Low-level OMAS--IMAS conversion machinery intentionally stays private in
``omas_imas``.  The public API detects local source formats automatically.
"""

from pathlib import Path

from .omas_imas import IMAS_DD_VERSION_CONVERSION

__all__ = ["IMASHandle", "load", "save", "IMAS_DD_VERSION_CONVERSION"]


def load(source, *, imas_version=None):
    """Open any supported local artifact as a native IMAS context manager."""
    from .._local_io import IMASHandle, open_imas

    _ = IMASHandle
    return open_imas(source, imas_version=imas_version)


def save(data, target, *, imas_version=None, occurrence=None):
    """Write OMAS, native IDS, or an IMAS handle to local IMAS HDF5/NetCDF."""
    import imas
    from .._local_io import IMASHandle
    from .omas_imas import save_omas_imas

    target_path = Path(target).expanduser()
    version = imas_version or IMAS_DD_VERSION_CONVERSION
    occurrence = occurrence or {}
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
            entry.put(data)
        return target_path

    if target_path.suffix.lower() == ".nc":
        # Convert through a temporary HDF5 entry, then copy each native IDS into
        # IMAS-Python's netCDF backend.
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="vaft-imas-save-") as temporary:
            root = Path(temporary)
            save_omas_imas(
                data,
                occurrence=occurrence,
                imas_version=version,
                new=True,
                verbose=False,
                uri="imas:hdf5?path=" + str(root),
            )
            with imas.DBEntry("imas:hdf5?path=" + str(root), "r", dd_version=version) as source_entry:
                with imas.DBEntry(str(target_path), "x", dd_version=version) as target_entry:
                    for name in sorted(path.stem for path in root.glob("*.h5") if path.name != "master.h5"):
                        try:
                            target_entry.put(source_entry.get(name, occurrence.get(name, 0)))
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
        occurrence=occurrence,
        imas_version=version,
        new=not (target_path / "master.h5").exists(),
        verbose=False,
        uri="imas:hdf5?path=" + str(target_path),
    )
    return target_path
