from .general import *
from .process_wrapper import *
from .formula_wrapper import *
from .update import *
from .sample import *


_REFERENCE_EXPORTS = {
    "ArtifactVerification",
    "ReferenceManifestError",
    "load_reference_manifest",
    "sha256_file",
    "verify_reference_artifacts",
}
_COMPARISON_EXPORTS = {
    "ComparisonEntry",
    "DifferenceKind",
    "ODSComparison",
    "ParityClassification",
    "Tolerance",
    "TolerancePolicy",
    "ToleranceRule",
    "compare_ods",
    "load_tolerance_policy",
    "write_comparison_reports",
}


def __getattr__(name):
    if name in _REFERENCE_EXPORTS:
        from . import reference

        value = getattr(reference, name)
        globals()[name] = value
        return value
    if name in _COMPARISON_EXPORTS:
        from . import comparison

        value = getattr(comparison, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _REFERENCE_EXPORTS | _COMPARISON_EXPORTS)


def load(source, *, imas_version=None):
    """Read any supported local artifact and return a normalized OMAS ODS.

    ``source`` may be OMAS JSON/HDF5, an IMAS netCDF file, an IMAS HDF5
    directory/image set, a GEQDSK file, or a sequence of GEQDSK files.
    """
    from ..io._local import load_ods

    ods, _info = load_ods(source, imas_version=imas_version)
    return ods


def save(ods, target):
    """Save an OMAS ODS as JSON or HDF5, chosen from ``target``'s suffix."""
    import gzip
    from pathlib import Path
    import shutil
    import tempfile

    target_path = Path(target).expanduser()
    suffixes = target_path.suffixes
    if (
        target_path.suffix.lower() not in {".h5", ".hdf5", ".json"}
        and suffixes[-2:] != [".json", ".gz"]
    ):
        raise ValueError("vaft.omas.save target must end in .json, .json.gz, .h5, or .hdf5")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if suffixes[-2:] == [".json", ".gz"]:
        with tempfile.NamedTemporaryFile(suffix=".json") as plain:
            ods.save(plain.name)
            plain.flush()
            plain.seek(0)
            with target_path.open("wb") as target_handle:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=target_handle,
                    compresslevel=9,
                    mtime=0,
                ) as compressed:
                    shutil.copyfileobj(plain, compressed)
    else:
        ods.save(str(target_path))
    return target_path
