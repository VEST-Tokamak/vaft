from os import fspath

from omas import load_omas_json as _omas_load_omas_json

from .general import *
from .process_wrapper import *
from .formula_wrapper import *
from .update import *
from .sample import *

#: Plotting adapters live in ``.plotting`` and are resolved lazily so that
#: importing ``vaft.omas`` does not pull in Matplotlib.
def _plotting_exports() -> frozenset:
    from .plotting import __all__ as names

    return frozenset(names)


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


_PLOTTING_EXPORTS: frozenset | None = None


def _is_plotting_export(name: str) -> bool:
    global _PLOTTING_EXPORTS
    if name != "plotting" and not (
        name.startswith("plot_")
        or name in {
            "available_plots",
            "disable_plot_methods",
            "enable_plot_methods",
            "extract_labels_from_odc",
            "normalize_entries",
            "render_plot",
        }
    ):
        return False
    if _PLOTTING_EXPORTS is None:
        _PLOTTING_EXPORTS = _plotting_exports()
    return name == "plotting" or name in _PLOTTING_EXPORTS or name == "render_plot"


def __getattr__(name):
    if _is_plotting_export(name):
        from . import plotting

        value = plotting if name == "plotting" else getattr(
            plotting, "render" if name == "render_plot" else name
        )
        globals()[name] = value
        return value
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
    return sorted(
        set(globals())
        | _REFERENCE_EXPORTS
        | _COMPARISON_EXPORTS
        | _plotting_exports()
        | {"plotting", "render_plot"}
    )


def load_omas_json(source, *args, **kwargs):
    """Load OMAS JSON from a string or any :class:`os.PathLike` path."""
    return _omas_load_omas_json(fspath(source), *args, **kwargs)


def load(source, *, imas_version=None):
    """Read any supported local artifact and return a normalized OMAS ODS.

    ``source`` may be OMAS JSON/HDF5, an IMAS netCDF file, an IMAS HDF5
    directory/image set, a GEQDSK file, or a sequence of GEQDSK files.
    """
    from ..database._local import load_ods

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


def to_equilibrium(ods, *, time_index=0, profile_index=0, convention=None):
    """Adapt one ODS equilibrium slice to the lightweight scientific model."""
    from vaft.process.equilibrium import as_equilibrium

    return as_equilibrium(
        ods, time_index=time_index, profile_index=profile_index, convention=convention
    )
