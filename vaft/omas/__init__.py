from .general import *
from .process_wrapper import *
from .formula_wrapper import *
from .update import *
from .sample import *


def load(source, *, imas_version=None):
    """Read any supported local artifact and return a normalized OMAS ODS.

    ``source`` may be OMAS JSON/HDF5, an IMAS netCDF file, an IMAS HDF5
    directory/image set, a GEQDSK file, or a sequence of GEQDSK files.
    """
    from .._local_io import load_ods

    ods, _info = load_ods(source, imas_version=imas_version)
    return ods


def save(ods, target):
    """Save an OMAS ODS as JSON or HDF5, chosen from ``target``'s suffix."""
    from pathlib import Path

    target_path = Path(target).expanduser()
    suffixes = target_path.suffixes
    if target_path.suffix.lower() not in {".h5", ".hdf5", ".json"} and suffixes[-2:] != [".json", ".gz"]:
        raise ValueError("vaft.omas.save target must end in .json, .json.gz, .h5, or .hdf5")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    ods.save(str(target_path))
    return target_path
