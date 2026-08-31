from vaft.data import sample as _sample_path

__all__ = ["sample_ods", "sample_gfile"]


def sample_ods():
    """Load the compact shot-39915 OMAS reference sample.

    This is the one-line entry point for examples, teaching material and quick
    checks: no arguments, no configuration, no network access. Tutorial 01 uses
    it throughout.

    Use ``vaft.data.sample`` with an explicit adapter when you need to choose
    the shot or the representation.
    """
    from ..database._local import load_ods

    ods, _ = load_ods(_sample_path(39915, representation="omas"), imas_version="3.41.0")
    return ods


def sample_gfile():
    """Load the historical packaged sample g-file as a VAFT GEQDSK object."""
    from vaft.data.resources import sample_geqdsk

    return sample_geqdsk("efit/g039915.00317")
