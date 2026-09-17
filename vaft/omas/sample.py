from vaft.data import sample as _sample_path

__all__ = ["sample_ods", "sample_gfile"]


def sample_ods(shot=39915):
    """Load a packaged VEST reference sample as an OMAS ODS.

    This is the one-line entry point for examples, teaching material and quick
    checks: no configuration, no network access. Tutorial 01 uses it throughout.

    Parameters
    ----------
    shot : int, optional
        One of ``vaft.data.available_samples()``. The default, 39915, is the
        compact sample that also ships in the wheel. 41524 and 41672 are full
        pipeline products (every EFIT slice, passive currents, coupling
        matrices) stored as repository-only ``omas.json.gz`` files, so they
        load from a Git checkout and raise ``FileNotFoundError`` with cloning
        instructions from an installed wheel.

    Use ``vaft.data.sample`` with an explicit adapter when you need to choose
    the representation.
    """
    from ..database._local import load_ods

    ods, _ = load_ods(_sample_path(int(shot), representation="omas"), imas_version="3.41.0")
    return ods


def sample_gfile():
    """Load the historical packaged sample g-file as a VAFT GEQDSK object."""
    from vaft.data.resources import sample_geqdsk

    return sample_geqdsk("efit/g039915.00317")
