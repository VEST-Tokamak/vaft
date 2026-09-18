from vaft.data import sample as _sample_path

__all__ = ["sample_ods", "sample_equilibria", "sample_gfile"]


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
        instructions from an installed wheel. 48224 is a repository-only
        kinetic slice at 300 ms (Thomson, charge exchange, core_profiles and
        the equilibrium they were mapped on); its other reconstructions of
        the same slice load with :func:`sample_equilibria`.

    Use ``vaft.data.sample`` with an explicit adapter when you need to choose
    the representation.
    """
    from ..database._local import load_ods

    ods, _ = load_ods(_sample_path(int(shot), representation="omas"), imas_version="3.41.0")
    return ods


def sample_equilibria(shot=48224):
    """Load the labelled equilibria a packaged sample carries beside its ODS.

    Some samples keep more than one reconstruction of the same slice, so that a
    tutorial can put them side by side. They are listed under ``equilibria`` in
    the sample manifest, each with the g-file it comes from and that file's
    SHA-256, and are converted here with ``read_geqdsk(path).to_omas()`` (psi in
    Wb, as the IMAS DD defines it).

    Shot 48224 at 300 ms returns ``efit_reference`` (the EFIT equilibrium the
    kinetic chain started from), ``efit_kinetic`` (VAFT's kinetic EFIT with the
    Thomson pressure constraint) and ``chease`` (the CHEASE refinement of the
    kinetic EFIT). The manifest's ``role`` entries say what each one is,
    including what cannot be established from the files themselves.

    Parameters
    ----------
    shot : int, optional
        A sample whose manifest declares ``equilibria``; 48224 is the only one.

    Returns
    -------
    dict
        ``{label: ODS}`` in manifest order.

    Raises
    ------
    ValueError
        The sample declares no equilibria, or a file's checksum does not match.
    FileNotFoundError
        The g-files are repository-only and absent from an installed wheel.
    """
    import hashlib

    from vaft.data import read_geqdsk
    from vaft.data.resources import data_path, require_repository_sample, sample_manifest

    manifest = sample_manifest(int(shot))
    entries = (manifest.get("equilibria") or {}).get("entries") or {}
    if not entries:
        raise ValueError(f"VAFT sample shot {int(shot)} declares no equilibria")
    equilibria = {}
    for label, record in entries.items():
        path = require_repository_sample(data_path(record["path"]))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(
                f"Checksum mismatch for sample {int(shot)} equilibrium {label!r}: {path}"
            )
        equilibria[label] = read_geqdsk(path).to_omas()
    return equilibria


def sample_gfile():
    """Load the historical packaged sample g-file as a VAFT GEQDSK object."""
    from vaft.data.resources import sample_geqdsk

    return sample_geqdsk("efit/g039915.00317")
