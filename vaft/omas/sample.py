from vaft.data import sample as _sample_path

__all__ = ["sample_ods", "sample_equilibria", "sample_pressure_weight_scan", "sample_gfile"]


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


def _declared_ods(shot, label, record):
    """Load one manifest-declared equilibrium file after checking its SHA-256."""
    import hashlib

    from vaft.data import read_geqdsk
    from vaft.data.resources import data_path, require_repository_sample

    path = require_repository_sample(data_path(f"samples/{int(shot)}/{record['path']}"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Checksum mismatch for sample {int(shot)} {label!r}: {path}")
    if record.get("format", "geqdsk") == "omas":
        from vaft.omas import load

        return load(path, imas_version="3.41.0")
    return read_geqdsk(path).to_omas()


def sample_equilibria(shot=48224):
    """Load the labelled equilibria a packaged sample carries beside its ODS.

    Some samples keep more than one reconstruction of the same slice so that a
    tutorial can put them side by side. They are listed under ``equilibria`` in
    the sample manifest, each with its file and that file's SHA-256: compact
    equilibrium ODSs (``format: omas``) or g-files converted with
    ``read_geqdsk(path).to_omas()`` (``format: geqdsk``). psi is in Wb.

    Shot 48224 at 300 ms returns

    ``efit_magnetic``
        EFIT on the magnetic k-file alone (no pressure constraint), with its
        constraints: measured, reconstructed and chi-square per channel.
    ``efit_kinetic``
        VAFT's kinetic EFIT on the same k-file with the Thomson pressure points
        added -- the nominal point of :func:`sample_pressure_weight_scan`.
    ``chease``
        The CHEASE refinement of an earlier kinetic EFIT of this slice.

    The manifest's ``role`` and ``notes`` entries say what each one is and what
    its constraints do and do not mean (the magnetic probes of this k-file are
    weighted out, and EFIT's chi-square is its plasma-current term).

    Returns
    -------
    dict
        ``{label: ODS}`` in manifest order.

    Raises
    ------
    ValueError
        The sample declares no equilibria, or a file's checksum does not match.
    FileNotFoundError
        The files are repository-only and absent from an installed wheel.
    """
    from vaft.data.resources import sample_manifest

    manifest = sample_manifest(int(shot))
    entries = (manifest.get("equilibria") or {}).get("entries") or {}
    if not entries:
        raise ValueError(f"VAFT sample shot {int(shot)} declares no equilibria")
    return {label: _declared_ods(shot, label, record) for label, record in entries.items()}


def sample_pressure_weight_scan(shot=48224):
    """Load a precomputed kinetic-EFIT pressure-weight scan as ``{factor: ODS}``.

    Each ODS is the same instant reconstructed with every pressure point's
    ``FWTPRE`` multiplied by ``factor``; the dict goes straight into
    ``vaft.plot.equilibrium_overview_pressure_weight_scan``. The scan settings
    (points, sigmas, base k-file) and a table of the results are under
    ``pressure_weight_scan`` in the sample manifest.

    For shot 48224 the three factors 0.1, 1 and 10 give the same equilibrium to
    1e-8: the magnetic channels are weighted out, so the pressure points are the
    only information on p', and scaling all of their weights together leaves the
    least-squares solution unchanged. A weight only matters against a
    constraint that competes with it.

    Returns
    -------
    dict
        ``{float factor: ODS}`` in ascending factor order.
    """
    from vaft.data.resources import sample_manifest

    manifest = sample_manifest(int(shot))
    scan = manifest.get("pressure_weight_scan") or {}
    entries = scan.get("entries") or {}
    if not entries:
        raise ValueError(f"VAFT sample shot {int(shot)} declares no pressure-weight scan")
    return {
        float(factor): _declared_ods(shot, f"pressure_weight_scan[{factor}]", record)
        for factor, record in sorted(entries.items(), key=lambda item: float(item[0]))
    }


def sample_gfile():
    """Load the historical packaged sample g-file as a VAFT GEQDSK object."""
    from vaft.data.resources import sample_geqdsk

    return sample_geqdsk("efit/g039915.00317")
