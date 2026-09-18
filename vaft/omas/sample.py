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
        the same slice load with :func:`sample_equilibria`. 45531 and 40600
        are repository-only fluctuation samples: 45531 carries the three-angle
        outboard fluctuation Mirnov array at its native 2 MHz / 500 kHz and a
        976.6 kHz soft X-ray subset around its plasma window; 40600 carries
        50 kFrames/s FAST-camera frames beside the outboard probe
        ``b_field_pol_probe.36`` (DAQ field 171) at 250 kHz. Neither has an
        equilibrium.

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
        added.
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


def sample_gfile():
    """Load the historical packaged sample g-file as a VAFT GEQDSK object."""
    from vaft.data.resources import sample_geqdsk

    return sample_geqdsk("efit/g039915.00317")
