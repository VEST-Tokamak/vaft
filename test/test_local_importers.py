from pathlib import Path

import pytest

import vaft
import vaft.database._local as _local_io


DATA = Path(__file__).resolve().parents[1] / "vaft" / "data"
GFILE = DATA / "efit" / "g039915.00317"
GFILES = [DATA / "efit" / "g039915.00317", DATA / "efit" / "g039915.00319"]
IMAS_NC = DATA / "samples" / "39915" / "imas.nc"


def test_omas_load_detects_single_and_multiple_geqdsk():
    single = vaft.omas.load(GFILE)
    multiple = vaft.omas.load(GFILES)

    assert "equilibrium" in single
    assert len(multiple["equilibrium.time_slice"]) == 2


def test_omas_json_and_hdf5_round_trip(tmp_path):
    source = vaft.omas.load(GFILE)
    for suffix in (".json", ".h5"):
        target = tmp_path / f"equilibrium{suffix}"
        vaft.omas.save(source, target)
        restored = vaft.omas.load(target)
        assert (
            restored["equilibrium.time_slice.0.profiles_2d.0.psi"].shape
            == source["equilibrium.time_slice.0.profiles_2d.0.psi"].shape
        )


@pytest.mark.parametrize("name", ["magnetics.json.gz", "magnetics.json", "magnetics.h5"])
def test_a_failed_save_leaves_the_previous_product_intact(tmp_path, monkeypatch, name):
    """A writer that dies mid-way must not disturb the canonical file.

    ``save`` used to open the target in place, so a disk-full or a killed
    worker left a short file under the canonical name that the next reader
    failed on with ``JSONDecodeError`` (cold review 0.7.0 data F16, #1888).
    Every format now writes beside the target and ``os.replace``s it, so a
    failure at any point of the write leaves the previous product as it was.
    The fault here fires after the bytes of the new product were written, the
    latest point a failure can occur.
    """
    import shutil

    from omas import ODS

    ods = ODS()
    ods["magnetics.time"] = [0.0, 1.0, 2.0]
    target = tmp_path / name
    vaft.omas.save(ods, target)
    good = target.read_bytes()

    def failing(real):
        def wrapper(*args, **kwargs):
            real(*args, **kwargs)
            raise OSError("disk full mid-write")
        return wrapper

    if name.endswith(".gz"):
        monkeypatch.setattr(shutil, "copyfileobj", failing(shutil.copyfileobj))
    else:
        monkeypatch.setattr(ODS, "save", failing(ODS.save))
    bigger = ODS()
    bigger["magnetics.time"] = list(range(50))
    with pytest.raises(OSError, match="disk full"):
        vaft.omas.save(bigger, target)

    assert target.read_bytes() == good
    assert sorted(p.name for p in tmp_path.iterdir()) == [name], "no staging file left behind"


def test_threads_saving_the_same_target_do_not_share_a_staging_file(tmp_path):
    """Four threads, each saving the same product repeatedly, all succeed.

    The staging name was `.<stem>.tmp-<pid><suffix>`, the same path for every
    thread of a process: one thread's `os.replace` moved another's half-written
    bytes onto the canonical name, and the other then failed with
    FileNotFoundError (PR #1926 review F1).  The token is now random per call.
    """
    import threading

    from omas import ODS

    ods = ODS()
    ods["magnetics.time"] = list(range(2000))
    target = tmp_path / "shared.json"
    errors: list[BaseException] = []

    def worker():
        try:
            for _ in range(5):
                vaft.omas.save(ods, target)
        except BaseException as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shared.json"]
    assert list(vaft.omas.load(target)["magnetics.time"]) == list(range(2000))


def test_imas_handle_detects_netcdf_and_converts_omas_source(tmp_path):
    with vaft.imas.load(IMAS_NC) as handle:
        assert handle.info.format == "imas_netcdf"
        assert "equilibrium" in handle.ids
        assert "equilibrium" in handle.info.available_ids
        assert type(handle.get("equilibrium")).__name__ == "IDSToplevel"

    json_source = tmp_path / "equilibrium.json"
    vaft.omas.save(vaft.omas.load(GFILE), json_source)
    with vaft.imas.load(json_source) as handle:
        assert handle.info.converted is True
        assert type(handle.get("equilibrium")).__name__ == "IDSToplevel"


def test_imas_hdf5_round_trip(tmp_path):
    source = vaft.omas.load(GFILE)
    target = tmp_path / "imas_entry"
    vaft.imas.save(source, target)
    assert (target / "master.h5").exists()
    with vaft.imas.load(target) as handle:
        assert handle.info.format == "imas_hdf5"
        assert type(handle.get("equilibrium")).__name__ == "IDSToplevel"

    # A single external image is usable without a manually prepared master.
    with vaft.imas.load(target / "equilibrium.h5") as handle:
        assert handle.info.format == "imas_images"
        assert handle.ids == ("equilibrium",)
        assert type(handle.get("equilibrium")).__name__ == "IDSToplevel"

    native_target = tmp_path / "native_occurrence"
    with vaft.imas.load(target) as handle:
        native = handle.get("equilibrium")
    vaft.imas.save(native, native_target, occurrence=2)
    with vaft.imas.load(native_target) as handle:
        assert type(handle.get("equilibrium", occurrence=2)).__name__ == "IDSToplevel"


def test_imas_netcdf_save_round_trip(tmp_path):
    source = vaft.omas.load(GFILE)
    target = tmp_path / "equilibrium.nc"
    vaft.imas.save(source, target, occurrence={"equilibrium": 2})
    # Existing NetCDF targets are replaced with the same occurrence mapping.
    vaft.imas.save(source, target, occurrence={"equilibrium": 2})
    with vaft.imas.load(target) as handle:
        assert type(handle.get("equilibrium", occurrence=2)).__name__ == "IDSToplevel"


def test_unknown_local_source_is_actionable(tmp_path):
    unknown = tmp_path / "unknown.bin"
    unknown.write_bytes(b"not a supported data source")
    with pytest.raises(ValueError, match="Unsupported local source"):
        vaft.omas.load(unknown)


def test_netcdf_without_version_metadata_uses_imas_fallback(tmp_path):
    source = tmp_path / "minimal.nc"
    with _local_io.h5py.File(source, "w") as handle:
        handle.create_group("equilibrium")
    descriptor = _local_io._detect(source)
    assert descriptor.format == "imas_netcdf"
    with pytest.warns(RuntimeWarning, match="using 3.41.0"):
        assert _local_io._resolved_version(descriptor, None) == ("3.41.0", True)


def test_imas_handle_is_available_to_star_import():
    namespace = {}
    exec("from vaft.imas import *", namespace)
    assert namespace["IMASHandle"] is vaft.imas.IMASHandle


def _scratch_count(prefix: str) -> int:
    import tempfile

    return len(list(Path(tempfile.gettempdir()).glob(f"{prefix}*")))


def test_partial_entry_reclaims_its_scratch_tree_when_staging_fails(monkeypatch):
    """mkdtemp has no finalizer, so every failure path must clean up itself.

    A TemporaryDirectory used to reclaim the tree at garbage-collection time;
    without that safety net a mid-loop copy failure would leak megabytes of
    staged IDS images for the life of the machine.
    """
    images = [IMAS_NC.with_name("magnetics.h5"), IMAS_NC.with_name("wall.h5")]

    def explode(*_args, **_kwargs):
        raise OSError("staging device is full")

    monkeypatch.setattr(_local_io.shutil, "copy2", explode)
    before = _scratch_count("vaft-local-imas-")
    with pytest.raises(OSError, match="staging device is full"):
        _local_io._make_partial_entry(images)
    assert _scratch_count("vaft-local-imas-") == before


def test_imas_handle_reclaims_its_scratch_tree_when_open_fails(monkeypatch):
    """`__enter__` raising means `__exit__` never runs, so open() must clean up."""
    handle = _local_io.open_imas(
        DATA / "samples" / "39915" / "omas.json.gz", imas_version="3.41.0"
    )

    def explode(*_args, **_kwargs):
        raise RuntimeError("the DD rejected an IDS")

    import vaft.imas.omas_imas as omas_imas

    monkeypatch.setattr(omas_imas, "save_omas_imas", explode)
    before = _scratch_count("vaft-imas-handle-")
    with pytest.raises(RuntimeError, match="the DD rejected an IDS"):
        handle.open()
    assert _scratch_count("vaft-imas-handle-") == before
