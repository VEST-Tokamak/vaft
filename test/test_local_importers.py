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
