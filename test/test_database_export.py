"""``vaft.database.export``: one staging, many portable backends (issue #450).

HSDS is replaced by a real local IMAS HDF5 entry served through a fake
``hsget``, so every backend converts genuine IMAS data offline.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil

import h5py
import imas
import numpy as np
import pytest

import vaft
from vaft.data.eqdsk import GFILE_NAME_PATTERN, geqdsk_filenames, read_geqdsk
from vaft.database import _export, staging


DATA = Path(vaft.__file__).resolve().parent / "data"
EFIT = DATA / "efit"
SHOT = 39915
DD = "3.41.0"
# The first file twice: two slices at 317 ms exercise the collision suffix.
GFILES = [EFIT / "g039915.00317", EFIT / "g039915.00319", EFIT / "g039915.00317"]


@pytest.fixture(scope="module")
def remote_entry(tmp_path_factory):
    root = tmp_path_factory.mktemp("remote") / str(SHOT)
    vaft.imas.save(vaft.omas.load(GFILES), root, imas_version=DD)
    return root


@pytest.fixture
def fake_hsds(monkeypatch, remote_entry):
    calls: list[str] = []

    def fake_hsget(uri: str, target: Path) -> Path:
        calls.append(uri.rsplit("/", 1)[-1])
        shutil.copy2(remote_entry / calls[-1], target)
        return target

    monkeypatch.setattr(staging, "run_hsget", fake_hsget)
    monkeypatch.setattr(staging, "ensure_imas_hdf5_userblock", lambda *_: None)
    monkeypatch.setattr(
        staging,
        "materialize_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            staging.H5ImageUnavailableError("no derived fixture")
        ),
    )
    return calls


def _export_all(tmp_path, **kwargs):
    return vaft.database.export(
        SHOT, "public", backend=list(_export.BACKENDS), output=tmp_path, cache="off", **kwargs
    )


def test_every_backend_from_one_staging(fake_hsds, remote_entry, tmp_path):
    written = _export_all(tmp_path)

    assert list(written) == list(_export.BACKENDS)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [f"imas_{SHOT}_hdf5", f"imas_{SHOT}.nc", f"omas_{SHOT}.json", f"omas_{SHOT}.h5",
         f"omas_{SHOT}.nc", f"geqdsk_{SHOT}"]
    )
    # One hsget per stored file, however many backends were requested.
    remote_files = sorted(p.name for p in remote_entry.iterdir())
    assert sorted(fake_hsds) == remote_files

    # imas-hdf5 is the staged entry itself: every linked image, natively readable.
    copied = written["imas-hdf5"]
    assert sorted(p.name for p in copied.iterdir()) == remote_files
    with imas.DBEntry("imas:hdf5?path=" + str(copied), "r", dd_version=DD) as entry:
        reference = entry.get("equilibrium")
    from vaft.database._local import _detect

    assert _detect(written["imas-nc"]).imas_version == DD
    with imas.DBEntry(str(written["imas-nc"]), "r", dd_version=DD) as entry:
        from_nc = entry.get("equilibrium")
    np.testing.assert_array_equal(from_nc.time, reference.time)
    ip = [slice_.global_quantities.ip for slice_ in reference.time_slice]

    for name in ("omas-json", "omas-hdf5", "omas-nc"):
        ods = vaft.omas.load(written[name], imas_version=DD)
        assert {"equilibrium", "wall"} <= set(ods.keys()), name
        assert len(ods["equilibrium.time_slice"]) == len(GFILES), name
        np.testing.assert_allclose(
            [ods[f"equilibrium.time_slice.{i}.global_quantities.ip"] for i in range(len(GFILES))],
            ip,
            err_msg=name,
        )

    gfiles = sorted(written["geqdsk"].iterdir())
    assert [p.name for p in gfiles] == ["g039915.00317_0", "g039915.00317_1", "g039915.00319"]
    np.testing.assert_allclose(sorted(read_geqdsk(p)["CURRENT"] for p in gfiles), sorted(ip))


def test_imas_netcdf_and_omas_netcdf_are_told_apart(fake_hsds, tmp_path):
    written = vaft.database.export(
        SHOT, "public", backend=["imas-nc", "omas-nc"], output=tmp_path, cache="off"
    )
    from vaft.database._local import _detect

    assert _detect(written["imas-nc"]).format == "imas_netcdf"
    assert _detect(written["omas-nc"]).format == "omas_netcdf"


def test_existing_artifacts_are_refused_before_any_download(fake_hsds, tmp_path):
    (tmp_path / f"omas_{SHOT}.json").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match=f"omas_{SHOT}.json"):
        vaft.database.export(SHOT, "public", backend=["omas-json"], output=tmp_path, cache="off")
    assert fake_hsds == []
    assert (tmp_path / f"omas_{SHOT}.json").read_text(encoding="utf-8") == "keep"

    vaft.database.export(
        SHOT, "public", backend=["omas-json", "imas-hdf5"], output=tmp_path, cache="off", overwrite=True
    )
    assert vaft.omas.load(tmp_path / f"omas_{SHOT}.json", imas_version=DD)["equilibrium.time_slice"]
    # Overwriting a directory artifact replaces it rather than nesting into it.
    vaft.database.export(SHOT, "public", backend="imas-hdf5", output=tmp_path, cache="off", overwrite=True)
    assert (tmp_path / f"imas_{SHOT}_hdf5" / "master.h5").is_file()
    assert not (tmp_path / f"imas_{SHOT}_hdf5" / f"imas_{SHOT}_hdf5").exists()


def test_a_failed_backend_leaves_nothing_that_looks_complete(fake_hsds, monkeypatch, tmp_path):
    def broken(ods, shot, directory):
        directory.mkdir()
        (directory / "g039915.00317").write_text("half", encoding="utf-8")
        raise RuntimeError("serializer failed")

    monkeypatch.setattr(_export, "_write_geqdsk", broken)
    with pytest.raises(RuntimeError, match="serializer failed"):
        vaft.database.export(
            SHOT, "public", backend=["omas-json", "geqdsk"], output=tmp_path, cache="off"
        )
    names = sorted(p.name for p in tmp_path.iterdir())
    # The backend that finished is complete; the one that failed left nothing.
    assert names == [f"omas_{SHOT}.json"]


def test_default_output_is_the_working_directory(fake_hsds, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    written = vaft.database.export(SHOT, "public", backend="imas-hdf5", cache="off")
    assert written == {"imas-hdf5": tmp_path / f"imas_{SHOT}_hdf5"}


def test_converted_backends_refuse_other_occurrences(fake_hsds, monkeypatch, remote_entry, tmp_path):
    with pytest.raises(ValueError, match="occurrence 0 only"):
        vaft.database.export(SHOT, "public", backend="omas-json", output=tmp_path, occurrence=1)

    remote = tmp_path / "remote"
    shutil.copytree(remote_entry, remote)
    with imas.DBEntry("imas:hdf5?path=" + str(remote), "r", dd_version=DD) as entry:
        equilibrium = entry.get("equilibrium")
    with imas.DBEntry("imas:hdf5?path=" + str(remote), "a", dd_version=DD) as entry:
        entry.put(equilibrium, 2)

    def fake_hsget(uri, target):
        shutil.copy2(remote / uri.rsplit("/", 1)[-1], target)
        return target

    monkeypatch.setattr(staging, "run_hsget", fake_hsget)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=r"equilibrium: \[2\]"):
        vaft.database.export(SHOT, "public", backend="omas-json", output=out, cache="off")
    assert not out.exists() or not any(out.iterdir())
    # The native copy keeps every occurrence.
    written = vaft.database.export(SHOT, "public", backend="imas-hdf5", output=out, cache="off")
    assert (written["imas-hdf5"] / "equilibrium_2.h5").is_file()


def test_invalid_requests_fail_before_staging(fake_hsds, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="choose from"):
        vaft.database.export(SHOT, "public", backend=["imas-hdf5", "nc"], output=tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        vaft.database.export(SHOT, "public", backend=[], output=tmp_path)

    import builtins

    real_import = builtins.__import__

    def no_netcdf(name, *args, **kwargs):
        if name == "netCDF4":
            raise ImportError("No module named 'netCDF4'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_netcdf)
    with pytest.raises(ImportError, match="pip install netCDF4"):
        vaft.database.export(SHOT, "public", backend=["omas-nc"], output=tmp_path)
    assert fake_hsds == []


def test_geqdsk_filenames_are_unique_and_parse_back():
    assert geqdsk_filenames(41672, [0.319, 0.320]) == ["g041672.00319", "g041672.00320"]
    # 319.1 ms and 319.4 ms quantize to one name: both get a rank suffix.
    assert geqdsk_filenames(41672, [0.3191, 0.320, 0.3194]) == [
        "g041672.00319_0", "g041672.00320", "g041672.00319_1"
    ]
    for name in geqdsk_filenames(41672, [0.3191, 0.3194]):
        match = GFILE_NAME_PATTERN.match(name)
        assert (match["shot"], match["time"]) == ("041672", "00319")


@pytest.mark.integration
def test_public_hsds_export(tmp_path):
    if os.environ.get("VAFT_RUN_HSDS_INTEGRATION") != "1":
        pytest.skip("set VAFT_RUN_HSDS_INTEGRATION=1 for the read-only public HSDS export")
    shot = 41672
    written = vaft.database.export(
        shot, "public", backend=["imas-hdf5", "imas-nc", "omas-json", "geqdsk"], output=tmp_path
    )
    with h5py.File(written["imas-hdf5"] / "master.h5", "r") as master:
        assert "equilibrium" in master
    with imas.DBEntry("imas:hdf5?path=" + str(written["imas-hdf5"]), "r") as entry:
        slices = len(entry.get("equilibrium").time_slice)
    with imas.DBEntry(str(written["imas-nc"]), "r") as entry:
        assert len(entry.get("equilibrium").time_slice) == slices
    assert len(vaft.omas.load(written["omas-json"])["equilibrium.time_slice"]) == slices
    gfiles = sorted(written["geqdsk"].iterdir())
    assert len(gfiles) == slices
    for path in gfiles:
        read_geqdsk(path)
