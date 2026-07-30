from pathlib import Path
from types import SimpleNamespace

import pytest

from vaft.database import transport


def test_hsget_reports_remote_size_and_progress(monkeypatch, tmp_path, caplog):
    calls = []

    def fake_which(command):
        return f"/tools/{command}"

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0].endswith("hsstat"):
            return SimpleNamespace(returncode=0, stdout="total_size: 1234\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(transport.shutil, "which", fake_which)
    monkeypatch.setattr(transport.subprocess, "run", fake_run)

    target = tmp_path / "shot" / "equilibrium.h5"
    with caplog.at_level("INFO", logger=transport.__name__):
        assert transport.run_hsget("hdf5://public/1/equilibrium.h5", target) == target
    assert calls[-1] == [
        "/tools/hsget",
        "hdf5://public/1/equilibrium.h5",
        str(target),
    ]
    output = caplog.text
    assert "total_size=1234" in output
    assert f"staging={target}" in output
    assert "status=starting" in output
    assert "status=complete" in output


def test_missing_command_has_actionable_cross_platform_message(monkeypatch, tmp_path):
    monkeypatch.setattr(transport.shutil, "which", lambda command: None)
    with pytest.raises(transport.HSDSCommandNotFoundError) as excinfo:
        transport.run_hsget("hdf5://public/1/master.h5", tmp_path / "master.h5")
    message = str(excinfo.value)
    assert "hsget" in message
    assert "PATH" in message
    assert "bin/Scripts" in message


def test_nonzero_exit_is_wrapped_with_code_and_remediation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        transport.shutil,
        "which",
        lambda command: None if command == "hsstat" else f"/tools/{command}",
    )
    monkeypatch.setattr(
        transport.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=13, stdout="", stderr="permission denied"
        ),
    )
    with pytest.raises(transport.HSDSCommandError) as excinfo:
        transport.run_hsget("hdf5://private/1/master.h5", tmp_path / "master.h5")
    error = excinfo.value
    assert error.exit_code == 13
    assert "ACL" in str(error)
    assert "permission denied" in str(error)


def test_hsload_uses_local_size(monkeypatch, tmp_path, caplog):
    source = tmp_path / "ids.h5"
    source.write_bytes(b"123456")
    monkeypatch.setattr(transport.shutil, "which", lambda command: f"/tools/{command}")
    monkeypatch.setattr(
        transport.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    remote = "hdf5://public/1/ids.h5"
    with caplog.at_level("INFO", logger=transport.__name__):
        assert transport.run_hsload(source, remote) == remote
    assert "total_size=6" in caplog.text


class _LocalRemoteAPI:
    def __init__(self, path):
        self.path = path

    def File(self, uri, mode):
        assert uri.startswith("hdf5://")
        return transport.h5py.File(self.path, mode)


def _verification_file(path: Path, *, shape=(2, 3), link_target="/equilibrium"):
    with transport.h5py.File(path, "w") as handle:
        handle.attrs["HDF5_BACKEND_VERSION"] = "1.0"
        handle.create_dataset(
            "equilibrium/psi",
            data=transport.np.arange(6).reshape(shape),
            chunks=True,
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        handle["linked"] = transport.h5py.ExternalLink("equilibrium.h5", link_target)


def test_verify_uploaded_image_checks_datasets_filters_version_and_links(tmp_path):
    local = tmp_path / "local.h5"
    remote = tmp_path / "remote.h5"
    _verification_file(local)
    _verification_file(remote)
    transport.verify_uploaded_image(
        local,
        "hdf5://public/1/equilibrium.h5",
        h5pyd_module=_LocalRemoteAPI(remote),
    )


def test_verify_uploaded_image_reports_mismatch(tmp_path):
    local = tmp_path / "local.h5"
    remote = tmp_path / "remote.h5"
    _verification_file(local)
    _verification_file(remote, shape=(3, 2), link_target="/wrong")
    with pytest.raises(transport.HSDSTransportVerificationError) as excinfo:
        transport.verify_uploaded_image(
            local,
            "hdf5://public/1/equilibrium.h5",
            h5pyd_module=_LocalRemoteAPI(remote),
        )
    assert "shape equilibrium/psi" in str(excinfo.value)
    assert "external link target linked" in str(excinfo.value)
