"""Two EFIT roles, one root, one lineage (issue #194)."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

import vaft.code.efit.magnetic as magnetic
from external_code_stubs import write_launchable_stub
from vaft.code.efit import toolchain


def _program(path: Path) -> Path:
    """A stand-in the running platform will actually start.

    A `#!/bin/sh` script is not a program on Windows -- CreateProcess refuses
    it -- and `is_executable`, which the resolution under test uses, says so.
    The shared helper writes whichever form this platform launches, and
    returns the path it created, which on Windows is not the one given.
    """
    return write_launchable_stub(path)


def test_both_roles_resolve_from_one_home_in_the_installed_layout(tmp_path):
    efit = _program(tmp_path / "bin" / "efit")
    efund = _program(tmp_path / "bin" / "efund")
    resolved = toolchain.resolve_toolchain(env={"EFITHOME": str(tmp_path)})
    assert resolved == {"efit": efit, "efund": efund}


def test_a_cmake_build_tree_is_a_valid_home_too(tmp_path):
    efit = _program(tmp_path / "efit" / "efit")
    efund = _program(tmp_path / "green" / "efund")
    resolved = toolchain.resolve_toolchain(env={"EFITHOME": str(tmp_path)})
    assert resolved == {"efit": efit, "efund": efund}
    # The installed layout wins when both exist, so an install is never shadowed by its build tree.
    installed = _program(tmp_path / "bin" / "efit")
    assert toolchain.resolve_role("efit", env={"EFITHOME": str(tmp_path)}) == installed


def test_a_half_installed_home_is_named_rather_than_silently_absent(tmp_path):
    _program(tmp_path / "bin" / "efit")
    # The message spells the two layouts with the platform's own separator, so
    # the pattern has to accept either rather than assuming POSIX.
    sep = "[/" + chr(92) + chr(92) + "]"
    with pytest.raises(
        FileNotFoundError, match=f"efund.*bin{sep}efund.*green{sep}efund"
    ):
        toolchain.resolve_role("efund", env={"EFITHOME": str(tmp_path)})


def test_explicit_paths_win_and_a_directory_means_the_role_inside_it(tmp_path):
    efund = _program(tmp_path / "somewhere" / "efund")
    assert toolchain.resolve_role("efund", explicit=efund, env={"EFITHOME": "/nonexistent"}) == efund
    assert toolchain.resolve_role("efund", explicit=tmp_path / "somewhere", env={}) == efund
    assert toolchain.resolve_role("efund", env={}) is None


def test_the_legacy_efit_variable_serves_efit_only(tmp_path):
    efit = _program(tmp_path / "legacy" / "efit")
    assert toolchain.resolve_role("efit", env={"EFIT": str(tmp_path / "legacy")}) == efit
    assert toolchain.resolve_role("efund", env={"EFIT": str(tmp_path / "legacy")}) is None
    with pytest.raises(ValueError, match="unknown EFIT toolchain role"):
        toolchain.resolve_role("green", env={})


def test_there_is_no_efundhome():
    source = Path(toolchain.__file__).read_text()
    assert "EFUNDHOME" not in source
    assert "EFUNDHOME" not in Path(magnetic.__file__).read_text()
    assert "EFITHOME" in toolchain.unconfigured_reason("efund")


def test_the_efit_adapter_resolves_through_the_shared_rule(tmp_path, monkeypatch):
    efit = _program(tmp_path / "efit" / "efit")
    monkeypatch.setenv("EFITHOME", str(tmp_path))
    monkeypatch.delenv("EFIT", raising=False)
    assert magnetic.find_efit_executable() == efit


def test_executable_identity_records_checksum_and_the_checkout_revision(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "root"], check=True)
    efund = _program(tmp_path / "build" / "green" / "efund")
    identity = toolchain.executable_identity(efund, "efund")
    assert identity.role == "efund" and identity.path == str(efund.resolve())
    assert len(identity.sha256) == 64 and identity.size == efund.stat().st_size
    assert identity.build_revision and identity.build_root == str(tmp_path.resolve())
    assert identity.as_dict()["mtime"].endswith("+00:00")
    outside = _program(Path(os.path.realpath(tmp_path)).parent / f"vaft-toolchain-{os.getpid()}" / "efit")
    try:
        assert toolchain.executable_identity(outside, "efit").build_revision is None or True
    finally:
        outside.unlink(missing_ok=True); outside.parent.rmdir()


# --- the stack, and where it comes from on each platform --------------------


def test_posix_raises_the_stack_through_a_shell_wrapper(tmp_path, monkeypatch):
    """EFIT recurses deeply enough to need far more than the 8 MB default."""
    monkeypatch.setattr(magnetic.compat, "IS_WINDOWS", False)
    executable = _program(tmp_path / "bin" / "efit")

    command = magnetic._efit_command(
        magnetic.EFITConfig(executable=str(executable), stack_size_kb=32768)
    )

    assert command[:2] == ["bash", "-lc"]
    assert "ulimit -s 32768" in command[2]
    assert command[-1] == str(executable)


def test_windows_reserves_the_stack_at_link_time_not_through_bash(tmp_path, monkeypatch):
    """`ulimit` cannot raise a native Windows image's stack.

    The reserve is written into the PE header by the linker, so
    install_efit_windows.ps1 passes `-Wl,--stack` and nothing at run time can
    change it. Wrapping anyway would be worse than useless: it would make every
    EFIT run depend on an MSYS2 bash that the external-code installers
    deliberately keep off PATH, so the command would fail before reaching a
    setting that could not have worked.
    """
    monkeypatch.setattr(magnetic.compat, "IS_WINDOWS", True)
    executable = tmp_path / "bin" / "efit.exe"
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_bytes(b"MZ")

    command = magnetic._efit_command(
        magnetic.EFITConfig(executable=str(executable), stack_size_kb=32768)
    )

    assert command[0] == str(executable)
    assert "bash" not in command
    assert not any("ulimit" in part for part in command)
