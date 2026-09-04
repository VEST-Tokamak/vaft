import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch
from pathlib import Path

import pytest
import numpy as np
from scipy import integrate

from vaft import compat


class CompatRuntimeTests(unittest.TestCase):
    def test_trapz_compat_matches_numpy_reference(self):
        x = np.linspace(0.0, 1.0, 11)
        y = x**2
        expected = np.trapezoid(y, x=x) if hasattr(np, "trapezoid") else np.trapz(y, x=x)
        self.assertAlmostEqual(compat.trapz_compat(y, x=x), expected)

    def test_trapz_compat_falls_back_to_np_trapz(self):
        x = np.linspace(0.0, 1.0, 5)
        y = x
        with patch.object(np, "trapezoid", None):
            if hasattr(np, "trapz"):
                expected = np.trapz(y, x=x)
            else:
                expected = integrate.trapezoid(y, x=x)
            self.assertAlmostEqual(compat.trapz_compat(y, x=x), expected)

    def test_cumtrapz_compat_falls_back_to_legacy_symbol(self):
        y = np.array([0.0, 1.0, 4.0, 9.0])
        x = np.array([0.0, 1.0, 2.0, 3.0])
        modern = integrate.cumulative_trapezoid

        def _legacy_impl(values, x=None, dx=1.0, axis=-1, initial=0.0):
            # Delegate to the modern implementation to keep numerical expectation stable.
            return modern(values, x=x, dx=dx, axis=axis, initial=initial)

        with (
            patch.object(integrate, "cumulative_trapezoid", None),
            patch.object(integrate, "cumtrapz", _legacy_impl, create=True),
        ):
            result = compat.cumtrapz_compat(y, x=x, initial=0.0)

        expected = integrate.cumulative_trapezoid(y, x=x, initial=0.0)
        np.testing.assert_allclose(result, expected)

    def test_runtime_patch_is_idempotent(self):
        compat._RUNTIME_PATCH_APPLIED = False
        compat.apply_runtime_compat_patches()
        self.assertTrue(compat._RUNTIME_PATCH_APPLIED)
        compat.apply_runtime_compat_patches()
        self.assertTrue(compat._RUNTIME_PATCH_APPLIED)

    def test_process_magnetics_uses_compat_helper_for_cumulative_integral(self):
        magnetics_source = Path(__file__).resolve().parents[1] / "vaft" / "process" / "magnetics.py"
        text = magnetics_source.read_text(encoding="utf-8")
        self.assertNotIn("integrate.cumtrapz(", text)
        self.assertIn("cumtrapz_compat(", text)

    def test_home_is_exported_when_windows_leaves_it_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HOME", None)
            with patch.object(compat, "IS_WINDOWS", True):
                resolved = compat.ensure_home_environment()
            self.assertEqual(resolved, str(Path.home()))
            self.assertEqual(os.environ["HOME"], str(Path.home()))

    def test_existing_home_is_never_overwritten(self):
        with patch.dict(os.environ, {"HOME": ""}, clear=False):
            with patch.object(compat, "IS_WINDOWS", True):
                self.assertEqual(compat.ensure_home_environment(), "")
            self.assertEqual(os.environ["HOME"], "")

    @unittest.skipUnless(os.name == "nt", "native Windows import regression")
    def test_omas_import_succeeds_without_home_on_windows(self):
        environment = os.environ.copy()
        environment.pop("HOME", None)
        result = subprocess.run(
            [sys.executable, "-c", "import vaft.imas.omas_imas; assert __import__('os').environ['HOME']"],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_user_home_matches_pathlib(self):
        self.assertEqual(compat.user_home(), Path.home())

    def test_temporary_file_can_be_reopened_by_name(self):
        """NamedTemporaryFile cannot do this on Windows; the helper must."""
        with compat.reopenable_temporary_file(suffix=".json") as path:
            path.write_text('{"ok": true}', encoding="utf-8")
            # A second, independent open by name -- what ODS.load()/save() do.
            with open(path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), '{"ok": true}')
        self.assertFalse(path.exists())

    def test_temporary_file_survives_a_consumer_that_kept_it_open(self):
        """The helper must absorb the sharing violation, not raise it.

        h5py and imas_core both hold a file they were handed by name, so a raw
        TemporaryDirectory here would fail with WinError 32 on the way out.
        """
        held = None
        try:
            with compat.reopenable_temporary_file(suffix=".h5") as path:
                path.write_bytes(b"payload")
                held = open(path, "rb")
        finally:
            if held is not None:
                held.close()

    def test_remove_directory_treats_a_concurrent_delete_as_success(self):
        """A sibling stage or the temp reaper may win the race; that is fine."""
        scratch = Path(tempfile.mkdtemp(prefix="vaft-test-race-"))

        def vanish(*_args, **_kwargs):
            raise FileNotFoundError(2, "no such directory")

        with patch.object(compat, "IS_WINDOWS", False):
            with patch.object(compat.shutil, "rmtree", vanish):
                self.assertTrue(compat.remove_directory(scratch))
        shutil.rmtree(scratch, ignore_errors=True)

    def test_remove_directory_reports_success(self):
        scratch = Path(tempfile.mkdtemp(prefix="vaft-test-"))
        (scratch / "payload.txt").write_text("x", encoding="utf-8")
        self.assertTrue(compat.remove_directory(scratch))
        self.assertFalse(scratch.exists())

    def test_remove_directory_tolerates_a_missing_tree(self):
        missing = Path(tempfile.gettempdir()) / "vaft-test-does-not-exist"
        self.assertTrue(compat.remove_directory(missing))
        self.assertFalse(compat.remove_directory(missing, missing_ok=False))

    def test_remove_directory_does_not_raise_on_a_locked_file(self):
        """imas_core keeps large IDS files open after DBEntry.close().

        On Windows that makes the tree unremovable. Reclaiming scratch space is
        best-effort, so this must report failure rather than raise.
        """
        scratch = Path(tempfile.mkdtemp(prefix="vaft-test-locked-"))
        (scratch / "held.bin").write_bytes(b"x")

        def refuse(*_args, **kwargs):
            if kwargs.get("ignore_errors"):
                return None
            raise PermissionError(32, "file in use")

        with patch.object(compat, "IS_WINDOWS", True):
            with patch.object(compat.shutil, "rmtree", refuse):
                self.assertFalse(compat.remove_directory(scratch))
        self.assertTrue(scratch.exists())
        shutil.rmtree(scratch)

    def test_remove_directory_still_raises_on_posix(self):
        """A POSIX failure is a real bug, not a platform quirk."""
        scratch = Path(tempfile.mkdtemp(prefix="vaft-test-posix-"))
        try:
            def explode(*_args, **_kwargs):
                raise PermissionError(13, "denied")

            with patch.object(compat, "IS_WINDOWS", False):
                with patch.object(compat.shutil, "rmtree", explode):
                    with self.assertRaises(PermissionError):
                        compat.remove_directory(scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_temporary_directory_never_raises_on_cleanup(self):
        def refuse(*_args, **kwargs):
            if kwargs.get("ignore_errors"):
                return None
            raise PermissionError(32, "file in use")

        with patch.object(compat, "IS_WINDOWS", True):
            with patch.object(compat.shutil, "rmtree", refuse):
                with compat.temporary_directory(prefix="vaft-test-") as scratch:
                    (scratch / "held.bin").write_bytes(b"x")
        shutil.rmtree(scratch, ignore_errors=True)

    def test_cleanup_failure_never_replaces_the_body_exception(self):
        """Cleanup runs in a finally, so it must not mask the real error."""
        def explode(*_args, **_kwargs):
            raise PermissionError(13, "denied")

        with patch.object(compat, "IS_WINDOWS", False):
            with patch.object(compat.shutil, "rmtree", explode):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with self.assertRaises(ValueError) as raised:
                        with compat.temporary_directory(prefix="vaft-test-") as scratch:
                            raise ValueError("the caller's own failure")
        self.assertEqual(str(raised.exception), "the caller's own failure")
        self.assertTrue(any(w.category is RuntimeWarning for w in caught))
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# What this platform can launch
#
# The Windows behaviour is exercised on every platform by patching
# `compat.IS_WINDOWS`. That is deliberate: `executable_suffixes()` reads the
# flag at call time precisely so the Linux CI leg covers the Windows rules
# too, rather than leaving them to the one runner that happens to be Windows.
# ---------------------------------------------------------------------------


POSIX_SCRIPT = """#!/bin/sh
exit 0
"""

# The first two bytes of every Windows PE, which is what `is_executable` reads
# when the filename suffix says nothing.
PE_HEADER = bytes([0x4D, 0x5A]) + bytes(64)


@pytest.fixture()
def windows(monkeypatch):
    monkeypatch.setattr(compat, "IS_WINDOWS", True)


@pytest.fixture()
def posix(monkeypatch):
    monkeypatch.setattr(compat, "IS_WINDOWS", False)


# --- candidates ------------------------------------------------------------


def test_posix_asks_for_exactly_the_documented_name(posix):
    """No suffix may ever leak into a POSIX lookup."""
    assert compat.executable_suffixes() == ()
    assert compat.executable_candidates(Path("bin/dcon")) == (Path("bin/dcon"),)


def test_windows_prefers_the_native_build_over_a_bare_name(windows):
    candidates = compat.executable_candidates(Path("bin/dcon"))

    assert [path.name for path in candidates] == [
        "dcon.exe",
        "dcon.bat",
        "dcon.cmd",
        "dcon",
    ]


def test_a_name_that_already_names_its_kind_is_not_extended(windows):
    assert compat.executable_candidates(Path("bin/dcon.exe")) == (Path("bin/dcon.exe"),)


def test_a_dotted_version_is_a_name_and_not_a_suffix(windows):
    """`with_suffix` would turn a pinned `chease-3.1` into `chease.exe`."""
    names = [path.name for path in compat.executable_candidates(Path("chease-3.1"))]

    assert names[0] == "chease-3.1.exe"
    assert "chease.exe" not in names


# --- the probe -------------------------------------------------------------


def test_windows_rejects_a_shell_script_that_os_access_would_accept(windows, tmp_path):
    """The whole reason this helper exists.

    ``os.access(path, os.X_OK)`` is true for every readable file on Windows, so
    a POSIX build dropped into a Windows installation used to pass every check
    VAFT made and fail inside CreateProcess as an opaque WinError 193.
    """
    script = tmp_path / "dcon"
    script.write_text(POSIX_SCRIPT, encoding="utf-8")

    assert compat.is_executable(script) is False
    assert os.access(script, os.X_OK) or not compat.IS_WINDOWS


def test_windows_accepts_a_portable_executable(windows, tmp_path):
    binary = tmp_path / "dcon"
    binary.write_bytes(PE_HEADER)

    assert compat.is_executable(binary) is True


def test_windows_accepts_a_launchable_suffix_on_its_name_alone(windows, tmp_path):
    wrapper = tmp_path / "dcon.cmd"
    wrapper.write_text("@echo off", encoding="ascii")

    assert compat.is_executable(wrapper) is True


def test_a_directory_is_never_executable(windows, tmp_path):
    """POSIX marks directories executable, and `subprocess.run` cannot start one."""
    assert compat.is_executable(tmp_path) is False


def test_a_missing_file_is_never_executable(windows, tmp_path):
    assert compat.is_executable(tmp_path / "absent") is False


@pytest.mark.skipif(os.name == "nt", reason="Windows has no POSIX execute bit to clear")
def test_posix_still_answers_with_the_execute_bit(posix, tmp_path):
    script = tmp_path / "dcon"
    script.write_text(POSIX_SCRIPT, encoding="utf-8")
    script.chmod(0o644)
    assert compat.is_executable(script) is False

    script.chmod(0o755)
    assert compat.is_executable(script) is True


# --- resolution ------------------------------------------------------------


def test_the_documented_posix_name_finds_the_native_windows_build(windows, tmp_path):
    (tmp_path / "bin").mkdir()
    native = tmp_path / "bin/dcon.exe"
    native.write_bytes(PE_HEADER)

    assert compat.resolve_executable(tmp_path / "bin/dcon") == native


def test_a_native_build_wins_over_a_stray_extensionless_file(windows, tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/dcon").write_text(POSIX_SCRIPT, encoding="utf-8")
    native = tmp_path / "bin/dcon.exe"
    native.write_bytes(PE_HEADER)

    assert compat.resolve_executable(tmp_path / "bin/dcon") == native


def test_resolution_reports_nothing_rather_than_guessing(windows, tmp_path):
    assert compat.resolve_executable(tmp_path / "bin/dcon") is None


if __name__ == "__main__":
    unittest.main()
