import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
