import os
import subprocess
import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
