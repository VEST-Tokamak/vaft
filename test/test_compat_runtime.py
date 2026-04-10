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


if __name__ == "__main__":
    unittest.main()
