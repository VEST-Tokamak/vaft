import unittest

from fixtures import legacy_magnetics_fixture
from helpers import path_exists, validate_contract
from spec import CANONICAL_IDS_SPECS, LEGACY_PATHS


class LegacyRejectionTests(unittest.TestCase):
    def test_legacy_paths_exist_in_legacy_fixture(self):
        payload = legacy_magnetics_fixture()
        for path in LEGACY_PATHS:
            self.assertTrue(path_exists(payload, path), path)

    def test_legacy_magnetics_shape_fails_canonical_contract(self):
        payload = legacy_magnetics_fixture()
        failures = validate_contract(
            payload,
            CANONICAL_IDS_SPECS,
            ids_names=("magnetics",),
            strict_values=False,
        )

        self.assertIn("magnetics", failures)
        message = "\n".join(failures["magnetics"])
        self.assertIn("magnetics.ip.0.data", message)
        self.assertIn("magnetics.flux_loop.0.flux.data", message)
        self.assertIn("magnetics.b_field_pol_probe.0.field.data", message)


if __name__ == "__main__":
    unittest.main()
