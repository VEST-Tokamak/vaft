import unittest

from helpers import get_path, validate_contract, format_failures
from spec import CANONICAL_IDS_SPECS


class MagneticsModelTests(unittest.TestCase):
    def assertNoContractFailures(self, failures):
        self.assertEqual(failures, {}, format_failures(failures))

    def test_vfit_magnetics_static_populates_yaml_geometry(self):
        from vaft.machine_mapping import vfit_magnetics_static

        payload = {}
        vfit_magnetics_static(payload)

        self.assertEqual(get_path(payload, "magnetics.ids_properties.homogeneous_time"), 1)
        self.assertEqual(len(get_path(payload, "magnetics.flux_loop")), 11)
        self.assertEqual(len(get_path(payload, "magnetics.b_field_pol_probe")), 64)
        self.assertEqual(get_path(payload, "magnetics.flux_loop.0.position.0.r"), 0.592)
        self.assertEqual(get_path(payload, "magnetics.flux_loop.0.position.0.z"), 0.685)
        self.assertEqual(get_path(payload, "magnetics.b_field_pol_probe.0.position.r"), 0.089)

    def test_vfit_magnetics_for_shot_satisfies_contract_offline(self):
        from vaft.machine_mapping import vfit_magnetics_for_shot

        payload = {}
        vfit_magnetics_for_shot(payload, shot=41672, tstart=0.24, tend=0.34, dt=4e-5)

        failures = validate_contract(
            payload,
            CANONICAL_IDS_SPECS,
            ids_names=("magnetics",),
        )
        self.assertNoContractFailures(failures)
        self.assertEqual(len(get_path(payload, "magnetics.flux_loop")), 11)
        self.assertEqual(len(get_path(payload, "magnetics.b_field_pol_probe")), 64)


if __name__ == "__main__":
    unittest.main()
