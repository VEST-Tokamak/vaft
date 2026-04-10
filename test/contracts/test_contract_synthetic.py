import unittest

from fixtures import canonical_minimal_fixture
from helpers import format_failures, get_path, validate_contract
from spec import CANONICAL_IDS_SPECS


class SyntheticContractTests(unittest.TestCase):
    def assertNoContractFailures(self, failures):
        self.assertEqual(failures, {}, format_failures(failures))

    def test_import_vaft_is_lightweight(self):
        import vaft

        self.assertTrue(hasattr(vaft, "__version__"))

    def test_import_machine_mapping_namespace_exposes_canonical_builders(self):
        from vaft import machine_mapping

        self.assertTrue(hasattr(machine_mapping, "vfit_dataset_description"))
        self.assertTrue(hasattr(machine_mapping, "vfit_pf_active_for_shot"))
        self.assertTrue(hasattr(machine_mapping, "vfit_tf_static"))
        self.assertFalse(hasattr(machine_mapping, "builders"))

    def test_static_diagnostics_builders_write_expected_plain_dict_paths(self):
        from vaft.machine_mapping import (
            vfit_barometry_static,
            vfit_tf_static,
            vfit_thomson_scattering_static,
        )

        payload = {}
        vfit_barometry_static(payload)
        vfit_tf_static(payload)
        vfit_thomson_scattering_static(payload)

        self.assertEqual(get_path(payload, "barometry.ids_properties.homogeneous_time"), 1)
        self.assertEqual(get_path(payload, "barometry.gauge.0.name"), "PKR-251 Main Gauge")
        self.assertEqual(get_path(payload, "tf.ids_properties.homogeneous_time"), 1)
        self.assertEqual(get_path(payload, "tf.r0"), 0.4)
        self.assertEqual(get_path(payload, "thomson_scattering.ids_properties.homogeneous_time"), 1)
        self.assertEqual(get_path(payload, "thomson_scattering.channel.0.position.r"), 0.475)
        self.assertEqual(get_path(payload, "thomson_scattering.channel.0.name"), "Polychrometer 1R1")

    def test_pf_active_builder_populates_full_contract_offline(self):
        from vaft.machine_mapping import vfit_pf_active_for_shot

        payload = {}
        vfit_pf_active_for_shot(payload, shot=41672, tstart=0.24, tend=0.34, dt=4e-5)

        failures = validate_contract(
            payload,
            CANONICAL_IDS_SPECS,
            ids_names=("pf_active",),
        )
        self.assertNoContractFailures(failures)
        self.assertEqual(len(get_path(payload, "pf_active.coil")), 10)
        self.assertTrue(len(get_path(payload, "pf_active.coil.0.element")) > 0)
        self.assertEqual(
            len(get_path(payload, "pf_active.coil.0.current.time")),
            len(get_path(payload, "pf_active.time")),
        )

    def test_dataset_description_builder_populates_contract_on_plain_dict(self):
        from vaft.machine_mapping import vfit_dataset_description

        payload = {}
        vfit_dataset_description(payload, shot=39915, run=1, user="tester")

        failures = validate_contract(
            payload,
            CANONICAL_IDS_SPECS,
            ids_names=("dataset_description",),
        )
        self.assertNoContractFailures(failures)
        self.assertEqual(get_path(payload, "dataset_description.data_entry.machine"), "VEST")
        self.assertEqual(get_path(payload, "dataset_description.data_entry.pulse"), 39915)
        self.assertEqual(get_path(payload, "dataset_description.data_entry.run"), 1)
        self.assertEqual(get_path(payload, "dataset_description.data_entry.user"), "tester")

    def test_canonical_minimal_fixture_satisfies_all_contracts(self):
        payload = canonical_minimal_fixture()
        failures = validate_contract(payload, CANONICAL_IDS_SPECS)
        self.assertNoContractFailures(failures)


if __name__ == "__main__":
    unittest.main()
