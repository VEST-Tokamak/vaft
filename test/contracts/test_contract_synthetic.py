import unittest
from unittest.mock import patch

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

        # barometry stores time per gauge (`gauge.0.pressure.time`) and never
        # writes a root `barometry.time`, so the DD requires 0 here -- unlike
        # tf and thomson_scattering below, which do write a root `.time`.
        self.assertEqual(get_path(payload, "barometry.ids_properties.homogeneous_time"), 0)
        self.assertEqual(get_path(payload, "barometry.gauge.0.name"), "PKR-251 Main Gauge")
        self.assertEqual(get_path(payload, "tf.ids_properties.homogeneous_time"), 1)
        self.assertEqual(get_path(payload, "tf.r0"), 0.4)
        self.assertEqual(get_path(payload, "thomson_scattering.ids_properties.homogeneous_time"), 1)
        self.assertEqual(get_path(payload, "thomson_scattering.channel.0.position.r"), 0.475)
        self.assertEqual(get_path(payload, "thomson_scattering.channel.0.name"), "Polychrometer 1R1")

    @patch("vaft.machine_mapping.pf_active._safe_vest_load", return_value=None)
    def test_pf_active_builder_rejects_missing_raw_data(self, _load):
        from vaft.database.raw import RawSignalUnavailableError
        from vaft.machine_mapping import vfit_pf_active_for_shot

        payload = {}
        with self.assertRaisesRegex(RawSignalUnavailableError, "shot 41672, field 5"):
            vfit_pf_active_for_shot(payload, shot=41672, tstart=0.24, tend=0.34, dt=4e-5)

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

    def test_dataset_description_preserves_source_type_and_description_options(self):
        from vaft.machine_mapping.dataset_description import dataset_description

        payload = {}
        dataset_description(
            payload,
            39915,
            {"source_type": "shot", "description": "raw fixture", "run": 2},
        )

        self.assertEqual(get_path(payload, "dataset_description.data_entry.pulse_type"), "shot")
        self.assertEqual(get_path(payload, "dataset_description.ids_properties.comment"), "raw fixture")

    def test_canonical_minimal_fixture_satisfies_all_contracts(self):
        payload = canonical_minimal_fixture()
        failures = validate_contract(payload, CANONICAL_IDS_SPECS)
        self.assertNoContractFailures(failures)


if __name__ == "__main__":
    unittest.main()
