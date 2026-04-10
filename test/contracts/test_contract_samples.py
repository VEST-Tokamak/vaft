import json
import unittest
from pathlib import Path

from helpers import format_failures, validate_contract
from spec import CANONICAL_IDS_SPECS, SAMPLE_FILE_IDS


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "vaft" / "data"


def load_json(name: str):
    with open(DATA_DIR / name, "r", encoding="utf-8") as handle:
        return json.load(handle)


class SampleContractTests(unittest.TestCase):
    def assertNoContractFailures(self, failures):
        self.assertEqual(failures, {}, format_failures(failures))

    def test_core_sample_smoke_on_41672(self):
        payload = load_json("41672.json")
        failures = validate_contract(
            payload,
            CANONICAL_IDS_SPECS,
            ids_names=SAMPLE_FILE_IDS["41672.json"],
            strict_values=False,
        )
        self.assertNoContractFailures(failures)

    def test_spectrometer_sample_smoke_on_39915(self):
        payload = load_json("39915.json")
        failures = validate_contract(
            payload,
            CANONICAL_IDS_SPECS,
            ids_names=SAMPLE_FILE_IDS["39915.json"],
            strict_values=False,
        )
        self.assertNoContractFailures(failures)

    def test_thomson_scattering_sample_smoke(self):
        payload = load_json("thomson_scattering.json")
        failures = validate_contract(
            payload,
            CANONICAL_IDS_SPECS,
            ids_names=SAMPLE_FILE_IDS["thomson_scattering.json"],
            strict_values=False,
        )
        self.assertNoContractFailures(failures)

    def test_charge_exchange_single_sample_smoke(self):
        payload = load_json("vfit_ion_doppler_single.json")
        failures = validate_contract(
            payload,
            CANONICAL_IDS_SPECS,
            ids_names=SAMPLE_FILE_IDS["vfit_ion_doppler_single.json"],
            strict_values=False,
        )
        self.assertNoContractFailures(failures)

    def test_charge_exchange_profile_sample_smoke(self):
        payload = load_json("vfit_ion_doppler_profile.json")
        failures = validate_contract(
            payload,
            CANONICAL_IDS_SPECS,
            ids_names=SAMPLE_FILE_IDS["vfit_ion_doppler_profile.json"],
            strict_values=False,
        )
        self.assertNoContractFailures(failures)


if __name__ == "__main__":
    unittest.main()
